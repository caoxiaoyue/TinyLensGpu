"""Shared provenance and handoff export for the J0946 macro drivers."""

import hashlib
import json

import jax.numpy as jnp
import numpy as np

from TinyLensGpu.example_support import (
    pixelized_objective_provenance,
    require_valid_reconstruction,
    source_axes_from_bbox,
    resolve_source_interpolation,
    resolve_stage_source_interpolation,
    source_interpolation_metadata,
    stage_from_payload,
)
from TinyLensGpu.example_support.pixelized_stage_cache import matching_pixelized_checkpoint


_STAGE_TAGS = ("a", "m0", "m1", "m2", "m3")
_SAMPLER_TAGS = ("a", "m1", "m2", "m3")


def resolve_run_source_interpolation(output_dir, requested, *, resume):
    """Check the manifest, sampler checkpoints, handoff, and saved stages."""
    if resume:
        manifest = output_dir / "manifest.json"
        if manifest.exists():
            requested = resolve_source_interpolation(
                requested, metadata=json.loads(manifest.read_text())["settings"],
                context=str(manifest),
            )
        for tag in _SAMPLER_TAGS:
            path = output_dir / f"sampler_{tag}.json"
            if path.exists():
                requested = resolve_source_interpolation(
                    requested, metadata=json.loads(path.read_text()), context=str(path),
                )
            elif (output_dir / f"sampler_{tag}.h5").exists():
                raise RuntimeError(f"Sampler checkpoint has no matching settings: {path}")
        handoff = output_dir / "m3_handoff.json"
        if handoff.exists():
            requested = resolve_source_interpolation(
                requested, metadata=json.loads(handoff.read_text()), context=str(handoff),
            )
    return resolve_stage_source_interpolation(
        output_dir, _STAGE_TAGS, requested=requested, resume=resume,
    )


def source_settings_match(saved, current, path):
    """Compare complete settings with the legacy bilinear default applied."""
    mode = resolve_source_interpolation(
        current["source_interpolation"], metadata=saved, context=str(path),
    )
    # Compare old settings in memory without rewriting their original metadata.
    return {**saved, **source_interpolation_metadata(mode)} == current


def require_sampler_objective(output_dir, tag, objective):
    """Check an existing sampler sidecar before reusing a scored stage."""
    path = output_dir / f"sampler_{tag}.json"
    if path.exists() and json.loads(path.read_text()).get("objective") != objective["fingerprint"]:
        raise RuntimeError(f"Incompatible existing sampler: {path}. Start a fresh run.")


def restore_adaptive_stage(output_dir, tag, cached, objective):
    """Restore the stage after validating its fit and sampler objectives."""
    cached = matching_pixelized_checkpoint(cached, objective, tag)
    if cached is None:
        return None
    require_sampler_objective(output_dir, tag, objective)
    return stage_from_payload(cached), cached["extra"]["medians"]


def require_current_handoff_schema(output_dir):
    """Keep completed historical exports available to fixed-scale consumers."""
    path = output_dir / "m3_handoff.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("complete") and "objective" not in record:
            raise RuntimeError("This is an archived M3 handoff. Start a fresh run in a new output directory.")


def write_macro_handoff(
    output_dir, likelihood, stage, reference, image, lens_light, *, fitted_objective=None,
):
    """Export the fitted fields and precision from one valid M3 evaluation."""
    names = stage.param_names
    posterior_path = output_dir / "posterior_m3.npz"
    with np.load(posterior_path) as posterior:
        samples = posterior["samples"]
        scores = posterior["log_likelihood"]
        if (
            posterior["param_names"].tolist() != names
            or not np.array_equal(samples, stage.samples)
            or scores.shape != (len(samples),)
        ):
            raise ValueError("The retained draws do not match the saved M3 stage.")
    valid = np.isfinite(scores) & (scores > -1e9) & np.isfinite(samples).all(axis=1)
    if not np.any(valid):
        raise ValueError("The M3 posterior contains no valid scored draw.")
    best_index = int(np.argmax(np.where(valid, scores, -np.inf)))

    fitted = fitted_objective or {}
    objective = pixelized_objective_provenance(
        likelihood, reference=reference, sampling=fitted.get("sampling"),
        upstream=fitted.get("upstream"),
    )
    if fitted_objective is not None and objective != fitted_objective:
        raise ValueError("M3 handoff configuration differs from the fitted objective.")
    if names != [param.name for param in likelihood.get_dynamic_params()]:
        raise ValueError("The retained parameter order differs from the M3 model.")
    posterior_hash = hashlib.sha256(posterior_path.read_bytes()).hexdigest()
    archive = output_dir / "m3_handoff.npz"
    metadata = output_dir / "m3_handoff.json"
    if fitted_objective is not None and metadata.exists() and archive.exists():
        saved = json.loads(metadata.read_text())
        if (
            saved.get("complete") and saved.get("objective") == objective
            and saved.get("evidence_logdet") == objective["evidence_logdet"]
            and saved.get("selected_sample_index") == best_index
            and saved.get("source_scale_provenance") == "selected_M3_native_reference_projection"
            and saved.get("retained_posterior_sha256") == posterior_hash
            and saved.get("handoff_sha256") == hashlib.sha256(archive.read_bytes()).hexdigest()
        ):
            return False

    theta = jnp.asarray(samples[best_index])
    result, diagnostics = likelihood.evaluate_with_diagnostics(theta)
    require_valid_reconstruction(result, diagnostics, context="exporting the M3 handoff")
    if not diagnostics.evidence_valid:
        raise ValueError(f"M3 handoff score failed: {diagnostics.evidence_invalid_reason}")
    score = float(result.log_evidence)
    if not np.isclose(score, scores[best_index], rtol=1e-6, atol=1e-3):
        raise ValueError("The selected M3 score differs from the retained score.")

    def float32(value):
        return np.asarray(value, dtype=np.float32)

    source = likelihood.source_model
    n = int(source.n)
    scale = float32(diagnostics.regularization_scale).ravel()
    if scale.shape != (n**2,) or not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("M3 source_scale must be finite, positive and match the source grid.")
    if diagnostics.evidence_logdet != objective["evidence_logdet"]:
        raise ValueError("The selected M3 score used a different determinant method.")
    bbox = float32(result.source_bbox)
    x_axis, y_axis = source_axes_from_bbox(bbox, n)
    mask = ~np.asarray(likelihood.unmask, dtype=bool)
    np.savez_compressed(
        archive,
        image_data=float32(image),
        noise_map=float32(likelihood.noise_map),
        psf_kernel=float32(likelihood.sim_obj.sim_config.psf_kernel),
        lens_light_model=float32(lens_light),
        lens_subtracted_image=float32(likelihood.image_data),
        mask=mask,
        source_values=float32(result.source_pixels),
        source_bbox=bbox,
        source_x_axis=float32(x_axis),
        source_y_axis=float32(y_axis),
        source_scale=scale,
        adapt_image=np.asarray(reference["adapt_image"]),
        macro_model_image=float32(result.model_image),
    )
    mode = source.interpolation
    record = dict(
        complete=True,
        **source_interpolation_metadata(mode),
        representative="maximum_likelihood_retained_M3_sample",
        selected_sample_index=best_index,
        best_sample_log_likelihood=score,
        retained_sample_log_likelihood=float(scores[best_index]),
        retained_posterior_sha256=posterior_hash,
        macro_parameters=dict(zip(names, np.asarray(theta).tolist())),
        posterior_medians=stage.medians(),
        source_n=n,
        dpix=objective["observation"]["dpix"],
        nsub=objective["observation"]["nsub"],
        source_regularization=source.regularization_type,
        adaptive_reg_rho=diagnostics.adaptive_reg_rho,
        log_lambda_source=objective["fixed_strengths"]["log_lambda_reg"],
        lambda_source=diagnostics.lambda_reg,
        reference_provenance=reference["reference_provenance"],
        source_scale_provenance="selected_M3_native_reference_projection",
        evidence_logdet=diagnostics.evidence_logdet,
        objective=objective,
        gradient_provenance=f"{mode}_derivative_of_new_M3_reconstruction",
        macro_backend="operator",
        dtype="float32",
        n_data=int((~mask).sum()),
        macro_log_z=float(stage.log_z),
        handoff_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    return True
