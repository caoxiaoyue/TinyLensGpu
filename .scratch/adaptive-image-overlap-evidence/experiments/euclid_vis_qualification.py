"""Verify the real VIS inputs and summarize a maintained Euclid stage run."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from astropy.io import fits


CHECKOUT = Path(__file__).resolve().parents[3]
EXAMPLE = CHECKOUT / "examples/pix_src_demo_operator/pipe/euclid_examples"
STAGES = ("m0", "m1", "m2", "m3")
SEED = 73


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(source_fits):
    """Check that the maintained example uses the original VIS data."""
    metadata = json.loads(Path("metadata.json").read_text())
    assert metadata["lens_id"] == source_fits.stem and metadata["band"] == "VIS"
    inputs = {"original_fits": sha256(source_fits), "metadata.json": sha256(Path("metadata.json"))}
    with fits.open(source_fits) as hdus:
        for filename, extension in metadata["source_extensions"].items():
            path = Path("data") / filename
            values = fits.getdata(path)
            np.testing.assert_array_equal(values, hdus[extension].data)
            assert np.isfinite(values).all()
            inputs[filename] = sha256(path)
        dpix = abs(hdus["VIS_FLUX"].header["CD1_1"]) * 3600
        np.testing.assert_allclose(dpix, metadata["pixel_scale_arcsec"], rtol=0, atol=1e-12)
    mask = Path("data/mask.fits")
    assert mask.read_bytes() == source_fits.with_name("mask_extra_galaxies.fits").read_bytes()
    inputs["mask.fits"] = sha256(mask)
    return {"lens_id": metadata["lens_id"], "band": metadata["band"], "sha256": inputs}


def observation(pipeline):
    image, noise, psf, bad = pipeline.load_lens_data(
        image_path="data/image.fits",
        noise_path="data/noise.fits",
        psf_path="data/psf.fits",
        mask_path="data/mask.fits",
    )
    circle = pipeline._make_circular_mask(image.shape, pipeline.DPIX, radius_arcsec=pipeline.MASK_RADIUS)
    if bad is not None:
        circle = circle | bad
    return image, noise, psf, circle


def prepare_upstream(pipeline, output):
    """Run the maintained parametric A and L stages with a recorded seed."""
    if any((output / f"stage_{tag}.pkl").exists() for tag in ("a", "l")):
        raise FileExistsError("Use a fresh output directory for upstream preparation.")
    pipeline.OUT_DIR = output
    pipeline.Sampler = partial(pipeline.Sampler, seed=SEED)
    image, noise, psf, circle = observation(pipeline)
    _, _, lens = pipeline.run_stage_a(
        image, noise, psf, circular_mask=circle, source_interpolation="bilinear",
    )
    feature = pipeline.run_stage_b(image, noise, lens, circular_mask=circle)
    pipeline.run_stage_l(
        image, noise, psf, feature, circular_mask=circle, source_interpolation="bilinear",
    )


def evaluate_stage(model, payload, medians):
    """Record one diagnostic solve at the stage's saved reporting point."""
    import jax.numpy as jnp

    theta = jnp.asarray([medians[name] for name in payload["param_names"]])
    result, diagnostics = model.evaluate_with_diagnostics(theta, include_effective_dof=True)
    if not diagnostics.valid or not diagnostics.evidence_valid:
        raise RuntimeError(
            f"Invalid reporting point. Solve {diagnostics.invalid_reason}, "
            f"evidence {diagnostics.evidence_invalid_reason}"
        )
    assert np.isfinite(result.log_evidence)
    scale = np.asarray(diagnostics.regularization_scale)
    assert np.isfinite(scale).all() and np.all(scale > 0)
    info = result.solver_info
    return {
        "log_evidence": float(result.log_evidence),
        "chi2": diagnostics.chi2,
        "reduced_chi2": diagnostics.reduced_chi2,
        "pcg_iterations": int(info.n_iter),
        "pcg_residual_norm": float(info.residual_norm),
        "reconstruction_valid": bool(diagnostics.valid),
        "evidence_valid": bool(diagnostics.evidence_valid),
        "evidence_logdet": diagnostics.evidence_logdet,
        "position_penalty": diagnostics.position.penalty,
        "max_source_separation_arcsec": diagnostics.position.max_separation,
        "lambda_reg": diagnostics.lambda_reg,
        "adaptive_reg_rho": diagnostics.adaptive_reg_rho,
        "scale_range": [float(scale.min()), float(scale.max())],
        "source_bbox": [float(value) for value in result.source_bbox],
    }


def summarize(pipeline, output):
    """Check the saved M1 producer, M2 calibration and M3 fit together."""
    import caskade as ck
    import jax.numpy as jnp

    from TinyLensGpu.example_support import artifact_fingerprint, pixelized_observation_provenance

    pipeline.OUT_DIR = output
    image, noise, psf, circle = observation(pipeline)
    feature_path = Path("data/feature_mask.fits")
    feature = fits.getdata(feature_path).astype(bool)
    upstream_mask = output / "upstream_feature_mask.fits"
    if upstream_mask.exists():
        np.testing.assert_array_equal(feature, fits.getdata(upstream_mask).astype(bool))
    active = ~(feature | circle)
    a = pipeline._load_stage("a")
    l = pipeline._load_stage("l")
    payloads = {tag: pipeline._load_stage(tag) for tag in STAGES}
    extras = {tag: payload["extra"] for tag, payload in payloads.items()}
    objectives = {tag: extra["objective"] for tag, extra in extras.items()}
    source = extras["m1"]["s1"]
    provenance = source["reference_provenance"]
    arc = np.asarray(source["adapt_image"])
    assert arc.shape == image.shape and np.isfinite(arc).all()
    assert np.any(arc[active] > 0) and np.all(arc[~active] == 0)
    np.testing.assert_array_equal(source["source_image"], source["source_pixels"].reshape(30, 30))

    reference_id = artifact_fingerprint(provenance)
    assert provenance["producer_evidence_logdet"] == "overlap"
    assert objectives["m2"]["reference_id"] == objectives["m3"]["reference_id"] == reference_id
    assert objectives["m2"]["grid"] == objectives["m3"]["grid"]
    assert objectives["m2"]["grid"]["source_bbox"] == list(source["source_bbox"])
    assert objectives["m2"]["adaptation"] == objectives["m3"]["adaptation"]
    calibrated = extras["m2"]["reg_hyperparams"]
    assert extras["m3"]["reg_hyperparams_fixed"] == calibrated
    # The posterior stores Float64 medians. The fit records the Float32 values it uses.
    strengths_float32 = {key: float(np.float32(value)) for key, value in calibrated.items()}
    assert objectives["m3"]["fixed_strengths"] == strengths_float32
    assert all(extras["m2"]["medians"][key] == value for key, value in calibrated.items())
    assert set(payloads["m2"]["param_names"]) == set(calibrated)
    assert set(payloads["m3"]["param_names"]).isdisjoint(calibrated)
    fixed_mass = extras["m2"]["mass_medians_fixed"]
    assert all(fixed_mass[key] == value for key, value in extras["m1"]["medians"].items())

    # Restore the actual two image positions used by the saved score.
    position = objectives["m1"]["position_likelihood"]
    position_config = dict(
        positions=list(zip(position["x"], position["y"])),
        threshold_arcsec=position["threshold"],
        min_log_like=position["min_log_like"],
    )
    args = (image - l["extra"]["lens_light_model"], noise, psf, feature)
    stage_a = pipeline._stage_from_payload(a)
    stage_m1 = pipeline._stage_from_payload(payloads["m1"])
    log_lambda_m0 = extras["m0"]["log_lambda_best"]
    builders = {
        "m0": (pipeline.build_stage_m0_likelihood, (stage_a, position_config)),
        "m1": (pipeline.build_stage_m1_likelihood, (stage_a, position_config, log_lambda_m0)),
        "m2": (pipeline.build_stage_m2_likelihood, (fixed_mass, position_config, source)),
        "m3": (pipeline.build_stage_m3_likelihood, (stage_m1, stage_a, position_config, calibrated, source)),
    }
    records = {}
    for tag, (builder, parameters) in builders.items():
        objective = objectives[tag]
        assert objective["evidence_logdet"] == "overlap" and objective["evidence_block_size"] == 10
        assert objective["grid"]["n"] == 30 and objective["grid"]["source_interpolation"] == "bilinear"
        assert objective["observation"]["nsub"] == 4 and objective["sampling"]["vectorized_chunk_size"] == 5
        model = builder(
            *args, *parameters, circular_mask=circle, evidence_logdet="overlap", evidence_block_size=10,
        )
        assert pixelized_observation_provenance(model) == objective["observation"]
        payload = payloads[tag]
        medians = {"log_lambda_reg": log_lambda_m0} if tag == "m0" else extras[tag]["medians"]
        fit = evaluate_stage(model, payload, medians)
        assert fit["evidence_logdet"] == objective["evidence_logdet"]
        if tag != "m0":
            np.testing.assert_allclose(fit["source_bbox"], source["source_bbox"], atol=1e-7, rtol=1e-7)
        if tag in ("m2", "m3"):
            np.testing.assert_allclose(fit["lambda_reg"], np.exp(calibrated["log_lambda_reg"]), rtol=1e-6)
            np.testing.assert_allclose(fit["adaptive_reg_rho"], calibrated["adaptive_reg_rho"], rtol=1e-6)
        if tag == "m1":
            theta = jnp.asarray([medians[name] for name in payload["param_names"]])
            with ck.ActiveContext(model):
                model.fill_params(theta)
                rendered, _, bbox = model.reconstruct_component_images(
                    jnp.asarray(source["source_pixels"], dtype=provenance["source_dtype"]), None,
                )
            np.testing.assert_allclose(bbox, source["source_bbox"], atol=1e-7, rtol=1e-7)
            np.testing.assert_allclose(rendered, arc, atol=1e-6, rtol=1e-6)
            fit["reference_renderer_max_abs_error"] = float(np.max(np.abs(rendered - arc)))

        weights = payload["weights"]
        if weights is not None:
            assert np.isfinite(payload["samples"]).all() and np.isfinite(weights).all()
            np.testing.assert_allclose(np.sum(weights), 1)
        assert np.isfinite(payload["log_z"])
        records[tag] = {
            "stage_score": float(payload["log_z"]),
            "posterior_draws": None if weights is None else int(len(weights)),
            "posterior_effective_samples": None if weights is None else float(1 / np.sum(weights**2)),
            "time_taken_seconds": extras[tag]["time_taken"],
            "medians": medians,
            "objective": {key: objective[key] for key in (
                "fingerprint", "evidence_logdet", "evidence_block_size", "block_size", "solver",
                "sampling", "grid", "reference_id", "adaptation", "fixed_strengths", "upstream",
                "regularization_type", "precision",
            )},
            "diagnostics": fit,
            "artifact_sha256": sha256(output / f"stage_{tag}.pkl"),
        }

    return {
        "active_pixels": int(np.count_nonzero(active)),
        "feature_mask_sha256": sha256(feature_path),
        "upstream_artifact_sha256": {
            f"stage_{tag}.pkl": sha256(output / f"stage_{tag}.pkl") for tag in ("a", "l")
        },
        "reference_id": reference_id,
        "reference": provenance,
        "reference_positive_percentile_99_5": float(np.percentile(np.maximum(arc[active], 0), 99.5)),
        "selected_m2_strengths": calibrated,
        "stages": records,
    }


def verify_resume(output):
    """Check the maintained resume log and the stage artifacts it reused."""
    before = {}
    for line in (output / "before_resume.sha256").read_text().splitlines():
        digest, filename = line.split()
        before[filename] = digest
    assert before == {
        f"stage_{tag}.pkl": sha256(output / f"stage_{tag}.pkl") for tag in ("a", "l", *STAGES)
    }
    log = (output / "vis_resume.log").read_text()
    assert "Pipeline complete" in log and "[stage-M1] reusing verified source reference" in log
    assert all(f"[stage-{tag.upper()}] reusing matching objective" in log for tag in ("m0", "m2", "m3"))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fits", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("verify", "upstream", "summarize"), default="summarize")
    parser.add_argument("--example", type=Path, default=EXAMPLE)
    args = parser.parse_args()
    source_fits = args.source_fits.resolve()
    output = args.out_dir.resolve()
    example = args.example.resolve()
    os.chdir(example)
    inputs = verify_inputs(source_fits)
    if args.phase == "verify":
        print(json.dumps(inputs, indent=2))
        return

    sys.path.insert(0, str(example))
    pipeline = importlib.import_module("pipeline")
    if args.phase == "upstream":
        prepare_upstream(pipeline, output)
        return

    import jax

    record = {
        **inputs,
        "checkout": subprocess.check_output(["git", "-C", str(example), "rev-parse", "HEAD"], text=True).strip(),
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "nautilus": version("nautilus-sampler"),
        "matched_resume_verified": verify_resume(output),
        **summarize(pipeline, output),
    }
    (output / "vis_results.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"Verified VIS stage products in {output}")


if __name__ == "__main__":
    main()
