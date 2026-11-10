"""Reference-arc and objective provenance for pixelized example workflows."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .source import (
    normalize_source_bbox,
    resolve_source_interpolation,
    source_interpolation_metadata,
)


def array_fingerprint(value):
    """Identify an array by its dtype, shape and contents."""
    array = np.asarray(value)
    digest = hashlib.sha256(f"{array.dtype.str}:{array.shape}:".encode())
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return {"array_sha256": array_fingerprint(value)}
    return value


def artifact_fingerprint(value):
    """Identify a stage product or structured settings record."""
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def pixelized_observation_provenance(likelihood):
    """Describe the native observation and the geometry used by the model."""
    config = likelihood.sim_obj.sim_config
    return {
        "image_data": array_fingerprint(likelihood.image_data),
        "noise_map": array_fingerprint(likelihood.noise_map),
        "psf_kernel": array_fingerprint(config.psf_kernel),
        "mask": array_fingerprint(config.mask),
        "source_seed_mask": array_fingerprint(config.source_seed_mask),
        "image_shape": list(likelihood.image_data.shape),
        "dpix": float(config.dpix),
        "nsub": int(config.nsub),
    }


def _grid_provenance(package):
    interpolation = resolve_source_interpolation(metadata=package)
    return {
        "n": int(package["n"]),
        "source_bbox": list(normalize_source_bbox(package["source_bbox"])),
        "source_x_axis": array_fingerprint(package["source_x_axis"]),
        "source_y_axis": array_fingerprint(package["source_y_axis"]),
        **source_interpolation_metadata(interpolation),
    }


def _component_parameters(components, *, fitted=False):
    return [
        {
            "profile": type(component).__qualname__,
            "parameters": [
                [param.name, float(np.asarray(param.value))]
                for param in (component.all_params if fitted else component.static_params)
            ],
        }
        for component in components
    ]


def _parent_point(likelihood, theta, param_names):
    import caskade as ck

    expected_names = [param.name for param in likelihood.get_dynamic_params()]
    if list(param_names) != expected_names:
        raise ValueError("Source reference parameters do not match the producer's order.")
    with ck.ActiveContext(likelihood):
        if expected_names:
            likelihood.fill_params(theta)
        return {
            "param_names": expected_names,
            "values": np.asarray(theta).tolist(),
            "lens_mass": _component_parameters(likelihood.phys_model.lens_mass, fitted=True),
            "lens_light": _component_parameters(likelihood.phys_model.lens_light, fitted=True),
        }


def _native_arc(arc, image_shape):
    arc = np.asarray(arc)
    if (arc.shape != tuple(image_shape) or not np.isrealobj(arc)
            or not np.all(np.isfinite(arc))):
        raise ValueError("Source reference requires a finite native adapt_image.")
    return arc


def record_source_reference(likelihood, package, result, diagnostics, theta, param_names):
    """Add the native source component and its producer identity to an export."""
    from TinyLensGpu.ObservationModel.LensImage.inversion_result import InvalidReconstructionError

    if getattr(diagnostics, "evidence_valid", None) is False:
        raise InvalidReconstructionError(
            f"Evidence failed while exporting the source reference: {diagnostics.evidence_invalid_reason}"
        )
    arc = _native_arc(diagnostics.source_image, likelihood.image_data.shape)
    package["adapt_image"] = arc
    if result.lens_amplitudes is not None:
        package["lens_light_intensities"] = np.asarray(result.lens_amplitudes)
    package["reference_provenance"] = {
        **_source_reconstruction_record(
            likelihood, package, theta, param_names, result.source_pixels.dtype,
        ),
        "source_bbox_dtype": str(np.asarray(result.source_bbox).dtype),
        "adapt_image": array_fingerprint(arc),
    }


def _source_reconstruction_record(likelihood, package, theta, param_names, source_dtype):
    intensities = package.get("lens_light_intensities")
    return {
        "version": 1,
        "observation": pixelized_observation_provenance(likelihood),
        "grid": _grid_provenance(package),
        "parent_point": _parent_point(likelihood, theta, param_names),
        "producer_evidence_logdet": getattr(likelihood, "evidence_logdet", "exact"),
        "producer_evidence_block_size": getattr(likelihood, "evidence_block_size", None),
        "source_dtype": str(source_dtype),
        "source_pixels": array_fingerprint(package["source_pixels"]),
        "lens_light_intensities": None if intensities is None else array_fingerprint(intensities),
    }


def source_reconstruction_provenance(likelihood, package, medians, param_names, *, source_dtype):
    """Record the inputs of a source fit that predates native arc export.

    The caller supplies the original observation, model, and fitted point.
    Save this record separately with the historical stage to allow recovery.
    """
    import jax.numpy as jnp

    theta = jnp.asarray([medians[name] for name in param_names])
    return _source_reconstruction_record(likelihood, package, theta, param_names, source_dtype)


def _validate_reference(package, expected_n, source_interpolation=None):
    from .source import validate_source_package

    validate_source_package(
        package, expected_n=expected_n, rho=0.0, check_axes=True,
        require_regularization=False, source_interpolation=source_interpolation,
    )
    record = package.get("reference_provenance")
    if not isinstance(record, dict) or record.get("version") != 1:
        raise ValueError("Source package is missing verifiable reference provenance.")
    if record.get("producer_evidence_logdet") not in ("block", "overlap", "exact"):
        raise ValueError("Source package lacks a recognized producer evidence method.")
    if record["grid"] != _grid_provenance(package):
        raise ValueError("Source package grid differs from its reference provenance.")
    pixels = np.asarray(package["source_pixels"])
    if not np.all(np.isfinite(pixels)) or record["source_pixels"] != array_fingerprint(pixels):
        raise ValueError("Source pixels differ from the verified reference producer.")
    intensities = package.get("lens_light_intensities")
    if intensities is not None and not np.all(np.isfinite(intensities)):
        raise ValueError("Source reference lens-light intensities must be finite.")
    if record["lens_light_intensities"] != (
        None if intensities is None else array_fingerprint(intensities)
    ):
        raise ValueError("Source reference lens-light intensities differ from its producer.")
    return record


def adapt_image_kwargs(package, *, expected_n, source_interpolation=None):
    """Validate a source package and return the two adaptive constructor inputs."""
    record = _validate_reference(package, expected_n, source_interpolation)
    arc = _native_arc(package.get("adapt_image"), record["observation"]["image_shape"])
    if record.get("adapt_image") != array_fingerprint(arc):
        raise ValueError("adapt_image differs from its reference provenance.")
    bbox_dtype = record.get("source_bbox_dtype", record["source_dtype"])
    bbox = np.asarray(record["grid"]["source_bbox"], dtype=np.dtype(bbox_dtype))
    return {"adapt_image": arc, "fixed_source_bbox": tuple(bbox)}


def require_reference_observation(likelihood, package, *, check_seed_mask=False):
    """Check the observation and sampling used to render the reference.

    A source seed mask also matters when replaying an inferred producer grid.
    Adaptive consumers use the fitted grid directly.
    """
    observed = pixelized_observation_provenance(likelihood)
    recorded = package["reference_provenance"]["observation"]
    if any(
        value != recorded.get(key) for key, value in observed.items()
        if check_seed_mask or key != "source_seed_mask"
    ):
        raise ValueError("Source reference does not match the observation, mask, PSF, or subpixel sampling.")


def pixelized_objective_provenance(likelihood, *, reference=None, sampling=None, upstream=None):
    """Record the actual model settings used to decide whether a stage can resume.

    Supply the exported reference package for adaptive stages and upstream
    product identities for the stage transitions. Sampling may be a
    SamplingConfig or an equivalent settings dict.
    """
    from dataclasses import asdict

    import jax

    from TinyLensGpu.Inference.build_likelihood import DEFAULT_OVERLAP_CHUNK_SIZE
    from TinyLensGpu.Inference.build_prior import extract_prior_specs
    from TinyLensGpu.ObservationModel.LensImage._pixelized_regularization import (
        ADAPT_REFERENCE_FLOOR,
        ADAPT_REFERENCE_PERCENTILE,
        BILINEAR_COVERAGE_THRESHOLD,
        HERMITE_COVERAGE_OFFSET,
        adaptive_reg_enabled,
        is_fixed,
    )
    from .sampling import SamplingConfig

    config = likelihood.sim_obj.sim_config
    source = likelihood.source_model
    interpolation = getattr(source, "interpolation", "bilinear")
    evidence = getattr(likelihood, "evidence_logdet", "exact")
    if isinstance(sampling, SamplingConfig):
        sampling = {"method": "nautilus", **asdict(sampling)}
    sampling = dict(sampling or {})
    if (sampling.get("vectorized") and sampling.get("vectorized_chunk_size") is None
            and evidence == "overlap"):
        sampling["vectorized_chunk_size"] = DEFAULT_OVERLAP_CHUNK_SIZE

    fixed_bbox = likelihood._fixed_source_bbox
    grid = {
        "n": int(source.n),
        "source_bbox": list(normalize_source_bbox(fixed_bbox)) if fixed_bbox is not None else None,
        **source_interpolation_metadata(interpolation),
    }
    if fixed_bbox is None:
        grid.update(
            source_bbox_padding=config.source_bbox_padding,
            source_bbox_outlier_frac=config.source_bbox_outlier_frac,
        )

    adaptation = None
    reference_id = None
    if reference is not None:
        adapt_image_kwargs(reference, expected_n=source.n, source_interpolation=interpolation)
        require_reference_observation(likelihood, reference)
        reference_id = artifact_fingerprint(reference["reference_provenance"])
        if adaptive_reg_enabled(source.adaptive_reg_rho):
            coverage = (
                {"offset": HERMITE_COVERAGE_OFFSET} if interpolation == "cubic_hermite"
                else {"threshold": BILINEAR_COVERAGE_THRESHOLD}
            )
            adaptation = {
                "recipe": "native_source_arc_v1",
                "percentile": ADAPT_REFERENCE_PERCENTILE,
                "normalization_floor": ADAPT_REFERENCE_FLOOR,
                "coverage": coverage,
            }

    fixed_strengths = {}
    for name in ("log_lambda_reg", "adaptive_reg_rho"):
        strength = getattr(source, name)
        if is_fixed(strength):
            fixed_strengths[name] = float(getattr(strength, "value", strength))

    solver = {"type": likelihood.solver_type}
    for name in ("pcg_max_iter", "pcg_rtol"):
        if hasattr(likelihood, name):
            solver[name] = getattr(likelihood, name)
    if likelihood.solver_type == "pnpg":
        for name in ("pnpg_max_iter", "pnpg_rtol", "pnpg_power_iter", "pnpg_comp_rel_tol"):
            solver[name] = getattr(likelihood, name)
    initial = getattr(likelihood, "_initial_linear", None)
    if initial is not None:
        solver["initial_linear"] = array_fingerprint(initial)
    position = None
    if likelihood._has_pos_penalty:
        position = {
            "x": np.asarray(likelihood._pos_px).tolist(),
            "y": np.asarray(likelihood._pos_py).tolist(),
            "threshold": float(likelihood._pos_thr),
            "min_log_like": float(likelihood._pos_minl),
        }

    record = {
        "version": 1,
        "observation": pixelized_observation_provenance(likelihood),
        "grid": grid,
        "regularization_type": source.regularization_type,
        "reference_id": reference_id,
        "adaptation": adaptation,
        "evidence_logdet": evidence,
        "evidence_block_size": getattr(likelihood, "evidence_block_size", None) if evidence == "overlap" else None,
        "block_size": getattr(likelihood, "block_size", None),
        "solver": solver,
        "lens_light_regularization": getattr(likelihood, "lens_light_regularization", None),
        "position_likelihood": position,
        "precision": {
            "enable_x64": jax.config.jax_enable_x64,
            "default_matmul": jax.config.jax_default_matmul_precision,
        },
        "lens_mass": _component_parameters(likelihood.phys_model.lens_mass),
        "lens_light": _component_parameters(likelihood.phys_model.lens_light),
        "priors": [asdict(spec) for spec in extract_prior_specs(likelihood)],
        "fixed_strengths": fixed_strengths,
        "sampling": sampling,
        "upstream": dict(upstream or {}),
    }
    record = _json_value(record)
    return {**record, "fingerprint": artifact_fingerprint(record)}


def recover_source_reference(likelihood, package, medians, param_names, *, historical_provenance=None):
    """Verify a saved producer and render a missing arc from its fitted pixels.

    Historical packages may supply a separate reconstruction record from
    the original fit. Its observation, source grid, pixels and fitted point
    must all match before rendering.
    """
    import caskade as ck
    import jax.numpy as jnp

    from .source import source_bboxes_match

    interpolation = getattr(likelihood.source_model, "interpolation", "bilinear")
    n = int(likelihood.source_model.n)
    if "reference_provenance" not in package and historical_provenance is not None:
        package = {**package, "reference_provenance": historical_provenance}
    record = _validate_reference(package, n, interpolation)
    require_reference_observation(likelihood, package, check_seed_mask=True)
    theta = jnp.asarray([medians[name] for name in param_names])
    if record["parent_point"] != _parent_point(likelihood, theta, param_names):
        raise ValueError("Source reference does not match the fitted producer point.")

    if package.get("adapt_image") is not None:
        adapt_image_kwargs(package, expected_n=n, source_interpolation=interpolation)
        return package

    pixels = jnp.asarray(package["source_pixels"], dtype=np.dtype(record["source_dtype"]))
    with ck.ActiveContext(likelihood):
        if len(param_names):
            likelihood.fill_params(theta)
        arc, _, bbox = likelihood.reconstruct_component_images(
            pixels, package.get("lens_light_intensities"),
        )
    if not source_bboxes_match(bbox, package["source_bbox"]):
        raise ValueError("Rendered source grid does not match the historical source package.")
    arc = _native_arc(arc, likelihood.image_data.shape)
    return {
        **package,
        "adapt_image": arc,
        "reference_provenance": {
            **record,
            "source_bbox_dtype": str(np.asarray(bbox).dtype),
            "adapt_image": array_fingerprint(arc),
        },
    }
