"""Shared pixelized-source export and validation for example pipelines."""

from __future__ import annotations

import numpy as np


def source_axes_from_bbox(source_bbox, n):
    xmin, xmax, ymin, ymax = [float(v) for v in source_bbox]
    return (
        np.linspace(xmin, xmax, int(n), dtype=np.float64),
        np.linspace(ymin, ymax, int(n), dtype=np.float64),
    )


def is_square_bbox(source_bbox, *, rtol=1.0e-6, atol=1.0e-7):
    xmin, xmax, ymin, ymax = [float(v) for v in source_bbox]
    return np.isclose(xmax - xmin, ymax - ymin, rtol=rtol, atol=atol)


def normalize_source_bbox(source_bbox, *, label="Source bbox"):
    if source_bbox is None or len(source_bbox) != 4:
        raise ValueError(f"{label} must contain four values.")
    bbox = tuple(float(v) for v in source_bbox)
    if not np.all(np.isfinite(bbox)):
        raise ValueError(f"{label} values must be finite.")
    if not (bbox[0] < bbox[1] and bbox[2] < bbox[3]):
        raise ValueError(f"{label} must satisfy xmin < xmax and ymin < ymax.")
    if not is_square_bbox(bbox):
        raise ValueError(f"{label} must be square.")
    return bbox


def source_bboxes_match(left, right):
    if left is None or right is None:
        return False
    try:
        left_bbox = normalize_source_bbox(left)
        right_bbox = normalize_source_bbox(right)
    except (TypeError, ValueError):
        return False
    return bool(np.allclose(left_bbox, right_bbox, rtol=1.0e-6, atol=1.0e-7))


def require_valid_reconstruction(result, diagnostics, *, context):
    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InvalidReconstructionError,
    )

    if not diagnostics.valid:
        raise InvalidReconstructionError(
            f"Reconstruction failed while {context}: {diagnostics.invalid_reason}"
        )


def export_source_package(likelihood, medians, param_names, *, n):
    """Solve the source template once and package it in the single-n schema."""
    import jax.numpy as jnp

    theta = jnp.asarray([medians[name] for name in param_names])
    result, diagnostics = likelihood.evaluate_with_diagnostics(theta)
    require_valid_reconstruction(
        result, diagnostics, context="exporting source package"
    )
    pixels = np.asarray(result.source_pixels, dtype=np.float64)
    bbox = tuple(float(v) for v in result.source_bbox)
    x, y = source_axes_from_bbox(bbox, n)
    return dict(
        source_pixels=pixels,
        source_image=pixels.reshape(n, n),
        source_bbox=bbox,
        source_x_axis=x,
        source_y_axis=y,
        n=int(n),
    )


def validate_source_package(
    package, *, expected_n, rho, expected_source_bbox=None,
    normalize_bbox=False, check_axes=False, require_regularization=True,
):
    """Validate/normalize a source package in place; return the same dict."""
    legacy_keys = [k for k in ("nx", "ny") if k in package]
    if legacy_keys:
        raise KeyError(
            "Source package uses legacy source-grid keys "
            f"{', '.join(legacy_keys)}; regenerate S0 with the single-n "
            "source-grid schema."
        )

    required = [
        "source_pixels", "source_bbox", "source_x_axis", "source_y_axis", "n",
    ]
    if require_regularization:
        required += ["lambda_best", "log_lambda_best"]
    missing = [k for k in required if k not in package]
    if missing:
        raise KeyError("Source package missing required keys: " + ", ".join(missing))

    n = int(package["n"])
    if n != expected_n:
        raise ValueError(
            f"Source grid dimension n={n} does not match configured n={expected_n}."
        )
    source_pixels = np.asarray(package["source_pixels"])
    if source_pixels.shape != (n * n,):
        raise ValueError(
            f"Source source_pixels must have shape ({n * n},), "
            f"got {source_pixels.shape}."
        )
    for axis_name in ("source_x_axis", "source_y_axis"):
        axis = np.asarray(package[axis_name])
        if axis.shape != (n,):
            raise ValueError(
                f"S0 {axis_name} must have shape ({n},), got {axis.shape}."
            )
    if normalize_bbox:
        bbox = normalize_source_bbox(
            package["source_bbox"], label="Source source_bbox"
        )
        package["source_bbox"] = bbox
    else:
        bbox = tuple(float(v) for v in package["source_bbox"])
        if len(bbox) != 4 or not np.all(np.isfinite(bbox)):
            raise ValueError("Source source_bbox must contain four finite values.")
        if not (bbox[0] < bbox[1] and bbox[2] < bbox[3]):
            raise ValueError(
                "Source source_bbox must satisfy xmin < xmax and ymin < ymax."
            )
        if not is_square_bbox(bbox):
            raise ValueError(
                "Source source_bbox is rectangular; regenerate S0 with a square "
                "pixelized source bbox."
            )
    if expected_source_bbox is not None and not source_bboxes_match(
        bbox, expected_source_bbox
    ):
        raise ValueError(
            "Source package bbox does not match the fixed stage-A source bbox."
        )
    if check_axes:
        expected_x_axis, expected_y_axis = source_axes_from_bbox(bbox, n)
        for axis_name, expected_axis in (
            ("source_x_axis", expected_x_axis),
            ("source_y_axis", expected_y_axis),
        ):
            if not np.allclose(
                np.asarray(package[axis_name], dtype=np.float64),
                expected_axis, rtol=1.0e-6, atol=1.0e-7,
            ):
                raise ValueError(f"Source {axis_name} does not match source_bbox.")

    if not require_regularization:
        return package
    from TinyLensGpu.utils.inversion.regularization import (
        source_template_scale_map,
    )

    scale_map = package.get("scale_map")
    if scale_map is None:
        scale_map = np.asarray(
            source_template_scale_map(package["source_pixels"], n, rho=rho),
            dtype=np.float32,
        )
        package["scale_map"] = scale_map
    else:
        scale_map = np.asarray(scale_map, dtype=np.float32)
        if scale_map.shape != (n * n,):
            raise ValueError(
                f"Source scale_map must have shape ({n * n},), got {scale_map.shape}."
            )
        if not np.all(np.isfinite(scale_map) & (scale_map > 0.0)):
            raise ValueError("Source scale_map values must be finite and positive.")
        package["scale_map"] = scale_map
    return package
