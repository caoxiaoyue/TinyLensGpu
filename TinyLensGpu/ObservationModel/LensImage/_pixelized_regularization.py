"""Shared source regularization preparation and projection."""

import caskade as ck
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.utils.inversion.regularization import (
    _normalize_source_template,
    _source_template_scale_from_normalized,
)


ADAPT_REFERENCE_PERCENTILE = 99.5
ADAPT_REFERENCE_FLOOR = 1e-10
BILINEAR_COVERAGE_THRESHOLD = 1e-8


def is_fixed(value):
    """Include transitive parameter and registered pointer dependencies."""
    if not isinstance(value, (ck.Module, ck.Param)):
        return True
    params = (p for p in value.topological_ordering() if isinstance(p, ck.Param))
    return all(not p.dynamic and (not p.pointer or bool(p.children)) for p in params)


def _param_value(value):
    return value.value if hasattr(value, "value") else value


def adaptive_reg_enabled(rho):
    """Whether adaptive regularization is active for the given rho.

    A rho with dynamic or unresolved dependencies enables adaptation, even
    when its current value is zero. A fixed rho — including a pointer whose
    registered dependencies are all fixed — enables adaptation only when its
    magnitude is at least ``1e-10``.
    """
    if not is_fixed(rho):
        return True
    try:
        return abs(float(_param_value(rho))) >= 1.0e-10
    except TypeError:
        return True


def prepare_reg_template(template, n, rho, fixed_source_bbox):
    """Validate the source-grid contract and normalize a fixed S0 once.

    Adaptive regularization (a positive fixed rho or a rho with dynamic or
    unresolved dependencies) requires both ``fixed_source_bbox`` and a finite
    template of shape ``(n * n,)`` or ``(n, n)``. A disabled (fixed, near-zero)
    rho accepts a missing template and returns ``None`` for it.
    """
    if adaptive_reg_enabled(rho):
        if fixed_source_bbox is None:
            raise ValueError("adaptive regularization requires fixed_source_bbox.")
        if template is None:
            raise ValueError("adaptive regularization requires adapt_image or fixed_reg_template.")
    if template is None:
        return None
    template = jnp.asarray(template, dtype=jnp.float32)
    if template.shape not in ((int(n) * int(n),), (int(n), int(n))):
        raise ValueError(
            "fixed_reg_template must have shape "
            f"({int(n) * int(n)},) or ({int(n)}, {int(n)}), got {template.shape}."
        )
    if not bool(np.asarray(jnp.all(jnp.isfinite(template)))):
        raise ValueError("fixed_reg_template values must be finite.")
    return _normalize_source_template(template, n)


def template_reg_scale(template_u, rho):
    """Evaluate weights at current rho; fixed zero keeps the uniform fast path."""
    if not adaptive_reg_enabled(rho):
        return None
    if template_u is None:
        raise ValueError("adaptive regularization requires adapt_image or fixed_reg_template.")
    return _source_template_scale_from_normalized(template_u, _param_value(rho))


def prepare_adaptive_inputs(
    adapt_image, fixed_reg_template, image_data, simulator, rho, fixed_source_bbox,
):
    """Return normalized active-ray brightness or a transitional source template."""
    if adapt_image is None:
        return None, prepare_reg_template(
            fixed_reg_template, simulator.source_n, rho, fixed_source_bbox,
        )
    if fixed_reg_template is not None:
        raise ValueError("adapt_image and fixed_reg_template are mutually exclusive.")

    reference = jnp.asarray(adapt_image)
    if reference.ndim != 2 or reference.shape != image_data.shape:
        raise ValueError(f"adapt_image must have shape {image_data.shape}, got {reference.shape}.")
    if jnp.issubdtype(reference.dtype, jnp.complexfloating):
        raise ValueError("adapt_image values must be real.")
    reference = reference.astype(jnp.result_type(image_data, jnp.float32))
    if not bool(np.asarray(jnp.all(jnp.isfinite(reference)))):
        raise ValueError("adapt_image values must be finite.")
    if not adaptive_reg_enabled(rho):
        return None, None
    if fixed_source_bbox is None:
        raise ValueError("adaptive regularization requires fixed_source_bbox.")
    if simulator.source_model.interpolation != "bilinear":
        raise ValueError("adapt_image currently requires bilinear source interpolation.")

    positive = jnp.maximum(reference.ravel()[simulator.flat_indices], 0.0)
    if positive.size:
        reference_brightness = jnp.maximum(
            jnp.percentile(positive, ADAPT_REFERENCE_PERCENTILE), ADAPT_REFERENCE_FLOOR,
        )
        positive = jnp.clip(positive / reference_brightness, 0.0, 1.0)
    ray_brightness = (
        positive[simulator._agg_segment_ids] if simulator.nsub > 1 else positive
    )
    return ray_brightness, None


def project_adapt_image(ray_brightness, weights, indices, n_source, nsub):
    """Project the prepared reference with the same bilinear stencil as the fit."""
    ray_weights = weights.astype(ray_brightness.dtype) / nsub**2
    zeros = jnp.zeros(n_source, dtype=ray_weights.dtype)
    coverage = zeros.at[indices].add(ray_weights)
    signal = zeros.at[indices].add(ray_weights * ray_brightness[:, None])
    covered = coverage > BILINEAR_COVERAGE_THRESHOLD
    return jnp.where(covered, signal / jnp.where(covered, coverage, 1.0), 0.0)
