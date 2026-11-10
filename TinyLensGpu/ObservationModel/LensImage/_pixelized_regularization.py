"""Shared source regularization preparation and projection."""

import caskade as ck
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.utils.inversion.regularization import _source_template_scale_from_normalized


ADAPT_REFERENCE_PERCENTILE = 99.5
ADAPT_REFERENCE_FLOOR = 1e-10
BILINEAR_COVERAGE_THRESHOLD = 1e-8
HERMITE_COVERAGE_OFFSET = 1e-4


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


def adaptive_reg_scale(projected_brightness, rho):
    """Evaluate the proposal scale, with None representing a uniform prior."""
    if projected_brightness is None:
        return None
    return _source_template_scale_from_normalized(projected_brightness, _param_value(rho))


def prepare_adapt_image(adapt_image, image_data, simulator, rho, fixed_source_bbox):
    """Validate the reference and normalize its active native pixels once."""
    enabled = adaptive_reg_enabled(rho)
    if enabled and fixed_source_bbox is None:
        raise ValueError("adaptive regularization requires fixed_source_bbox.")
    if adapt_image is None:
        if enabled:
            raise ValueError("adaptive regularization requires adapt_image.")
        return None

    reference = jnp.asarray(adapt_image)
    if reference.ndim != 2 or reference.shape != image_data.shape:
        raise ValueError(f"adapt_image must have shape {image_data.shape}, got {reference.shape}.")
    if jnp.issubdtype(reference.dtype, jnp.complexfloating):
        raise ValueError("adapt_image values must be real.")
    reference = reference.astype(jnp.result_type(image_data, jnp.float32))
    if not bool(np.asarray(jnp.all(jnp.isfinite(reference)))):
        raise ValueError("adapt_image values must be finite.")
    if not enabled:
        return None
    positive = jnp.maximum(reference.ravel()[simulator.flat_indices], 0.0)
    if positive.size:
        reference_brightness = jnp.maximum(
            jnp.percentile(positive, ADAPT_REFERENCE_PERCENTILE), ADAPT_REFERENCE_FLOOR,
        )
        positive = jnp.clip(positive / reference_brightness, 0.0, 1.0)
    return positive[simulator._agg_segment_ids] if simulator.nsub > 1 else positive


def project_adapt_image(ray_brightness, weights, indices, n_source, nsub, *, interpolation):
    """Project the reference before the signed imaging stencil is integrated.

    Hermite projection squares each ray's weights and uses an offset to
    smooth the transition to uncovered nodes.
    """
    hermite = interpolation == "cubic_hermite"
    ray_weights = weights.astype(ray_brightness.dtype)
    if hermite:
        ray_weights = jnp.square(ray_weights)
    ray_weights = ray_weights / nsub**2
    zeros = jnp.zeros(n_source, dtype=ray_weights.dtype)
    coverage = zeros.at[indices].add(ray_weights)
    signal = zeros.at[indices].add(ray_weights * ray_brightness[:, None])
    if hermite:
        return signal / (coverage + HERMITE_COVERAGE_OFFSET)
    covered = coverage > BILINEAR_COVERAGE_THRESHOLD
    return jnp.where(covered, signal / jnp.where(covered, coverage, 1.0), 0.0)
