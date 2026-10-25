"""Shared fixed-template preparation for dense and operator pixelized models."""

import caskade as ck
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.utils.inversion.regularization import (
    _normalize_source_template,
    _source_template_scale_from_normalized,
)


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
            raise ValueError("adaptive regularization requires fixed_source_bbox from an S0 source template.")
        if template is None:
            raise ValueError("adaptive regularization requires fixed_reg_template from an S0 source template.")
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
        raise ValueError("adaptive regularization requires fixed_reg_template from an S0 source template.")
    return _source_template_scale_from_normalized(template_u, _param_value(rho))
