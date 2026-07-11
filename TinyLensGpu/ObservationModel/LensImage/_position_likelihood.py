"""Shared position-likelihood penalty helpers for observation models.

These are used by ``ImageProbModel``, ``PixelizedImageProbModel``, and
``PixelizedImageProbModelOperator`` to avoid duplicating identical logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp
from jax import Array


def resolve_position_likelihood_attrs(
    config: Optional[Dict[str, Any]],
) -> tuple[Array | None, Array | None, Array, Array, bool]:
    """Parse position-likelihood config into ready-to-use attributes.

    Parameters
    ----------
    config : dict or None
        Configuration dictionary that may contain keys ``positions``,
        ``threshold_arcsec`` / ``position_threshold``, and
        ``min_log_like`` / ``min_position_likelihood``.

    Returns
    -------
    pos_px : Array or None
        X-coordinates of position-constrained image pixels.
    pos_py : Array or None
        Y-coordinates of position-constrained image pixels.
    pos_thr : Array (scalar)
        Separation threshold in arcsec.
    pos_minl : Array (scalar)
        Penalty amplitude (non-positive).
    has_penalty : bool
        ``True`` when a position penalty should be applied.
    """
    if config is None:
        return None, None, jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32), False

    positions = config.get("positions", [])
    if positions is None or len(positions) < 2:
        return None, None, jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32), False

    pos_px = jnp.array([p[0] for p in positions], dtype=jnp.float32)
    pos_py = jnp.array([p[1] for p in positions], dtype=jnp.float32)
    pos_thr = jnp.array(
        float(config.get("threshold_arcsec", config.get("position_threshold", 0.0))),
        dtype=jnp.float32,
    )
    pos_minl = jnp.array(
        float(config.get("min_log_like", config.get("min_position_likelihood", 0.0))),
        dtype=jnp.float32,
    )
    return pos_px, pos_py, pos_thr, pos_minl, True


def compute_position_penalty_jax(
    phys_model,
    pos_px: Array,
    pos_py: Array,
    pos_thr: Array,
    pos_minl: Array,
) -> Array:
    r"""Evaluate the JAX-compatible position-likelihood penalty.

    Penalizes models where ray-traced image-plane positions do not map
    to a common source-plane position:

    .. math::

        \text{Penalty} = \text{min\_log\_like} \cdot (1 - e^{-\text{ratio}})

    where :math:`\text{ratio} = \max(0, \max(\text{separation}) - \text{threshold}) / \text{threshold}`.

    Parameters
    ----------
    phys_model : PhysicalModel
        Lens mass model with a ``deflection(x, y)`` method.
    pos_px, pos_py : Array
        Image-plane coordinates of the constrained pixels.
    pos_thr : Array (scalar)
        Separation threshold in arcsec.
    pos_minl : Array (scalar)
        Penalty amplitude (non-positive).

    Returns
    -------
    Array (scalar)
        Log-likelihood penalty value (≤ 0).
    """
    beta_x, beta_y = phys_model.deflection(pos_px, pos_py)

    dx = beta_x[:, None] - beta_x[None, :]
    dy = beta_y[:, None] - beta_y[None, :]
    dist = jnp.sqrt(dx * dx + dy * dy)
    max_sep = jnp.max(dist)

    exceed = jnp.maximum(0.0, max_sep - pos_thr)
    ratio = jnp.where(pos_thr > 0.0, exceed / pos_thr, 0.0)
    pen_continuous = pos_minl * (1.0 - jnp.exp(-ratio))

    return jnp.clip(pen_continuous, min=pos_minl, max=0.0)
