"""Shared position-likelihood penalty helpers for observation models.

These are used by ``ImageProbModel``, ``PixelizedImageProbModel``, and
``PixelizedImageProbModelOperator`` to avoid duplicating identical logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array


def resolve_position_likelihood_attrs(
    config: Optional[Dict[str, Any]],
) -> tuple[Array | None, Array | None, Array, Array, Array, bool]:
    """Parse position-likelihood config into ready-to-use attributes.

    Parameters
    ----------
    config : dict or None
        Configuration dictionary that may contain keys ``positions`` and
        either ``sigma_arcsec`` / ``position_sigma`` for a Gaussian penalty,
        or the legacy ``threshold_arcsec`` / ``position_threshold`` and
        ``min_log_like`` / ``min_position_likelihood`` pair.

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
    pos_sigma : Array (scalar)
        Gaussian source-plane separation uncertainty. Zero selects the legacy
        threshold penalty.
    has_penalty : bool
        ``True`` when a position penalty should be applied.
    """
    if config is None:
        zero = jnp.array(0.0, dtype=jnp.float32)
        return None, None, zero, zero, zero, False

    positions = config.get("positions", [])
    if positions is None or len(positions) < 2:
        zero = jnp.array(0.0, dtype=jnp.float32)
        return None, None, zero, zero, zero, False

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
    sigma_value = float(
        config.get("sigma_arcsec", config.get("position_sigma", 0.0))
    )
    if ("sigma_arcsec" in config or "position_sigma" in config) and (
        not np.isfinite(sigma_value) or sigma_value <= 0.0
    ):
        raise ValueError("position-likelihood sigma must be finite and positive")
    pos_sigma = jnp.array(sigma_value, dtype=jnp.float32)
    return pos_px, pos_py, pos_thr, pos_minl, pos_sigma, True


def compute_position_penalty_jax(
    phys_model,
    pos_px: Array,
    pos_py: Array,
    pos_thr: Array,
    pos_minl: Array,
    pos_sigma: Array,
) -> Array:
    r"""Evaluate the JAX-compatible position-likelihood penalty.

    Penalizes models where ray-traced image-plane positions do not map
    to a common source-plane position. When ``pos_sigma > 0``, the penalty is
    Gaussian. It profiles over the unknown common source position using the
    centroid of all ray-traced positions, then sums their two-dimensional
    squared residuals. Otherwise the legacy threshold penalty is used:

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
    pos_sigma : Array (scalar)
        Gaussian source-plane separation uncertainty, or zero for threshold
        mode.

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

    threshold_penalty = jnp.clip(pen_continuous, min=pos_minl, max=0.0)
    beta_x_centered = beta_x - jnp.mean(beta_x)
    beta_y_centered = beta_y - jnp.mean(beta_y)
    gaussian_chi2 = jnp.sum(
        beta_x_centered ** 2 + beta_y_centered ** 2
    ) / jnp.maximum(pos_sigma, 1.0e-12) ** 2
    gaussian_penalty = -0.5 * gaussian_chi2
    return jnp.where(pos_sigma > 0.0, gaussian_penalty, threshold_penalty)
