"""Mapping utilities for lensing reconstruction."""

from functools import partial

import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=("n_source",))
def dense_mapping_from_weights_indices(
    weights: jnp.ndarray,
    indices: jnp.ndarray,
    n_source: int,
) -> jnp.ndarray:
    """
    Construct a dense mapping matrix from sparse interpolation weights and indices.

    Parameters
    ----------
    weights : jnp.ndarray
        Interpolation weights array of shape ``(n_data, k_neighbors)``.
    indices : jnp.ndarray
        Neighbor indices array of shape ``(n_data, k_neighbors)``.
    n_source : int
        Total number of source points (columns in the output matrix).

    Returns
    -------
    jnp.ndarray
        Dense mapping matrix of shape ``(n_data, n_source)``.
    """
    n_data = int(weights.shape[0])
    n_neighbors = int(weights.shape[1])
    row_indices = jnp.repeat(jnp.arange(n_data, dtype=jnp.int32), n_neighbors)
    col_indices = indices.reshape(-1).astype(jnp.int32)
    weights_flat = weights.reshape(-1)
    mapping_matrix = jnp.zeros((n_data, n_source), dtype=weights_flat.dtype)
    return mapping_matrix.at[row_indices, col_indices].add(weights_flat)


@partial(jax.jit, static_argnames=("n",))
def lens_mapping_operator_bilinear_from(
    data_mesh_beta: jnp.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build bilinear interpolation weights/indices on a square source grid."""
    beta = jnp.asarray(data_mesh_beta, dtype=jnp.float32)

    dx = (x_max - x_min) / (n - 1)
    dy = (y_max - y_min) / (n - 1)

    ux = (beta[:, 0] - x_min) / dx
    uy = (beta[:, 1] - y_min) / dy

    ix0 = jnp.floor(ux).astype(jnp.int32)
    iy0 = jnp.floor(uy).astype(jnp.int32)
    fx = ux - ix0
    fy = uy - iy0

    valid = (ux >= 0) & (ux <= n - 1) & (uy >= 0) & (uy <= n - 1)

    ix0_c = jnp.clip(ix0, 0, n - 1)
    iy0_c = jnp.clip(iy0, 0, n - 1)
    ix1_c = jnp.clip(ix0_c + 1, 0, n - 1)
    iy1_c = jnp.clip(iy0_c + 1, 0, n - 1)

    i00 = iy0_c * n + ix0_c
    i10 = iy0_c * n + ix1_c
    i01 = iy1_c * n + ix0_c
    i11 = iy1_c * n + ix1_c

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    weights = jnp.where(valid[:, None],
                        jnp.stack([w00, w10, w01, w11], axis=1), 0.0)
    indices = jnp.stack([i00, i10, i01, i11], axis=1).astype(jnp.int32)
    return weights, indices, valid


def build_source_grid(n, xmin, xmax, ymin, ymax):
    """Build a square source-plane grid spanning [xmin, xmax] x [ymin, ymax].

    Returns (x_axis, y_axis, xgrid, ygrid) with ``jnp.meshgrid(..., indexing='xy')`` layout.
    """
    x_axis = jnp.linspace(xmin, xmax, int(n))
    y_axis = jnp.linspace(ymin, ymax, int(n))
    xgrid, ygrid = jnp.meshgrid(x_axis, y_axis, indexing="xy")
    return x_axis, y_axis, xgrid, ygrid


def make_square_bbox(xmin, xmax, ymin, ymax):
    """Expand bbox bounds to a square while preserving each axis center.

    The shorter span is expanded to match the longer span.  Inputs may be
    Python scalars or JAX scalar arrays; the returned values are JAX-compatible
    scalar arrays.
    """
    xmin = jnp.asarray(xmin)
    xmax = jnp.asarray(xmax)
    ymin = jnp.asarray(ymin)
    ymax = jnp.asarray(ymax)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    side = jnp.maximum(xmax - xmin, ymax - ymin)
    half = 0.5 * side
    return xmid - half, xmid + half, ymid - half, ymid + half


def infer_source_bbox(beta_x, beta_y, padding=0.05, outlier_frac=0.0):
    """Infer a square source-plane bounding box from ray-traced beta points.

    Computes robust quantile bounds of beta coordinates and adds a
    fractional padding margin on each side (default 0.05 — 5% per side).
    ``outlier_frac`` is trimmed from each tail.  The default of 0 uses the
    absolute min/max; set a positive fraction to enable robust quantile
    trimming.  Ensures a minimum span of 1e-6 in each direction so that
    downstream grid construction is well-defined.  The shorter span is
    always expanded around its center after padding/flooring so the
    returned source-plane bbox has equal x/y extent.
    """
    if not (0.0 <= outlier_frac < 0.5):
        raise ValueError(
            f"outlier_frac must be in [0, 0.5), got {outlier_frac}"
        )

    beta_x = jnp.ravel(jnp.asarray(beta_x))
    beta_y = jnp.ravel(jnp.asarray(beta_y))

    if outlier_frac == 0.0:
        xmin = jnp.min(beta_x)
        xmax = jnp.max(beta_x)
        ymin = jnp.min(beta_y)
        ymax = jnp.max(beta_y)
    else:
        q_low = float(outlier_frac)
        q_high = 1.0 - float(outlier_frac)
        xmin = jnp.quantile(beta_x, q_low)
        xmax = jnp.quantile(beta_x, q_high)
        ymin = jnp.quantile(beta_y, q_low)
        ymax = jnp.quantile(beta_y, q_high)

    span_x = xmax - xmin
    span_y = ymax - ymin
    xmin = xmin - padding * span_x
    xmax = xmax + padding * span_x
    ymin = ymin - padding * span_y
    ymax = ymax + padding * span_y
    # Floor at ±0.5e-6 to keep the grid well-defined for point-like sources.
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    min_half = 0.5e-6
    xmin = jnp.minimum(xmin, xmid - min_half)
    xmax = jnp.maximum(xmax, xmid + min_half)
    ymin = jnp.minimum(ymin, ymid - min_half)
    ymax = jnp.maximum(ymax, ymid + min_half)
    xmin, xmax, ymin, ymax = make_square_bbox(xmin, xmax, ymin, ymax)
    return xmin, xmax, ymin, ymax


def build_lens_mapping_matrix(beta_x, beta_y, source_x_axis, source_y_axis):
    """Build dense bilinear mapping matrix (N_d image pixels x N_s source pixels)."""
    beta_x = jnp.ravel(jnp.asarray(beta_x))
    beta_y = jnp.ravel(jnp.asarray(beta_y))
    data_mesh_beta = jnp.stack([beta_x, beta_y], axis=1)

    source_x_axis = jnp.asarray(source_x_axis)
    source_y_axis = jnp.asarray(source_y_axis)
    n = source_x_axis.shape[0]

    weights, indices, _ = lens_mapping_operator_bilinear_from(
        data_mesh_beta,
        source_x_axis[0],
        source_x_axis[-1],
        source_y_axis[0],
        source_y_axis[-1],
        n,
    )
    return dense_mapping_from_weights_indices(weights, indices, n * n)


__all__ = [
    "lens_mapping_operator_bilinear_from",
    "dense_mapping_from_weights_indices",
    "build_source_grid",
    "build_lens_mapping_matrix",
    "make_square_bbox",
    "infer_source_bbox",
]
