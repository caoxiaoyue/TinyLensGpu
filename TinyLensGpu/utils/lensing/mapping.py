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


@partial(jax.jit, static_argnames=("nx", "ny"))
def lens_mapping_operator_bilinear_rectangular_from(
    data_mesh_beta: jnp.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build bilinear interpolation weights/indices on a rectangular source grid."""
    beta = jnp.asarray(data_mesh_beta, dtype=jnp.float32)

    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)

    ux = (beta[:, 0] - x_min) / dx
    uy = (beta[:, 1] - y_min) / dy

    ix0 = jnp.floor(ux).astype(jnp.int32)
    iy0 = jnp.floor(uy).astype(jnp.int32)
    fx = ux - ix0
    fy = uy - iy0

    valid = (ux >= 0) & (ux <= nx - 1) & (uy >= 0) & (uy <= ny - 1)

    ix0_c = jnp.clip(ix0, 0, nx - 1)
    iy0_c = jnp.clip(iy0, 0, ny - 1)
    ix1_c = jnp.clip(ix0_c + 1, 0, nx - 1)
    iy1_c = jnp.clip(iy0_c + 1, 0, ny - 1)

    i00 = iy0_c * nx + ix0_c
    i10 = iy0_c * nx + ix1_c
    i01 = iy1_c * nx + ix0_c
    i11 = iy1_c * nx + ix1_c

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    weights = jnp.where(valid[:, None],
                        jnp.stack([w00, w10, w01, w11], axis=1), 0.0)
    indices = jnp.stack([i00, i10, i01, i11], axis=1).astype(jnp.int32)
    return weights, indices, valid


def build_source_grid(nx, ny, xmin, xmax, ymin, ymax):
    """Build a rectangular source-plane grid spanning [xmin, xmax] x [ymin, ymax].

    Returns (x_axis, y_axis, xgrid, ygrid) with ``jnp.meshgrid(..., indexing='xy')`` layout.
    """
    x_axis = jnp.linspace(xmin, xmax, int(nx))
    y_axis = jnp.linspace(ymin, ymax, int(ny))
    xgrid, ygrid = jnp.meshgrid(x_axis, y_axis, indexing="xy")
    return x_axis, y_axis, xgrid, ygrid


def infer_source_bbox(beta_x, beta_y, padding=0.0):
    """Infer source-plane bounding box from ray-traced beta points.

    Computes min/max of beta coordinates and adds a fractional padding
    margin on each side (default 0 — no padding).  Ensures a minimum
    span of 1e-6 in each direction so that downstream grid construction
    is well-defined.
    """
    beta_x = jnp.ravel(jnp.asarray(beta_x))
    beta_y = jnp.ravel(jnp.asarray(beta_y))
    xmin = jnp.min(beta_x)
    xmax = jnp.max(beta_x)
    ymin = jnp.min(beta_y)
    ymax = jnp.max(beta_y)
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
    return xmin, xmax, ymin, ymax


def build_lens_mapping_matrix(beta_x, beta_y, source_x_axis, source_y_axis):
    """Build dense bilinear mapping matrix (N_d image pixels x N_s source pixels)."""
    beta_x = jnp.ravel(jnp.asarray(beta_x))
    beta_y = jnp.ravel(jnp.asarray(beta_y))
    data_mesh_beta = jnp.stack([beta_x, beta_y], axis=1)

    source_x_axis = jnp.asarray(source_x_axis)
    source_y_axis = jnp.asarray(source_y_axis)
    nx = source_x_axis.shape[0]
    ny = source_y_axis.shape[0]

    weights, indices, _ = lens_mapping_operator_bilinear_rectangular_from(
        data_mesh_beta,
        source_x_axis[0],
        source_x_axis[-1],
        source_y_axis[0],
        source_y_axis[-1],
        nx,
        ny,
    )
    return dense_mapping_from_weights_indices(weights, indices, nx * ny)


__all__ = [
    "lens_mapping_operator_bilinear_rectangular_from",
    "dense_mapping_from_weights_indices",
    "build_source_grid",
    "build_lens_mapping_matrix",
    "infer_source_bbox",
]
