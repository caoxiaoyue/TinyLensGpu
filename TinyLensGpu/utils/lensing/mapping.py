"""Mapping utilities for pixelized-source lensing reconstruction."""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp

from TinyLensGpu.utils.interpolation.kernels import get_interpolation_weights


def _dense_mapping_from_weights_indices(
    weights: jnp.ndarray,
    indices: jnp.ndarray,
    n_source: int,
) -> jnp.ndarray:
    n_data = int(weights.shape[0])
    n_neighbors = int(weights.shape[1])
    row_indices = jnp.repeat(jnp.arange(n_data, dtype=jnp.int32), n_neighbors)
    col_indices = indices.reshape(-1).astype(jnp.int32)
    weights_flat = weights.reshape(-1)
    mapping_matrix = jnp.zeros((n_data, int(n_source)), dtype=weights_flat.dtype)
    return mapping_matrix.at[row_indices, col_indices].add(weights_flat)


@partial(jax.jit, static_argnames=("k_neighbors", "kernel", "radius_scale"))
def lens_mapping_matrix_from(
    source_mesh_beta: jnp.ndarray,
    data_mesh_beta: jnp.ndarray,
    k_neighbors: int = 5,
    kernel: Literal["wendland_c2", "wendland_c4", "wendland_c6"] = "wendland_c4",
    radius_scale: float = 1.5,
) -> jnp.ndarray:
    """Compute dense mapping matrix using KNN Wendland interpolation."""
    n_source = source_mesh_beta.shape[0]
    weights, indices, _ = get_interpolation_weights(
        points=source_mesh_beta,
        query_points=data_mesh_beta,
        k_neighbors=k_neighbors,
        kernel=kernel,
        radius_scale=radius_scale,
    )
    return _dense_mapping_from_weights_indices(weights, indices, int(n_source))


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
    """Build bilinear interpolation operator entries for a rectangular source grid."""
    nx_i = int(nx)
    ny_i = int(ny)

    query = jnp.asarray(data_mesh_beta, dtype=jnp.float32)
    x_min_f = jnp.asarray(x_min, dtype=jnp.float32)
    x_max_f = jnp.asarray(x_max, dtype=jnp.float32)
    y_min_f = jnp.asarray(y_min, dtype=jnp.float32)
    y_max_f = jnp.asarray(y_max, dtype=jnp.float32)

    dx = (x_max_f - x_min_f) / jnp.asarray(max(nx_i - 1, 1), dtype=jnp.float32)
    dy = (y_max_f - y_min_f) / jnp.asarray(max(ny_i - 1, 1), dtype=jnp.float32)

    ux = (query[:, 0] - x_min_f) / (dx + 1e-12)
    uy = (query[:, 1] - y_min_f) / (dy + 1e-12)

    ix0 = jnp.floor(ux).astype(jnp.int32)
    iy0 = jnp.floor(uy).astype(jnp.int32)
    fx = ux - jnp.floor(ux)
    fy = uy - jnp.floor(uy)

    valid = (
        (ux >= 0.0)
        & (ux <= float(nx_i - 1))
        & (uy >= 0.0)
        & (uy <= float(ny_i - 1))
    )

    ix0_c = jnp.clip(ix0, 0, nx_i - 1)
    iy0_c = jnp.clip(iy0, 0, ny_i - 1)
    ix1_c = jnp.clip(ix0_c + 1, 0, nx_i - 1)
    iy1_c = jnp.clip(iy0_c + 1, 0, ny_i - 1)

    i00 = iy0_c * nx_i + ix0_c
    i10 = iy0_c * nx_i + ix1_c
    i01 = iy1_c * nx_i + ix0_c
    i11 = iy1_c * nx_i + ix1_c

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    weights = jnp.stack([w00, w10, w01, w11], axis=1)
    weights = jnp.where(valid[:, None], weights, 0.0).astype(jnp.float32)
    indices = jnp.stack([i00, i10, i01, i11], axis=1).astype(jnp.int32)
    return weights, indices, valid


@partial(jax.jit, static_argnames=("nx", "ny"))
def lens_mapping_matrix_bilinear_rectangular_from(
    data_mesh_beta: jnp.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
) -> jnp.ndarray:
    """Compute dense bilinear mapping matrix for a rectangular source grid."""
    weights, indices, _ = lens_mapping_operator_bilinear_rectangular_from(
        data_mesh_beta=data_mesh_beta,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        nx=nx,
        ny=ny,
    )
    return _dense_mapping_from_weights_indices(weights, indices, int(nx) * int(ny))


__all__ = [
    "lens_mapping_matrix_from",
    "lens_mapping_operator_bilinear_rectangular_from",
    "lens_mapping_matrix_bilinear_rectangular_from",
]
