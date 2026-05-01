"""Helpers for pixelized source-plane grids and lens mappings."""

# pyright: reportMissingImports=false

from __future__ import annotations

import jax.numpy as jnp

from TinyLensGpu.utils.lensing.mapping import (
    dense_mapping_from_weights_indices,
    lens_mapping_operator_bilinear_rectangular_from,
)


def build_source_grid(
    nx: int,
    ny: int,
    half_size: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build a rectangular source-plane grid.

    Parameters
    ----------
    nx, ny : int
        Number of samples along the x and y axes.
    half_size : float
        Grid extends from ``-half_size`` to ``+half_size`` on both axes.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ``(x_axis, y_axis, xgrid, ygrid)`` where the grids follow row-major
        ``jnp.meshgrid(..., indexing='xy')`` layout.
    """
    x_axis = jnp.linspace(-half_size, half_size, int(nx))
    y_axis = jnp.linspace(-half_size, half_size, int(ny))
    xgrid, ygrid = jnp.meshgrid(x_axis, y_axis, indexing="xy")
    return x_axis, y_axis, xgrid, ygrid


def build_lens_mapping_matrix(
    beta_x: jnp.ndarray,
    beta_y: jnp.ndarray,
    source_x_axis: jnp.ndarray,
    source_y_axis: jnp.ndarray,
) -> jnp.ndarray:
    """Build a dense image-to-source bilinear mapping matrix.

    Parameters
    ----------
    beta_x, beta_y : jnp.ndarray
        Ray-traced source-plane coordinates for each image pixel.
    source_x_axis, source_y_axis : jnp.ndarray
        Rectangular source-grid axes.

    Returns
    -------
    jnp.ndarray
        Dense mapping matrix of shape ``(N_d, N_s)``.
    """
    beta_x = jnp.ravel(jnp.asarray(beta_x))
    beta_y = jnp.ravel(jnp.asarray(beta_y))
    data_mesh_beta = jnp.stack([beta_x, beta_y], axis=1)

    source_x_axis = jnp.asarray(source_x_axis)
    source_y_axis = jnp.asarray(source_y_axis)
    nx = int(source_x_axis.shape[0])
    ny = int(source_y_axis.shape[0])

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


__all__ = ["build_source_grid", "build_lens_mapping_matrix"]
