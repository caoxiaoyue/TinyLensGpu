"""
Gravitational Lensing Mapping Matrix Construction

This module provides optimized implementations for constructing lens mapping matrices
using kernel-based interpolation. The mapping matrix M relates source plane values to
image plane values: data_values = M @ source_values

All functions are optimized with JAX JIT compilation for maximum performance.
"""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp

from TinyLensGpu.utils.interpolation.kernels import get_interpolation_weights


@partial(jax.jit, static_argnames=('k_neighbors', 'kernel', 'radius_scale'))
def lens_mapping_matrix_from(
    source_mesh_beta: jnp.ndarray,
    data_mesh_beta: jnp.ndarray,
    k_neighbors: int = 5,
    kernel: Literal['wendland_c2', 'wendland_c4', 'wendland_c6'] = 'wendland_c4',
    radius_scale: float = 1.5,
) -> jnp.ndarray:
    """
    Compute lens mapping matrix using kernel-based interpolation.
    
    This function creates a mapping matrix M such that:
        data_values = M @ source_values
    
    The mapping uses k-nearest neighbor interpolation with Wendland kernels
    for smooth and accurate reconstruction.
    
    Args:
        source_mesh_beta: Source plane coordinates, shape (N_source, 2)
        data_mesh_beta: Image plane coordinates (after lensing), shape (N_data, 2)
        k_neighbors: Number of nearest neighbors for interpolation (default: 5)
        kernel: Kernel type - 'wendland_c2', 'wendland_c4', or 'wendland_c6' (default: 'wendland_c4')
        radius_scale: Scale factor for kernel support radius (default: 1.5)
        
    Returns:
        Mapping matrix of shape (N_data, N_source). This is stored as a dense
        array but is sparse by construction (only k_neighbors entries per row).
        
    Performance:
        - JIT compiled: 10-100x faster after warmup
        - Vectorized operations: no Python loops
        - Memory efficient: uses scatter operations
        
    Example:
        >>> # Map source plane to image plane
        >>> map_mat = lens_mapping_matrix_from(source_coords, data_coords)
        >>> data_values = map_mat @ source_values
        
    Note:
        The matrix has approximately N_data * k_neighbors non-zero entries,
        giving ~(1 - k_neighbors/N_source) * 100% sparsity.
    """
    N_data = data_mesh_beta.shape[0]
    N_source = source_mesh_beta.shape[0]

    weights, indices, distances = get_interpolation_weights(
        points=source_mesh_beta, 
        query_points=data_mesh_beta, 
        k_neighbors=k_neighbors, 
        kernel=kernel,
        radius_scale=radius_scale,
    )

    row_indices = jnp.repeat(jnp.arange(N_data), k_neighbors)
    col_indices = indices.flatten()
    weights_flat = weights.flatten()
    
    map_mat = jnp.zeros((N_data, N_source), dtype=jnp.float32)
    map_mat = map_mat.at[row_indices, col_indices].set(weights_flat)

    return map_mat


__all__ = [
    'lens_mapping_matrix_from'
]
