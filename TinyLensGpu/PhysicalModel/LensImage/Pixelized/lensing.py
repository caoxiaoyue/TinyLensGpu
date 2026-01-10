"""
Gravitational Lensing Operations using JAX

This module provides optimized implementations for:
1. Lens mapping matrix construction via kernel interpolation
2. PSF convolution operations (dense and sparse matrix methods)
3. Blurred lens mapping matrix computation

All functions are optimized with JAX JIT compilation for maximum performance.
"""

from functools import partial
from typing import Optional, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .interp_kernel import get_interpolation_weights


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


def build_psf_matrix_dense(
    mask: np.ndarray,
    psf_kernel: np.ndarray
) -> jnp.ndarray:
    """
    Build dense PSF convolution matrix using Numba-accelerated construction.
    
    This matrix P represents PSF convolution as a linear operation:
        blurred_image = P @ unblurred_image
    
    Args:
        mask: 2D boolean array where True indicates masked (invalid) pixels
        psf_kernel: 2D PSF kernel for convolution, assumed centered
        
    Returns:
        Dense matrix P of shape (n_valid_pixels, n_valid_pixels) as JAX array
        
    Memory:
        Dense storage: O(n_valid_pixels^2)
        For 1000 valid pixels: ~4 MB (float32)
        For 10000 valid pixels: ~400 MB (float32)
        
    Performance:
        Construction is accelerated with Numba JIT compilation.
        Once built, matrix-vector multiplication is highly optimized via BLAS.
        
    Note:
        Consider using build_psf_matrix_sparse() for large images to reduce
        memory usage and improve performance.
    """
    from numba import jit as numba_jit
    
    inv_mask_np = np.array(~mask)
    psf_kernel_np = np.array(psf_kernel, dtype=np.float32)
    
    h_indices_np, w_indices_np = np.where(inv_mask_np)
    n_data_pixels = len(h_indices_np)
    
    psf_h, psf_w = psf_kernel_np.shape
    psf_center_h = psf_h // 2
    psf_center_w = psf_w // 2
    
    @numba_jit(nopython=True)
    def _build_matrix_numba(
        h_indices: np.ndarray,
        w_indices: np.ndarray,
        psf_data: np.ndarray,
        psf_h: int,
        psf_w: int,
        psf_center_h: int,
        psf_center_w: int,
        n_pixels: int
    ) -> np.ndarray:
        """Build PSF matrix with Numba acceleration."""
        psf_matrix = np.zeros((n_pixels, n_pixels), dtype=np.float32)
        
        for i in range(n_pixels):
            hi, wi = h_indices[i], w_indices[i]
            
            for j in range(n_pixels):
                hj, wj = h_indices[j], w_indices[j]
                
                dh = hi - hj + psf_center_h
                dw = wi - wj + psf_center_w
                
                if 0 <= dh < psf_h and 0 <= dw < psf_w:
                    psf_matrix[i, j] = psf_data[dh, dw]
        
        return psf_matrix
    
    psf_matrix_np = _build_matrix_numba(
        h_indices_np, w_indices_np, 
        psf_kernel_np, psf_h, psf_w, 
        psf_center_h, psf_center_w, 
        n_data_pixels
    )
    
    return jnp.array(psf_matrix_np)


def build_psf_matrix_sparse(
    mask: np.ndarray,
    psf_kernel: np.ndarray
):
    """
    Build sparse PSF convolution matrix in JAX BCOO format.
    
    Like build_psf_matrix_dense() but uses sparse storage for efficiency.
    
    Args:
        mask: 2D boolean array where True indicates masked (invalid) pixels
        psf_kernel: 2D PSF kernel for convolution, assumed centered
        
    Returns:
        Sparse BCOO matrix of shape (n_valid_pixels, n_valid_pixels)
        
    Memory:
        Sparse storage: O(nnz) where nnz ≈ n_valid_pixels * psf_kernel_size
        Typically 100-1000x less memory than dense version for small PSF kernels.
        
    Performance:
        - Construction: Numba-accelerated, similar to dense version
        - Matrix-vector multiplication: Often faster than dense for sparse PSF
        - Scales better with PSF kernel size
        
    Advantages over dense:
        - Much lower memory usage
        - Faster sparse @ dense matrix multiplication
        - Better for large images or large PSF kernels
        
    Example:
        >>> psf_mat = build_psf_matrix_sparse(mask, psf_kernel)
        >>> blurred = psf_mat @ unblurred  # Sparse matrix multiplication
    """
    import jax.experimental.sparse as jsparse
    from numba import jit as numba_jit
    
    inv_mask_np = np.array(~mask)
    psf_kernel_np = np.array(psf_kernel, dtype=np.float32)
    
    h_indices_np, w_indices_np = np.where(inv_mask_np)
    n_data_pixels = len(h_indices_np)
    
    psf_h, psf_w = psf_kernel_np.shape
    psf_center_h = psf_h // 2
    psf_center_w = psf_w // 2
    
    max_nnz = psf_h * psf_w * n_data_pixels
    
    @numba_jit(nopython=True)
    def _build_sparse_numba(
        h_indices: np.ndarray,
        w_indices: np.ndarray,
        psf_data: np.ndarray,
        psf_h: int,
        psf_w: int,
        psf_center_h: int,
        psf_center_w: int,
        n_pixels: int,
        max_nnz: int
    ):
        """Build sparse PSF matrix with Numba acceleration."""
        rows = np.zeros(max_nnz, dtype=np.int32)
        cols = np.zeros(max_nnz, dtype=np.int32)
        values = np.zeros(max_nnz, dtype=np.float32)
        
        count = 0
        for i in range(n_pixels):
            hi, wi = h_indices[i], w_indices[i]
            
            for j in range(n_pixels):
                hj, wj = h_indices[j], w_indices[j]
                
                dh = hi - hj + psf_center_h
                dw = wi - wj + psf_center_w
                
                if 0 <= dh < psf_h and 0 <= dw < psf_w:
                    val = psf_data[dh, dw]
                    if abs(val) > 1e-10:
                        rows[count] = i
                        cols[count] = j
                        values[count] = val
                        count += 1
        
        return rows[:count], cols[:count], values[:count]
    
    rows_np, cols_np, values_np = _build_sparse_numba(
        h_indices_np, w_indices_np,
        psf_kernel_np, psf_h, psf_w,
        psf_center_h, psf_center_w,
        n_data_pixels, max_nnz
    )
    
    indices = jnp.array(np.stack([rows_np, cols_np], axis=1), dtype=jnp.int32)
    data = jnp.array(values_np, dtype=jnp.float32)
    
    psf_matrix_sparse = jsparse.BCOO(
        (data, indices),
        shape=(n_data_pixels, n_data_pixels)
    )
    
    return psf_matrix_sparse
