"""
PSF Convolution Operations

This module provides optimized implementations for PSF (Point Spread Function) 
convolution operations using matrix representations. Supports both dense and 
sparse matrix formats.

All functions are optimized with Numba JIT compilation for construction and
JAX for matrix operations.
"""

import jax.numpy as jnp
import numpy as np


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
    idx_map_np = np.full(inv_mask_np.shape, -1, dtype=np.int32)
    idx_map_np[h_indices_np, w_indices_np] = np.arange(n_data_pixels, dtype=np.int32)
    
    psf_h, psf_w = psf_kernel_np.shape
    psf_center_h = psf_h // 2
    psf_center_w = psf_w // 2
    
    @numba_jit(nopython=True)
    def _build_matrix_numba(
        h_indices: np.ndarray,
        w_indices: np.ndarray,
        idx_map: np.ndarray,
        psf_data: np.ndarray,
        psf_h: int,
        psf_w: int,
        psf_center_h: int,
        psf_center_w: int,
        n_pixels: int
    ) -> np.ndarray:
        psf_matrix = np.zeros((n_pixels, n_pixels), dtype=np.float32)
        
        for i in range(n_pixels):
            hi, wi = h_indices[i], w_indices[i]
            
            for kh in range(psf_h):
                for kw in range(psf_w):
                    val = psf_data[kh, kw]
                    if abs(val) <= 1e-10:
                        continue

                    hj = hi - (kh - psf_center_h)
                    wj = wi - (kw - psf_center_w)

                    if 0 <= hj < idx_map.shape[0] and 0 <= wj < idx_map.shape[1]:
                        j = idx_map[hj, wj]
                        if j >= 0:
                            psf_matrix[i, j] = val
        
        return psf_matrix
    
    psf_matrix_np = _build_matrix_numba(
        h_indices_np, w_indices_np, idx_map_np,
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
    idx_map_np = np.full(inv_mask_np.shape, -1, dtype=np.int32)
    idx_map_np[h_indices_np, w_indices_np] = np.arange(n_data_pixels, dtype=np.int32)
    
    psf_h, psf_w = psf_kernel_np.shape
    psf_center_h = psf_h // 2
    psf_center_w = psf_w // 2
    
    max_nnz = psf_h * psf_w * n_data_pixels
    
    @numba_jit(nopython=True)
    def _build_sparse_numba(
        h_indices: np.ndarray,
        w_indices: np.ndarray,
        idx_map: np.ndarray,
        psf_data: np.ndarray,
        psf_h: int,
        psf_w: int,
        psf_center_h: int,
        psf_center_w: int,
        n_pixels: int,
        max_nnz: int
    ):
        rows = np.zeros(max_nnz, dtype=np.int32)
        cols = np.zeros(max_nnz, dtype=np.int32)
        values = np.zeros(max_nnz, dtype=np.float32)
        
        count = 0
        for i in range(n_pixels):
            hi, wi = h_indices[i], w_indices[i]
            
            for kh in range(psf_h):
                for kw in range(psf_w):
                    val = psf_data[kh, kw]
                    if abs(val) <= 1e-10:
                        continue

                    hj = hi - (kh - psf_center_h)
                    wj = wi - (kw - psf_center_w)

                    if 0 <= hj < idx_map.shape[0] and 0 <= wj < idx_map.shape[1]:
                        j = idx_map[hj, wj]
                        if j >= 0:
                            rows[count] = i
                            cols[count] = j
                            values[count] = val
                            count += 1
        
        return rows[:count], cols[:count], values[:count]
    
    rows_np, cols_np, values_np = _build_sparse_numba(
        h_indices_np, w_indices_np, idx_map_np,
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


__all__ = [
    'build_psf_matrix_dense',
    'build_psf_matrix_sparse',
    'apply_psf_to_mapping_matrix'
]


from typing import Tuple
import functools
import jax
from jax import jit
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
import numpy as np


@functools.partial(jax.jit, static_argnames=('image_shape',))
def apply_psf_to_mapping_matrix(
    mapping_matrix: jnp.ndarray,
    psf_kernel: jnp.ndarray,
    image_shape: Tuple[int, int],
    unmasked_indices: Tuple[jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """
    Apply PSF convolution to the lens mapping matrix using dense 2D convolution.
    
    This function avoids constructing the large dense/sparse PSF matrix by
    treating the mapping matrix as a batch of source images and convolving
    them efficiently.
    
    Args:
        mapping_matrix: Mapping matrix of shape (n_unmasked, n_source)
        psf_kernel: 2D PSF kernel
        image_shape: Tuple (height, width) of the original image
        unmasked_indices: Tuple of (y_indices, x_indices) arrays indicating 
                         unmasked pixel positions.
        
    Returns:
        Blurred mapping matrix of shape (n_unmasked, n_source)
    """
    n_unmasked, n_source = mapping_matrix.shape
    h, w = image_shape
    y_indices, x_indices = unmasked_indices
    
    # 1. Scatter mapping matrix rows to full 2D grid
    # Initialize full grid (n_source, h, w)
    full_grid = jnp.zeros((n_source, h, w), dtype=mapping_matrix.dtype)
    
    # Scatter the mapping matrix values into the grid
    full_grid = full_grid.at[:, y_indices, x_indices].set(mapping_matrix.T)
    
    # 2. Convolve with PSF
    # We vmap over the source dimension (axis 0)
    convolve_fn = jax.vmap(
        lambda img: convolve2d(img, psf_kernel, mode='same'),
        in_axes=0, out_axes=0
    )
    
    blurred_grid = convolve_fn(full_grid)
    
    # 3. Gather back unmasked pixels
    blurred_unmasked = blurred_grid[:, y_indices, x_indices]
    
    # Transpose back to (n_unmasked, n_source)
    return blurred_unmasked.T
