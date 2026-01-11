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


__all__ = [
    'build_psf_matrix_dense',
    'build_psf_matrix_sparse'
]
