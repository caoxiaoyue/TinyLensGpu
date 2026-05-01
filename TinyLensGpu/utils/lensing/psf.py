"""
PSF Convolution Operations

This module provides optimized implementations for PSF (Point Spread Function)
convolution operations using matrix representations and FFT.
"""

import functools
from typing import Tuple, Literal

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
from numba import jit

__all__ = [
    'build_psf_matrix_dense',
    'build_psf_matrix_sparse',
    'apply_psf_to_mapping_matrix'
]

@jit(nopython=True)
def _build_matrix_numba(
    h_indices: np.ndarray,
    w_indices: np.ndarray,
    idx_map: np.ndarray,
    psf_data: np.ndarray,
) -> np.ndarray:
    """
    Build dense PSF matrix over unmasked pixels using Numba loops.

    Parameters
    ----------
    h_indices, w_indices : np.ndarray
        Row and column indices of unmasked pixels.
    idx_map : np.ndarray
        Dense image-to-unmasked index map with ``-1`` for masked pixels.
    psf_data : np.ndarray
        PSF kernel values.

    Returns
    -------
    np.ndarray
        Dense PSF matrix with shape ``(n_unmasked, n_unmasked)``.
    """
    n_pixels = len(h_indices)
    psf_h, psf_w = psf_data.shape
    psf_cy, psf_cx = psf_h // 2, psf_w // 2
    
    psf_matrix = np.zeros((n_pixels, n_pixels), dtype=np.float32)
    
    for i in range(n_pixels):
        hi, wi = h_indices[i], w_indices[i]
        
        for kh in range(psf_h):
            for kw in range(psf_w):
                val = psf_data[kh, kw]
                if abs(val) <= 1e-10:
                    continue

                hj = hi - (kh - psf_cy)
                wj = wi - (kw - psf_cx)

                if 0 <= hj < idx_map.shape[0] and 0 <= wj < idx_map.shape[1]:
                    j = idx_map[hj, wj]
                    if j >= 0:
                        psf_matrix[i, j] = val
    
    return psf_matrix


@jit(nopython=True)
def _build_sparse_numba(
    h_indices: np.ndarray,
    w_indices: np.ndarray,
    idx_map: np.ndarray,
    psf_data: np.ndarray,
    max_nnz: int
):
    """
    Build sparse PSF matrix entries (COO triplets) using Numba loops.

    Parameters
    ----------
    h_indices, w_indices : np.ndarray
        Row and column indices of unmasked pixels.
    idx_map : np.ndarray
        Dense image-to-unmasked index map with ``-1`` for masked pixels.
    psf_data : np.ndarray
        PSF kernel values.
    max_nnz : int
        Preallocated capacity for nonzero entries.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(rows, cols, values)`` arrays in COO format.
    """
    n_pixels = len(h_indices)
    psf_h, psf_w = psf_data.shape
    psf_cy, psf_cx = psf_h // 2, psf_w // 2

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

                hj = hi - (kh - psf_cy)
                wj = wi - (kw - psf_cx)

                if 0 <= hj < idx_map.shape[0] and 0 <= wj < idx_map.shape[1]:
                    j = idx_map[hj, wj]
                    if j >= 0:
                        rows[count] = i
                        cols[count] = j
                        values[count] = val
                        count += 1
    
    return rows[:count], cols[:count], values[:count]


def _get_indices_and_map(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract unmasked pixel indices and inverse lookup map.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask where ``True`` marks masked pixels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(h_indices, w_indices, idx_map)`` for unmasked-pixel addressing.
    """
    inv_mask = ~mask
    h_indices, w_indices = np.where(inv_mask)
    
    idx_map = np.full(mask.shape, -1, dtype=np.int32)
    idx_map[h_indices, w_indices] = np.arange(len(h_indices), dtype=np.int32)
    
    return h_indices, w_indices, idx_map


def build_psf_matrix_dense(
    mask: np.ndarray,
    psf_kernel: np.ndarray
) -> jnp.ndarray:
    """
    Build dense PSF convolution matrix using Numba-accelerated construction.
    
    Args:
        mask: 2D boolean array where True indicates masked (invalid) pixels.
        psf_kernel: 2D PSF kernel for convolution.
        
    Returns:
        Dense matrix P of shape (n_valid, n_valid) as JAX array.
    """
    h_indices, w_indices, idx_map = _get_indices_and_map(mask)
    psf_kernel = psf_kernel.astype(np.float32)
    
    psf_matrix_np = _build_matrix_numba(
        h_indices, w_indices, idx_map, psf_kernel
    )
    
    return jnp.array(psf_matrix_np)


def build_psf_matrix_sparse(
    mask: np.ndarray,
    psf_kernel: np.ndarray
):
    """
    Build sparse PSF convolution matrix in JAX BCOO format.
    
    Args:
        mask: 2D boolean array where True indicates masked (invalid) pixels.
        psf_kernel: 2D PSF kernel for convolution.
        
    Returns:
        Sparse BCOO matrix of shape (n_valid, n_valid).
    """
    h_indices, w_indices, idx_map = _get_indices_and_map(mask)
    psf_kernel = psf_kernel.astype(np.float32)
    n_pixels = len(h_indices)
    
    # Estimate max non-zero elements
    max_nnz = psf_kernel.size * n_pixels
    
    rows, cols, values = _build_sparse_numba(
        h_indices, w_indices, idx_map, psf_kernel, max_nnz
    )
    
    indices = jnp.array(np.stack([rows, cols], axis=1), dtype=jnp.int32)
    data = jnp.array(values, dtype=jnp.float32)
    
    return jsparse.BCOO((data, indices), shape=(n_pixels, n_pixels))


@functools.partial(jax.jit, static_argnames=('image_shape',))
def _apply_psf_fft(
    mapping_matrix: jnp.ndarray,
    psf_kernel: jnp.ndarray,
    image_shape: Tuple[int, int],
    unmasked_indices: Tuple[jnp.ndarray, jnp.ndarray],
    *,
    psf_fft: jnp.ndarray | None = None,
    psf_shape: Tuple[int, int] | None = None,
) -> jnp.ndarray:
    """Apply PSF convolution to mapping columns using FFT."""
    h, w = image_shape
    y_indices, x_indices = unmasked_indices
    _psf_shape = psf_shape if psf_shape is not None else psf_kernel.shape
    psf_h, psf_w = int(_psf_shape[0]), int(_psf_shape[1])

    full_grid = jnp.zeros((mapping_matrix.shape[1], h, w), dtype=mapping_matrix.dtype)
    full_grid = full_grid.at[:, y_indices, x_indices].set(mapping_matrix.T)

    fft_shape = (h + psf_h - 1, w + psf_w - 1)
    _psf_fft = psf_fft if psf_fft is not None else jnp.fft.rfft2(psf_kernel, s=fft_shape)
    full_fft = jnp.fft.rfft2(full_grid, s=fft_shape)
    blurred_full = jnp.fft.irfft2(full_fft * _psf_fft, s=fft_shape)

    start_h = (psf_h - 1) // 2
    start_w = (psf_w - 1) // 2
    blurred_grid = blurred_full[:, start_h : start_h + h, start_w : start_w + w]

    return blurred_grid[:, y_indices, x_indices].T


def apply_psf_to_mapping_matrix(
    mapping_matrix: jnp.ndarray,
    psf_kernel: jnp.ndarray,
    image_shape: Tuple[int, int],
    unmasked_indices: Tuple[jnp.ndarray, jnp.ndarray],
    method: Literal['fft', 'matrix'] = 'fft',
    *,
    psf_fft: jnp.ndarray | None = None,
    psf_shape: Tuple[int, int] | None = None,
) -> jnp.ndarray:
    """
    Apply PSF convolution to the lens mapping matrix.
    
    Args:
        mapping_matrix: Mapping matrix of shape (n_unmasked, n_source).
        psf_kernel: 2D PSF kernel.
        image_shape: Tuple (height, width) of the original image.
        unmasked_indices: Tuple of (y_indices, x_indices) unmasked pixel positions.
        method: 'fft' (faster) or 'matrix' (legacy/verification).
        
    Returns:
        Blurred mapping matrix of shape (n_unmasked, n_source).
    """
    if method == 'fft':
        kwargs = {}
        if psf_fft is not None:
            kwargs['psf_fft'] = psf_fft
        if psf_shape is not None:
            kwargs['psf_shape'] = psf_shape
        elif psf_fft is not None:
            kwargs['psf_shape'] = tuple(int(s) for s in psf_kernel.shape)
        return _apply_psf_fft(mapping_matrix, psf_kernel, image_shape, unmasked_indices, **kwargs)
    elif method == 'matrix':
        # Reconstruct mask on CPU for Numba
        y_idx, x_idx = unmasked_indices
        y_idx_np = np.array(y_idx)
        x_idx_np = np.array(x_idx)
        
        mask = np.ones(image_shape, dtype=bool)
        mask[y_idx_np, x_idx_np] = False
        
        psf_matrix = build_psf_matrix_dense(mask, np.array(psf_kernel))
        return psf_matrix @ mapping_matrix
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fft' or 'matrix'.")
