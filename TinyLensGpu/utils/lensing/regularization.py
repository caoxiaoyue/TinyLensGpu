"""
Regularization Matrix Construction for Source Reconstruction

This module provides various covariance kernels (exponential, Gaussian, Matérn)
for constructing regularization matrices used in pixelized source reconstruction.
The regularization matrix penalizes non-smooth source reconstructions.
"""

import jax.numpy as jnp
import jax
from functools import partial


def sparse_regularization_dense_from(
    rows: jnp.ndarray,
    cols: jnp.ndarray,
    values: jnp.ndarray,
    n_source: int,
) -> jnp.ndarray:
    """
    Convert sparse COO regularization entries to dense matrix form.

    Parameters
    ----------
    rows : jnp.ndarray
        Row indices of COO entries.
    cols : jnp.ndarray
        Column indices of COO entries.
    values : jnp.ndarray
        COO entry values.
    n_source : int
        Matrix size along each axis.

    Returns
    -------
    jnp.ndarray
        Dense matrix of shape ``(n_source, n_source)``.
    """
    dense = jnp.zeros((int(n_source), int(n_source)), dtype=jnp.asarray(values).dtype)
    return dense.at[(rows, cols)].add(values)


def _ridge_from_coefficient(coefficient: float) -> jnp.ndarray:
    """
    Return stabilizing diagonal ridge term from regularization strength.

    Parameters
    ----------
    coefficient : float
        User-provided regularization amplitude.

    Returns
    -------
    jnp.ndarray
        Positive ridge value used to avoid singular operators.
    """
    return jnp.maximum(1e-8, 1e-6 * jnp.maximum(1.0, jnp.asarray(coefficient, dtype=jnp.float32)))


@partial(jax.jit, static_argnames=['nx', 'ny'])
def _regularization_rect_zero_sparse(
    coefficient: float,
    nx: int,
    ny: int,
):
    """
    Build sparse zero-order rectangular regularization (diagonal only).

    Parameters
    ----------
    coefficient : float
        Regularization amplitude.
    nx : int
        Source-grid width.
    ny : int
        Source-grid height.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]
        COO rows, cols, values, and source size.
    """
    n_source = int(nx) * int(ny)
    rows = jnp.arange(n_source, dtype=jnp.int32)
    cols = jnp.arange(n_source, dtype=jnp.int32)
    values = jnp.full((n_source,), jnp.asarray(coefficient, dtype=jnp.float32))
    values = values + _ridge_from_coefficient(coefficient)
    return rows, cols, values.astype(jnp.float32), n_source


@partial(jax.jit, static_argnames=['nx', 'ny'])
def _regularization_rect_gradient_sparse(
    coefficient: float,
    nx: int,
    ny: int,
):
    """
    Build sparse first-order (gradient) rectangular regularization.

    Parameters
    ----------
    coefficient : float
        Regularization amplitude.
    nx : int
        Source-grid width.
    ny : int
        Source-grid height.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]
        COO rows, cols, values, and source size.
    """
    nx_i = int(nx)
    ny_i = int(ny)
    n_source = nx_i * ny_i

    if nx_i > 1:
        horiz_iy = jnp.repeat(jnp.arange(ny_i, dtype=jnp.int32), nx_i - 1)
        horiz_ix = jnp.tile(jnp.arange(nx_i - 1, dtype=jnp.int32), ny_i)
        horiz_a = horiz_iy * nx_i + horiz_ix
        horiz_b = horiz_a + 1
    else:
        horiz_a = jnp.zeros((0,), dtype=jnp.int32)
        horiz_b = jnp.zeros((0,), dtype=jnp.int32)

    if ny_i > 1:
        vert_iy = jnp.repeat(jnp.arange(ny_i - 1, dtype=jnp.int32), nx_i)
        vert_ix = jnp.tile(jnp.arange(nx_i, dtype=jnp.int32), ny_i - 1)
        vert_a = vert_iy * nx_i + vert_ix
        vert_b = vert_a + nx_i
    else:
        vert_a = jnp.zeros((0,), dtype=jnp.int32)
        vert_b = jnp.zeros((0,), dtype=jnp.int32)

    edge_a = jnp.concatenate([horiz_a, vert_a], axis=0)
    edge_b = jnp.concatenate([horiz_b, vert_b], axis=0)
    n_edges = edge_a.shape[0]

    pair_rows = jnp.stack([edge_a, edge_a, edge_b, edge_b], axis=1).reshape(-1)
    pair_cols = jnp.stack([edge_a, edge_b, edge_a, edge_b], axis=1).reshape(-1)
    pair_vals = jnp.tile(
        jnp.array([coefficient, -coefficient, -coefficient, coefficient], dtype=jnp.float32),
        (n_edges,),
    )

    diag_rows = jnp.arange(n_source, dtype=jnp.int32)
    diag_cols = jnp.arange(n_source, dtype=jnp.int32)
    diag_vals = jnp.full((n_source,), _ridge_from_coefficient(coefficient), dtype=jnp.float32)

    rows = jnp.concatenate([pair_rows, diag_rows], axis=0)
    cols = jnp.concatenate([pair_cols, diag_cols], axis=0)
    vals = jnp.concatenate([pair_vals, diag_vals], axis=0)
    return rows, cols, vals.astype(jnp.float32), n_source


@partial(jax.jit, static_argnames=['nx', 'ny'])
def _regularization_rect_curvature_sparse(
    coefficient: float,
    nx: int,
    ny: int,
):
    """
    Build sparse second-order (curvature) rectangular regularization.

    Parameters
    ----------
    coefficient : float
        Regularization amplitude.
    nx : int
        Source-grid width.
    ny : int
        Source-grid height.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]
        COO rows, cols, values, and source size.
    """
    nx_i = int(nx)
    ny_i = int(ny)
    n_source = nx_i * ny_i

    if nx_i < 3 or ny_i < 3:
        return _regularization_rect_zero_sparse(coefficient, nx_i, ny_i)

    xs = jnp.arange(1, nx_i - 1, dtype=jnp.int32)
    ys = jnp.arange(1, ny_i - 1, dtype=jnp.int32)
    xx, yy = jnp.meshgrid(xs, ys, indexing='xy')
    center = (yy * nx_i + xx).reshape(-1)
    left = center - 1
    right = center + 1
    up = center - nx_i
    down = center + nx_i

    stencil_idx = jnp.stack([center, left, right, up, down], axis=1)
    stencil_val = jnp.array([4.0, -1.0, -1.0, -1.0, -1.0], dtype=jnp.float32)

    rows = jnp.repeat(stencil_idx, repeats=5, axis=1).reshape(-1)
    cols = jnp.tile(stencil_idx, (1, 5)).reshape(-1)
    row_vals = (stencil_val[:, None] * stencil_val[None, :]).reshape(-1)
    vals = (jnp.tile(row_vals, (stencil_idx.shape[0],)) * coefficient).astype(jnp.float32)

    diag_rows = jnp.arange(n_source, dtype=jnp.int32)
    diag_cols = jnp.arange(n_source, dtype=jnp.int32)
    diag_vals = jnp.full((n_source,), _ridge_from_coefficient(coefficient), dtype=jnp.float32)

    rows = jnp.concatenate([rows, diag_rows], axis=0)
    cols = jnp.concatenate([cols, diag_cols], axis=0)
    vals = jnp.concatenate([vals, diag_vals], axis=0)
    return rows, cols, vals.astype(jnp.float32), n_source


def regularization_sparse_rectangular_from(
    coefficient: float,
    nx: int,
    ny: int,
    reg_scheme: str = 'gradient',
):
    """
    Build sparse rectangular-grid regularization in COO form.

    Parameters
    ----------
    coefficient : float
        Regularization amplitude.
    nx : int
        Source-grid width.
    ny : int
        Source-grid height.
    reg_scheme : str, optional
        One of ``'zero'``, ``'gradient'``, or ``'curvature'``.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]
        COO rows, cols, values, and source size.

    Raises
    ------
    ValueError
        If regularization scheme or grid shape is invalid.
    """
    scheme = str(reg_scheme).strip().lower()
    valid = {'zero', 'gradient', 'curvature'}
    if scheme not in valid:
        raise ValueError(f"Unknown reg_scheme: '{reg_scheme}'. Must be one of {valid}.")

    nx_i = int(nx)
    ny_i = int(ny)
    if nx_i <= 0 or ny_i <= 0:
        raise ValueError(f"Rectangular grid shape must be positive, got nx={nx_i}, ny={ny_i}.")

    if scheme == 'zero':
        return _regularization_rect_zero_sparse(coefficient, nx_i, ny_i)
    if scheme == 'gradient':
        return _regularization_rect_gradient_sparse(coefficient, nx_i, ny_i)
    return _regularization_rect_curvature_sparse(coefficient, nx_i, ny_i)


@jax.jit
def exp_cov_matrix_from(
    scale_coefficient: float,
    pixel_points: jnp.ndarray,
) -> jnp.ndarray:
    """
    Construct the source brightness covariance matrix using exponential kernel.
    This matrix determines the regularization pattern (i.e., how different source pixels are smoothed).

    The covariance matrix includes one non-linear parameter, the scale coefficient,
    which determines the typical scale of the regularization pattern.

    Parameters
    ----------
    scale_coefficient : float
        The typical scale of the regularization pattern.
    pixel_points : jnp.ndarray
        A 2D array with shape [N_source_pixels, 2], containing source pixelization
        coordinates on the source plane: [[x1,y1], [x2,y2], ...]

    Returns
    -------
    jnp.ndarray
        The source covariance matrix (2D array), shape [N_source_pixels, N_source_pixels].
    """
    diff = pixel_points[:, None, :] - pixel_points[None, :, :]

    distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))

    covariance_matrix = jnp.exp(-distances / scale_coefficient)

    # Use relative jitter for better numerical stability
    n_points = len(pixel_points)
    jitter = 1e-6 * jnp.trace(covariance_matrix) / n_points
    covariance_matrix = covariance_matrix + jnp.eye(n_points) * jitter

    return covariance_matrix


@jax.jit
def gauss_cov_matrix_from(
    scale_coefficient: float,
    pixel_points: jnp.ndarray,
) -> jnp.ndarray:
    """
    Construct the source brightness covariance matrix using Gaussian (RBF) kernel.
    This matrix determines the regularization pattern (i.e., how different source pixels are smoothed).

    The covariance matrix includes one non-linear parameter, the scale coefficient,
    which determines the typical scale of the regularization pattern.

    Parameters
    ----------
    scale_coefficient : float
        The typical scale of the regularization pattern.
    pixel_points : jnp.ndarray
        A 2D array with shape [N_source_pixels, 2], containing source pixelization
        coordinates on the source plane: [[x1,y1], [x2,y2], ...]

    Returns
    -------
    jnp.ndarray
        The source covariance matrix (2D array), shape [N_source_pixels, N_source_pixels].
    """
    diff = pixel_points[:, None, :] - pixel_points[None, :, :]

    distances_sq = jnp.sum(diff**2, axis=-1)

    covariance_matrix = jnp.exp(-distances_sq / (2 * scale_coefficient**2))

    # Use relative jitter for better numerical stability
    n_points = len(pixel_points)
    jitter = 1e-6 * jnp.trace(covariance_matrix) / n_points
    covariance_matrix = covariance_matrix + jnp.eye(n_points) * jitter

    return covariance_matrix


@jax.jit
def matern32_cov_matrix_from(
    scale_coefficient: float,
    pixel_points: jnp.ndarray,
) -> jnp.ndarray:
    """
    Construct the source brightness covariance matrix using Matern-3/2 kernel.
    This matrix determines the regularization pattern (i.e., how different source pixels are smoothed).

    The Matern-3/2 kernel is once differentiable and provides more flexibility than the exponential kernel.
    It is often used when the underlying process is expected to be smooth but not infinitely differentiable.

    Parameters
    ----------
    scale_coefficient : float
        The typical scale (length scale) of the regularization pattern.
    pixel_points : jnp.ndarray
        A 2D array with shape [N_source_pixels, 2], containing source pixelization
        coordinates on the source plane: [[x1,y1], [x2,y2], ...]

    Returns
    -------
    jnp.ndarray
        The source covariance matrix (2D array), shape [N_source_pixels, N_source_pixels].

    Notes
    -----
    The Matern-3/2 kernel is defined as:
        K(r) = (1 + sqrt(3) * r / ℓ) * exp(-sqrt(3) * r / ℓ)
    where r is the Euclidean distance and ℓ is the scale coefficient.
    """
    diff = pixel_points[:, None, :] - pixel_points[None, :, :]

    distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))

    sqrt3 = jnp.sqrt(3.0)
    scaled_dist = sqrt3 * distances / scale_coefficient

    covariance_matrix = (1.0 + scaled_dist) * jnp.exp(-scaled_dist)

    # Use relative jitter for better numerical stability
    n_points = len(pixel_points)
    jitter = 1e-6 * jnp.trace(covariance_matrix) / n_points
    covariance_matrix = covariance_matrix + jnp.eye(n_points) * jitter

    return covariance_matrix


@jax.jit
def matern52_cov_matrix_from(
    scale_coefficient: float,
    pixel_points: jnp.ndarray,
) -> jnp.ndarray:
    """
    Construct the source brightness covariance matrix using Matern-5/2 kernel.
    This matrix determines the regularization pattern (i.e., how different source pixels are smoothed).

    The Matern-5/2 kernel is twice differentiable and provides even smoother interpolation than Matern-3/2.
    It is commonly used in Gaussian process regression for modeling smooth functions.

    Parameters
    ----------
    scale_coefficient : float
        The typical scale (length scale) of the regularization pattern.
    pixel_points : jnp.ndarray
        A 2D array with shape [N_source_pixels, 2], containing source pixelization
        coordinates on the source plane: [[x1,y1], [x2,y2], ...]

    Returns
    -------
    jnp.ndarray
        The source covariance matrix (2D array), shape [N_source_pixels, N_source_pixels].

    Notes
    -----
    The Matern-5/2 kernel is defined as:
        K(r) = (1 + sqrt(5) * r / ℓ + 5 * r² / (3 * ℓ²)) * exp(-sqrt(5) * r / ℓ)
    where r is the Euclidean distance and ℓ is the scale coefficient.
    """
    diff = pixel_points[:, None, :] - pixel_points[None, :, :]

    distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))

    sqrt5 = jnp.sqrt(5.0)
    scaled_dist = sqrt5 * distances / scale_coefficient
    scaled_dist_sq = 5.0 * distances**2 / (3.0 * scale_coefficient**2)

    covariance_matrix = (1.0 + scaled_dist + scaled_dist_sq) * jnp.exp(-scaled_dist)

    # Use relative jitter for better numerical stability
    n_points = len(pixel_points)
    jitter = 1e-6 * jnp.trace(covariance_matrix) / n_points
    covariance_matrix = covariance_matrix + jnp.eye(n_points) * jitter

    return covariance_matrix


@partial(jax.jit, static_argnames=['reg_type'])
def _regularization_matrix_gp_from_jitted(
    scale: float,
    coefficient: float,
    points: jnp.ndarray,
    reg_type: str,
) -> jnp.ndarray:
    """
    Build dense GP regularization matrix from covariance kernel.

    Parameters
    ----------
    scale : float
        Kernel length scale.
    coefficient : float
        Regularization amplitude.
    points : jnp.ndarray
        Source-point coordinates with shape ``(n_source, 2)``.
    reg_type : str
        Covariance kernel type.

    Returns
    -------
    jnp.ndarray
        Dense regularization matrix with shape ``(n_source, n_source)``.
    """
    if reg_type == 'exp':
        covariance_matrix = exp_cov_matrix_from(scale, points)
    elif reg_type == 'gauss':
        covariance_matrix = gauss_cov_matrix_from(scale, points)
    elif reg_type == 'matern32':
        covariance_matrix = matern32_cov_matrix_from(scale, points)
    else:
        covariance_matrix = matern52_cov_matrix_from(scale, points)

    inverse_covariance_matrix = jnp.linalg.solve(
        covariance_matrix, 
        jnp.eye(covariance_matrix.shape[0], dtype=covariance_matrix.dtype)
    )
    
    regularization_matrix = coefficient * inverse_covariance_matrix

    return regularization_matrix


def regularization_matrix_gp_from(
    scale: float,
    coefficient: float,
    points: jnp.ndarray,
    reg_type: str = 'exp',
) -> jnp.ndarray:
    """
    Construct the regularization matrix from Gaussian Process covariance.
    
    The regularization matrix is the inverse of the covariance matrix scaled by a coefficient.
    This is used to penalize non-smooth source reconstructions.
    
    Parameters
    ----------
    scale : float
        The typical scale of the regularization pattern.
    coefficient : float
        The regularization strength coefficient.
    points : jnp.ndarray
        A 2D array with shape [N_source_pixels, 2], containing source pixelization 
        coordinates on the source plane: [[x1,y1], [x2,y2], ...]
    reg_type : str
        Type of covariance kernel: 'exp' for exponential, 'gauss' for Gaussian (RBF),
        'matern32' for Matern-3/2, or 'matern52' for Matern-5/2. Default is 'exp'.
    
    Returns
    -------
    jnp.ndarray
        The regularization matrix (2D array), shape [N_source_pixels, N_source_pixels].
    
    Raises
    ------
    ValueError
        If reg_type is not one of the supported kernel types.
    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> points = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> reg_matrix = regularization_matrix_gp_from(0.1, 1.0, points, 'matern32')
    >>> reg_matrix.shape
    (3, 3)
    """
    valid_types = {'exp', 'gauss', 'matern32', 'matern52'}
    if reg_type not in valid_types:
        raise ValueError(
            f"Unknown reg_type: '{reg_type}'. "
            f"Must be one of {valid_types}."
        )
    
    return _regularization_matrix_gp_from_jitted(scale, coefficient, points, reg_type)


__all__ = [
    'exp_cov_matrix_from',
    'gauss_cov_matrix_from',
    'matern32_cov_matrix_from',
    'matern52_cov_matrix_from',
    'regularization_matrix_gp_from',
    'regularization_sparse_rectangular_from',
    'sparse_regularization_dense_from',
]
