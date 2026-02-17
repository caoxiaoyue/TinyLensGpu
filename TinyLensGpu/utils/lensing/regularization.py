"""
Regularization Matrix Construction for Source Reconstruction

This module provides various covariance kernels (exponential, Gaussian, Matérn)
for constructing regularization matrices used in pixelized source reconstruction.
The regularization matrix penalizes non-smooth source reconstructions.
"""

import jax.numpy as jnp
import jax
from functools import partial


def _kernel_weight_jax(distance: jnp.ndarray, scale: float, reg_type: str) -> jnp.ndarray:
    """
    Evaluate pairwise kernel weights for regularization graphs.

    Parameters
    ----------
    distance : jnp.ndarray
        Pairwise distances.
    scale : float
        Correlation-length parameter.
    reg_type : str
        Kernel family: ``'exp'``, ``'gauss'``, ``'matern32'`` or ``'matern52'``.

    Returns
    -------
    jnp.ndarray
        Non-negative kernel weights.
    """
    scale = jnp.maximum(scale, 1e-6)
    
    if reg_type == 'exp':
        return jnp.exp(-distance / scale)
    elif reg_type == 'gauss':
        return jnp.exp(-(distance ** 2) / (2.0 * scale * scale))
    elif reg_type == 'matern32':
        sqrt3 = jnp.sqrt(3.0)
        z = sqrt3 * distance / scale
        return (1.0 + z) * jnp.exp(-z)
    elif reg_type == 'matern52':
        sqrt5 = jnp.sqrt(5.0)
        z = sqrt5 * distance / scale
        return (1.0 + z + (z * z) / 3.0) * jnp.exp(-z)
    else:
        # Should be unreachable if validated
        return jnp.exp(-distance / scale)


@partial(jax.jit, static_argnames=['reg_type', 'k_neighbors'])
def regularization_sparse_knn_from(
    scale: float,
    coefficient: float,
    points: jnp.ndarray,
    reg_type: str = 'exp',
    k_neighbors: int = 16,
):
    """Build sparse KNN-graph Laplacian regularization in COO edge-list form.
    
    This implementation uses JAX operations (brute-force distance + top_k) 
    to preserve differentiability w.r.t. points.

    Returns
    -------
    rows, cols, values, n_source
        COO entries for symmetric sparse regularization matrix H.
    """
    valid_types = {'exp', 'gauss', 'matern32', 'matern52'}
    if reg_type not in valid_types:
        raise ValueError(f"Unknown reg_type: '{reg_type}'. Must be one of {valid_types}.")

    n_source = points.shape[0]
    
    # Handle trivial cases
    if n_source == 0:
        return (
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,), dtype=jnp.float32),
            0,
        )

    if n_source == 1:
        diag = jnp.array([max(float(coefficient), 1e-6)], dtype=jnp.float32)
        return (
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            diag,
            1,
        )

    # 1. Compute pairwise distances (N, N)
    # Note: For very large N (>10k), this might be memory intensive on GPU.
    diff = points[:, None, :] - points[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    # Add epsilon to avoid sqrt(0) gradient NaN at i=j
    dist = jnp.sqrt(dist_sq + 1e-12)
    
    # 2. Find k-nearest neighbors
    # We want smallest distances. top_k finds largest values, so we negate.
    # We retrieve k+1 neighbors to ensure we can filter out the self-loop (distance 0).
    k = max(1, min(int(k_neighbors), int(n_source) - 1))
    search_k = k + 1

    neg_dist = -dist
    top_vals, top_idx = jax.lax.top_k(neg_dist, search_k)

    # 3. Filter out self-loops
    # We look for the index of the point itself (row index) in the top-k results.
    # If found, we remove it. If not found (e.g. all neighbors are duplicates),
    # we remove the last neighbor to keep exactly k neighbors.
    row_indices = jnp.arange(n_source)
    is_self = top_idx == row_indices[:, None]
    
    # Check if self is found in each row
    has_self = jnp.any(is_self, axis=1)
    
    # Find position of self. argmax returns first True. If all False, returns 0.
    self_pos = jnp.argmax(is_self, axis=1)
    
    # Determine which index to drop
    # If self is present, drop self_pos.
    # If self is NOT present, drop the last element (index k).
    drop_idx = jnp.where(has_self, self_pos, k)
    drop_idx = drop_idx[:, None]  # (N, 1)

    # Construct indices to gather k columns
    col_idx = jnp.arange(k)
    col_idx = jnp.tile(col_idx, (n_source, 1))
    
    # Shift indices to skip the dropped element
    gather_cols = col_idx + (col_idx >= drop_idx).astype(jnp.int32)
    
    # Gather values and indices
    neighbors_idx = jnp.take_along_axis(top_idx, gather_cols, axis=1)
    neighbors_vals = jnp.take_along_axis(top_vals, gather_cols, axis=1)
    
    neighbors_dist = -neighbors_vals  # (N, k)
    
    # 3. Compute weights
    weights = _kernel_weight_jax(neighbors_dist, scale, reg_type)
    weights = weights * coefficient
    
    # 4. Construct COO list (Directed Edges)
    # i -> neighbors[i]
    row_indices = jnp.repeat(jnp.arange(n_source), k)
    col_indices = neighbors_idx.flatten()
    edge_weights = weights.flatten()
    
    # 5. Symmetrization & Laplacian Construction
    # We sum the directed edges: W_sym = W_dir + W_dir.T
    # This means we include both (i, j) and (j, i) in the COO list.
    # Off-diagonal entries in Laplacian are -w.
    all_rows_off = jnp.concatenate([row_indices, col_indices])
    all_cols_off = jnp.concatenate([col_indices, row_indices])
    all_vals_off = jnp.concatenate([-edge_weights, -edge_weights])
    
    # Diagonal entries: D_ii = sum_{j!=i} |H_{ij}| + ridge
    # Since H_{ij} (off-diag) are negative, we sum their negations (which are positive weights).
    diag_sum = jax.ops.segment_sum(
        -all_vals_off,
        all_rows_off,
        num_segments=n_source
    )
    
    ridge = jnp.maximum(1e-8, 1e-6 * jnp.maximum(1.0, coefficient))
    diag_vals = diag_sum + ridge
    
    # Add diagonal elements to COO lists
    diag_rows = jnp.arange(n_source)
    diag_cols = jnp.arange(n_source)
    
    final_rows = jnp.concatenate([all_rows_off, diag_rows])
    final_cols = jnp.concatenate([all_cols_off, diag_cols])
    final_vals = jnp.concatenate([all_vals_off, diag_vals])
    
    return (
        final_rows.astype(jnp.int32),
        final_cols.astype(jnp.int32),
        final_vals.astype(jnp.float32),
        n_source,
    )


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
    'regularization_sparse_knn_from',
    'regularization_sparse_rectangular_from',
    'sparse_regularization_dense_from',
]
