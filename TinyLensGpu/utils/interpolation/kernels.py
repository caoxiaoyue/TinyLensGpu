"""
Optimized Kernel-Based Interpolation in JAX

Uses Wendland compactly supported kernels with normalized weights (partition of unity).
More robust and faster than MLS, better accuracy than simple IDW.
"""
import jax
import jax.numpy as jnp
from functools import partial


def get_interpolation_weights(points, query_points, k_neighbors=10, kernel='wendland_c4',
                             radius_scale=1.5):
    """
    Compute interpolation weights between source points and query points.
    
    This is a standalone function to get the weights used in kernel interpolation,
    useful when you want to analyze or reuse weights separately from interpolation.
    
    Args:
        points:       (N, 2) source point coordinates
        query_points: (M, 2) query point coordinates
        k_neighbors:  number of nearest neighbors (default: 10)
        kernel:       'wendland_c2', 'wendland_c4', or 'wendland_c6' (default: 'wendland_c4')
        radius_scale: multiplier for auto-computed radius (default: 1.5)
        
    Returns:
        weights:   (M, k) normalized weights for each query point
        indices:   (M, k) indices of K nearest neighbors in points array
        distances: (M, k) distances to K nearest neighbors
        
    Example:
        >>> weights, indices, distances = get_interpolation_weights(src_pts, query_pts)
        >>> # Now you can use weights and indices for custom interpolation
        >>> interpolated = jnp.sum(weights * values[indices], axis=1)
    """
    points = jnp.asarray(points)
    query_points = jnp.asarray(query_points)
    
    if kernel == 'wendland_c2':
        kernel_fn = wendland_c2
    elif kernel == 'wendland_c4':
        kernel_fn = wendland_c4
    elif kernel == 'wendland_c6':
        kernel_fn = wendland_c6
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    return compute_weights(points, query_points, k_neighbors, radius_scale, kernel_fn)


def wendland_c2(r, h):
    """
    Wendland C2: (1 - r/h)^4 * (4*r/h + 1)
    C2 continuous, compact support
    """
    s = r / (h + 1e-10)
    w = jnp.where(s < 1.0, (1.0 - s)**4 * (4.0 * s + 1.0), 0.0)
    return w


def wendland_c4(r, h):
    """
    Wendland C4: (1 - r/h)^6 * (35*(r/h)^2 + 18*r/h + 3)
    C4 continuous, smoother, compact support
    """
    s = r / (h + 1e-10)
    w = jnp.where(s < 1.0, (1.0 - s)**6 * (35.0 * s**2 + 18.0 * s + 3.0), 0.0)
    return w


def wendland_c6(r, h):
    """
    Wendland C6: (1 - r/h)^8 * (32*(r/h)^3 + 25*(r/h)^2 + 8*r/h + 1)
    C6 continuous, very smooth, compact support
    """
    s = r / (h + 1e-10)
    w = jnp.where(s < 1.0, (1.0 - s)**8 * (32.0 * s**3 + 25.0 * s**2 + 8.0 * s + 1.0), 0.0)
    return w


def compute_weights(points, query_points, k_neighbors, radius_scale, kernel_fn):
    """
    Compute normalized kernel weights for interpolation.
    
    This function computes the weights between source points and 
    query points using K-nearest neighbors and Wendland kernels.
    
    Args:
        points:       (N, 2) source point coordinates
        query_points: (M, 2) query point coordinates
        k_neighbors:  number of nearest neighbors
        radius_scale: multiplier for auto-computed radius
        kernel_fn:    kernel function (wendland_c2/c4/c6)
        
    Returns:
        weights:      (M, k) normalized weights for each query point
        indices:      (M, k) indices of K nearest neighbors for each query point
        distances:    (M, k) distances to K nearest neighbors
    """
    diff = query_points[:, None, :] - points[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1)

    top_k_vals, top_k_indices = jax.lax.top_k(-dist_sq, k_neighbors)
    knn_distances = jnp.sqrt(-top_k_vals)
    
    h = jnp.max(knn_distances, axis=1, keepdims=True) * radius_scale
    
    weights = kernel_fn(knn_distances, h)
    
    weight_sum = jnp.sum(weights, axis=1, keepdims=True) + 1e-10
    weights_normalized = weights / weight_sum
    
    return weights_normalized, top_k_indices, knn_distances


__all__ = [
    'get_interpolation_weights',
    'wendland_c2',
    'wendland_c4',
    'wendland_c6',
    'compute_weights'
]
