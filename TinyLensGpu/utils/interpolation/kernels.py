"""
Optimized Kernel-Based Interpolation in JAX

Uses Wendland compactly supported kernels with normalized weights (partition of unity).
More robust and faster than MLS, better accuracy than simple IDW.
"""
import jax
import jax.numpy as jnp


def get_interpolation_weights(points, query_points, k_neighbors=10, kernel='wendland_c4',
                             radius_scale=1.5):
    """
    Compute k-NN interpolation weights for query points.

    Parameters
    ----------
    points : array_like
        Source-node coordinates with shape ``(n_source, 2)``.
    query_points : array_like
        Query coordinates with shape ``(n_query, 2)``.
    k_neighbors : int, optional
        Number of nearest source nodes used for each query point.
    kernel : {'wendland_c2', 'wendland_c4', 'wendland_c6'}, optional
        Compactly supported kernel family.
    radius_scale : float, optional
        Scaling factor applied to the adaptive kernel radius.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Normalized weights, neighbor indices, and neighbor distances, each with
        shape ``(n_query, k_neighbors)``.
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
    Evaluate Wendland C2 kernel.

    Parameters
    ----------
    r : array_like
        Pairwise distance(s).
    h : array_like
        Support radius (same units as ``r``).

    Returns
    -------
    jnp.ndarray
        Kernel value(s), zero outside compact support.
    """
    s = r / (h + 1e-10)
    w = jnp.where(s < 1.0, (1.0 - s)**4 * (4.0 * s + 1.0), 0.0)
    return w


def wendland_c4(r, h):
    """
    Evaluate Wendland C4 kernel.

    Parameters
    ----------
    r : array_like
        Pairwise distance(s).
    h : array_like
        Support radius.

    Returns
    -------
    jnp.ndarray
        Kernel value(s), zero outside compact support.
    """
    s = r / (h + 1e-10)
    w = jnp.where(s < 1.0, (1.0 - s)**6 * (35.0 * s**2 + 18.0 * s + 3.0), 0.0)
    return w


def wendland_c6(r, h):
    """
    Evaluate Wendland C6 kernel.

    Parameters
    ----------
    r : array_like
        Pairwise distance(s).
    h : array_like
        Support radius.

    Returns
    -------
    jnp.ndarray
        Kernel value(s), zero outside compact support.
    """
    s = r / (h + 1e-10)
    w = jnp.where(s < 1.0, (1.0 - s)**8 * (32.0 * s**3 + 25.0 * s**2 + 8.0 * s + 1.0), 0.0)
    return w


def compute_weights(points, query_points, k_neighbors, radius_scale, kernel_fn):
    """
    Compute normalized kernel weights and k-NN indices.

    Parameters
    ----------
    points : jnp.ndarray
        Source-node coordinates with shape ``(n_source, 2)``.
    query_points : jnp.ndarray
        Query coordinates with shape ``(n_query, 2)``.
    k_neighbors : int
        Number of neighbors per query point.
    radius_scale : float
        Support-radius multiplier.
    kernel_fn : callable
        Kernel function taking ``(r, h)``.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Normalized weights, k-NN indices, and distances.
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
