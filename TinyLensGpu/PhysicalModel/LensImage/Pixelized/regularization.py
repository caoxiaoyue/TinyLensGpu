import jax.numpy as jnp
import jax
from functools import partial

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
    
    covariance_matrix = covariance_matrix + jnp.eye(len(pixel_points)) * 1e-6

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
    
    covariance_matrix = covariance_matrix + jnp.eye(len(pixel_points)) * 1e-6

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
    
    covariance_matrix = covariance_matrix + jnp.eye(len(pixel_points)) * 1e-6

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
    
    covariance_matrix = covariance_matrix + jnp.eye(len(pixel_points)) * 1e-6

    return covariance_matrix


@partial(jax.jit, static_argnames=['reg_type'])
def _regularization_matrix_gp_from_jitted(
    scale: float,
    coefficient: float,
    points: jnp.ndarray,
    reg_type: str,
) -> jnp.ndarray:
    """
    Internal JIT-compiled function for computing regularization matrix.
    
    This function should not be called directly. Use regularization_matrix_gp_from() instead.
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
