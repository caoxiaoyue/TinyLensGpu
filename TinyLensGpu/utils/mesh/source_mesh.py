"""
Source Mesh Sampling Utilities

This module provides utilities for generating source meshes for pixelized source
reconstruction. It supports weighted sampling based on image brightness with both
random and quasi-Monte Carlo (Sobol) sampling methods.
"""

import numpy as np
from scipy.stats import qmc


def apply_gaussian_blur(img, sigma):
    """
    Apply separable Gaussian smoothing to a 2D image.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image with shape ``(H, W)``.
    sigma : float
        Gaussian standard deviation in pixels. Non-positive values return
        the input unchanged.

    Returns
    -------
    np.ndarray
        Blurred image with the same shape as ``img``.
    """
    if sigma <= 0:
        return img
    
    r = int(max(1, round(3 * sigma)))
    xs = np.arange(-r, r + 1, dtype=np.float32)
    kernel = np.exp(-(xs**2) / (2 * (sigma**2 + 1e-8)))
    kernel /= kernel.sum()
    
    blurred = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode='same'), 1, img)
    blurred = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode='same'), 0, blurred)
    return blurred


def sample_points_weighted(img, mask, n_points=1500, alpha=1.5, blur_sigma_px=0.0,
                           replace=False, normalize_xy=False, pixel_jitter=False,
                           method='random', seed=None):
    """
    Sample source-plane seed points using brightness-weighted probabilities.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image with shape ``(H, W)``.
    mask : np.ndarray
        Boolean validity mask of shape ``(H, W)`` where ``True`` denotes
        regions eligible for sampling.
    n_points : int, optional
        Number of samples to draw.
    alpha : float, optional
        Exponent applied to normalized brightness before sampling.
    blur_sigma_px : float, optional
        Gaussian blur sigma (pixels) applied before weighting.
    replace : bool, optional
        Whether pixel indices can be sampled multiple times.
    normalize_xy : bool, optional
        If ``True``, return coordinates normalized to ``[0, 1]``.
    pixel_jitter : bool, optional
        If ``True``, add sub-pixel random offsets.
    method : {'random', 'sobol'}, optional
        Sampling strategy.
    seed : int, optional
        Random seed used by random or Sobol samplers.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int], np.ndarray]
        Sampled points ``(N, 2)``, image shape ``(H, W)``, and processed
        brightness map used for weighting.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy array")
    if mask.dtype != bool:
        raise TypeError("mask must be a boolean array")
    if mask.shape != img.shape:
        raise ValueError(f"mask shape {mask.shape} doesn't match image shape {img.shape}")
    
    Y = np.array(img, dtype=np.float64)
    H, W = Y.shape

    if blur_sigma_px > 0:
        Y = apply_gaussian_blur(Y, blur_sigma_px)

    Y /= np.sum(Y)
    Y = np.clip(Y, 0.0, 1.0)
    weights = np.power(Y, alpha)
    
    weights = weights * mask.astype(np.float64)
    
    weights_flat = weights.ravel()
    weights_flat += weights_flat.max() * 1e-6
    probabilities = weights_flat / weights_flat.sum()

    if method == 'random':
        return _sample_random(probabilities, n_points, W, H, replace, 
                            normalize_xy, pixel_jitter, seed, Y)
    elif method == 'sobol':
        return _sample_sobol(probabilities, n_points, W, H, 
                           normalize_xy, pixel_jitter, seed, Y)
    else:
        raise ValueError(f"Unknown sampling method: {method}. Use 'random' or 'sobol'.")


def _sample_random(probabilities, n_points, W, H, replace, normalize_xy, pixel_jitter, seed, Y):
    """
    Draw weighted samples using NumPy's discrete random choice.

    Parameters
    ----------
    probabilities : np.ndarray
        Flattened categorical probabilities over all pixels.
    n_points : int
        Number of requested samples.
    W, H : int
        Image width and height.
    replace : bool
        Sampling with/without replacement.
    normalize_xy : bool
        Whether to normalize output coordinates.
    pixel_jitter : bool
        Whether to add random sub-pixel offsets.
    seed : int, optional
        Random seed for reproducibility.
    Y : np.ndarray
        Processed weighting image returned to the caller.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int], np.ndarray]
        Sampled points, image shape, and processed brightness map.

    Raises
    ------
    ValueError
        If sampling without replacement requests more points than nonzero
        probability pixels.
    """
    if not replace:
        positive_pixels = int(np.count_nonzero(probabilities > 0))
        if n_points > positive_pixels:
            raise ValueError(f"Requested {n_points} points exceeds positive weight pixels {positive_pixels}. "
                           f"Set replace=True or reduce alpha/increase blur_sigma_px.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(probabilities.size, size=n_points, replace=replace, p=probabilities)
    ys, xs = divmod(idx, W)
    
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    if pixel_jitter:
        xs += rng.random(n_points)
        ys += rng.random(n_points)

    if normalize_xy:
        xs = xs / W
        ys = ys / H

    pts = np.column_stack([xs, ys])
    return pts, (H, W), Y


def _sample_sobol(probabilities, n_points, W, H, normalize_xy, pixel_jitter, seed, Y):
    """
    Draw weighted samples using Sobol quasi-random numbers.

    Parameters
    ----------
    probabilities : np.ndarray
        Flattened categorical probabilities over all pixels.
    n_points : int
        Number of requested samples.
    W, H : int
        Image width and height.
    normalize_xy : bool
        Whether to normalize output coordinates.
    pixel_jitter : bool
        Whether to add Sobol-derived sub-pixel offsets.
    seed : int, optional
        Sobol scrambler seed.
    Y : np.ndarray
        Processed weighting image returned to the caller.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int], np.ndarray]
        Sampled points, image shape, and processed brightness map.
    """
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(n_points)))
    U = sampler.random_base2(m)[:n_points]

    cdf = np.cumsum(probabilities)
    idx = np.searchsorted(cdf, U[:, 0], side='right')
    ys, xs = divmod(idx, W)

    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    
    if pixel_jitter:
        xs += U[:, 1]
        ys += U[:, 2]

    if normalize_xy:
        xs = xs / W
        ys = ys / H

    pts = np.column_stack([xs, ys])
    return pts, (H, W), Y


__all__ = [
    'apply_gaussian_blur',
    'sample_points_weighted'
]
