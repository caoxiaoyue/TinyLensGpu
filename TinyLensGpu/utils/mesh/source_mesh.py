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
    Compute apply gaussian blur.
    
    Parameters
    ----------
    img : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    sigma : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
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
    Sample points from image based on brightness weighting.
    Supports both random and quasi-Monte Carlo (Sobol) sampling.
    Only supports grayscale images, with sampling restricted to mask regions.
    
    Parameters:
    - img: Input grayscale image array (2D numpy array)
    - mask: Boolean array, same size as img, True for valid sampling regions
    - n_points: Number of points to sample
    - alpha: Density bias exponent, >1 favors bright areas, =1 linear, <1 more uniform
    - blur_sigma_px: Optional Gaussian smoothing std dev (pixels); 0 for no smoothing
    - replace: Whether to allow same pixel to be sampled multiple times
    - normalize_xy: Normalize coordinates to [0,1]x[0,1]
    - pixel_jitter: Add random jitter within pixels for continuous coordinates
    - method: Sampling method, 'random' or 'sobol' (quasi-Monte Carlo)
    - seed: Random seed
    
    Returns:
    - pts: (N,2) array with x,y coordinates (image coordinates, origin at top-left)
    - (H, W): Image dimensions
    - Y: Processed brightness distribution (only for method='random')
    
    Note:
    - Only supports grayscale images
    - Pixels outside mask regions have zero weight and won't be sampled
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
    Internal helper to sample random.
    
    Parameters
    ----------
    probabilities : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    n_points : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    W : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    H : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    replace : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    normalize_xy : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    pixel_jitter : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    seed : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    Y : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    Raises
    ------
    ValueError
        Raised when input validation fails or required runtime state is missing.
    
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
    Internal helper to sample sobol.
    
    Parameters
    ----------
    probabilities : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    n_points : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    W : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    H : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    normalize_xy : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    pixel_jitter : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    seed : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    Y : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
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
