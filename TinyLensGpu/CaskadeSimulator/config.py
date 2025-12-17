"""
Simulator configuration for caskade-based lens simulation.

This module provides configuration management for gravitational lens
simulations, including coordinate grids, PSF kernels, and masks.
"""

import numpy as np
import jax.numpy as jnp
from typing import Optional


def make_grid_2d(npix, dpix, nsub=1):
    """
    Generate 2D coordinate grids for image plane.

    Args:
        npix: Number of pixels per side
        dpix: Pixel scale in arcsec/pixel
        nsub: Subsampling factor

    Returns:
        xgrid, ygrid: 2D coordinate grids
    """
    npix_sub = npix * nsub
    dpix_sub = dpix / nsub

    # Create 1D coordinate arrays
    x_1d = (jnp.arange(npix_sub) - npix_sub / 2.0 + 0.5) * dpix_sub
    y_1d = (jnp.arange(npix_sub) - npix_sub / 2.0 + 0.5) * dpix_sub

    # Create 2D grids
    xgrid, ygrid = jnp.meshgrid(x_1d, y_1d)

    return xgrid, ygrid


class SimulatorConfig:
    """
    Configuration for gravitational lens simulation.

    This class manages all simulation parameters including image size,
    pixel scale, PSF kernel, subsampling, and masks.

    Parameters
    ----------
    dpix : float
        Pixel scale in arcsec/pixel
    npix : int
        Number of pixels per side (image is npix x npix)
    psf_kernel : array_like, optional
        PSF kernel for convolution. Default is delta function.
    nsub : int, optional
        Subsampling factor for higher resolution ray-tracing (default: 1)
    mask : array_like, optional
        Boolean mask array. True values are masked out. Default: no mask.

    Attributes
    ----------
    dpix : float
        Pixel scale
    npix : int
        Image size
    psf_kernel : ndarray
        PSF kernel
    nsub : int
        Subsampling factor
    mask : ndarray
        Boolean mask
    xgrid : ndarray
        X-coordinate grid at full resolution
    ygrid : ndarray
        Y-coordinate grid at full resolution
    xgrid_sub : ndarray
        X-coordinate grid at subsampled resolution
    ygrid_sub : ndarray
        Y-coordinate grid at subsampled resolution
    """

    def __init__(
        self,
        dpix: float,
        npix: int,
        psf_kernel: Optional[np.ndarray] = None,
        nsub: int = 1,
        mask: Optional[np.ndarray] = None,
    ):
        self.dpix = dpix
        self.npix = npix

        # Default PSF: delta function
        if psf_kernel is None:
            psf_kernel = np.array([[0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0]])
        self.psf_kernel = psf_kernel

        self.nsub = nsub

        # Default mask: no masking
        if mask is None:
            mask = np.zeros((npix, npix))
        self.mask = mask.astype('bool')

        # Generate coordinate grids
        self.xgrid, self.ygrid, self.xgrid_sub, self.ygrid_sub = self.get_coords(
            self.npix, self.dpix, self.nsub
        )

    @staticmethod
    def get_coords(npix: int, dpix: float, nsub: int = 1):
        """
        Generate coordinate grids for image simulation.

        Parameters
        ----------
        npix : int
            Number of pixels per side
        dpix : float
            Pixel scale
        nsub : int, optional
            Subsampling factor (default: 1)

        Returns
        -------
        tuple
            (xgrid, ygrid, xgrid_sub, ygrid_sub) as JAX arrays
        """
        xgrid, ygrid = make_grid_2d(npix, dpix, 1)
        xgrid_sub, ygrid_sub = make_grid_2d(npix, dpix, nsub)
        return xgrid, ygrid, xgrid_sub, ygrid_sub

    def __repr__(self):
        return (f"SimulatorConfig(npix={self.npix}, dpix={self.dpix}, "
                f"nsub={self.nsub}, psf_shape={self.psf_kernel.shape})")
