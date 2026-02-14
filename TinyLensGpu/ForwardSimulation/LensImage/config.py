"""
Simulator configuration for  lens simulation.

This module provides configuration management for gravitational lens
simulations, including coordinate grids, PSF kernels, and masks.
"""

import numpy as np
import jax.numpy as jnp
from jax import Array
from typing import Optional, Tuple


def make_grid_2d(npix: int, dpix: float, nsub: int = 1) -> Tuple[Array, Array]:
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

    x_1d = (jnp.arange(npix_sub) - npix_sub / 2.0 + 0.5) * dpix_sub
    y_1d = (jnp.arange(npix_sub) - npix_sub / 2.0 + 0.5) * dpix_sub

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
        psf_kernel: Optional[Array] = None,
        nsub: int = 1,
        mask: Optional[Array] = None,
    ) -> None:
        """
        Initialize a `SimulatorConfig` instance with validated configuration.

        Parameters
        ----------
        dpix : float
            Pixel scale in arcsec/pixel.
        npix : int
            Number of pixels per side (image is npix x npix).
        psf_kernel : array_like, optional
            PSF kernel for convolution. Default is a delta function.
        nsub : int, optional
            Subsampling factor for higher resolution ray-tracing (default: 1).
        mask : array_like, optional
            Boolean mask array. True values are masked out. Default: no mask.
        """
        self.dpix = dpix
        self.npix = npix

        if psf_kernel is None:
            psf_kernel = jnp.array([[0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0]])
        self.psf_kernel = psf_kernel

        self.nsub = nsub

        if mask is None:
            mask = jnp.zeros((npix, npix))
        self.mask = jnp.array(mask, dtype=bool)

        self.xgrid, self.ygrid, self.xgrid_sub, self.ygrid_sub = self.get_coords(
            self.npix, self.dpix, self.nsub
        )

        self._prepare_1d_subgrid()

    def _prepare_1d_subgrid(self):
        """
        Internal helper to prepare 1d subgrid.
        
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        if self.nsub > 1:
            self.mask_sub = jnp.repeat(jnp.repeat(self.mask, self.nsub, axis=0), self.nsub, axis=1)
        else:
            self.mask_sub = self.mask

        self.unmask_sub = ~self.mask_sub
        self.flat_indices = jnp.flatnonzero(self.unmask_sub)

        x_flat = self.xgrid_sub.flatten()
        y_flat = self.ygrid_sub.flatten()

        self.xgrid_sub_1d = x_flat[self.flat_indices]
        self.ygrid_sub_1d = y_flat[self.flat_indices]

        self.subgrid_shape = self.xgrid_sub.shape

    @staticmethod
    def get_coords(npix: int, dpix: float, nsub: int = 1) -> Tuple[Array, Array, Array, Array]:
        """
        Compute get coords.
        
        Parameters
        ----------
        npix : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        dpix : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        nsub : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        xgrid, ygrid = make_grid_2d(npix, dpix, 1)
        xgrid_sub, ygrid_sub = make_grid_2d(npix, dpix, nsub)
        return xgrid, ygrid, xgrid_sub, ygrid_sub

    def __repr__(self) -> str:
        """
        Internal helper to repr.
        
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        return (f"SimulatorConfig(npix={self.npix}, dpix={self.dpix}, "
                f"nsub={self.nsub}, psf_shape={self.psf_kernel.shape})")
