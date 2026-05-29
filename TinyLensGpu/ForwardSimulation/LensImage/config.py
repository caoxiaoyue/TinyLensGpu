"""
Simulator configuration for  lens simulation.

This module provides configuration management for gravitational lens
simulations, including coordinate grids, PSF kernels, and masks.
"""

import numpy as np
import jax.numpy as jnp
from jax import Array
from typing import Optional, Tuple, Union, cast


def make_grid_2d(npix: int, dpix: float, nsub: int = 1) -> Tuple[Array, Array]:
    """
    Generate 2D coordinate grids for image plane.

    Parameters
    ----------
    npix : int
        Number of pixels per side in the native image grid.
    dpix : float
        Pixel scale in arcsec per pixel.
    nsub : int, optional
        Subsampling factor used to construct a finer grid.

    Returns:
        xgrid, ygrid: 2D coordinate grids with shape ``(npix * nsub, npix * nsub)``.
    """
    npix_sub = npix * nsub
    dpix_sub = dpix / nsub

    x_1d = (jnp.arange(npix_sub) - npix_sub / 2.0 + 0.5) * dpix_sub
    y_1d = (jnp.arange(npix_sub) - npix_sub / 2.0 + 0.5) * dpix_sub

    xgrid, ygrid = jnp.meshgrid(x_1d, y_1d)

    return xgrid, ygrid


def make_grid_2d_transformed(
    npix: int,
    dpix: float,
    nsub: int = 1,
    shift_x: Union[float, Array] = 0.0,
    shift_y: Union[float, Array] = 0.0,
    rotation: Union[float, Array] = 0.0,
) -> Tuple[Array, Array]:
    """
    Generate transformed 2D coordinate grids for a square image plane.

    The base grid follows the same zero-centered convention as ``make_grid_2d``.
    A clockwise rotation around the origin is applied first. The shift terms
    follow apparent image-plane semantics: positive ``shift_x`` and ``shift_y``
    correspond to positive image-plane offsets, and are therefore applied with
    opposite sign in the sampling grid transform.

    Parameters
    ----------
    npix : int
        Number of pixels per side in the native image grid.
    dpix : float
        Pixel scale in arcsec per pixel.
    nsub : int, optional
        Subsampling factor used to construct a finer grid.
    shift_x : float, optional
        Apparent image-plane offset along +x in arcsec.
    shift_y : float, optional
        Apparent image-plane offset along +y in arcsec.
    rotation : float, optional
        Clockwise rotation angle in degrees.

    Returns
    -------
    tuple[Array, Array]
        Transformed ``(xgrid, ygrid)`` with shape
        ``(npix * nsub, npix * nsub)``.
    """
    xgrid, ygrid = make_grid_2d(npix=npix, dpix=dpix, nsub=nsub)

    rotation_rad = jnp.deg2rad(rotation)
    cos_phi = jnp.cos(rotation_rad)
    sin_phi = jnp.sin(rotation_rad)
    x_rot = xgrid * cos_phi + ygrid * sin_phi
    y_rot = -xgrid * sin_phi + ygrid * cos_phi

    return x_rot - shift_x, y_rot - shift_y


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
        source_seed_mask: Optional[Array] = None,
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
        source_seed_mask : array_like, optional
            Boolean mask array used to infer the source-plane bounding box.
            True values are masked out. If None, defaults to ``mask``.
            This mask should be a subset of ``mask`` (i.e. have the same or
            fewer unmasked pixels) and only cover the lensed arc region.
        """
        self.dpix = dpix
        self.npix = npix

        if psf_kernel is None:
            psf_kernel = jnp.array([[0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0]])
        self.psf_kernel = cast(Array, jnp.array(psf_kernel))

        self.nsub = nsub

        if mask is None:
            mask = jnp.zeros((npix, npix))
        self.mask = jnp.array(mask, dtype=bool)

        if source_seed_mask is None:
            source_seed_mask = self.mask
        self.source_seed_mask = jnp.array(source_seed_mask, dtype=bool)

        self.xgrid, self.ygrid, self.xgrid_sub, self.ygrid_sub = self.get_coords(
            self.npix, self.dpix, self.nsub
        )

        self._prepare_1d_subgrid()

    def _prepare_1d_subgrid(self):
        """
        Precompute flattened unmasked sub-grid coordinates.

        The method caches index/coordinate arrays frequently reused by forward
        simulation and inversion routines.
        """
        self.unmask = ~self.mask
        self.flat_indices_native = jnp.flatnonzero(self.unmask)

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
        Build native and subsampled image-plane coordinate grids.

        Parameters
        ----------
        npix : int
            Number of pixels per side in the native image grid.
        dpix : float
            Pixel scale in arcsec per pixel.
        nsub : int, optional
            Subsampling factor for high-resolution grids.

        Returns
        -------
        tuple[Array, Array, Array, Array]
            ``(xgrid, ygrid, xgrid_sub, ygrid_sub)`` where ``xgrid/ygrid`` have
            shape ``(npix, npix)`` and ``xgrid_sub/ygrid_sub`` have shape
            ``(npix * nsub, npix * nsub)``.
        """
        xgrid, ygrid = make_grid_2d(npix, dpix, 1)
        xgrid_sub, ygrid_sub = make_grid_2d(npix, dpix, nsub)
        return xgrid, ygrid, xgrid_sub, ygrid_sub

    def __repr__(self) -> str:
        """
        Return compact string representation for logging/debugging.

        Returns
        -------
        str
            Human-readable summary including image size, pixel scale, subsampling,
            and PSF shape.
        """
        psf_shape = np.shape(self.psf_kernel)
        return (f"SimulatorConfig(npix={self.npix}, dpix={self.dpix}, "
                f"nsub={self.nsub}, psf_shape={psf_shape})")
