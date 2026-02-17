"""
Gaussian light profile .

This module implements the elliptical Gaussian light profile using the
caskade framework. Gaussian profiles are commonly used in Multi-Gaussian
Expansion (MGE) decompositions of galaxy light distributions.
"""

from typing import Optional
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.utils.geometry import ellipse2circle_transform
from TinyLensGpu.Inference.param_u import ParamU


class GaussianEllipse(ck.Module):
    """
    Elliptical Gaussian light profile.

    The Gaussian profile is frequently used in Multi-Gaussian Expansion (MGE)
    to model complex galaxy light distributions as a sum of Gaussian components.

    Parameters
    ----------
    flux : float, optional
        Total flux/luminosity (can be linear parameter)
    sigma : float, optional
        Width (standard deviation) of the Gaussian in arcseconds
    e1 : float, optional
        Ellipticity component 1
    e2 : float, optional
        Ellipticity component 2
    center_x : float, optional
        Center x-coordinate in arcseconds
    center_y : float, optional
        Center y-coordinate in arcseconds
    """

    def __init__(self, flux: Optional[float] = None, sigma: Optional[float] = None, 
                 e1: Optional[float] = None, e2: Optional[float] = None,
                 center_x: Optional[float] = None, center_y: Optional[float] = None) -> None:
        """
        Initialize elliptical Gaussian profile parameters.

        Parameters
        ----------
        flux : float, optional
            Total flux normalization.
        sigma : float, optional
            Gaussian width (standard deviation) in arcseconds.
        e1, e2 : float, optional
            Ellipticity components.
        center_x, center_y : float, optional
            Profile center coordinates in arcseconds.
        """
        super().__init__()

        # Define parameters using ParamU (or convert if already ParamU)
        self.flux = flux if isinstance(flux, ParamU) else ParamU("flux", flux)
        self.sigma = sigma if isinstance(sigma, ParamU) else ParamU("sigma", sigma)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

    @ck.forward
    def light(self, x: Array, y: Array, flux: Optional[Array] = None, 
              sigma: Optional[Array] = None, e1: Optional[Array] = None, 
              e2: Optional[Array] = None, center_x: Optional[Array] = None, 
              center_y: Optional[Array] = None) -> Array:
        """
        Evaluate elliptical Gaussian surface brightness on the image plane.

        Parameters
        ----------
        x : array_like
            x-coordinates where to evaluate surface brightness
        y : array_like
            y-coordinates where to evaluate surface brightness
        flux : float, optional
            Total flux (defaults to self.flux.value)
        sigma : float, optional
            Width of Gaussian (defaults to self.sigma.value)
        e1 : float, optional
            Ellipticity component 1 (defaults to self.e1.value)
        e2 : float, optional
            Ellipticity component 2 (defaults to self.e2.value)
        center_x : float, optional
            Center x-coordinate (defaults to self.center_x.value)
        center_y : float, optional
            Center y-coordinate (defaults to self.center_y.value)

        Returns
        -------
        surface_brightness : array_like
            Surface-brightness values at the requested coordinates.
        """
        # Ensure parameters are JAX arrays
        flux = jnp.asarray(flux)
        sigma = jnp.asarray(sigma)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)

        # Normalization constant
        c = flux / (2 * jnp.pi * sigma**2)

        # Transform ellipse to circle
        x_, y_ = ellipse2circle_transform(x, y, e1, e2, center_x, center_y)

        # Gaussian profile
        factor = x_**2 / sigma**2 + y_**2 / sigma**2
        return c * jnp.exp(-factor / 2.0)
