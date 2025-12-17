"""
Gaussian light profile using caskade.

This module implements the elliptical Gaussian light profile using the
caskade framework. Gaussian profiles are commonly used in Multi-Gaussian
Expansion (MGE) decompositions of galaxy light distributions.
"""

import caskade as ck
import jax.numpy as jnp
from ..utils import ellipse2circle_transform


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

    def __init__(self, flux=None, sigma=None, e1=None, e2=None,
                 center_x=None, center_y=None):
        super().__init__()

        # Define parameters using ck.Param
        self.flux = ck.Param("flux", flux)
        self.sigma = ck.Param("sigma", sigma)
        self.e1 = ck.Param("e1", e1)
        self.e2 = ck.Param("e2", e2)
        self.center_x = ck.Param("center_x", center_x)
        self.center_y = ck.Param("center_y", center_y)

    @ck.forward
    def light(self, x, y, flux=None, sigma=None, e1=None, e2=None,
              center_x=None, center_y=None):
        """
        Compute surface brightness at given positions.

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
            Surface brightness at the given positions
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
