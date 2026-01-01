"""
Sersic light profile .

This module implements the elliptical Sersic light profile using the
caskade framework for modular, composable light distributions.
"""

from typing import Optional
import caskade as ck
import jax.numpy as jnp
from jax import Array
from ..utils import ellipse2circle_transform
from ..param_u import ParamU


class SersicEllipse(ck.Module):
    """
    Elliptical Sersic light profile.

    The Sersic profile is a generalization of the exponential and
    de Vaucouleurs profiles, commonly used to model galaxy light
    distributions.

    Parameters
    ----------
    R_sersic : float, optional
        Effective (half-light) radius in arcseconds
    n_sersic : float, optional
        Sersic index (n=1 for exponential, n=4 for de Vaucouleurs)
    e1 : float, optional
        Ellipticity component 1
    e2 : float, optional
        Ellipticity component 2
    center_x : float, optional
        Center x-coordinate in arcseconds
    center_y : float, optional
        Center y-coordinate in arcseconds
    Ie : float, optional
        Intensity at the effective radius (can be linear parameter)
    """

    def __init__(self, R_sersic: Optional[float] = None, n_sersic: Optional[float] = None, 
                 e1: Optional[float] = None, e2: Optional[float] = None,
                 center_x: Optional[float] = None, center_y: Optional[float] = None, 
                 Ie: Optional[float] = None) -> None:
        super().__init__()

        # Define parameters using ParamU (or convert if already ParamU)
        self.R_sersic = R_sersic if isinstance(R_sersic, ParamU) else ParamU("R_sersic", R_sersic)
        self.n_sersic = n_sersic if isinstance(n_sersic, ParamU) else ParamU("n_sersic", n_sersic)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)
        self.Ie = Ie if isinstance(Ie, ParamU) else ParamU("Ie", Ie)

    @ck.forward
    def light(self, x: Array, y: Array, R_sersic: Optional[Array] = None, 
              n_sersic: Optional[Array] = None, e1: Optional[Array] = None, 
              e2: Optional[Array] = None, center_x: Optional[Array] = None, 
              center_y: Optional[Array] = None, Ie: Optional[Array] = None) -> Array:
        """
        Compute surface brightness at given positions.

        Parameters
        ----------
        x : array_like
            x-coordinates where to evaluate surface brightness
        y : array_like
            y-coordinates where to evaluate surface brightness
        R_sersic : float, optional
            Effective radius (defaults to self.R_sersic.value)
        n_sersic : float, optional
            Sersic index (defaults to self.n_sersic.value)
        e1 : float, optional
            Ellipticity component 1 (defaults to self.e1.value)
        e2 : float, optional
            Ellipticity component 2 (defaults to self.e2.value)
        center_x : float, optional
            Center x-coordinate (defaults to self.center_x.value)
        center_y : float, optional
            Center y-coordinate (defaults to self.center_y.value)
        Ie : float, optional
            Intensity at effective radius (defaults to self.Ie.value)

        Returns
        -------
        surface_brightness : array_like
            Surface brightness at the given positions
        """
        # Ensure parameters are JAX arrays
        R_sersic = jnp.asarray(R_sersic)
        n_sersic = jnp.asarray(n_sersic)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        Ie = jnp.asarray(Ie)

        # Transform ellipse to circle
        x_, y_ = ellipse2circle_transform(x, y, e1, e2, center_x, center_y)
        R = jnp.sqrt(x_**2 + y_**2)

        # Calculate bn coefficient using Ciotti & Bertin (1999) higher-order approximation
        inv_n = 1.0 / n_sersic
        bn = (2.0 * n_sersic - 1.0 / 3.0 +
              4.0 / 405.0 * inv_n +
              46.0 / 25515.0 * inv_n**2 +
              131.0 / 1148175.0 * inv_n**3 -
              2194697.0 / 30690717750.0 * inv_n**4)

        # Sersic profile
        return Ie * jnp.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))
