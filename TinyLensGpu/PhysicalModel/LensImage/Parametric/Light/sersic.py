"""
Sersic light profile .

This module implements the elliptical Sersic light profile using the
caskade framework for modular, composable light distributions.
"""

from typing import Optional
import caskade as ck
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln
from TinyLensGpu.utils.geometry import ellipse2circle_transform
from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.utils.photometry import mag2cps


class SersicEllipse(ck.Module):
    """
    Elliptical Sersic light profile.

    The Sersic profile is a generalization of the exponential and
    de Vaucouleurs profiles, commonly used to model galaxy light
    distributions.

    Brightness can be specified either via Ie (intensity at the
    effective radius) or via magnitude (total apparent magnitude).
    These two options are mutually exclusive.

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
    magnitude : float, optional
        Total apparent magnitude. Mutually exclusive with Ie.
    mag_zero_point : float, optional
        Magnitude zero point for converting magnitude to counts.
        Default is 22.0.
    """

    def __init__(self, R_sersic: Optional[float] = None, n_sersic: Optional[float] = None,
                 e1: Optional[float] = None, e2: Optional[float] = None,
                 center_x: Optional[float] = None, center_y: Optional[float] = None,
                 Ie: Optional[float] = None,
                 magnitude: Optional[float] = None,
                 mag_zero_point: float = 22.0) -> None:
        """
        Initialize the SersicEllipse profile.

        Parameters
        ----------
        R_sersic : float, optional
            Effective radius (half-light radius) in arcseconds.
        n_sersic : float, optional
            Sersic index (n=1 is exponential, n=4 is de Vaucouleurs).
        e1 : float, optional
            Ellipticity component 1.
        e2 : float, optional
            Ellipticity component 2.
        center_x : float, optional
            Center x-coordinate in arcseconds.
        center_y : float, optional
            Center y-coordinate in arcseconds.
        Ie : float, optional
            Intensity at the effective radius (R_sersic).
        magnitude : float, optional
            Total apparent magnitude. Mutually exclusive with Ie.
        mag_zero_point : float, optional
            Magnitude zero point. Default is 22.0.
        """
        super().__init__()

        # Validate brightness parameter specification
        has_ie = Ie is not None
        has_mag = magnitude is not None

        if has_ie and has_mag:
            raise ValueError(
                "Must provide exactly one of 'Ie' or 'magnitude'"
            )
        if not has_ie and not has_mag:
            raise ValueError(
                "Must provide exactly one of 'Ie' or 'magnitude'"
            )

        # Define shape parameters using ParamU
        self.R_sersic = R_sersic if isinstance(R_sersic, ParamU) else ParamU("R_sersic", R_sersic)
        self.n_sersic = n_sersic if isinstance(n_sersic, ParamU) else ParamU("n_sersic", n_sersic)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

        # Handle brightness parameterization
        if has_mag:
            self.magnitude = magnitude if isinstance(magnitude, ParamU) else ParamU("magnitude", magnitude)
            self.mag_zero_point = mag_zero_point
        else:
            self.magnitude = None
            self.mag_zero_point = None
            self.Ie = Ie if isinstance(Ie, ParamU) else ParamU("Ie", Ie)

    @staticmethod
    def total_flux_analytic_from(R_sersic: float, Ie: float, n_sersic: float) -> float:
        """
        Compute total flux of a Sersic profile analytically.

        Formula derived from integrating the Sersic surface brightness
        profile over the full plane (Ciotti 1991):

        L = Ie * Re^2 * 2*pi*n*e^bn * Gamma(2n) / bn^(2n)

        where bn is the Sersic coefficient.

        Parameters
        ----------
        R_sersic : float
            Effective radius (half-light radius).
        Ie : float
            Intensity at the effective radius.
        n_sersic : float
            Sersic index.

        Returns
        -------
        float
            Total integrated flux (counts per second).
        """
        inv_n = 1.0 / n_sersic
        bn = (2.0 * n_sersic - 1.0 / 3.0 +
              4.0 / 405.0 * inv_n +
              46.0 / 25515.0 * inv_n**2 +
              131.0 / 1148175.0 * inv_n**3 -
              2194697.0 / 30690717750.0 * inv_n**4)

        gamma_2n = jnp.exp(gammaln(2.0 * n_sersic))
        factor = (bn ** (2.0 * n_sersic) /
                  (2.0 * jnp.pi * R_sersic ** 2 * jnp.exp(bn) * n_sersic * gamma_2n))
        return Ie / factor

    @staticmethod
    def Ie_from_magnitude(magnitude: float, mag_zero_point: float,
                          R_sersic: float, n_sersic: float) -> float:
        """
        Compute Ie from total apparent magnitude.

        Steps:
        1. Convert magnitude to total counts per second (cps).
        2. Compute total flux for a unit-Ie Sersic profile.
        3. Scale Ie such that the total flux matches the target cps.

        Parameters
        ----------
        magnitude : float
            Total apparent magnitude.
        mag_zero_point : float
            Magnitude zero point.
        R_sersic : float
            Effective radius.
        n_sersic : float
            Sersic index.

        Returns
        -------
        float
            Intensity at the effective radius (Ie).
        """
        total_cps_tmp = SersicEllipse.total_flux_analytic_from(
            R_sersic=R_sersic, Ie=1.0, n_sersic=n_sersic
        )
        total_cps = mag2cps(magnitude, mag_zero_point)
        rescale_factor = total_cps / total_cps_tmp
        return rescale_factor

    @ck.forward
    def light(self, x: Array, y: Array, R_sersic: Optional[Array] = None,
              n_sersic: Optional[Array] = None, e1: Optional[Array] = None,
              e2: Optional[Array] = None, center_x: Optional[Array] = None,
              center_y: Optional[Array] = None, Ie: Optional[Array] = None,
              magnitude: Optional[Array] = None,
              mag_zero_point: Optional[Array] = None) -> Array:
        """
        Evaluate elliptical Sersic surface brightness on the image plane.

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
        magnitude : float, optional
            Total apparent magnitude. If the model is initialized with magnitude,
            Ie is always computed from magnitude dynamically.
        mag_zero_point : float, optional
            Magnitude zero point (defaults to self.mag_zero_point)

        Returns
        -------
        surface_brightness : array_like
            Surface-brightness values at the requested coordinates.
        """
        # Ensure parameters are JAX arrays
        R_sersic = jnp.asarray(R_sersic)
        n_sersic = jnp.asarray(n_sersic)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        Ie = jnp.asarray(Ie) if Ie is not None else None
        magnitude = jnp.asarray(magnitude) if magnitude is not None else None
        if mag_zero_point is None and self.mag_zero_point is not None:
            mag_zero_point = jnp.asarray(self.mag_zero_point)
        elif mag_zero_point is not None:
            mag_zero_point = jnp.asarray(mag_zero_point)

        # If the model was initialized with magnitude, ALWAYS compute Ie dynamically
        if self.magnitude is not None:
            Ie = self.Ie_from_magnitude(magnitude, mag_zero_point, R_sersic, n_sersic)

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
        light_profile = Ie * jnp.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))

        # Guard against unphysical parameters (R_sersic <= 0 or n_sersic <= 0)
        is_valid = (R_sersic > 0.0) & (n_sersic > 0.0)
        return jnp.where(is_valid, light_profile, jnp.nan)
