"""
Singular Isothermal Ellipsoid (SIE) mass profile using caskade.

This module implements the SIE mass distribution profile using the caskade
framework for modular, composable gravitational lensing models.
"""

import caskade as ck
import jax.numpy as jnp
from ..utils import ellipticity2phi_q, xy_transform, relocate_radii
from ..param_u import ParamU


class SIE(ck.Module):
    """
    Singular Isothermal Ellipsoid mass profile.

    Parameters
    ----------
    theta_E : float, optional
        Einstein radius in arcseconds
    e1 : float, optional
        Ellipticity component 1
    e2 : float, optional
        Ellipticity component 2
    center_x : float, optional
        Center x-coordinate in arcseconds
    center_y : float, optional
        Center y-coordinate in arcseconds
    """

    def __init__(self, theta_E=None, e1=None, e2=None,
                 center_x=None, center_y=None):
        super().__init__()

        # Define parameters using ParamU (or convert if already ParamU)
        self.theta_E = theta_E if isinstance(theta_E, ParamU) else ParamU("theta_E", theta_E)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

    @ck.forward
    def deriv(self, x, y, theta_E=None, e1=None, e2=None,
              center_x=None, center_y=None):
        """
        Calculate deflection angles for SIE profile.

        Parameters
        ----------
        x : array_like
            x-coordinates where to evaluate deflection
        y : array_like
            y-coordinates where to evaluate deflection
        theta_E : float, optional
            Einstein radius (defaults to self.theta_E.value)
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
        alpha_x : array_like
            Deflection angle in x-direction
        alpha_y : array_like
            Deflection angle in y-direction
        """
        # Ensure parameters are JAX arrays (convert from torch if needed)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        theta_E = jnp.asarray(theta_E)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)

        # Convert ellipticity to position angle and axis ratio
        PA, q = ellipticity2phi_q(e1, e2)

        # Transform coordinates to aligned frame
        PA = PA - jnp.pi/2.0  # Convention adjustment
        x_new, y_new = xy_transform(x, y, center_x, center_y, PA)
        x_new, y_new, r_new = relocate_radii(x_new, y_new)

        # Calculate deflection in rotated frame
        qfact = jnp.sqrt(1.0/q - q)
        eps = 1e-8  # Small value for numerical stability

        # Handle special case when q ≈ 1 (SIS case)
        is_sis = jnp.abs(qfact) <= eps

        # SIS deflection
        alpha_x_sis = x_new/r_new * theta_E
        alpha_y_sis = y_new/r_new * theta_E

        # SIE deflection
        psi = jnp.sqrt(1.0/q**2.0 - 1.0) * x_new/r_new
        phi = jnp.sqrt(1.0 - q**2.0) * y_new/r_new

        # Handle numerical stability for arcsinh and arcsin
        psi = jnp.clip(psi, -1e10, 1e10)
        phi = jnp.clip(phi, -1.0 + eps, 1.0 - eps)

        alpha_x_sie = jnp.arcsinh(psi)/qfact * theta_E
        alpha_y_sie = jnp.arcsin(phi)/qfact * theta_E

        # Select between SIS and SIE based on q value
        alpha_x = jnp.where(is_sis, alpha_x_sis, alpha_x_sie)
        alpha_y = jnp.where(is_sis, alpha_y_sis, alpha_y_sie)

        # Transform back to original frame
        return xy_transform(alpha_x, alpha_y, 0.0, 0.0, -PA)
