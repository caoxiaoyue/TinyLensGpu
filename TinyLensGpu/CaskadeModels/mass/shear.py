"""
External Shear mass profile using caskade.

This module implements the external shear contribution to the lensing
deflection field using the caskade framework.
"""

import caskade as ck
import jax.numpy as jnp


class Shear(ck.Module):
    """
    External shear mass profile.

    The external shear represents the tidal gravitational field from
    large-scale structure around the lens system.

    Parameters
    ----------
    gamma1 : float, optional
        Shear component 1
    gamma2 : float, optional
        Shear component 2
    """

    def __init__(self, gamma1=None, gamma2=None):
        super().__init__()

        # Define parameters using ck.Param
        self.gamma1 = ck.Param("gamma1", gamma1)
        self.gamma2 = ck.Param("gamma2", gamma2)

    @ck.forward
    def deriv(self, x, y, gamma1=None, gamma2=None):
        """
        Calculate deflection angles for external shear.

        Parameters
        ----------
        x : array_like
            x-coordinates where to evaluate deflection
        y : array_like
            y-coordinates where to evaluate deflection
        gamma1 : float, optional
            Shear component 1 (defaults to self.gamma1.value)
        gamma2 : float, optional
            Shear component 2 (defaults to self.gamma2.value)

        Returns
        -------
        alpha_x : array_like
            Deflection angle in x-direction
        alpha_y : array_like
            Deflection angle in y-direction
        """
        # Ensure parameters are JAX arrays
        gamma1 = jnp.asarray(gamma1)
        gamma2 = jnp.asarray(gamma2)

        alpha_x = gamma1 * x + gamma2 * y
        alpha_y = gamma2 * x - gamma1 * y
        return alpha_x, alpha_y
