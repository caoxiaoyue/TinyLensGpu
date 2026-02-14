"""
External Shear mass profile .

This module implements the external shear contribution to the lensing
deflection field using the caskade framework.
"""

from typing import Optional, Tuple
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.Inference.param_u import ParamU


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

    def __init__(self, gamma1: Optional[float] = None, gamma2: Optional[float] = None) -> None:
        """
        Initialize a `Shear` instance with validated configuration.
        
        Parameters
        ----------
        gamma1 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        gamma2 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        super().__init__()

        # Define parameters using ParamU (or convert if already ParamU)
        self.gamma1 = gamma1 if isinstance(gamma1, ParamU) else ParamU("gamma1", gamma1)
        self.gamma2 = gamma2 if isinstance(gamma2, ParamU) else ParamU("gamma2", gamma2)

    @ck.forward
    def deriv(self, x: Array, y: Array, gamma1: Optional[Array] = None, 
              gamma2: Optional[Array] = None) -> Tuple[Array, Array]:
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
