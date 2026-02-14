"""
Elliptical Power Law (EPL) mass profile.

This module implements the EPL mass distribution profile using the caskade
framework for modular, composable gravitational lensing models.
"""

from typing import Optional, Tuple
import caskade as ck
import jax
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.utils.geometry import ellipticity2phi_q, xy_transform
from TinyLensGpu.Inference.param_u import ParamU


def omega_iterative(phi: Array, t: Array, q: Array, niter: int = 50) -> Array:
    """
    Compute omega iterative.
    
    Parameters
    ----------
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    t : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    niter : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    f = (1.0 - q) / (1.0 + q)
    Omega_0 = jnp.exp(1j * phi)
    fact = -f * jnp.exp(2j * phi)
    
    def body_fun(carry, n):
        """
        Compute body fun.
        
        Parameters
        ----------
        carry : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        n : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        current_sum, current_term = carry
        # n is 1-based index from scan over range
        factor = (2.0 * n - (2.0 - t)) / (2.0 * n + (2.0 - t)) * fact
        next_term = current_term * factor
        next_sum = current_sum + next_term
        return (next_sum, next_term), None

    # Scan from n=1 to niter
    init_val = (Omega_0, Omega_0)
    # xs must be same length as iterations.
    xs = jnp.arange(1, niter + 1)
    
    final_state, _ = jax.lax.scan(body_fun, init_val, xs)
    return final_state[0]


class EPL(ck.Module):
    """
    Elliptical Power Law mass profile.

    Parameters
    ----------
    theta_E : float, optional
        Einstein radius in arcseconds
    gamma : float, optional
        Power-law slope
    e1 : float, optional
        Ellipticity component 1
    e2 : float, optional
        Ellipticity component 2
    center_x : float, optional
        Center x-coordinate in arcseconds
    center_y : float, optional
        Center y-coordinate in arcseconds
    """

    def __init__(self, theta_E: Optional[float] = None, gamma: Optional[float] = None,
                 e1: Optional[float] = None, e2: Optional[float] = None, 
                 center_x: Optional[float] = None, center_y: Optional[float] = None) -> None:
        """
        Initialize a `EPL` instance with validated configuration.
        
        Parameters
        ----------
        theta_E : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        gamma : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        e1 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        e2 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        center_x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        center_y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        super().__init__()

        # Define parameters using ParamU (or convert if already ParamU)
        self.theta_E = theta_E if isinstance(theta_E, ParamU) else ParamU("theta_E", theta_E)
        self.gamma = gamma if isinstance(gamma, ParamU) else ParamU("gamma", gamma)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

    @ck.forward
    def deriv(self, x: Array, y: Array, theta_E: Optional[Array] = None, 
              gamma: Optional[Array] = None, e1: Optional[Array] = None, 
              e2: Optional[Array] = None, center_x: Optional[Array] = None, 
              center_y: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Calculate deflection angles for EPL profile.

        Parameters
        ----------
        x : array_like
            x-coordinates where to evaluate deflection
        y : array_like
            y-coordinates where to evaluate deflection
        theta_E : float, optional
            Einstein radius
        gamma : float, optional
            Power-law slope
        e1 : float, optional
            Ellipticity component 1
        e2 : float, optional
            Ellipticity component 2
        center_x : float, optional
            Center x-coordinate
        center_y : float, optional
            Center y-coordinate

        Returns
        -------
        alpha_x : array_like
            Deflection angle in x-direction
        alpha_y : array_like
            Deflection angle in y-direction
        """
        # Ensure parameters are JAX arrays
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        theta_E = jnp.asarray(theta_E)
        gamma = jnp.asarray(gamma)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)

        # Convert parameters
        phi_rot, q = ellipticity2phi_q(e1, e2)
        b = theta_E * jnp.sqrt(q)
        t = gamma - 1.0

        # Transform coordinates to aligned frame
        # lenstronomy uses rotation by phi_G (angle of major axis)
        x_new, y_new = xy_transform(x, y, center_x, center_y, phi_rot)
        
        # Calculate deflection in aligned frame
        # Logic from lenstronomy.LensModel.Profiles.epl_numba
        
        # zz = x * q + 1j * y
        zz = x_new * q + 1j * y_new
        R = jnp.abs(zz)
        R = jnp.maximum(R, 1e-9) # Numerical stability
        phi_ell = jnp.angle(zz)
        
        # Calculate Omega
        # Use 50 iterations which is usually sufficient
        Omega = omega_iterative(phi_ell, t, q, niter=50)
        
        term = (b / R)**(t - 1.0)
        prefactor = (2.0 * b) / (1.0 + q)
        
        alpha_complex = prefactor * term * Omega
        
        alpha_x_new = jnp.real(alpha_complex)
        alpha_y_new = jnp.imag(alpha_complex)

        # Transform back to original frame
        return xy_transform(alpha_x_new, alpha_y_new, 0.0, 0.0, -phi_rot)
