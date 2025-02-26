import functools
import jax.numpy as jnp
from jax import jit
from TinyLensGpu.Profile.ProfileBase import MassProfile
from TinyLensGpu.Profile.util import xy_transform, relocate_radii, hyp2f1_series, ellipticity2phi_q

class EPL(MassProfile):
    _name = "EPL"
    _params = ["theta_E", "e1", "e2", "center_x", "center_y", "slope"]


    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, e1, e2, center_x, center_y, slope):
        """Calculate deflection angles for EPL profile.
        
        Following Tessore & Metcalf 2015 (arXiv:1507.01819)
        The convergence has the form of kappa(x, y)=0.5*(2-t)*(b/sqrt(q^2*x^2+y^2))^t
        In this form, b/sqrt(q) is the Einstein radius in the intermediate-axis convention
        """
        PA, q = ellipticity2phi_q(e1, e2)
        # Transform coordinates
        x_new, y_new = xy_transform(x, y, center_x, center_y, PA)
        x_new, y_new, _ = relocate_radii(x_new, y_new)
        
        # Convert 3D slope to 2D slope
        slope = slope - 1
        
        # Calculate parameters needed for Tessore & Metcalf formula
        f = (1.0 - q)/(1.0 + q)
        thetaE_tessore = theta_E * jnp.sqrt(q)  # Convert to intermediate-axis convention
        
        # Calculate polar coordinates
        phi = jnp.arctan2(y_new, q * x_new)  # eq.4
        R = jnp.sqrt(q**2 * x_new**2 + y_new**2)  # eq.3
        
        # Calculate complex coordinates (Euler's formula)
        z = jnp.cos(phi) + 1j * jnp.sin(phi)
        
        # Calculate hypergeometric function using series expansion
        z2 = -f * z**2
        hyp2f1 = hyp2f1_series(1.0, 0.5*slope, 2.0-0.5*slope, z2)
        
        # Calculate deflection (eq.13)
        prefactor = 2.0 * thetaE_tessore/(1.0 + q) * (thetaE_tessore/R)**(slope-1.)
        tmp = prefactor * z * hyp2f1
        
        # Extract real and imaginary parts
        alpha_x = jnp.real(tmp)
        alpha_y = jnp.imag(tmp)
        
        # Transform back to original frame
        alpha_x, alpha_y = xy_transform(alpha_x, alpha_y, 0.0, 0.0, -PA)
        
        return alpha_x, alpha_y
