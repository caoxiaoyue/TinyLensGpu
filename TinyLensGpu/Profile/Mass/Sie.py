import functools
import jax.numpy as jnp
from jax import jit
from TinyLensGpu.Profile.ProfileBase import MassProfile
from TinyLensGpu.Profile.util import ellipticity2phi_q, xy_transform, relocate_radii
import jax

class SIE(MassProfile):
    _name = "SIE"
    _params = ["theta_E", "e1", "e2", "center_x", "center_y"]

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, e1, e2, center_x, center_y):
        """Calculate deflection angles for SIE profile."""
        PA, q = ellipticity2phi_q(e1, e2)
        # jax.debug.print("PA: {}", PA/jnp.pi*180.0)
        # jax.debug.print("q: {}", q)

        # Transform coordinates
        PA = PA - jnp.pi/2.0  # Convert to the convention used in the Python reference
        x_new, y_new = xy_transform(x, y, center_x, center_y, PA)
        x_new, y_new, r_new = relocate_radii(x_new, y_new)
        
        # Calculate deflection in rotated frame
        qfact = jnp.sqrt(1.0/q - q)
        eps = 1e-8  # Small value for numerical stability
        
        # Handle special case when q approx 1 (SIS case)
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
        # jax.debug.print("alpha_x max: {}", jnp.max(alpha_x))
        # jax.debug.print("alpha_y max: {}", jnp.max(alpha_y))
        
        # Transform back to original frame
        return xy_transform(alpha_x, alpha_y, 0.0, 0.0, -PA)

