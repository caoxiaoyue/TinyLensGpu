"""
Dipole mass profile.
"""
from typing import Optional, Tuple
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.Inference.param_u import ParamU


class Dipole(ck.Module):
    """
    Dipole perturbation to the lens deflection field.

    This component models an m=1 angular perturbation with a preferred
    orientation and coupling strength.

    Parameters
    ----------
    com_x, com_y : float, optional
        Dipole center in arcsec.
    phi_dipole : float, optional
        Dipole orientation angle in radians.
    coupling : float, optional
        Deflection amplitude of the dipole term.
    """

    def __init__(self, com_x: Optional[float] = None, com_y: Optional[float] = None,
                 phi_dipole: Optional[float] = None, coupling: Optional[float] = None) -> None:
        """
        Initialize dipole perturbation component.

        Parameters
        ----------
        com_x, com_y : float, optional
            Dipole center in arcsec.
        phi_dipole : float, optional
            Dipole orientation in radians.
        coupling : float, optional
            Dipole amplitude.
        """
        super().__init__()
        self.com_x = com_x if isinstance(com_x, ParamU) else ParamU("com_x", com_x)
        self.com_y = com_y if isinstance(com_y, ParamU) else ParamU("com_y", com_y)
        self.phi_dipole = phi_dipole if isinstance(phi_dipole, ParamU) else ParamU("phi_dipole", phi_dipole)
        self.coupling = coupling if isinstance(coupling, ParamU) else ParamU("coupling", coupling)

    @ck.forward
    def deriv(self, x: Array, y: Array, com_x: Optional[Array] = None, 
              com_y: Optional[Array] = None, phi_dipole: Optional[Array] = None, 
              coupling: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Evaluate dipole deflection field.

        Parameters
        ----------
        x, y : Array
            Image-plane coordinates in arcsec.
        com_x, com_y : Array, optional
            Dipole center.
        phi_dipole : Array, optional
            Orientation angle in radians.
        coupling : Array, optional
            Dipole amplitude.

        Returns
        -------
        tuple[Array, Array]
            Deflection components ``(alpha_x, alpha_y)``.
        """
        com_x = jnp.asarray(com_x)
        com_y = jnp.asarray(com_y)
        phi_dipole = jnp.asarray(phi_dipole)
        coupling = jnp.asarray(coupling)

        # coordinate shift
        x_shift = x - com_x
        y_shift = y - com_y

        # rotation angle
        sin_phi = jnp.sin(phi_dipole)
        cos_phi = jnp.cos(phi_dipole)
        x_ = cos_phi * x_shift + sin_phi * y_shift
        y_ = -sin_phi * x_shift + cos_phi * y_shift

        r = jnp.sqrt(x_**2 + y_**2)
        r = jnp.maximum(r, 1e-9)

        f_x_prim = coupling * x_ / r
        f_y_prim = jnp.zeros_like(x_)
        
        # rotate back
        f_x = cos_phi * f_x_prim - sin_phi * f_y_prim
        f_y = sin_phi * f_x_prim + cos_phi * f_y_prim
        
        return f_x, f_y
