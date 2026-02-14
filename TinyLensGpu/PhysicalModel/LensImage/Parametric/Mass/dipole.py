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
    Represent the `Dipole` component in the TinyLensGpu pipeline.
    
    Parameters
    ----------
    com_x : Any
        Configuration argument consumed during construction of this component.
    com_y : Any
        Configuration argument consumed during construction of this component.
    phi_dipole : Any
        Configuration argument consumed during construction of this component.
    coupling : Any
        Configuration argument consumed during construction of this component.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    def __init__(self, com_x: Optional[float] = None, com_y: Optional[float] = None,
                 phi_dipole: Optional[float] = None, coupling: Optional[float] = None) -> None:
        """
        Initialize a `Dipole` instance with validated configuration.
        
        Parameters
        ----------
        com_x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        com_y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        phi_dipole : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        coupling : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
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
        Compute deriv.
        
        Parameters
        ----------
        x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        com_x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        com_y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        phi_dipole : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        coupling : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
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
