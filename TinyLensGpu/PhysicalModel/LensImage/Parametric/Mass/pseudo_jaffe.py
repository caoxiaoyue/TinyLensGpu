"""
Pseudo Jaffe mass profile.
"""
from typing import Optional, Tuple
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.Inference.param_u import ParamU


class PseudoJaffe(ck.Module):
    """
    Represent the `PseudoJaffe` component in the TinyLensGpu pipeline.
    
    Parameters
    ----------
    sigma0 : Any
        Configuration argument consumed during construction of this component.
    Ra : Any
        Configuration argument consumed during construction of this component.
    Rs : Any
        Configuration argument consumed during construction of this component.
    center_x : Any
        Configuration argument consumed during construction of this component.
    center_y : Any
        Configuration argument consumed during construction of this component.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    def __init__(self, sigma0: Optional[float] = None, Ra: Optional[float] = None, 
                 Rs: Optional[float] = None, center_x: Optional[float] = None, 
                 center_y: Optional[float] = None) -> None:
        """
        Initialize a `PseudoJaffe` instance with validated configuration.
        
        Parameters
        ----------
        sigma0 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        Ra : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        Rs : Any
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
        self.sigma0 = sigma0 if isinstance(sigma0, ParamU) else ParamU("sigma0", sigma0)
        self.Ra = Ra if isinstance(Ra, ParamU) else ParamU("Ra", Ra)
        self.Rs = Rs if isinstance(Rs, ParamU) else ParamU("Rs", Rs)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

    @staticmethod
    def _sort_ra_rs(Ra, Rs):
        # Ra < Rs
        """
        Internal helper to sort ra rs.
        
        Parameters
        ----------
        Ra : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        Rs : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        Ra_new = jnp.minimum(Ra, Rs)
        Rs_new = jnp.maximum(Ra, Rs)
        
        # Avoid singularities
        Ra_new = jnp.maximum(Ra_new, 1e-8)
        # Ensure Rs > Ra
        Rs_new = jnp.where(Rs_new < Ra_new + 1e-8, Ra_new + 2e-8, Rs_new)
        
        return Ra_new, Rs_new

    @staticmethod
    def _f_A20(r_a, r_s):
        """
        Internal helper to f A20.
        
        Parameters
        ----------
        r_a : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        r_s : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        return r_a / (1 + jnp.sqrt(1 + r_a**2)) - r_s / (1 + jnp.sqrt(1 + r_s**2))

    @ck.forward
    def deriv(self, x: Array, y: Array, sigma0: Optional[Array] = None, 
              Ra: Optional[Array] = None, Rs: Optional[Array] = None, 
              center_x: Optional[Array] = None, center_y: Optional[Array] = None) -> Tuple[Array, Array]:
        
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
        sigma0 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        Ra : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        Rs : Any
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
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        sigma0 = jnp.asarray(sigma0)
        Ra = jnp.asarray(Ra)
        Rs = jnp.asarray(Rs)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)

        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        r_safe = jnp.maximum(r, 1e-4) # self._s = 0.0001 in lenstronomy
        
        alpha_r = 2 * sigma0 * Ra * Rs / (Rs - Ra) * self._f_A20(r_safe / Ra, r_safe / Rs)
        
        f_x = alpha_r * x_ / r_safe
        f_y = alpha_r * y_ / r_safe
        
        return f_x, f_y
