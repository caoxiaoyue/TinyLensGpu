"""
Pseudo-Jaffe light profile.

This module implements the elliptical Pseudo-Jaffe light profile using the
caskade framework.
"""

from typing import Optional
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.utils.geometry import transform_e1e2_square_average
from TinyLensGpu.Inference.param_u import ParamU


class PseudoJaffeEllipse(ck.Module):
    """
    Elliptical Pseudo-Jaffe light profile.

    The Pseudo-Jaffe profile is a dual pseudo isothermal mass distribution
    projected onto 2D.

    Parameters
    ----------
    amp : float, optional
        Surface brightness amplitude
    Ra : float, optional
        Core radius
    Rs : float, optional
        Scale radius (transition radius)
    e1 : float, optional
        Ellipticity component 1
    e2 : float, optional
        Ellipticity component 2
    center_x : float, optional
        Center x-coordinate in arcseconds
    center_y : float, optional
        Center y-coordinate in arcseconds
    """

    def __init__(self, amp: Optional[float] = None, Ra: Optional[float] = None, 
                 Rs: Optional[float] = None, e1: Optional[float] = None, 
                 e2: Optional[float] = None, center_x: Optional[float] = None, 
                 center_y: Optional[float] = None) -> None:
        """
        Initialize a `PseudoJaffeEllipse` instance with validated configuration.
        
        Parameters
        ----------
        amp : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        Ra : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        Rs : Any
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

        self.amp = amp if isinstance(amp, ParamU) else ParamU("amp", amp)
        self.Ra = Ra if isinstance(Ra, ParamU) else ParamU("Ra", Ra)
        self.Rs = Rs if isinstance(Rs, ParamU) else ParamU("Rs", Rs)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

    @ck.forward
    def light(self, x: Array, y: Array, amp: Optional[Array] = None, 
              Ra: Optional[Array] = None, Rs: Optional[Array] = None, 
              e1: Optional[Array] = None, e2: Optional[Array] = None, 
              center_x: Optional[Array] = None, center_y: Optional[Array] = None) -> Array:
        """
        Compute surface brightness at given positions.

        Parameters
        ----------
        x : array_like
            x-coordinates where to evaluate surface brightness
        y : array_like
            y-coordinates where to evaluate surface brightness
        amp : float, optional
            Amplitude (defaults to self.amp.value)
        Ra : float, optional
            Core radius (defaults to self.Ra.value)
        Rs : float, optional
            Scale radius (defaults to self.Rs.value)
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
        surface_brightness : array_like
            Surface brightness at the given positions
        """
        amp = jnp.asarray(amp)
        Ra = jnp.asarray(Ra)
        Rs = jnp.asarray(Rs)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)

        # Sort Ra and Rs to ensure Rs > Ra (avoiding singularity)
        Ra_new = jnp.where(Ra >= Rs, Rs, Ra)
        Rs_new = jnp.where(Ra >= Rs, Ra, Rs)
        Ra_new = jnp.maximum(Ra_new, 1e-9)
        Rs_new = jnp.maximum(Rs_new, Ra_new + 1e-9)

        # Transform coordinates (square average) to match lenstronomy's PseudoJaffeEllipse
        x_, y_ = transform_e1e2_square_average(x, y, e1, e2, center_x, center_y)
        r = jnp.sqrt(x_**2 + y_**2)

        # Projected density formula:
        # sigma(r) = amp * Ra * Rs / (Rs - Ra) * (1/sqrt(Ra^2 + r^2) - 1/sqrt(Rs^2 + r^2))
        return amp * Ra_new * Rs_new / (Rs_new - Ra_new) * (
            1.0 / jnp.sqrt(Ra_new**2 + r**2) - 1.0 / jnp.sqrt(Rs_new**2 + r**2)
        )
