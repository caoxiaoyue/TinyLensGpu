"""
Ellipsoid light profile.

This module implements the elliptical light profile with constant surface brightness
within an ellipsoid using the caskade framework.
"""

from typing import Optional
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.utils.geometry import ellipse2circle_transform
from TinyLensGpu.Inference.param_u import ParamU


class Ellipsoid(ck.Module):
    """
    Constant surface brightness within an elliptical region.

    Parameters
    ----------
    amp : float, optional
        Total integrated flux within the ellipsoid
    radius : float, optional
        Radius of the ellipsoid (product average of semi-major and semi-minor axes)
    e1 : float, optional
        Ellipticity component 1
    e2 : float, optional
        Ellipticity component 2
    center_x : float, optional
        Center x-coordinate in arcseconds
    center_y : float, optional
        Center y-coordinate in arcseconds
    """

    def __init__(self, amp: Optional[float] = None, radius: Optional[float] = None, 
                 e1: Optional[float] = None, e2: Optional[float] = None,
                 center_x: Optional[float] = None, center_y: Optional[float] = None) -> None:
        """
        Initialize a `Ellipsoid` instance with validated configuration.
        
        Parameters
        ----------
        amp : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        radius : Any
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
        self.radius = radius if isinstance(radius, ParamU) else ParamU("radius", radius)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

    @ck.forward
    def light(self, x: Array, y: Array, amp: Optional[Array] = None, 
              radius: Optional[Array] = None, e1: Optional[Array] = None, 
              e2: Optional[Array] = None, center_x: Optional[Array] = None, 
              center_y: Optional[Array] = None) -> Array:
        """
        Compute surface brightness at given positions.

        Parameters
        ----------
        x : array_like
            x-coordinates where to evaluate surface brightness
        y : array_like
            y-coordinates where to evaluate surface brightness
        amp : float, optional
            Total integrated flux (defaults to self.amp.value)
        radius : float, optional
            Radius (defaults to self.radius.value)
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
        radius = jnp.asarray(radius)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)

        # Transform ellipse to circle (product average)
        x_, y_ = ellipse2circle_transform(x, y, e1, e2, center_x, center_y)
        r2 = x_**2 + y_**2

        # Surface brightness: amp / (pi * radius^2) inside, 0 outside
        area = jnp.pi * radius**2
        flux_mask = jnp.where(r2 <= radius**2, 1.0, 0.0)
        
        return (amp / area) * flux_mask
