"""
Moffat light profile.

This module implements the elliptical Moffat light profile using the
caskade framework.
"""

from typing import Optional
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.utils.geometry import ellipse2circle_transform
from TinyLensGpu.Inference.param_u import ParamU


class MoffatEllipse(ck.Module):
    """
    Elliptical Moffat light profile.

    The Moffat profile is often used to model the Point Spread Function (PSF)
    or galaxy light distributions with broader wings than a Gaussian.

    Parameters
    ----------
    amp : float, optional
        Normalization (surface brightness at the center)
    alpha : float, optional
        Scale parameter related to the FWHM
    beta : float, optional
        Exponent parameter (beta=1 for Cauchy, beta->inf for Gaussian)
    e1 : float, optional
        Ellipticity component 1
    e2 : float, optional
        Ellipticity component 2
    center_x : float, optional
        Center x-coordinate in arcseconds
    center_y : float, optional
        Center y-coordinate in arcseconds
    """

    def __init__(self, amp: Optional[float] = None, alpha: Optional[float] = None, 
                 beta: Optional[float] = None, e1: Optional[float] = None, 
                 e2: Optional[float] = None, center_x: Optional[float] = None, 
                 center_y: Optional[float] = None) -> None:
        """
        Initialize a `MoffatEllipse` instance with validated configuration.
        
        Parameters
        ----------
        amp : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        alpha : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        beta : Any
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
        self.alpha = alpha if isinstance(alpha, ParamU) else ParamU("alpha", alpha)
        self.beta = beta if isinstance(beta, ParamU) else ParamU("beta", beta)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

    @ck.forward
    def light(self, x: Array, y: Array, amp: Optional[Array] = None, 
              alpha: Optional[Array] = None, beta: Optional[Array] = None, 
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
            Normalization (defaults to self.amp.value)
        alpha : float, optional
            Scale parameter (defaults to self.alpha.value)
        beta : float, optional
            Exponent (defaults to self.beta.value)
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
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)

        # Transform ellipse to circle (product average)
        x_, y_ = ellipse2circle_transform(x, y, e1, e2, center_x, center_y)
        r2 = x_**2 + y_**2

        # Moffat profile formula: I(r) = amp * (1 + (r/alpha)^2)**(-beta)
        return amp * (1.0 + r2 / alpha**2) ** (-beta)
