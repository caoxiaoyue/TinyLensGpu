"""
Hernquist light profile.

This module implements the elliptical Hernquist light profile using the
caskade framework.
"""

from typing import Optional
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.utils.geometry import ellipse2circle_transform
from TinyLensGpu.Inference.param_u import ParamU


class HernquistEllipse(ck.Module):
    """
    Elliptical Hernquist light profile.

    The Hernquist profile is a 3D density profile projected onto 2D.
    It is often used to model the stellar distribution of galaxies.

    Parameters
    ----------
    amp : float, optional
        Surface brightness amplitude
    Rs : float, optional
        Scale radius (half-light radius = Rs / 0.551)
    e1 : float, optional
        Ellipticity component 1
    e2 : float, optional
        Ellipticity component 2
    center_x : float, optional
        Center x-coordinate in arcseconds
    center_y : float, optional
        Center y-coordinate in arcseconds
    """

    def __init__(self, amp: Optional[float] = None, Rs: Optional[float] = None, 
                 e1: Optional[float] = None, e2: Optional[float] = None,
                 center_x: Optional[float] = None, center_y: Optional[float] = None) -> None:
        """
        Initialize a `HernquistEllipse` instance with validated configuration.
        
        Parameters
        ----------
        amp : Any
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
        self.Rs = Rs if isinstance(Rs, ParamU) else ParamU("Rs", Rs)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

    @ck.forward
    def light(self, x: Array, y: Array, amp: Optional[Array] = None, 
              Rs: Optional[Array] = None, e1: Optional[Array] = None, 
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
            Amplitude (defaults to self.amp.value)
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
        Rs = jnp.asarray(Rs)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)

        # Transform ellipse to circle (product average)
        x_, y_ = ellipse2circle_transform(x, y, e1, e2, center_x, center_y)
        r = jnp.sqrt(x_**2 + y_**2)
        
        # Handle numerical singularity at r=0
        r = jnp.maximum(r, 1e-10)
        X = r / Rs
        
        # Handle X=1 case for F(X)
        # In lenstronomy, X[X == 1] = 1.000001
        X = jnp.where(jnp.abs(X - 1.0) < 1e-10, 1.000001, X)
        
        def F(X):
            """
            Compute F.
            
            Parameters
            ----------
            X : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            cond1 = X < 1
            # For X < 1: 1 / sqrt(1 - X^2) * arctanh(sqrt(1 - X^2))
            u1 = jnp.sqrt(1 - X**2)
            val1 = 1 / u1 * jnp.arctanh(u1)
            
            # For X > 1: 1 / sqrt(X^2 - 1) * arctan(sqrt(X^2 - 1))
            u2 = jnp.sqrt(X**2 - 1)
            val2 = 1 / u2 * jnp.arctan(u2)
            
            return jnp.where(cond1, val1, val2)

        # Projected density formula
        # sigma = amp / (X^2 - 1)^2 * (-3 + (2 + X^2) * F(X))
        return amp / (X**2 - 1)**2 * (-3 + (2 + X**2) * F(X))
