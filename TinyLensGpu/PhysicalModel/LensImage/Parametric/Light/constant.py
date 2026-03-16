"""
Constant background light profile.

This module implements a spatially uniform light component using the caskade
framework. The profile is useful for modeling sky background or any other
image-plane emission that does not vary with position.
"""

from typing import Optional

import caskade as ck
import jax.numpy as jnp
from jax import Array

from TinyLensGpu.Inference.param_u import ParamU


class ConstantBackground(ck.Module):
    """
    Spatially uniform background light profile.

    The profile represents a constant surface-brightness component across the
    full image plane. It is primarily intended for modeling sky background in
    forward simulations and linear light-profile fitting.

    Parameters
    ----------
    intensity : float, optional
        Uniform surface-brightness value returned at every input coordinate.
    """

    def __init__(self, intensity: Optional[float] = None) -> None:
        """
        Initialize the constant background parameter.

        Parameters
        ----------
        intensity : float, optional
            Uniform surface-brightness normalization.
        """
        super().__init__()

        self.intensity = intensity if isinstance(intensity, ParamU) else ParamU("intensity", intensity)

    @ck.forward
    def light(self, x: Array, y: Array, intensity: Optional[Array] = None) -> Array:
        """
        Evaluate the uniform background on the image plane.

        Parameters
        ----------
        x : array_like
            x-coordinates where to evaluate surface brightness.
        y : array_like
            y-coordinates where to evaluate surface brightness. The values are
            accepted for interface consistency but do not affect the result.
        intensity : float, optional
            Uniform surface-brightness value to return.

        Returns
        -------
        surface_brightness : array_like
            Array with the same shape as ``x`` filled with ``intensity``.
        """
        del y

        intensity = jnp.asarray(intensity)
        return jnp.ones_like(x) * intensity
