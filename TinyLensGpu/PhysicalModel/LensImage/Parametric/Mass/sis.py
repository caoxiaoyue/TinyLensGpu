"""Singular Isothermal Sphere (SIS) mass profile."""

from typing import Optional, Tuple

import caskade as ck
import jax.numpy as jnp
from jax import Array

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.utils.geometry import relocate_radii


def _sis_deflection(
    x: Array,
    y: Array,
    theta_E: Array,
    center_x: Array,
    center_y: Array,
) -> Tuple[Array, Array]:
    """Evaluate SIS deflection, including the shared central regularization."""
    x_local = x - center_x
    y_local = y - center_y
    x_local, y_local, radius = relocate_radii(x_local, y_local)
    scale = theta_E / radius
    return x_local * scale, y_local * scale


class SIS(ck.Module):
    """Circular singular isothermal mass profile.

    Parameters
    ----------
    theta_E : float, optional
        Einstein radius in arcseconds.
    center_x : float, optional
        Center x-coordinate in arcseconds.
    center_y : float, optional
        Center y-coordinate in arcseconds.
    """

    def __init__(
        self,
        theta_E: Optional[float] = None,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.theta_E = (
            theta_E if isinstance(theta_E, ParamU) else ParamU("theta_E", theta_E)
        )
        self.center_x = (
            center_x
            if isinstance(center_x, ParamU)
            else ParamU("center_x", center_x)
        )
        self.center_y = (
            center_y
            if isinstance(center_y, ParamU)
            else ParamU("center_y", center_y)
        )

    @ck.forward
    def deriv(
        self,
        x: Array,
        y: Array,
        theta_E: Optional[Array] = None,
        center_x: Optional[Array] = None,
        center_y: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Calculate the deflection angles at coordinates ``(x, y)``."""
        theta_E = jnp.asarray(theta_E)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        return _sis_deflection(x, y, theta_E, center_x, center_y)
