"""Vegetti-style singular Pseudo-Jaffe with an Einstein-radius tidal approximation."""

from typing import Optional, Tuple

import caskade as ck
import jax.numpy as jnp
from jax import Array

from TinyLensGpu.Inference.param_u import ParamU


class PseudoJaffeSubhalo(ck.Module):
    """Singular Pseudo-Jaffe subhalo with an approximate tidal-radius relation.

    This is a Vegetti-style profile, not an exact reproduction of Vegetti et al.
    (2010). As an intentional model choice, its tidal radius uses the host
    Einstein radius rather than the paper's main-lens density normalization:

    ``rt = sqrt(b_sub * theta_E_host)``.

    ``theta_E_host`` is the host's Einstein-radius parameter; it is not generally
    equal to the paper's lens strength for a non-isothermal EPL. No conversion
    to that normalization is applied. The square-root relation is an adopted
    approximation, not a general tidal-stripping calculation.

    Consequently, ``rt`` is a caskade pointer parameter rather than an
    independent inference parameter.  The convergence and radial deflection
    follow the singular Pseudo-Jaffe convention used by Vegetti et al.:

    ``kappa(R) = b_sub / 2 * (1/R - 1/sqrt(R**2 + rt**2))``

    ``alpha(R) = b_sub / R * (R + rt - sqrt(R**2 + rt**2))``.

    Parameters
    ----------
    b_sub : float or ParamU, optional
        Subhalo lens strength. This is the sole radial/mass degree of freedom.
    theta_E_host : float or ParamU, optional
        Host Einstein radius used by the approximate tidal relation. Pass the
        same ``ParamU`` object used by the host mass profile to keep the relation
        live during inference. Use the same angular units as ``b_sub``.
    center_x, center_y : float or ParamU, optional
        Subhalo center coordinates in the lens plane.
    """

    def __init__(
        self,
        b_sub: Optional[float] = None,
        theta_E_host: Optional[float] = None,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.b_sub = (
            b_sub if isinstance(b_sub, ParamU) else ParamU("b_sub", b_sub)
        )
        self.theta_E_host = (
            theta_E_host
            if isinstance(theta_E_host, ParamU)
            else ParamU("theta_E_host", theta_E_host)
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

        b_sub_param = self.b_sub
        theta_e_host_param = self.theta_E_host

        def tidal_radius(_):
            product = b_sub_param.value * theta_e_host_param.value
            # ``sqrt`` has an infinite derivative at zero. Evaluate that branch
            # at a finite dummy value, then select the zero-strength limit rt=0,
            # so the zero-strength profile remains safe under autodiff.
            safe_product = jnp.where(product == 0, jnp.ones_like(product), product)
            return jnp.where(
                product == 0,
                jnp.zeros_like(product),
                jnp.sqrt(safe_product),
            )

        self.rt = ck.Param(
            "rt",
            tidal_radius,
            link=(b_sub_param, theta_e_host_param),
        )

    @ck.forward
    def deriv(
        self,
        x: Array,
        y: Array,
        b_sub: Optional[Array] = None,
        rt: Optional[Array] = None,
        center_x: Optional[Array] = None,
        center_y: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Evaluate the singular Pseudo-Jaffe deflection field."""
        b_sub = jnp.asarray(b_sub)
        rt = jnp.asarray(rt)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)

        dx = x - center_x
        dy = y - center_y
        radius = jnp.hypot(dx, dy)
        radius_safe = jnp.maximum(radius, 1.0e-8)

        # Cancellation-free form of the singular Pseudo-Jaffe deflection:
        # R + rt - hypot(R, rt) = 2 R rt / (R + rt + hypot(R, rt)).
        root = jnp.hypot(radius_safe, rt)
        alpha_r = 2.0 * b_sub * rt / (radius_safe + rt + root)
        scale = alpha_r / radius_safe

        return scale * dx, scale * dy
