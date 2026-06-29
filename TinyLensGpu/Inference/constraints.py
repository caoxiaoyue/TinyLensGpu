"""
Parameter constraints for inference.

This module provides Caskade modules that enforce parameter constraints
during sampling by returning NaN when constraints are violated, which
propagates to negative infinity log-likelihood and effectively rejects
the sample during nested sampling.
"""

import caskade as ck
import jax.numpy as jnp

from .param_u import ParamU


class EllipticityConstraint(ck.Module):
    """
    Caskade module to reparametrize ellipticity parameters and enforce constraints.

    Wraps two raw ellipticity parameters (``e1_raw``, ``e2_raw``) and produces
    constrained outputs (``out_e1``, ``out_e2``). When the derived axis ratio
    ``q`` or position angle ``phi`` fall outside the specified bounds, the
    module returns NaN. This NaN propagates through the forward model and
    results in a negative infinity log-likelihood, causing nested samplers to
    reject the sample automatically.

    The axis ratio is computed as

    .. math::
        q = \\frac{1 - e}{1 + e}, \\quad e = \\sqrt{e_1^2 + e_2^2}

    and the position angle (in degrees, wrapped to ``[0, 180)``) is

    .. math::
        \\phi = \\frac{1}{2} \\arctan2(e_2, e_1) \\times \\frac{180}{\\pi}

    Parameters
    ----------
    e1 : ParamU
        Raw ellipticity parameter along the x-axis.
    e2 : ParamU
        Raw ellipticity parameter at 45 degrees.
    q_min : float, optional
        Lower bound on the axis ratio (default 0.0).
    q_max : float, optional
        Upper bound on the axis ratio (default 1.0).
    phi_min : float, optional
        Lower bound on the position angle in degrees (default 0.0).
    phi_max : float, optional
        Upper bound on the position angle in degrees (default 180.0).

    Examples
    --------
    >>> from TinyLensGpu.Inference import ParamU, EllipticityConstraint
    >>> e1 = ParamU("e1", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3])
    >>> e2 = ParamU("e2", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3])
    >>> constraint = EllipticityConstraint(e1, e2, q_min=0.3, phi_min=70.0, phi_max=95.0)
    """

    def __init__(
        self,
        e1: ParamU,
        e2: ParamU,
        q_min: float = 0.0,
        q_max: float = 1.0,
        phi_min: float = 0.0,
        phi_max: float = 180.0,
    ):
        super().__init__()
        self.e1_raw = e1
        self.e2_raw = e2
        self.q_min = q_min
        self.q_max = q_max
        self.phi_min = phi_min
        self.phi_max = phi_max

    def _check_valid(self, e1_raw, e2_raw):
        """Return a boolean mask for constraint satisfaction.

        Parameters
        ----------
        e1_raw, e2_raw : Array
            Raw ellipticity components.

        Returns
        -------
        Array (bool)
            ``True`` where the ellipticity and position-angle constraints
            are all satisfied.
        """
        e = jnp.sqrt(e1_raw**2 + e2_raw**2)
        q = (1 - e) / (1 + e)

        phi_rad = 0.5 * jnp.arctan2(e2_raw, e1_raw)
        phi_deg = jnp.degrees(phi_rad)
        phi_deg_wrapped = jnp.mod(phi_deg, 180.0)

        return (
            (q >= self.q_min)
            & (q <= self.q_max)
            & (phi_deg_wrapped >= self.phi_min)
            & (phi_deg_wrapped <= self.phi_max)
        )

    @ck.forward
    def out_e1(self, e1_raw, e2_raw):
        """Return ``e1_raw`` when constraints are satisfied, otherwise NaN."""
        return jnp.where(self._check_valid(e1_raw, e2_raw), e1_raw, jnp.nan)

    @ck.forward
    def out_e2(self, e1_raw, e2_raw):
        """Return ``e2_raw`` when constraints are satisfied, otherwise NaN."""
        return jnp.where(self._check_valid(e1_raw, e2_raw), e2_raw, jnp.nan)


__all__ = ["EllipticityConstraint"]
