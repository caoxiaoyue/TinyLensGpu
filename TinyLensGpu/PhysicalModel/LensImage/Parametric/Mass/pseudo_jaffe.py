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
    Spherical Pseudo-Jaffe mass profile.

    This profile is controlled by a normalization ``sigma0`` and two scale radii
    ``Ra`` and ``Rs`` with ``Ra < Rs``.

    Parameters
    ----------
    sigma0 : float, optional
        Surface-density normalization.
    Ra : float, optional
        Inner/core scale radius.
    Rs : float, optional
        Outer/truncation scale radius.
    center_x, center_y : float, optional
        Lens center coordinates.
    """

    def __init__(self, sigma0: Optional[float] = None, Ra: Optional[float] = None, 
                 Rs: Optional[float] = None, center_x: Optional[float] = None, 
                 center_y: Optional[float] = None) -> None:
        """
        Initialize spherical Pseudo-Jaffe profile.

        Parameters
        ----------
        sigma0, Ra, Rs, center_x, center_y : float, optional
            Model parameters converted to :class:`ParamU` when provided as scalars.
        """
        super().__init__()
        self.sigma0 = sigma0 if isinstance(sigma0, ParamU) else ParamU("sigma0", sigma0)
        self.Ra = Ra if isinstance(Ra, ParamU) else ParamU("Ra", Ra)
        self.Rs = Rs if isinstance(Rs, ParamU) else ParamU("Rs", Rs)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

    @staticmethod
    def _sort_ra_rs(Ra, Rs):
        """
        Enforce ordered radii and minimum separation.

        Parameters
        ----------
        Ra : array_like
            Candidate inner radius.
        Rs : array_like
            Candidate outer radius.

        Returns
        -------
        tuple[array_like, array_like]
            Ordered radii ``(Ra_new, Rs_new)`` with ``Ra_new < Rs_new``.
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
        Evaluate Pseudo-Jaffe radial helper term.

        Parameters
        ----------
        r_a : array_like
            Radius normalized by ``Ra``.
        r_s : array_like
            Radius normalized by ``Rs``.

        Returns
        -------
        array_like
            Auxiliary term entering the deflection amplitude.
        """
        return r_a / (1 + jnp.sqrt(1 + r_a**2)) - r_s / (1 + jnp.sqrt(1 + r_s**2))

    @ck.forward
    def deriv(self, x: Array, y: Array, sigma0: Optional[Array] = None, 
              Ra: Optional[Array] = None, Rs: Optional[Array] = None, 
              center_x: Optional[Array] = None, center_y: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Evaluate spherical Pseudo-Jaffe deflection field.

        Parameters
        ----------
        x, y : Array
            Image-plane coordinates.
        sigma0, Ra, Rs, center_x, center_y : Array, optional
            Runtime parameter values injected by caskade.

        Returns
        -------
        tuple[Array, Array]
            Deflection components ``(alpha_x, alpha_y)``.
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
