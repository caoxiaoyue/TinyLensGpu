"""
Truncated Navarro-Frenk-White (TNFW) mass profile with elliptical potential.

This module implements the TNFW mass distribution profile using the caskade
framework for modular, composable gravitational lensing models.
References:
    - Baltz et al. 2009: https://arxiv.org/abs/0705.0682
    - lenstronomy implementation: https://github.com/lenstronomy/lenstronomy
"""

from typing import Optional, Tuple, Union
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.utils.geometry import ellipticity2phi_q
from TinyLensGpu.Inference.param_u import ParamU


def _L(x: Array, tau: Array, s: float = 0.001) -> Array:
    """
    Evaluate TNFW auxiliary logarithmic term.

    Parameters
    ----------
    x : Array
        Dimensionless radius ``R/Rs``.
    tau : Array
        Dimensionless truncation ratio ``r_trunc/Rs``.
    s : float, optional
        Lower clipping threshold for numerical stability.

    Returns
    -------
    Array
        Auxiliary function ``L(x, tau)`` used in TNFW analytic expressions.
    """
    x = jnp.maximum(x, s)
    return jnp.log(x * (tau + jnp.sqrt(tau**2 + x**2))**-1)


def _F(x: Array, s: float = 0.001) -> Array:
    """
    Evaluate NFW projection helper function ``F(x)``.

    Parameters
    ----------
    x : Array
        Dimensionless radius ``R/Rs``.
    s : float, optional
        Lower clipping threshold for stability.

    Returns
    -------
    Array
        Piecewise analytic value of ``F(x)``.
    """
    x = jnp.maximum(x, s)
    
    def f_less_1(val):
        """Branch for ``x < 1``."""
        return (1 - val**2)**-0.5 * jnp.arctanh((1 - val**2)**0.5)
    
    def f_more_1(val):
        """Branch for ``x > 1``."""
        return (val**2 - 1)**-0.5 * jnp.arctan((val**2 - 1)**0.5)
    
    def f_equal_1(val):
        """Branch for ``x == 1`` limit value."""
        return jnp.ones_like(val)

    # Use where for element-wise conditional logic
    res = jnp.where(x < 1, f_less_1(x), f_more_1(x))
    res = jnp.where(x == 1, f_equal_1(x), res)
    return res


def _tnfw_F_analytic(x: Array, tau: Array, s: float = 0.001) -> Array:
    """
    Evaluate analytic TNFW surface-density term.

    Parameters
    ----------
    x : Array
        Dimensionless radius ``R/Rs``.
    tau : Array
        Dimensionless truncation ratio.
    s : float, optional
        Lower clipping threshold.

    Returns
    -------
    Array
        Analytic projected term used by TNFW deflection expressions.
    """
    t2 = tau**2
    x = jnp.maximum(x, s)
    _F_val = _F(x, s)
    
    a = t2 * (t2 + 1)**-2
    
    # Calculate b
    b_neq_1 = (t2 + 1) * (x**2 - 1)**-1 * (1 - _F_val)
    b_eq_1 = (t2 + 1) * 1.0 / 3
    b = jnp.where(x == 1, b_eq_1, b_neq_1)
    
    c = 2 * _F_val
    d = -jnp.pi * (t2 + x**2)**-0.5
    e = (t2 - 1) * (tau * (t2 + x**2)**0.5)**-1 * _L(x, tau, s)
    
    return a * (b + c + d + e)


def _tnfw_g(x: Array, tau: Array, s: float = 0.001) -> Array:
    """
    Evaluate TNFW deflection helper function ``g(x, tau)``.

    Parameters
    ----------
    x : Array
        Dimensionless radius ``R/Rs``.
    tau : Array
        Dimensionless truncation ratio.
    s : float, optional
        Lower clipping threshold.

    Returns
    -------
    Array
        Helper function used to compute TNFW deflection amplitude.
    """
    x = jnp.maximum(x, s)
    F_x = _F(x, s)
    L_x = _L(x, tau, s)
    
    term1 = (tau**2 + 1 + 2 * (x**2 - 1)) * F_x
    term2 = tau * jnp.pi
    term3 = (tau**2 - 1) * jnp.log(tau)
    term4 = jnp.sqrt(tau**2 + x**2) * (-jnp.pi + L_x * (tau**2 - 1) * tau**-1)
    
    return tau**2 * (tau**2 + 1)**-2 * (term1 + term2 + term3 + term4)


def alpha2rho0(alpha_Rs: Array, Rs: Array) -> Array:
    """
    Convert deflection normalization to TNFW density normalization.

    Parameters
    ----------
    alpha_Rs : Array
        Deflection at scale radius ``Rs``.
    Rs : Array
        TNFW scale radius.

    Returns
    -------
    Array
        Characteristic density normalization ``rho0``.
    """
    return alpha_Rs / (4.0 * Rs**2 * (1.0 + jnp.log(0.5)))


class TNFWSpherical(ck.Module):
    """
    Spherical Truncated NFW profile.
    
    Parameters
    ----------
    Rs : float, optional
        Scale radius in arcseconds
    alpha_Rs : float, optional
        Deflection angle at Rs in arcseconds
    r_trunc : float, optional
        Truncation radius in arcseconds
    center_x : float, optional
        Center x-coordinate in arcseconds
    center_y : float, optional
        Center y-coordinate in arcseconds
    """

    def __init__(self, Rs: Optional[float] = None, alpha_Rs: Optional[float] = None,
                 r_trunc: Optional[float] = None, center_x: Optional[float] = None,
                 center_y: Optional[float] = None) -> None:
        """
        Initialize spherical TNFW mass profile.

        Parameters
        ----------
        Rs, alpha_Rs, r_trunc : float, optional
            Scale radius, deflection normalization, and truncation radius.
        center_x, center_y : float, optional
            Lens center coordinates.
        """
        super().__init__()
        
        self.Rs = Rs if isinstance(Rs, ParamU) else ParamU("Rs", Rs)
        self.alpha_Rs = alpha_Rs if isinstance(alpha_Rs, ParamU) else ParamU("alpha_Rs", alpha_Rs)
        self.r_trunc = r_trunc if isinstance(r_trunc, ParamU) else ParamU("r_trunc", r_trunc)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)
        
        self._s = 0.001

    @ck.forward
    def deriv(self, x: Array, y: Array, Rs: Optional[Array] = None,
              alpha_Rs: Optional[Array] = None, r_trunc: Optional[Array] = None,
              center_x: Optional[Array] = None, center_y: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Calculate deflection angles for Spherical TNFW profile.
        
        Returns
        -------
        alpha_x : array_like
            Deflection angle in x-direction
        alpha_y : array_like
            Deflection angle in y-direction
        """
        # Ensure parameters are JAX arrays
        Rs = jnp.asarray(Rs)
        alpha_Rs = jnp.asarray(alpha_Rs)
        r_trunc = jnp.asarray(r_trunc)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        
        # Stability for Rs
        Rs = jnp.maximum(Rs, 1e-7)

        # Coordinate shift
        x_ = x - center_x
        y_ = y - center_y
        
        R_ = jnp.sqrt(x_**2 + y_**2)
        
        # Calculate rho0
        rho0_input = alpha2rho0(alpha_Rs, Rs)
        
        # Calculate spherical TNFW deflection
        R_safe = jnp.maximum(R_, self._s * Rs)
        x_scaled = R_safe / Rs
        x_scaled = jnp.maximum(x_scaled, self._s)
        tau = r_trunc / Rs
        
        gx = _tnfw_g(x_scaled, tau, self._s)
        a = 4 * rho0_input * Rs * gx / x_scaled**2
        
        f_x = a * x_
        f_y = a * y_
        
        return f_x, f_y


class TNFWEllipsePotential(ck.Module):
    """
    Truncated NFW profile with elliptical potential.
    
    Parameters
    ----------
    Rs : float, optional
        Scale radius in arcseconds
    alpha_Rs : float, optional
        Deflection angle at Rs in arcseconds
    r_trunc : float, optional
        Truncation radius in arcseconds
    e1 : float, optional
        Ellipticity component 1
    e2 : float, optional
        Ellipticity component 2
    center_x : float, optional
        Center x-coordinate in arcseconds
    center_y : float, optional
        Center y-coordinate in arcseconds
    """

    def __init__(self, Rs: Optional[float] = None, alpha_Rs: Optional[float] = None,
                 r_trunc: Optional[float] = None, e1: Optional[float] = None,
                 e2: Optional[float] = None, center_x: Optional[float] = None,
                 center_y: Optional[float] = None) -> None:
        """
        Initialize TNFW profile with elliptical potential approximation.

        Parameters
        ----------
        Rs, alpha_Rs, r_trunc : float, optional
            Scale radius, deflection normalization, and truncation radius.
        e1, e2 : float, optional
            Ellipticity components of the potential.
        center_x, center_y : float, optional
            Lens center coordinates.
        """
        super().__init__()
        
        self.Rs = Rs if isinstance(Rs, ParamU) else ParamU("Rs", Rs)
        self.alpha_Rs = alpha_Rs if isinstance(alpha_Rs, ParamU) else ParamU("alpha_Rs", alpha_Rs)
        self.r_trunc = r_trunc if isinstance(r_trunc, ParamU) else ParamU("r_trunc", r_trunc)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)
        
        self._s = 0.001

    @ck.forward
    def deriv(self, x: Array, y: Array, Rs: Optional[Array] = None,
              alpha_Rs: Optional[Array] = None, r_trunc: Optional[Array] = None,
              e1: Optional[Array] = None, e2: Optional[Array] = None,
              center_x: Optional[Array] = None, center_y: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Calculate deflection angles for TNFW elliptical potential profile.
        
        Returns
        -------
        alpha_x : array_like
            Deflection angle in x-direction
        alpha_y : array_like
            Deflection angle in y-direction
        """
        # Ensure parameters are JAX arrays
        Rs = jnp.asarray(Rs)
        alpha_Rs = jnp.asarray(alpha_Rs)
        r_trunc = jnp.asarray(r_trunc)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        
        # Stability for Rs
        Rs = jnp.maximum(Rs, 1e-7)

        # Coordinate transformation (square average)
        phi_G, q = ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = jnp.cos(phi_G)
        sin_phi = jnp.sin(phi_G)
        
        # e = abs(1 - q**2) / (1 + q**2)
        # Note: ellipticity2phi_q returns q = (1-c)/(1+c), so q is axis ratio minor/major
        # q2e logic: e = (1-q^2)/(1+q^2)
        e = jnp.abs(1 - q**2) / (1 + q**2)
        
        x_ = (cos_phi * x_shift + sin_phi * y_shift) * jnp.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) * jnp.sqrt(1 + e)
        
        R_ = jnp.sqrt(x_**2 + y_**2)
        
        # Calculate rho0
        rho0_input = alpha2rho0(alpha_Rs, Rs)
        
        # Calculate spherical TNFW deflection
        # tnfw_alpha(R, Rs, rho0, r_trunc, ax_x, ax_y)
        # Here ax_x = x_, ax_y = y_
        
        R_safe = jnp.maximum(R_, self._s * Rs)
        x_scaled = R_safe / Rs
        x_scaled = jnp.maximum(x_scaled, self._s)
        tau = r_trunc / Rs
        
        gx = _tnfw_g(x_scaled, tau, self._s)
        a = 4 * rho0_input * Rs * gx / x_scaled**2
        
        f_x_prim = a * x_
        f_y_prim = a * y_
        
        # Transform back
        f_x_prim *= jnp.sqrt(1 - e)
        f_y_prim *= jnp.sqrt(1 + e)
        
        f_x = cos_phi * f_x_prim - sin_phi * f_y_prim
        f_y = sin_phi * f_x_prim + cos_phi * f_y_prim
        
        return f_x, f_y
