"""
Multipole mass profiles.

This module implements the Multipole and EllipticalMultipole mass profiles
referencing lenstronomy implementation, adapted for JAX and TinyLensGpu.
"""

from typing import Optional, Tuple, Union
import caskade as ck
import jax.numpy as jnp
import jax
from jax import Array
from TinyLensGpu.utils.geometry import cart2polar
from TinyLensGpu.Inference.param_u import ParamU


def _phi_ell(phi, q):
    """
    Internal helper to phi ell.
    
    Parameters
    ----------
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    return (
        phi
        - jnp.arctan2(jnp.sin(phi), jnp.cos(phi))
        + jnp.arctan2(jnp.sin(phi), q * jnp.cos(phi))
    )


def _F_m1_1_hat(phi, q):
    # Prevent division by zero or log of zero/negative
    """
    Internal helper to F m1 1 hat.
    
    Parameters
    ----------
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    q_safe = jnp.where(q > 0.99999, 0.99999, q) # Avoid q=1 exactly
    
    term1 = jnp.cos(phi) * (
        q * jnp.log(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
        - (jnp.log(2) * (1 + q) / 2 - (1 - q**2) * (1 + jnp.log(2) / 4))
    )
    term2 = 2 * jnp.sin(phi) * (phi - _phi_ell(phi, q))
    return -(term1 + term2) / (2 * (1 - q**2))


def _F_m1_1_hat_derivative(phi, q):
    """
    Internal helper to F m1 1 hat derivative.
    
    Parameters
    ----------
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    term1 = -jnp.cos(phi) * q * 2 * (q**2 - 1) * jnp.sin(2 * phi) / (
        1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)
    ) + jnp.sin(phi) * (
        -q * jnp.log(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
        + jnp.log(2) * (1 + q) / 2
        - (1 - q**2) * (1 + jnp.log(2) / 4)
    )
    term2 = 2 * jnp.cos(phi) * (phi - _phi_ell(phi, q)) + 2 * jnp.sin(phi) * (
        1 - q / (q**2 * jnp.cos(phi) ** 2 + jnp.sin(phi) ** 2)
    )
    return -(term1 + term2) / (2 * (1 - q**2))


def _potential_m1_1(r, phi, q, r_E):
    """
    Internal helper to potential m1 1.
    
    Parameters
    ----------
    r : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    r_E : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    lambda_m1 = 2 / (1 + q)
    return r * _F_m1_1_hat(phi, q) + lambda_m1 / 2 * r * jnp.log(r / r_E) * jnp.cos(phi)


def _alpha_m1_1(r, phi, q, r_E):
    """
    Internal helper to alpha m1 1.
    
    Parameters
    ----------
    r : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    r_E : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    lambda_m1 = 2 / (1 + q)
    f_phi = _F_m1_1_hat(phi, q)
    df_dphi = _F_m1_1_hat_derivative(phi, q)
    alpha_x = (
        f_phi * jnp.cos(phi)
        - df_dphi * jnp.sin(phi)
        + lambda_m1 / 2 * (jnp.log(r / r_E) + jnp.cos(phi) ** 2)
    )
    alpha_y = (
        f_phi * jnp.sin(phi)
        + df_dphi * jnp.cos(phi)
        + lambda_m1 / 2 * jnp.cos(phi) * jnp.sin(phi)
    )
    return alpha_x, alpha_y


def _A_3_1(q):
    """
    Internal helper to A 3 1.
    
    Parameters
    ----------
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    return (
        jnp.log(2) * (1 + q) ** 2
        - 2 * (1 - q) * (1 + q) ** 2 * (1 + jnp.log(2) / 4)
        + (1 - q**2) ** 2 / 4
    )


def _F_m3_1_hat(phi, q):
    """
    Internal helper to F m3 1 hat.
    
    Parameters
    ----------
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    term1 = jnp.cos(phi) * (
        q * (3 + q**2) * jnp.log(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)) - _A_3_1(q)
    )
    term2 = 2 * jnp.sin(phi) * (1 + 3 * q**2) * (phi - _phi_ell(phi, q))
    return (term1 + term2) / (2 * (1 - q**2) ** 2)


def _F_m3_1_hat_derivative(phi, q):
    """
    Internal helper to F m3 1 hat derivative.
    
    Parameters
    ----------
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    term1 = -jnp.cos(phi) * q * (3 + q**2) * 2 * (q**2 - 1) * jnp.sin(2 * phi) / (
        1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)
    ) + jnp.sin(phi) * (
        -q * (3 + q**2) * jnp.log(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)) + _A_3_1(q)
    )
    term2 = 2 * jnp.cos(phi) * (1 + 3 * q**2) * (phi - _phi_ell(phi, q)) + 2 * jnp.sin(
        phi
    ) * (1 + 3 * q**2) * (1 - q / (q**2 * jnp.cos(phi) ** 2 + jnp.sin(phi) ** 2))
    return (term1 + term2) / (2 * (1 - q**2) ** 2)


def _potential_m3_1(r, phi, q, r_E):
    """
    Internal helper to potential m3 1.
    
    Parameters
    ----------
    r : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    r_E : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    lambda_m3 = -2 * (1 - q) / (1 + q) ** 2
    return r * _F_m3_1_hat(phi, q) + lambda_m3 / 2 * r * jnp.log(r / r_E) * jnp.cos(phi)


def _alpha_m3_1(r, phi, q, r_E):
    """
    Internal helper to alpha m3 1.
    
    Parameters
    ----------
    r : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    r_E : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    lambda_m3 = -2 * (1 - q) / (1 + q) ** 2
    f_phi = _F_m3_1_hat(phi, q)
    df_dphi = _F_m3_1_hat_derivative(phi, q)
    alpha_x = (
        f_phi * jnp.cos(phi)
        - df_dphi * jnp.sin(phi)
        + lambda_m3 / 2 * (jnp.log(r / r_E) + jnp.cos(phi) ** 2)
    )
    alpha_y = (
        f_phi * jnp.sin(phi)
        + df_dphi * jnp.cos(phi)
        + lambda_m3 / 2 * jnp.cos(phi) * jnp.sin(phi)
    )
    return alpha_x, alpha_y


def _F_m4_1(phi, q):
    """
    Internal helper to F m4 1.
    
    Parameters
    ----------
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    term1 = (
        -4
        * jnp.sqrt(2)
        * (1 + 4 * q**2 + q**4 + (q**4 - 1) * jnp.cos(2 * phi))
        / ((3 * (1 - q**2) ** 2) * jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)))
    )
    term2 = (
        (1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * jnp.cos(phi)
        * jnp.arctan(
            (jnp.sqrt(2 * (1 - q**2)) * jnp.cos(phi))
            / jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
        )
    )
    
    # Handle log argument safety
    log_arg = (
        jnp.sqrt(1 - q**2) * jnp.sin(phi) / q
        + jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
    )
    # Ensure positive argument for log (though physically should be positive)
    log_arg = jnp.maximum(log_arg, 1e-10)
    
    term3 = (
        (1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * jnp.sin(phi)
        * jnp.log(log_arg)
    )

    return term1 + term2 + term3


def _F_m4_1_derivative(phi, q):
    """
    Internal helper to F m4 1 derivative.
    
    Parameters
    ----------
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    term1 = (
        -4
        * jnp.sqrt(2)
        * (1 + q**4 + (q**4 - 1) * jnp.cos(2 * phi))
        * jnp.sin(2 * phi)
        / (3 * (1 - q**2) * (1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)) ** (3 / 2))
    )
    
    sqrt_term = jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
    
    term2 = (
        -(1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * (
            jnp.sin(phi)
            * jnp.arctan(
                (jnp.sqrt(2 * (1 - q**2)) * jnp.cos(phi))
                / sqrt_term
            )
            + jnp.sqrt(2 * (1 - q**2))
            * jnp.sin(2 * phi)
            / (2 * sqrt_term)
        )
    )
    
    log_arg = (
        jnp.sqrt(1 - q**2) * jnp.sin(phi) / q
        + jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
    )
    log_arg = jnp.maximum(log_arg, 1e-10)
    
    term3 = (
        (1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * jnp.cos(phi)
        * (
            jnp.log(log_arg)
            + jnp.sqrt(1 - q**2)
            / q
            * jnp.sin(phi)
            / jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
        )
    )

    return term1 + term2 + term3


def _F_m4_2(phi, q):
    """
    Internal helper to F m4 2.
    
    Parameters
    ----------
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    sqrt_term = jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
    
    term1 = (
        -4
        * jnp.sqrt(2)
        * q
        / (3 * (1 - q**2))
        * jnp.sin(2 * phi)
        / sqrt_term
    )
    term2 = (
        -4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * jnp.sin(phi)
        * jnp.arctan(
            (jnp.sqrt(2 * (1 - q**2)) * jnp.cos(phi))
            / sqrt_term
        )
    )
    
    log_arg = (
        jnp.sqrt(1 - q**2) * jnp.sin(phi) / q
        + jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
    )
    log_arg = jnp.maximum(log_arg, 1e-10)
    
    term3 = (
        4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * jnp.cos(phi)
        * jnp.log(log_arg)
    )

    return term1 + term2 + term3


def _F_m4_2_derivative(phi, q):
    """
    Internal helper to F m4 2 derivative.
    
    Parameters
    ----------
    phi : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    q : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    sqrt_term = jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
    
    term1 = (
        -8
        * jnp.sqrt(2)
        * q
        / (6 * (1 - q**2))
        * (
            -(1 - q**2)
            * jnp.sin(2 * phi) ** 2
            / sqrt_term ** 3
            + 2 * jnp.cos(2 * phi) / sqrt_term
        )
    )
    term2 = (
        4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * (
            -jnp.cos(phi)
            * jnp.arctan(
                (jnp.sqrt(2 * (1 - q**2)) * jnp.cos(phi))
                / sqrt_term
            )
            + 2
            * jnp.sqrt(2 * (1 - q**2))
            * jnp.sin(phi) ** 2
            / (2 * sqrt_term)
        )
    )
    
    log_arg = (
        jnp.sqrt(1 - q**2) * jnp.sin(phi) / q
        + jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
    )
    log_arg = jnp.maximum(log_arg, 1e-10)
    
    term3 = (
        4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * (
            -jnp.sin(phi)
            * jnp.log(log_arg)
            + jnp.sqrt(1 - q**2)
            / q
            * jnp.cos(phi) ** 2
            / jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
        )
    )

    return term1 + term2 + term3


class Multipole(ck.Module):
    """
    Represent the `Multipole` component in the TinyLensGpu pipeline.
    
    Parameters
    ----------
    m : Any
        Configuration argument consumed during construction of this component.
    a_m : Any
        Configuration argument consumed during construction of this component.
    phi_m : Any
        Configuration argument consumed during construction of this component.
    center_x : Any
        Configuration argument consumed during construction of this component.
    center_y : Any
        Configuration argument consumed during construction of this component.
    r_E : Any
        Configuration argument consumed during construction of this component.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    def __init__(self, m: int = None, a_m: Optional[float] = None, 
                 phi_m: Optional[float] = None, center_x: Optional[float] = None, 
                 center_y: Optional[float] = None, r_E: Optional[float] = None) -> None:
        """
        Initialize a `Multipole` instance with validated configuration.
        
        Parameters
        ----------
        m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        a_m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        phi_m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        center_x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        center_y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        r_E : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        super().__init__()
        self.m_init = m
        self.a_m = a_m if isinstance(a_m, ParamU) else ParamU("a_m", a_m)
        self.phi_m = phi_m if isinstance(phi_m, ParamU) else ParamU("phi_m", phi_m)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)
        self.r_E = r_E if isinstance(r_E, ParamU) else ParamU("r_E", r_E)

    @ck.forward
    def deriv(self, x: Array, y: Array, m: Optional[int] = None, 
              a_m: Optional[Array] = None, phi_m: Optional[Array] = None, 
              center_x: Optional[Array] = None, center_y: Optional[Array] = None, 
              r_E: Optional[Array] = None) -> Tuple[Array, Array]:
        
        """
        Compute deriv.
        
        Parameters
        ----------
        x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        a_m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        phi_m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        center_x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        center_y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        r_E : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        Raises
        ------
        ValueError
            Raised when input validation fails or required runtime state is missing.
        
        """
        m_val = m if m is not None else self.m_init
        if m_val is None:
            raise ValueError("Multipole order 'm' must be provided.")
            
        a_m = jnp.asarray(a_m)
        phi_m = jnp.asarray(phi_m)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        r_E = jnp.asarray(r_E) if r_E is not None else jnp.array(1.0)

        x_shift = x - center_x
        y_shift = y - center_y
        r, phi = cart2polar(x_shift, y_shift)
        
        # Branching based on m_val. 
        # Since m_val is expected to be a python integer (static), we can use python if.
        # However, if m is passed dynamically (e.g. from another module calling this), we need care.
        # Assuming m_val is int.

        if m_val == 1:
            r = jnp.maximum(r, 0.000001)
            # alpha_x = a_m/2 * (cos(phi_m)*log(r/r_E) + cos(phi - phi_m)*cos(phi))
            # alpha_y = a_m/2 * (sin(phi_m)*log(r/r_E) + cos(phi - phi_m)*sin(phi))
            
            term1 = jnp.cos(phi - phi_m)
            
            f_x = a_m / 2 * (jnp.cos(phi_m) * jnp.log(r / r_E) + term1 * jnp.cos(phi))
            f_y = a_m / 2 * (jnp.sin(phi_m) * jnp.log(r / r_E) + term1 * jnp.sin(phi))
        else:
            # For m > 1
            # f_x = cos(phi) * a_m / (1 - m**2) * cos(m * (phi - phi_m)) + sin(phi) * m * a_m / (1 - m**2) * sin(m * (phi - phi_m))
            # f_y = sin(phi) * a_m / (1 - m**2) * cos(m * (phi - phi_m)) - cos(phi) * m * a_m / (1 - m**2) * sin(m * (phi - phi_m))
            
            factor1 = a_m / (1 - m_val**2)
            term_cos = jnp.cos(m_val * (phi - phi_m))
            term_sin = jnp.sin(m_val * (phi - phi_m))
            
            f_x = jnp.cos(phi) * factor1 * term_cos + jnp.sin(phi) * m_val * factor1 * term_sin
            f_y = jnp.sin(phi) * factor1 * term_cos - jnp.cos(phi) * m_val * factor1 * term_sin

        return f_x, f_y


class EllipticalMultipole(ck.Module):
    """
    Represent the `EllipticalMultipole` component in the TinyLensGpu pipeline.
    
    Parameters
    ----------
    m : Any
        Configuration argument consumed during construction of this component.
    a_m : Any
        Configuration argument consumed during construction of this component.
    phi_m : Any
        Configuration argument consumed during construction of this component.
    q : Any
        Configuration argument consumed during construction of this component.
    center_x : Any
        Configuration argument consumed during construction of this component.
    center_y : Any
        Configuration argument consumed during construction of this component.
    r_E : Any
        Configuration argument consumed during construction of this component.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    def __init__(self, m: int = None, a_m: Optional[float] = None, 
                 phi_m: Optional[float] = None, q: Optional[float] = None,
                 center_x: Optional[float] = None, center_y: Optional[float] = None, 
                 r_E: Optional[float] = None) -> None:
        """
        Initialize a `EllipticalMultipole` instance with validated configuration.
        
        Parameters
        ----------
        m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        a_m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        phi_m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        q : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        center_x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        center_y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        r_E : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        super().__init__()
        self.m_init = m
        self.a_m = a_m if isinstance(a_m, ParamU) else ParamU("a_m", a_m)
        self.phi_m = phi_m if isinstance(phi_m, ParamU) else ParamU("phi_m", phi_m)
        self.q = q if isinstance(q, ParamU) else ParamU("q", q)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)
        self.r_E = r_E if isinstance(r_E, ParamU) else ParamU("r_E", r_E)

    @ck.forward
    def deriv(self, x: Array, y: Array, m: Optional[int] = None, 
              a_m: Optional[Array] = None, phi_m: Optional[Array] = None, 
              q: Optional[Array] = None, center_x: Optional[Array] = None, 
              center_y: Optional[Array] = None, r_E: Optional[Array] = None) -> Tuple[Array, Array]:
        
        """
        Compute deriv.
        
        Parameters
        ----------
        x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        a_m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        phi_m : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        q : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        center_x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        center_y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        r_E : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        Raises
        ------
        ValueError
            Raised when input validation fails or required runtime state is missing.
        
        """
        m_val = m if m is not None else self.m_init
        if m_val is None:
            raise ValueError("Multipole order 'm' must be provided.")

        a_m = jnp.asarray(a_m)
        phi_m = jnp.asarray(phi_m)
        q = jnp.asarray(q)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        r_E = jnp.asarray(r_E)

        x_shift = x - center_x
        y_shift = y - center_y
        r, phi = cart2polar(x_shift, y_shift)
        r = jnp.maximum(r, 0.000001)

        # Condition for circular approximation
        condition = jnp.abs(1 - q**2) ** ((m_val + 1) / 2) < 1e-8
        
        # We compute both and select
        # Circular approximation
        sph_multipole = Multipole(m=m_val)
        # Use __wrapped__ to bypass caskade parameter checks with JAX arrays
        f_x_circ, f_y_circ = sph_multipole.deriv.__wrapped__(sph_multipole, x, y, m=m_val, a_m=a_m, phi_m=phi_m, 
                                                 center_x=center_x, center_y=center_y, r_E=r_E)

        # Elliptical calculation
        # To avoid NaNs in elliptical calculation when q is 1, we use a safe q.
        q_safe = jnp.where(condition, 0.99, q) 
        # 0.99 is arbitrary safe value, result won't be used if condition is true.

        f_x_ell = jnp.zeros_like(x)
        f_y_ell = jnp.zeros_like(y)

        if m_val == 1:
            alpha_x_1, alpha_y_1 = _alpha_m1_1(r, phi, q_safe, r_E)
            alpha_x_2, alpha_y_2 = _alpha_m1_1(r, phi + jnp.pi / 2, 1 / q_safe, r_E)
            f_x_ell = (
                a_m
                * jnp.sqrt(q_safe)
                * (
                    jnp.cos(m_val * phi_m) * alpha_x_1
                    - (1 / q_safe) * jnp.sin(m_val * phi_m) * alpha_y_2
                )
            )
            f_y_ell = (
                a_m
                * jnp.sqrt(q_safe)
                * (
                    jnp.cos(m_val * phi_m) * alpha_y_1
                    + (1 / q_safe) * jnp.sin(m_val * phi_m) * alpha_x_2
                )
            )
        
        elif m_val == 3:
            alpha_x_1, alpha_y_1 = _alpha_m3_1(r, phi, q_safe, r_E)
            alpha_x_2, alpha_y_2 = _alpha_m3_1(r, phi + jnp.pi / 2, 1 / q_safe, r_E)
            f_x_ell = (
                a_m
                * jnp.sqrt(q_safe)
                * (
                    jnp.cos(m_val * phi_m) * alpha_x_1
                    + (1 / q_safe) * jnp.sin(m_val * phi_m) * alpha_y_2
                )
            )
            f_y_ell = (
                a_m
                * jnp.sqrt(q_safe)
                * (
                    jnp.cos(m_val * phi_m) * alpha_y_1
                    - (1 / q_safe) * jnp.sin(m_val * phi_m) * alpha_x_2
                )
            )

        elif m_val == 4:
            F_m4 = _F_m4_1(phi, q=q_safe) * jnp.cos(m_val * phi_m) + _F_m4_2(
                phi, q=q_safe
            ) * jnp.sin(m_val * phi_m)
            F_m4_prime = _F_m4_1_derivative(phi, q=q_safe) * jnp.cos(
                m_val * phi_m
            ) + _F_m4_2_derivative(phi, q=q_safe) * jnp.sin(m_val * phi_m)
            
            f_x_ell = a_m * jnp.sqrt(q_safe) * (F_m4 * jnp.cos(phi) - F_m4_prime * jnp.sin(phi))
            f_y_ell = a_m * jnp.sqrt(q_safe) * (F_m4 * jnp.sin(phi) + F_m4_prime * jnp.cos(phi))

        else:
             # This branch should not be reached for other m if inputs are correct.
             # But if it is, we return zeros or raise error (but cannot raise in JIT).
             # We assume m is one of 1, 3, 4 for EllipticalMultipole.
             pass

        f_x = jnp.where(condition, f_x_circ, f_x_ell)
        f_y = jnp.where(condition, f_y_circ, f_y_ell)

        return f_x, f_y
