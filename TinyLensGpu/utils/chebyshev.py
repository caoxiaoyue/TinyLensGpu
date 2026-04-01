"""
Chebyshev polynomial utilities for wavelength-dependent model parameters.

This module implements Chebyshev polynomial evolution of parameters across
multiple bands, following the GALFITM method (Häußler et al. 2013, MNRAS 430, 330).

The Chebyshev polynomial of the first kind T_n(z) is used to describe how
parameters evolve with wavelength:
    p(λ) = Σ c_n * T_n(z(λ))

where z(λ) maps the wavelength range to [-1, +1].
"""

from typing import Sequence, Callable
import numpy as np
import jax.numpy as jnp
from jax import Array


def chebyshev_node(wavelength: float, lambda_min: float, lambda_max: float) -> float:
    """
    Map wavelength to Chebyshev domain z ∈ [-1, +1].
    
    Parameters
    ----------
    wavelength : float
        Wavelength in Angstroms (or any consistent unit)
    lambda_min : float
        Minimum wavelength in the band set
    lambda_max : float
        Maximum wavelength in the band set
        
    Returns
    -------
    z : float
        Normalized wavelength in [-1, +1]
    """
    return 2.0 * (wavelength - lambda_min) / (lambda_max - lambda_min) - 1.0


def chebyshev_polynomial(z: Array, order: int) -> Array:
    """
    Evaluate Chebyshev polynomial of the first kind T_n(z).
    
    Uses recurrence relation:
        T_0(z) = 1
        T_1(z) = z
        T_n(z) = 2*z*T_{n-1}(z) - T_{n-2}(z)
    
    Parameters
    ----------
    z : array_like
        Normalized wavelength in [-1, +1]
    order : int
        Polynomial order (0 = constant, 1 = linear, 2 = quadratic)
        
    Returns
    -------
    T_n : array_like
        Chebyshev polynomial evaluated at z
    """
    z = jnp.asarray(z)
    
    if order == 0:
        return jnp.ones_like(z)
    elif order == 1:
        return z
    elif order == 2:
        return 2.0 * z**2 - 1.0
    elif order == 3:
        return 4.0 * z**3 - 3.0 * z
    else:
        # General recurrence for higher orders
        T_prev2 = jnp.ones_like(z)  # T_0
        T_prev1 = z  # T_1
        
        for n in range(2, order + 1):
            T_current = 2.0 * z * T_prev1 - T_prev2
            T_prev2 = T_prev1
            T_prev1 = T_current
            
        return T_prev1


def evaluate_chebyshev_series(
    z: Array,
    coefficients: Sequence[float],
) -> Array:
    """
    Evaluate Chebyshev series: p(z) = Σ c_n * T_n(z)
    
    Parameters
    ----------
    z : array_like
        Normalized wavelength in [-1, +1]
    coefficients : sequence of float
        Chebyshev coefficients [c_0, c_1, c_2, ...]
        
    Returns
    -------
    value : array_like
        Parameter value at wavelength(s) z
    """
    z = jnp.asarray(z)
    result = jnp.zeros_like(z)
    
    for order, coeff in enumerate(coefficients):
        result = result + coeff * chebyshev_polynomial(z, order)
        
    return result


def create_chebyshev_parameter_function(
    wavelength: float,
    lambda_min: float,
    lambda_max: float,
    max_order: int = 2,
) -> Callable:
    """
    Create a function that evaluates a Chebyshev series at a fixed wavelength.
    
    This factory function returns a callable suitable for use with caskade's
    parameter linking mechanism. The returned function takes coefficients
    as input and returns the parameter value at the specified wavelength.
    
    Parameters
    ----------
    wavelength : float
        Wavelength at which to evaluate the parameter
    lambda_min : float
        Minimum wavelength in the band set
    lambda_max : float
        Maximum wavelength in the band set
    max_order : int, optional
        Maximum Chebyshev polynomial order (default: 2 for quadratic)
        
    Returns
    -------
    param_func : callable
        Function that takes Chebyshev coefficients and returns parameter value
        
    Example
    -------
    >>> # For second-order Chebyshev (3 coefficients: c0, c1, c2)
    >>> func = create_chebyshev_parameter_function(
    ...     wavelength=6231.0,  # r-band
    ...     lambda_min=3543.0,  # u-band
    ...     lambda_max=22010.0, # K-band
    ...     max_order=2
    ... )
    >>> # Use with caskade: param = lambda p: func([p.c0.value, p.c1.value, p.c2.value])
    """
    z = chebyshev_node(wavelength, lambda_min, lambda_max)
    
    def param_func(coefficients: Sequence[float]) -> Array:
        """Evaluate Chebyshev series with given coefficients."""
        return evaluate_chebyshev_series(z, coefficients)
    
    return param_func


def get_band_wavelengths(band_names: Sequence[str]) -> dict[str, float]:
    """
    Get effective wavelengths for common photometric bands.
    
    Values are in Angstroms, following the GALFITM paper conventions:
    u=3543, g=4770, r=6231, i=7625, z=9134, Y=10305, J=12483, H=16313, K=22010
    
    Parameters
    ----------
    band_names : sequence of str
        Band identifiers (e.g., 'g', 'r', 'i')
        
    Returns
    -------
    wavelengths : dict
        Dictionary mapping band names to effective wavelengths in Angstroms
    """
    # Standard SDSS/2MASS effective wavelengths in Angstroms
    standard_wavelengths = {
        'u': 3543.0,
        'g': 4770.0,
        'r': 6231.0,
        'i': 7625.0,
        'z': 9134.0,
        'Y': 10305.0,
        'J': 12483.0,
        'H': 16313.0,
        'K': 22010.0,
        # HST/WFC3 common bands
        'F606W': 5770.0,
        'F814W': 8050.0,
        'F125W': 12486.0,
        'F140W': 13930.0,
        'F160W': 15420.0,
        # Generic optical
        'B': 4450.0,
        'V': 5510.0,
        'R': 6580.0,
        'I': 8060.0,
    }
    
    return {band: standard_wavelengths.get(band, 5500.0) for band in band_names}


def compute_wavelength_range(
    band_wavelengths: Sequence[float],
    padding: float = 0.1
) -> tuple[float, float]:
    """
    Compute wavelength range with optional padding.
    
    Parameters
    ----------
    band_wavelengths : sequence of float
        Wavelengths of observed bands
    padding : float, optional
        Fractional padding to add to wavelength range (default: 0.1)
        
    Returns
    -------
    lambda_min, lambda_max : tuple of float
        Wavelength range for Chebyshev normalization
    """
    lambda_min = min(band_wavelengths)
    lambda_max = max(band_wavelengths)
    
    # Add padding to avoid edge effects
    range_width = lambda_max - lambda_min
    lambda_min = lambda_min - padding * range_width
    lambda_max = lambda_max + padding * range_width
    
    return lambda_min, lambda_max