"""
Chebyshev polynomial utilities for wavelength-dependent model parameters.

This module implements Chebyshev polynomial evolution of parameters across
multiple bands, following the GALFITM method (Häußler et al. 2013, MNRAS 430, 330).

The Chebyshev polynomial of the first kind T_n(z) is used to describe how
parameters evolve with wavelength:
    p(λ) = Σ c_n * T_n(z(λ))

where z(λ) maps the wavelength range to [-1, +1].
"""

from typing import Sequence
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
    if order == 1:
        return z

    T_prev2 = jnp.ones_like(z)  # T_0
    T_prev1 = z  # T_1
    for _ in range(2, order + 1):
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