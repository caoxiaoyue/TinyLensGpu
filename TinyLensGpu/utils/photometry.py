"""Photometry utility functions for magnitude conversions."""

import jax.numpy as jnp


def mag2cps(magnitude, mag_zero_point=22.0):
    """
    Convert apparent magnitude to counts per second.

    Uses the standard astronomical magnitude formula:
        cps = 10^(-(magnitude - mag_zero_point) / 2.5)

    Parameters
    ----------
    magnitude : float or array_like
        Apparent magnitude.
    mag_zero_point : float, optional
        Magnitude zero point. Default is 22.0.

    Returns
    -------
    float or array_like
        Counts per second.
    """
    delta_M = magnitude - mag_zero_point
    return 10.0 ** (-delta_M / 2.5)


def cps2mag(cps, mag_zero_point=22.0):
    """
    Convert counts per second to apparent magnitude.

    Uses the standard astronomical magnitude formula:
        magnitude = -2.5 * log10(cps) + mag_zero_point

    Parameters
    ----------
    cps : float or array_like
        Counts per second.
    mag_zero_point : float, optional
        Magnitude zero point. Default is 22.0.

    Returns
    -------
    float or array_like
        Apparent magnitude.
    """
    delta_M = -2.5 * jnp.log10(cps)
    return delta_M + mag_zero_point
