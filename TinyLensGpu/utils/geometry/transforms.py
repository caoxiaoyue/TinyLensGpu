"""
Geometric transformation utilities for lens modeling.

These functions were originally from the PhysicalModel.LensImage.Parametric.utils module
and are now centralized here for reuse across the codebase.
"""

from typing import Tuple
import jax.numpy as jnp
from jax import Array


def phi_q2_ellipticity(phi: Array, q: Array) -> Tuple[Array, Array]:
    """
    Convert orientation and axis ratio to ellipticity components.

    Parameters
    ----------
    phi : Array
        Position angle in radians.
    q : Array
        Minor-to-major axis ratio.

    Returns
    -------
    tuple[Array, Array]
        Ellipticity components ``(e1, e2)``.
    """
    e = (1. - q) / (1. + q)
    e1 = e * jnp.cos(2 * phi)
    e2 = e * jnp.sin(2 * phi)
    return e1, e2


def ellipticity2phi_q(e1: Array, e2: Array) -> Tuple[Array, Array]:
    """
    Convert ellipticity components to orientation and axis ratio.

    Parameters
    ----------
    e1 : Array
        Ellipticity component along x-axis.
    e2 : Array
        Ellipticity component at 45 degrees.

    Returns
    -------
    tuple[Array, Array]
        Position angle ``phi`` (radians) and axis ratio ``q``.
    """
    dtype = jnp.result_type(e1, e2, 1.0)
    e1, e2 = jnp.asarray(e1, dtype=dtype), jnp.asarray(e2, dtype=dtype)
    c2 = e1**2 + e2**2
    # The orientation is arbitrary at the circular point. Only that point
    # needs a substitute; replacing either zero component kills its derivative
    # along an otherwise perfectly regular axis.
    phi = jnp.arctan2(jnp.where(c2 == 0, 0., e2),
                      jnp.where(c2 == 0, 1., e1)) / 2.
    c = jnp.sqrt(jnp.maximum(c2, jnp.finfo(c2.dtype).tiny))
    c = jnp.minimum(c, 0.9999)
    q = (1. - c) / (1. + c)
    return phi, q


def _bounded_ellipticity(e1: Array, e2: Array) -> Tuple[Array, Array]:
    """Apply the same radial bound as ``ellipticity2phi_q`` to components."""
    dtype = jnp.result_type(e1, e2, 1.0)
    e1, e2 = jnp.asarray(e1, dtype=dtype), jnp.asarray(e2, dtype=dtype)
    c2 = e1**2 + e2**2
    c_safe = jnp.sqrt(jnp.maximum(c2, jnp.finfo(c2.dtype).tiny))
    scale = jnp.minimum(1., 0.9999 / c_safe)
    return e1 * scale, e2 * scale


def xy_transform(x: Array, y: Array, xc: Array, yc: Array, phi: Array) -> Tuple[Array, Array]:
    """
    Shift and rotate coordinates into a local lens frame.

    Parameters
    ----------
    x, y : Array
        Input coordinates in arcsec.
    xc, yc : Array
        Lens center in arcsec.
    phi : Array
        Clockwise rotation angle in radians.

    Returns
    -------
    tuple[Array, Array]
        Rotated coordinates ``(x_rot, y_rot)``.
    """
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    # Translate to center
    x_shift = x - xc
    y_shift = y - yc
    # Rotate clockwise
    x_rot = x_shift * cos_phi + y_shift * sin_phi
    y_rot = -x_shift * sin_phi + y_shift * cos_phi

    return x_rot, y_rot


def cart2polar(x: Array, y: Array) -> Tuple[Array, Array]:
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : Array
        x-coordinate.
    y : Array
        y-coordinate.

    Returns
    -------
    tuple[Array, Array]
        Radius ``r`` and angle ``phi`` (radians).
    """
    r = jnp.sqrt(x**2+y**2)
    phi = jnp.arctan2(y, x)
    return r, phi


def polar2cart(r: Array, phi: Array) -> Tuple[Array, Array]:
    """
    Convert polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : Array
        Radius.
    phi : Array
        Polar angle in radians.

    Returns
    -------
    tuple[Array, Array]
        Cartesian coordinates ``(x, y)``.
    """
    x = r*jnp.cos(phi)
    y = r*jnp.sin(phi)
    return x, y


def relocate_radii(x: Array, y: Array) -> Tuple[Array, Array, Array]:
    """
    Enforce a minimum radius to avoid singular behavior at the origin.

    Parameters
    ----------
    x : Array
        x-coordinate.
    y : Array
        y-coordinate.

    Returns
    -------
    tuple[Array, Array, Array]
        Stabilized coordinates ``(x_new, y_new)`` and radius ``r`` with
        ``r >= 1e-5``.
    """
    r, theta = cart2polar(x, y)
    r = jnp.where(r < 1e-5, 1e-5, r)
    x, y = polar2cart(r, theta)
    return x, y, r


def ellipse2circle_transform(x: Array, y: Array, e1: Array, e2: Array, 
                            center_x: Array, center_y: Array) -> Tuple[Array, Array]:
    """
    Transform elliptical isophotes into circularized coordinates.

    Parameters
    ----------
    x, y : Array
        Coordinates to transform.
    e1, e2 : Array
        Ellipticity components.
    center_x, center_y : Array
        Coordinate origin.

    Returns
    -------
    tuple[Array, Array]
        Circularized coordinates ``(x_circ, y_circ)``. Their orientation is
        the symmetric square-root convention; their radius matches the
        rotated principal-axis convention.
    """
    # Symmetric square root of the elliptical metric. Its squared norm is
    # q*x_rot**2 + y_rot**2/q, without an undefined position angle at e=0.
    e1, e2 = _bounded_ellipticity(e1, e2)
    dx, dy = x - center_x, y - center_y
    normalizer = jnp.sqrt(1. - e1**2 - e2**2)
    return ((1. - e1) * dx - e2 * dy) / normalizer, \
           ((1. + e1) * dy - e2 * dx) / normalizer


def q2e(q: Array) -> Array:
    """
    Convert axis ratio to ellipticity magnitude.

    Parameters
    ----------
    q : Array
        Axis ratio.

    Returns
    -------
    Array
        Ellipticity magnitude ``e = |1-q^2|/(1+q^2)``.
    """
    e = jnp.abs(1 - q**2) / (1 + q**2)
    return e


def transform_e1e2_square_average(x: Array, y: Array, e1: Array, e2: Array, 
                                 center_x: Array, center_y: Array) -> Tuple[Array, Array]:
    """
    Apply square-averaged ellipticity coordinate transform.

    Parameters
    ----------
    x, y : Array
        Input coordinates.
    e1, e2 : Array
        Ellipticity components.
    center_x, center_y : Array
        Coordinate origin.

    Returns
    -------
    tuple[Array, Array]
        Transformed coordinates in the square-averaged convention, using the
        symmetric square root of its metric.
    """
    # The symmetric square root gives the same radius as the rotated
    # square-average metric and stays differentiable at circularity.
    e1, e2 = _bounded_ellipticity(e1, e2)
    dx, dy = x - center_x, y - center_y
    normalizer = jnp.sqrt(1. + e1**2 + e2**2)
    return ((1. - e1) * dx - e2 * dy) / normalizer, \
           ((1. + e1) * dy - e2 * dx) / normalizer


__all__ = [
    'phi_q2_ellipticity',
    'ellipticity2phi_q',
    'xy_transform',
    'cart2polar',
    'polar2cart',
    'relocate_radii',
    'ellipse2circle_transform',
    'q2e',
    'transform_e1e2_square_average'
]
