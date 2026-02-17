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
    e1 = jnp.where(e1 == 0., 1e-12, e1)
    e2 = jnp.where(e2 == 0., 1e-12, e2)
    phi = jnp.arctan2(e2, e1) / 2.
    c = jnp.sqrt(e1**2 + e2**2)
    c = jnp.minimum(c, 0.9999)
    q = (1. - c) / (1. + c)
    return phi, q


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
        Circularized coordinates ``(x_circ, y_circ)``.
    """
    phi_G, q = ellipticity2phi_q(e1, e2)
    xt1, xt2 = xy_transform(x, y, center_x, center_y, phi_G)
    return xt1 * jnp.sqrt(q), xt2 / jnp.sqrt(q)


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
        Transformed coordinates in the square-averaged convention.
    """
    phi_g, q = ellipticity2phi_q(e1, e2)
    x_shift = x - center_x
    y_shift = y - center_y
    cos_phi = jnp.cos(phi_g)
    sin_phi = jnp.sin(phi_g)
    e = q2e(q)
    x_ = (cos_phi * x_shift + sin_phi * y_shift) * jnp.sqrt(1 - e)
    y_ = (-sin_phi * x_shift + cos_phi * y_shift) * jnp.sqrt(1 + e)
    return x_, y_


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
