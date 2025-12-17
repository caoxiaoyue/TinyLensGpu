"""
Utility functions for caskade models.

These functions were originally from the Profile.util module but are now
copied here to make the Caskade implementation independent of legacy code.
"""

import jax.numpy as jnp


def ellipticity2phi_q(e1, e2):
    """
    Convert ellipticity to phi and q.
    Args:
        e1: ellipticity in x direction
        e2: ellipticity in y direction
    Returns:
        phi: rotation angle, in radians
        q: axis ratio
    """
    e1 = jnp.where(e1 == 0., 1e-4, e1)
    e2 = jnp.where(e2 == 0., 1e-4, e2)
    phi = jnp.arctan2(e2, e1) / 2.
    c = jnp.sqrt(e1**2 + e2**2)
    c = jnp.minimum(c, 0.9999)
    q = (1. - c) / (1. + c)
    return phi, q


def xy_transform(x, y, xc, yc, phi):
    """
    Transform coordinates to the lens frame.
    Args:
        x: x coordinate, in arcsec
        y: y coordinate, in arcsec
        xc: center x coordinate, in arcsec
        yc: center y coordinate, in arcsec
        phi: rotation angle, in radians
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


def cart2polar(x, y):
    r = jnp.sqrt(x**2+y**2)
    phi = jnp.arctan2(y, x)
    return r, phi


def polar2cart(r, phi):
    x = r*jnp.cos(phi)
    y = r*jnp.sin(phi)
    return x, y


def relocate_radii(x, y):
    """Handle numerical singularity at origin."""
    r, theta = cart2polar(x, y)
    r = jnp.where(r < 1e-5, 1e-5, r)
    x, y = polar2cart(r, theta)
    return x, y, r


def ellipse2circle_transform(x, y, e1, e2, center_x, center_y):
    phi_G, q = ellipticity2phi_q(e1, e2)
    xt1, xt2 = xy_transform(x, y, center_x, center_y, phi_G)
    return xt1 * jnp.sqrt(q), xt2 / jnp.sqrt(q)


__all__ = [
    'ellipticity2phi_q',
    'xy_transform',
    'relocate_radii',
    'ellipse2circle_transform'
]
