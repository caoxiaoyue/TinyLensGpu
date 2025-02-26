import jax.numpy as jnp


def phi_q2_ellipticity(phi, q):
    """
    Convert phi and q to ellipticity.
    Args:
        phi: rotation angle, in radians
        q: axis ratio
    Returns:
        e1: ellipticity in x direction
        e2: ellipticity in y direction
    """
    e1 = (1. - q) / (1. + q) * jnp.cos(2 * phi)
    e2 = (1. - q) / (1. + q) * jnp.sin(2 * phi)
    return e1, e2


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


def shear_polar2cartesian(phi, gamma):
    gamma1 = gamma*jnp.cos(2*phi)
    gamma2 = gamma*jnp.sin(2*phi)
    return gamma1, gamma2


def shear_cartesian2polar(gamma1, gamma2):
    phi = jnp.arctan2(gamma2, gamma1) / 2
    gamma = jnp.sqrt(gamma1 ** 2 + gamma2 ** 2)
    return phi, gamma

    
def cart2polar(x, y):
    r = jnp.sqrt(x**2+y**2)
    phi = jnp.arctan2(y, x)
    return r, phi


def polar2cart(r, phi):
    x = r*jnp.cos(phi)
    y = r*jnp.sin(phi)
    return x, y


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


def ellipse2circle_transform(x, y, e1, e2, center_x, center_y):
    phi_G, q = ellipticity2phi_q(e1, e2)
    xt1, xt2 = xy_transform(x, y, center_x, center_y, phi_G)
    return xt1 * jnp.sqrt(q), xt2 / jnp.sqrt(q)


def relocate_radii(x, y):
    """Handle numerical singularity at origin."""
    r, theta = cart2polar(x, y)
    r = jnp.where(r < 1e-5, 1e-5, r)
    x, y = polar2cart(r, theta)
    return x, y, r


def hyp2f1_series(a, b, c, z, max_terms=10):
    """Compute hypergeometric function using series expansion."""
    result = 1.0 + 0j
    term = 1.0 + 0j
    for n in range(1, max_terms):
        term = term * (a + n - 1) * (b + n - 1) * z / ((c + n - 1) * n)
        result = result + term
    return result