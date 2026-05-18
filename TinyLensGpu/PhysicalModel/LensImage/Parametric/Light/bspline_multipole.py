"""
B-spline radial basis with multipole angular light profiles.

This module provides a linear basis representation for two-dimensional galaxy
surface brightness.  Each ``BsplineMultipoleBasis`` instance contributes one column
to the design matrix used by ``ImageProbModel(use_linear=True)``: a cubic
B-spline in elliptical radius multiplied by a sine, cosine, or monopole angular
term.
"""

from typing import List, Optional

import caskade as ck
import jax.numpy as jnp
from jax import Array

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.utils import generate_radial_basis_knots
from TinyLensGpu.utils.geometry.transforms import ellipticity2phi_q, xy_transform


def bspline_bkpts(rbkpt, degree=3):
    """Generate full clamped knot vector from breakpoints.

    Parameters
    ----------
    rbkpt : array-like
        1D array of radial breakpoints (interior knots).
    degree : int
        B-spline degree (default 3 for cubic).

    Returns
    -------
    jnp.ndarray
        Full knot vector with ``degree`` boundary padding at both ends.
    """
    rbkpt = jnp.asarray(rbkpt)
    left_pad = jnp.repeat(rbkpt[0:1], degree)
    right_pad = jnp.repeat(rbkpt[-1:], degree)
    return jnp.concatenate([left_pad, rbkpt, right_pad])


def bspline_basis_k(r, full_knots, k, degree=3):
    """Evaluate the k-th B-spline basis function B_{k,degree}(r) in pure JAX.

    Uses unrolled Cox-de Boor recursion for arbitrary degree. Returns 0 for
    ``r`` outside ``[full_knots[k], full_knots[k+degree+1])`` except that the
    final knot is included for the last basis interval.

    Parameters
    ----------
    r : jnp.ndarray
        Radial coordinate values (scalar or array).
    full_knots : jnp.ndarray
        Full clamped knot vector from ``bspline_bkpts``.
    k : int
        Basis function index (0 <= k < len(full_knots) - degree - 1).
    degree : int, optional
        B-spline degree (default 3 for cubic).

    Returns
    -------
    jnp.ndarray
        ``B_{k,degree}(r)`` values, same shape as ``r``.
    """
    r = jnp.asarray(r)
    t = jnp.asarray(full_knots)
    last_knot = t[-1]

    # Initialize degree 0 basis functions
    b = []
    for i in range(k, k + degree + 1):
        in_interval = (t[i] <= r) & (r < t[i + 1])
        at_final_edge = (r == last_knot) & (t[i + 1] == last_knot)
        b.append(jnp.where(in_interval | at_final_edge, 1.0, 0.0))

    # Recursively build higher degree basis functions
    for p in range(1, degree + 1):
        next_b = []
        for i in range(k, k + degree - p + 1):
            denom1 = t[i + p] - t[i]
            denom2 = t[i + p + 1] - t[i + 1]

            safe1 = jnp.where(denom1 == 0, 1.0, denom1)
            safe2 = jnp.where(denom2 == 0, 1.0, denom2)

            term1 = jnp.where(denom1 == 0, 0.0, (r - t[i]) / safe1 * b[i - k])
            term2 = jnp.where(denom2 == 0, 0.0, (t[i + p + 1] - r) / safe2 * b[i - k + 1])

            next_b.append(term1 + term2)
        b = next_b

    return b[0]


class BsplineMultipoleBasis(ck.Module):
    """Single B-spline radial basis × multipole angular basis function.

    One instance is one column in the linear design matrix.  The amplitude
    ``amp`` is solved analytically by ``ImageProbModel(use_linear=True)``.

    Parameters
    ----------
    k : int
        Radial B-spline basis index (0 <= k < n_radial_bases).
    m : int
        Multipole identifier: 0→monopole (1), -N→sin(|N|*theta),
        +N→cos(N*theta).
    rbkpt : array-like
        Radial breakpoints shared across components.
    degree : int, optional
        B-spline degree (default 3, frozen).
    center_x, center_y : ParamU or float, optional
        Shared center coordinates.
    e1, e2 : ParamU or float, optional
        Shared ellipticity components.
    """

    def __init__(
        self,
        k: int,
        m: int,
        rbkpt,
        degree: int = 3,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        e1: Optional[float] = None,
        e2: Optional[float] = None,
    ) -> None:
        """Initialize one B-spline multipole basis component."""
        super().__init__()
        object.__setattr__(self, '_k', k)
        object.__setattr__(self, '_m', m)
        object.__setattr__(self, '_rbkpt', jnp.asarray(rbkpt))
        object.__setattr__(self, '_degree', degree)
        object.__setattr__(self, '_full_knots', bspline_bkpts(self._rbkpt, self._degree))

        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", 0.0 if center_x is None else center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", 0.0 if center_y is None else center_y)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", 0.0 if e1 is None else e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", 0.0 if e2 is None else e2)

        # amp=1.0 returns the raw column before the linear solver overwrites it.
        self.amp = ParamU("amp", 1.0)

    @ck.forward
    def light(
        self,
        x: Array,
        y: Array,
        amp: Optional[Array] = None,
        center_x: Optional[Array] = None,
        center_y: Optional[Array] = None,
        e1: Optional[Array] = None,
        e2: Optional[Array] = None,
    ) -> Array:
        """Evaluate the weighted basis function at image-plane coordinates.

        Parameters
        ----------
        x, y : Array
            Coordinates where the light profile is evaluated, in arcseconds.
        amp : Array, optional
            Linear amplitude injected by caskade.
        center_x, center_y : Array, optional
            Center coordinates injected by caskade.
        e1, e2 : Array, optional
            Ellipticity components injected by caskade.

        Returns
        -------
        Array
            ``amp * B_k(r) * f_m(theta)`` evaluated at ``(x, y)``.
        """
        amp = jnp.asarray(amp)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)

        phi_G, q = ellipticity2phi_q(e1, e2)
        xt1, xt2 = xy_transform(x, y, center_x, center_y, phi_G)
        r = jnp.sqrt(xt1**2 * q + xt2**2 / q)

        # When e1 ≈ e2 ≈ 0 (circular), ellipticity2phi_q introduces a
        # numerical jitter that rotates theta by ~π/8.  Compute theta
        # directly from (dx, dy) in that regime so that angular multipole
        # terms evaluate at the correct geometric angle.
        dx = x - center_x
        dy = y - center_y
        c = jnp.sqrt(e1**2 + e2**2)
        theta = jnp.where(
            c < 1e-10,
            jnp.arctan2(dy, dx),
            jnp.arctan2(xt2, xt1),
        )

        basis_r = bspline_basis_k(r, self._full_knots, self._k, self._degree)

        m = self._m
        if m == 0:
            fm = jnp.ones_like(theta)
        elif m < 0:
            fm = jnp.sin(jnp.abs(m) * theta)
        else:
            fm = jnp.cos(m * theta)

        return amp * basis_r * fm


def build_bspline_multipole_set(
    dpix: float,
    r_min: float = 0.01,
    r_max: float = 5.0,
    n_radial: int = 15,
    ntheta: Optional[List[int]] = None,
    degree: int = 3,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    e1: Optional[float] = None,
    e2: Optional[float] = None,
    mask: Optional[jnp.ndarray] = None,
) -> List[BsplineMultipoleBasis]:
    """Create all B-spline multipole basis components.

    Parameters
    ----------
    dpix : float
        Pixel scale in arcsec.
    r_min, r_max : float
        Radial range in arcseconds.
    n_radial : int
        Number of radial breakpoints (log-spaced).
    ntheta : list of int, optional
        Multipole orders. Default is ``[0, -2, 2]``.
    degree : int
        B-spline degree (default 3).
    center_x, center_y : ParamU or float, optional
        Shared center coordinates. If not ``ParamU``, new ``ParamU`` objects
        are created and shared across all components.
    e1, e2 : ParamU or float, optional
        Shared ellipticity. Same sharing behavior as ``center_x`` and
        ``center_y``.
    mask : jnp.ndarray, optional
        Boolean mask where True indicates pixels to avoid when distributing
        knots.

    Returns
    -------
    list of BsplineMultipoleBasis
        All basis components, ready to pass as lens_light to ``PhysicalModel``.
    """
    if ntheta is None:
        ntheta = [0, -2, 2]

    # Knots are static; use the float values of centers for distribution
    cx_val = float(center_x.value) if isinstance(center_x, ParamU) else (0.0 if center_x is None else float(center_x))
    cy_val = float(center_y.value) if isinstance(center_y, ParamU) else (0.0 if center_y is None else float(center_y))

    rbkpt = generate_radial_basis_knots(
        dpix=dpix,
        center_x=cx_val,
        center_y=cy_val,
        n_sigmas=n_radial,
        log_rmin=jnp.log10(r_min),
        log_rmax=jnp.log10(r_max),
        arc_mask=mask,
    )

    center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", 0.0 if center_x is None else center_x)
    center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", 0.0 if center_y is None else center_y)
    e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", 0.0 if e1 is None else e1)
    e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", 0.0 if e2 is None else e2)

    components: List[BsplineMultipoleBasis] = []
    n_bases = len(rbkpt) + degree - 1
    for k in range(n_bases):
        for m in ntheta:
            comp = BsplineMultipoleBasis(k, m, rbkpt, degree, center_x, center_y, e1, e2)
            object.__setattr__(comp, '__name__', f'bspline_{k}_{m}')
            components.append(comp)
    return components
