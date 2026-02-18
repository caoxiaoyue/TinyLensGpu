"""
Shapelet basis functions for galaxy surface brightness reconstruction.

Implements the Refregier (2003) shapelet formalism: a complete orthonormal
basis built from Hermite polynomials times a Gaussian envelope.  Each
``ShapeletBasisFunction`` contributes one column to the linear system solved
by ``ImageProbModel(use_linear=True, solver_type='normal')``.

Usage
-----
Build a set of basis functions and pass them as ``source_light``::

    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light.shapelet import (
        build_shapelet_set, build_shapelet_basis_matrix
    )

    basis = build_shapelet_set(n_max=8, beta=0.2, center_x=0.0, center_y=0.3)
    phys_model = PhysicalModel(lens_mass=[sie], source_light=basis, lens_light=[])
    prob_model = ImageProbModel(..., use_linear=True, solver_type='normal')
    image, X_vec = prob_model.forward_model(use_linear=True, return_intensity=True)
    # X_vec[i] is the solved amplitude for basis[i]
"""

from __future__ import annotations

import functools
import math
from typing import List, Optional

import caskade as ck
import jax
import jax.numpy as jnp
from jax import Array

from TinyLensGpu.Inference.param_u import ParamU


# ---------------------------------------------------------------------------
# Low-level 1-D primitives
# ---------------------------------------------------------------------------

def _hermite_e(n: int, x: Array) -> Array:
    """Probabilist's Hermite polynomial He_n(x) via three-term recurrence.

    He_0 = 1,  He_1 = x,  He_k = x * He_{k-1} - (k-1) * He_{k-2}.

    Parameters
    ----------
    n : int
        Polynomial order (static; loop is unrolled at JIT compile time).
    x : Array
        Evaluation points.

    Returns
    -------
    Array
        He_n(x), same shape as *x*.
    """
    if n == 0:
        return jnp.ones_like(x)
    if n == 1:
        return x
    He_prev = jnp.ones_like(x)
    He_curr = x
    for k in range(2, n + 1):
        He_next = x * He_curr - (k - 1) * He_prev
        He_prev = He_curr
        He_curr = He_next
    return He_curr


def _shapelet_1d(n: int, x: Array) -> Array:
    """Normalised 1-D shapelet basis function phi_n(x).

    phi_n(x) = [2^n * sqrt(pi) * n!]^{-1/2} * He_n(x) * exp(-x^2 / 2)

    Parameters
    ----------
    n : int
        Basis order (static).
    x : Array
        Scaled coordinate (x - center) / beta.

    Returns
    -------
    Array
        phi_n(x), same shape as *x*.
    """
    norm_sq = float(2**n) * math.sqrt(math.pi) * float(math.factorial(n))
    return _hermite_e(n, x) * jnp.exp(-x**2 / 2) / math.sqrt(norm_sq)


def _shapelet_1d_all(n_max: int, x: Array) -> Array:
    """Compute all 1-D shapelet basis functions phi_0 ... phi_{n_max} at once.

    Parameters
    ----------
    n_max : int
        Maximum order (static).
    x : Array
        Scaled coordinates, shape ``(n_pixels,)``.

    Returns
    -------
    Array
        Shape ``(n_max + 1, n_pixels)``.  Row *n* is phi_n(x).
    """
    exp_factor = jnp.exp(-x**2 / 2)

    # Build He_n values via Python loop (unrolled at JIT time since n_max is static)
    He = [jnp.ones_like(x), x]
    for k in range(2, n_max + 1):
        He.append(x * He[-1] - (k - 1) * He[-2])

    rows = []
    for n in range(n_max + 1):
        norm_sq = float(2**n) * math.sqrt(math.pi) * float(math.factorial(n))
        rows.append(He[n] * exp_factor / math.sqrt(norm_sq))

    return jnp.stack(rows, axis=0)  # (n_max+1, n_pixels)


# ---------------------------------------------------------------------------
# Fast vectorised basis-matrix builder (JIT-compiled)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(2,))
def build_shapelet_basis_matrix(
    x: Array,
    y: Array,
    n_max: int,
    beta: float,
    center_x: float,
    center_y: float,
) -> Array:
    """Build the full ``(n_pixels, n_basis)`` shapelet basis matrix.

    Ordering of columns matches ``build_shapelet_set``:
    (n_max, 0), (n_max-1, 1), ..., (0, n_max) for each total order,
    i.e. (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...

    Parameters
    ----------
    x, y : Array
        Flat coordinate arrays, shape ``(n_pixels,)``.
    n_max : int
        Maximum shapelet order (static for JIT).
    beta : float
        Shapelet scale radius in arcseconds.
    center_x, center_y : float
        Shapelet centre coordinates in arcseconds.

    Returns
    -------
    Array
        Shape ``(n_pixels, n_basis)`` where
        ``n_basis = (n_max + 1) * (n_max + 2) // 2``.
    """
    x_scaled = (x - center_x) / beta
    y_scaled = (y - center_y) / beta

    phi_x = _shapelet_1d_all(n_max, x_scaled)  # (n_max+1, n_pixels)
    phi_y = _shapelet_1d_all(n_max, y_scaled)  # (n_max+1, n_pixels)

    columns = []
    for n in range(n_max + 1):
        for n1 in range(n, -1, -1):
            n2 = n - n1
            columns.append(phi_x[n1] * phi_y[n2])  # (n_pixels,)

    return jnp.stack(columns, axis=-1)  # (n_pixels, n_basis)


# ---------------------------------------------------------------------------
# Caskade module: one basis function = one linear amplitude
# ---------------------------------------------------------------------------

class ShapeletBasisFunction(ck.Module):
    """Single 2-D shapelet basis function phi_{n1}(x/beta) * phi_{n2}(y/beta).

    Each instance contributes one column to the linear system.  The amplitude
    ``amp`` is the only caskade-tracked parameter; it is solved analytically
    by ``ImageProbModel(use_linear=True, solver_type='normal')``.

    Parameters
    ----------
    n1, n2 : int
        Shapelet orders along x and y.
    beta : float
        Scale radius in arcseconds.
    center_x, center_y : float
        Centre of the shapelet basis in arcseconds.
    """

    def __init__(
        self,
        n1: int,
        n2: int,
        beta: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> None:
        super().__init__()
        # Use object.__setattr__ to bypass caskade's __setattr__ interception
        # (same pattern as PhysicalModel._lens_mass_list).
        object.__setattr__(self, '_n1', n1)
        object.__setattr__(self, '_n2', n2)
        object.__setattr__(self, '_beta', float(beta))
        object.__setattr__(self, '_center_x', float(center_x))
        object.__setattr__(self, '_center_y', float(center_y))
        # amp=1.0 so the column returned by light() is the raw basis function;
        # the linear solver overwrites this with the optimal amplitude.
        self.amp = ParamU("amp", 1.0)

    @ck.forward
    def light(self, x: Array, y: Array, amp: Optional[Array] = None) -> Array:
        """Evaluate the weighted basis function at image-plane coordinates.

        Parameters
        ----------
        x, y : Array
            Coordinates (source-plane after deflection), arcseconds.
        amp : Array, optional
            Amplitude injected by caskade (defaults to ``self.amp.value``).

        Returns
        -------
        Array
            ``amp * phi_{n1}((x - cx) / beta) * phi_{n2}((y - cy) / beta)``.
        """
        amp = jnp.asarray(amp)
        x_scaled = (x - self._center_x) / self._beta
        y_scaled = (y - self._center_y) / self._beta
        return amp * _shapelet_1d(self._n1, x_scaled) * _shapelet_1d(self._n2, y_scaled)

    def __repr__(self) -> str:
        return (
            f"ShapeletBasisFunction(n1={self._n1}, n2={self._n2}, "
            f"beta={self._beta}, cx={self._center_x}, cy={self._center_y})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_shapelet_set(
    n_max: int,
    beta: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> List[ShapeletBasisFunction]:
    """Create all shapelet basis functions up to total order *n_max*.

    Ordering: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...
    Total number of basis functions: ``(n_max + 1) * (n_max + 2) // 2``.

    Parameters
    ----------
    n_max : int
        Maximum total order n1 + n2.
    beta : float
        Shapelet scale radius in arcseconds.
    center_x, center_y : float
        Centre of the shapelet basis in arcseconds.

    Returns
    -------
    list of ShapeletBasisFunction
        Ready to pass as ``source_light`` to ``PhysicalModel``.
    """
    basis: List[ShapeletBasisFunction] = []
    for n in range(n_max + 1):
        for n1 in range(n, -1, -1):
            n2 = n - n1
            basis.append(
                ShapeletBasisFunction(n1=n1, n2=n2, beta=beta, center_x=center_x, center_y=center_y)
            )
    return basis
