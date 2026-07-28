"""
Critical line and caustic computation for gravitational lensing.

Provides utilities to compute critical lines (image-plane curves where
det(Jacobian) = 0) and caustics (their source-plane mappings) from
lens mass models.

Theory
------
The lens equation maps image-plane coordinates θ to source-plane
coordinates β:

    β(θ) = θ - α(θ)

where α(θ) is the total deflection from all mass components.

The Jacobian (magnification matrix) is:

    A = ∂β/∂θ = I - ∂α/∂θ

Critical lines are the curves in the image plane where det(A) = 0.
Caustics are the corresponding curves in the source plane, obtained by
mapping critical-line points through the lens equation.

Usage
-----
Both :class:`PhysicalModel` instances and plain lists of mass-profile
modules are supported::

    from TinyLensGpu.utils.lensing.critical_line import (
        compute_critical_and_caustics,
    )

    # With a PhysicalModel
    crit_lines, caustics = compute_critical_and_caustics(phys_model)

    # With a plain list of mass profiles
    crit_lines, caustics = compute_critical_and_caustics([sie, shear])
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from caskade import Module as ckModule
from matplotlib.path import Path

from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_lens_mass_list(
    source: Union[PhysicalModel, List[ckModule]],
) -> List[ckModule]:
    """Accept either a PhysicalModel or a plain list of mass modules."""
    if isinstance(source, PhysicalModel):
        return source.lens_mass
    if isinstance(source, list):
        return list(source)
    raise TypeError(
        f"Expected PhysicalModel or list of mass modules, got {type(source)}."
    )


def _build_deflection_fn(lens_mass: List[ckModule]):
    """Build a pure JAX-callable deflection function ``(x, y) -> (beta_x, beta_y)``.

    The returned function sums the ``deriv`` contributions from every mass
    profile in *lens_mass*, bypassing caskade's ``@ck.forward`` wrapper so
    that ``jax.jacfwd`` can trace through it.

    Concrete parameter values are snapshotted once when this function is
    called; later changes to the model parameters are not reflected.
    """

    # Pre-extract concrete parameter values and the raw (unwrapped) deriv
    # function for each mass model.  This is necessary because jax.jacfwd
    # cannot trace through caskade's @ck.forward parameter resolution.
    #
    # The mapping from caskade Param to function argument uses the *module
    # attribute name* (e.g. ``self.e1`` → kwarg ``e1``), NOT the Param's
    # ``.name`` (which may differ, e.g. ``"e1_mass"``).
    import inspect

    mass_configs: list[tuple[ckModule, callable, dict[str, jnp.ndarray]]] = []
    for m in lens_mass:
        deriv_fn = m.deriv.__wrapped__  # raw JAX function (no @ck.forward)

        # Get the function's parameter names (skip self, x, y)
        sig = inspect.signature(deriv_fn)
        func_arg_names = list(sig.parameters.keys())[3:]

        # Build concrete kwargs by looking up each arg name on the module
        param_kwargs: dict[str, jnp.ndarray] = {}
        for arg_name in func_arg_names:
            param_obj = getattr(m, arg_name, None)
            if param_obj is not None and hasattr(param_obj, "value"):
                param_kwargs[arg_name] = jnp.asarray(param_obj.value)

        mass_configs.append((m, deriv_fn, param_kwargs))

    def _deflection(x: jnp.ndarray, y: jnp.ndarray):
        beta_x, beta_y = x, y
        for model, deriv_fn, kwargs in mass_configs:
            ax, ay = deriv_fn(model, x, y, **kwargs)
            beta_x = beta_x - ax
            beta_y = beta_y - ay
        return beta_x, beta_y

    return _deflection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_jacobian_det(
    lens_mass: Union[PhysicalModel, List[ckModule]],
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """Compute det(Jacobian of the lens equation) on a coordinate grid.

    Parameters
    ----------
    lens_mass : PhysicalModel or list of ck.Module
        Lens mass components.
    x : jnp.ndarray
        x-coordinates (image plane, arcsec).  1-D or 2-D.
    y : jnp.ndarray
        y-coordinates (image plane, arcsec).  Same shape as *x*.

    Returns
    -------
    jnp.ndarray
        Determinant of the Jacobian matrix at each point, same shape as
        *x* and *y*.  Points where this value crosses zero belong to a
        critical line.
    """
    mass_list = _resolve_lens_mass_list(lens_mass)
    deflection_fn = _build_deflection_fn(mass_list)

    # Flatten → compute Jacobian per point → reshape back
    x_flat = jnp.ravel(x)
    y_flat = jnp.ravel(y)

    def _f_single(xy: jnp.ndarray) -> jnp.ndarray:
        """Single-point lens equation, (2,) → (2,)."""
        bx, by = deflection_fn(xy[0], xy[1])
        return jnp.array([bx, by])

    xy_batch = jnp.stack([x_flat, y_flat], axis=-1)  # (N, 2)
    jac = jax.vmap(jax.jacfwd(_f_single))(xy_batch)  # (N, 2, 2)

    det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
    return det.reshape(x.shape)


def _extract_contours(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z: np.ndarray,
    level: float = 0.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract contour paths at *level* from a 2-D scalar field *z*.

    Uses matplotlib's contour engine internally (no figure is kept).
    Returns a list of ``(x_path, y_path)`` 1-D arrays.
    """
    if x_grid.ndim != 2 or y_grid.ndim != 2:
        raise ValueError("x_grid and y_grid must be 2-D arrays.")
    if z.shape != x_grid.shape:
        raise ValueError("z must have the same shape as x_grid/y_grid.")

    # Use a temporary figure — the contour engine requires an Axes context.
    fig = plt.figure()
    try:
        cs = plt.contour(x_grid, y_grid, z, levels=[level])
        paths: List[Tuple[np.ndarray, np.ndarray]] = []
        # Matplotlib ≥ 3.8 exposes get_paths() directly on QuadContourSet,
        # with disconnected contours combined into compound Path objects.
        for path in cs.get_paths():
            vertices = np.asarray(path.vertices)
            codes = path.codes
            starts = (
                np.flatnonzero(codes == Path.MOVETO)
                if codes is not None
                else np.array([0])
            )
            stops: np.ndarray = np.append(starts[1:], len(vertices))
            for start, stop in zip(starts, stops):
                segment = vertices[start:stop]
                if len(segment) <= 1:
                    continue
                paths.append(
                    (
                        np.asarray(segment[:, 0]),
                        np.asarray(segment[:, 1]),
                    )
                )
    finally:
        plt.close(fig)

    return paths


def find_critical_lines(
    lens_mass: Union[PhysicalModel, List[ckModule]],
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 512,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Find critical line contours in the image plane.

    Evaluates det(Jacobian) on a regular *n_grid* × *n_grid* mesh and
    extracts the zero-level contour(s).

    Parameters
    ----------
    lens_mass : PhysicalModel or list of ck.Module
        Lens mass components.
    x_range : tuple (xmin, xmax)
        Image-plane x-coordinate range (arcsec).
    y_range : tuple (ymin, ymax)
        Image-plane y-coordinate range (arcsec).
    n_grid : int
        Number of grid points along each axis (default 512).

    Returns
    -------
    list of (x_path, y_path)
        Each element is a ``(1-D x, 1-D y)`` tuple tracing one connected
        critical-line segment.
    """
    mass_list = _resolve_lens_mass_list(lens_mass)

    xs = jnp.linspace(x_range[0], x_range[1], n_grid)
    ys = jnp.linspace(y_range[0], y_range[1], n_grid)
    xv, yv = jnp.meshgrid(xs, ys, indexing="xy")

    det = compute_jacobian_det(mass_list, xv, yv)
    det_np = np.asarray(det)
    xv_np = np.asarray(xv)
    yv_np = np.asarray(yv)

    return _extract_contours(xv_np, yv_np, det_np, level=0.0)


def find_caustics(
    lens_mass: Union[PhysicalModel, List[ckModule]],
    critical_line_paths: List[Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Map critical-line points to the source plane to obtain caustics.

    Each critical-line path is ray-traced through the lens equation
    β(θ) = θ - α(θ).

    Parameters
    ----------
    lens_mass : PhysicalModel or list of ck.Module
        Lens mass components.
    critical_line_paths : list of (x, y)
        Critical line paths returned by :func:`find_critical_lines`.

    Returns
    -------
    list of (beta_x, beta_y)
        Caustic paths in the source plane.  The list has the same length
        and ordering as *critical_line_paths*.
    """
    mass_list = _resolve_lens_mass_list(lens_mass)
    deflection_fn = _build_deflection_fn(mass_list)

    caustic_paths: List[Tuple[np.ndarray, np.ndarray]] = []
    for x_path, y_path in critical_line_paths:
        x_j = jnp.asarray(x_path)
        y_j = jnp.asarray(y_path)
        bx, by = deflection_fn(x_j, y_j)
        caustic_paths.append(
            (np.asarray(bx), np.asarray(by))
        )
    return caustic_paths


def compute_critical_and_caustics(
    lens_mass: Union[PhysicalModel, List[ckModule]],
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 512,
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    List[Tuple[np.ndarray, np.ndarray]],
]:
    """Compute critical lines and caustics in one call.

    This is the primary high-level entry point for visualization use.

    Parameters
    ----------
    lens_mass : PhysicalModel or list of ck.Module
        Lens mass components whose parameter values have already been
        set to the desired state (e.g., posterior median).
    x_range : tuple (xmin, xmax)
        Image-plane x-coordinate search range (arcsec).
    y_range : tuple (ymin, ymax)
        Image-plane y-coordinate search range (arcsec).
    n_grid : int
        Grid resolution for critical-line contour extraction (default 512).

    Returns
    -------
    critical_lines : list of (x, y)
        Critical line paths in the image plane.
    caustics : list of (beta_x, beta_y)
        Caustic paths in the source plane.
    """
    crit = find_critical_lines(lens_mass, x_range, y_range, n_grid)
    caus = find_caustics(lens_mass, crit)
    return crit, caus


__all__ = [
    "compute_jacobian_det",
    "find_critical_lines",
    "find_caustics",
    "compute_critical_and_caustics",
]
