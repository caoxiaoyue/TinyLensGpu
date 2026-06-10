"""
Critical-line and caustic overlay helpers.

These thin wrappers call :mod:`TinyLensGpu.utils.lensing.critical_line`
and plot the resulting paths onto matplotlib axes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.lensing.critical_line import (
    compute_critical_and_caustics,
    find_caustics,
    find_critical_lines,
)
from ._helpers import CAUSTIC_DEFAULTS, CRITICAL_LINE_DEFAULTS


def _get_lens_mass_from_likelihood(likelihood_obj) -> List:
    """Best-effort extraction of lens mass list from a likelihood object.

    Supports both parametric (``ImageProbModel``) and operator-based
    (``PixelizedImageProbModelOperator``) likelihoods.
    """
    # Parametric path: likelihood_obj.sim_obj.phys_model
    if hasattr(likelihood_obj, "sim_obj") and hasattr(
        likelihood_obj.sim_obj, "phys_model"
    ):
        return likelihood_obj.sim_obj.phys_model.lens_mass

    # Operator path: likelihood_obj.phys_model
    if hasattr(likelihood_obj, "phys_model"):
        return likelihood_obj.phys_model.lens_mass

    raise TypeError(
        "Cannot extract lens mass list from likelihood object "
        f"of type {type(likelihood_obj)}."
    )


def overlay_critical_lines(
    ax,
    lens_mass: Union[PhysicalModel, List],
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 512,
    line_kwargs: Optional[dict] = None,
):
    """Compute and overlay critical lines on a matplotlib *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (image-plane coordinates).
    lens_mass : PhysicalModel or list of mass modules
        Lens mass components with parameters already set.
    x_range, y_range : tuple
        Image-plane coordinate range (arcsec).
    n_grid : int
        Grid resolution for contour extraction.
    line_kwargs : dict, optional
        Passed to :meth:`~matplotlib.axes.Axes.plot` for each path.
        Defaults to white dashed lines.
    """
    kw = dict(CRITICAL_LINE_DEFAULTS)
    if line_kwargs:
        kw.update(line_kwargs)

    paths = find_critical_lines(lens_mass, x_range, y_range, n_grid)
    for xp, yp in paths:
        ax.plot(xp, yp, **kw)


def overlay_caustics(
    ax,
    lens_mass: Union[PhysicalModel, List],
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 512,
    line_kwargs: Optional[dict] = None,
    critical_paths: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
):
    """Compute and overlay caustics on a matplotlib *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (source-plane coordinates).
    lens_mass : PhysicalModel or list of mass modules
        Lens mass components with parameters already set.
    x_range, y_range : tuple
        Image-plane coordinate range used for critical-line search.
    n_grid : int
        Grid resolution for critical-line extraction.
    line_kwargs : dict, optional
        Passed to :meth:`~matplotlib.axes.Axes.plot` for each path.
        Defaults to gold dashed lines.
    critical_paths : list of (x, y), optional
        Pre-computed critical-line paths.  When provided, *x_range*,
        *y_range* and *n_grid* are ignored — caustics are mapped directly
        from these paths.
    """
    kw = dict(CAUSTIC_DEFAULTS)
    if line_kwargs:
        kw.update(line_kwargs)

    if critical_paths is None:
        critical_paths = find_critical_lines(lens_mass, x_range, y_range, n_grid)

    caustic_paths = find_caustics(lens_mass, critical_paths)
    for bx, by in caustic_paths:
        ax.plot(bx, by, **kw)


def overlay_critical_and_caustics(
    image_axes,
    source_ax,
    lens_mass: Union[PhysicalModel, List],
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 512,
    crit_kwargs: Optional[dict] = None,
    caus_kwargs: Optional[dict] = None,
):
    """Convenience: overlay critical lines and caustics in one call.

    Critical lines are computed once; the same paths are used for the
    image-plane overlay and for mapping to caustics.

    Parameters
    ----------
    image_axes : list of matplotlib.axes.Axes
        One or more image-plane axes to overlay critical lines on.
    source_ax : matplotlib.axes.Axes
        Source-plane axis to overlay caustics on.
    lens_mass : PhysicalModel or list of mass modules
        Lens mass components.
    x_range, y_range : tuple
        Image-plane range for critical-line search.
    n_grid : int
        Grid resolution.
    crit_kwargs, caus_kwargs : dict, optional
        Style overrides for critical lines and caustics respectively.
    """
    crit_paths = find_critical_lines(lens_mass, x_range, y_range, n_grid)

    # Critical lines on all image-plane axes
    crit_kw = dict(CRITICAL_LINE_DEFAULTS)
    if crit_kwargs:
        crit_kw.update(crit_kwargs)
    for ax in image_axes:
        for xp, yp in crit_paths:
            ax.plot(xp, yp, **crit_kw)

    # Caustics on source-plane axis
    caus_kw = dict(CAUSTIC_DEFAULTS)
    if caus_kwargs:
        caus_kw.update(caus_kwargs)
    caustic_paths = find_caustics(lens_mass, crit_paths)
    for bx, by in caustic_paths:
        source_ax.plot(bx, by, **caus_kw)
