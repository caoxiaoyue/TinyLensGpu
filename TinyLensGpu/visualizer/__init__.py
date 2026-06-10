"""
Visualization package for TinyLensGpu lens modeling results.

Provides reusable plotting functions for common diagnostic panels:

- :func:`plot_model_results` — 2×3 parametric-model diagnostics
- :func:`plot_pixelized_source_results` — 1×4 pixelized-source diagnostics
- :func:`overlay_critical_lines` / :func:`overlay_caustics` — low-level
  helpers for adding critical-line and caustic curves to any axes.
"""

from ._plot_two_by_three import plot_model_results
from ._plot_pix_src import plot_pixelized_source_results
from ._overlays import (
    overlay_critical_lines,
    overlay_caustics,
    overlay_critical_and_caustics,
)

__all__ = [
    "plot_model_results",
    "plot_pixelized_source_results",
    "overlay_critical_lines",
    "overlay_caustics",
    "overlay_critical_and_caustics",
]
