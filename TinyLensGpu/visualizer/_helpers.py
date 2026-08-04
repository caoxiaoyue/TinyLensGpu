"""
Shared plotting helpers for the visualizer package.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from TinyLensGpu.utils.misc import get_mask_bounding_box


# ---------------------------------------------------------------------------
# Colormap factories
# ---------------------------------------------------------------------------

def _make_cmap(base_name: str, bad_color: str = "white"):
    """Return a matplotlib colormap copy with masked-pixel colour set."""
    cmap = plt.get_cmap(base_name).copy()
    cmap.set_bad(color=bad_color)
    return cmap


CMAP_IMAGE = _make_cmap("inferno")
CMAP_RESIDUALS = _make_cmap("RdBu_r")
CMAP_SOURCE = _make_cmap("inferno")
CMAP_VIRIDIS = _make_cmap("viridis")
CMAP_VIRIDIS_RESIDUALS = _make_cmap("RdBu_r")


# ---------------------------------------------------------------------------
# Extent builders
# ---------------------------------------------------------------------------

def image_extent(npix: int, dpix: float) -> Tuple[float, float, float, float]:
    """Return ``[left, right, bottom, top]`` for an image-plane imshow.

    The coordinate origin (0, 0) is at the image centre.
    """
    half = npix * dpix / 2.0
    return (-half, half, -half, half)



# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def apply_mask(data, mask, fill_value=np.nan):
    """Return a masked or NaN-filled array for display.

    When *mask* is ``None`` the data is returned unchanged.
    """
    if mask is not None:
        return np.ma.masked_array(data, mask=mask)
    return data


def compute_residuals(
    data: np.ndarray,
    model: np.ndarray,
    noise: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute normalised residuals ``(data - model) / noise``.

    Masked pixels are filled with ``NaN`` for clean display with
    ``cmap.set_bad('white')``.
    """
    res = (data - model) / noise
    if mask is not None:
        res = np.where(mask, np.nan, res)
    return res


def apply_bounding_box(
    axes,
    mask: Optional[np.ndarray],
    npix: int,
    dpix: float,
):
    """Set *axes* limits to the unmasked-pixel bounding box.

    Does nothing when *mask* is ``None``.
    """
    if mask is None:
        return
    xlim, ylim = get_mask_bounding_box(mask, npix, dpix)
    if xlim is not None and ylim is not None:
        for ax in axes:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

CRITICAL_LINE_DEFAULTS = dict(
    color="white",
    linewidth=1.2,
    linestyle="--",
    alpha=0.85,
)

CAUSTIC_DEFAULTS = dict(
    color="gold",
    linewidth=1.2,
    linestyle="--",
    alpha=0.85,
)
