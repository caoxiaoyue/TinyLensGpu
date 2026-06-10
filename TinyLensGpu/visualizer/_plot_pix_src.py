"""
1×4 diagnostic plot for pixelized-source inversions.

Provides :func:`plot_pixelized_source_results` which abstracts the common
four-panel visualisation pattern found across pixelized-source demos.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ._helpers import CMAP_VIRIDIS, CMAP_VIRIDIS_RESIDUALS
from ._overlays import overlay_critical_and_caustics


def plot_pixelized_source_results(
    image_data: np.ndarray,
    model_image: np.ndarray,
    resid_norm: np.ndarray,
    mask: np.ndarray,
    source_image: np.ndarray,
    source_extent: Tuple[float, float, float, float],
    image_dpix: float = 0.05,
    chi2_nu: Optional[float] = None,
    lambda_reg: Optional[float] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show_critical_lines: bool = False,
    show_caustics: bool = False,
    lens_mass=None,
    crit_x_range: Tuple[float, float] = (-3.0, 3.0),
    crit_y_range: Tuple[float, float] = (-3.0, 3.0),
    crit_n_grid: int = 512,
):
    """Four-panel plot for pixelized-source inversion results.

    Panels
    ------
    1. Observed data (or lens-subtracted data)
    2. Model image (pixelized source lensed forward)
    3. Normalised residuals
    4. Source reconstruction (source plane)

    Parameters
    ----------
    image_data : np.ndarray
        2-D observed (or lens-subtracted) image.
    model_image : np.ndarray
        2-D model image reconstructed from the pixelized source.
    resid_norm : np.ndarray
        2-D normalised residuals ``(data - model) / noise``.
    mask : np.ndarray
        Boolean mask where ``True`` means excluded pixels.
    source_image : np.ndarray
        2-D source-plane pixel array.
    source_extent : tuple (xmin, xmax, ymin, ymax)
        Source-plane extent in arcsec.
    image_dpix : float
        Pixel scale of the image plane (arcsec/pixel).
    chi2_nu : float, optional
        Reduced χ² to display in the residual panel title.
    lambda_reg : float, optional
        Regularisation strength for annotation.
    save_path : str, optional
        Output file path.
    title : str, optional
        Overall figure suptitle.
    show_critical_lines : bool
        Overlay critical lines on image-plane panels (0–2).
    show_caustics : bool
        Overlay caustics on the source-plane panel (3).
    lens_mass : PhysicalModel or list of mass modules, optional
        Required when *show_critical_lines* or *show_caustics* is
        ``True``.  Parameters must be set to the desired state beforehand.
    crit_x_range, crit_y_range : tuple
        Image-plane range for critical-line search.
    crit_n_grid : int
        Grid resolution for critical-line contour extraction.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    npix = image_data.shape[0]
    ext_i = [
        -npix * image_dpix / 2,
        npix * image_dpix / 2,
        -npix * image_dpix / 2,
        npix * image_dpix / 2,
    ]
    ext_s = list(source_extent)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    # Colormap range from unmasked data
    vmax = np.nanpercentile(image_data[~mask], 99.5) if mask is not None else np.nanmax(image_data)

    # Panel 1: data
    im0 = axes[0].imshow(
        image_data, origin="lower", extent=ext_i, cmap=CMAP_VIRIDIS, vmin=0, vmax=vmax,
    )
    axes[0].set_title("Lensed arc (data)", fontsize=11)
    axes[0].set_xlabel("arcsec")
    axes[0].set_ylabel("arcsec")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: model
    im1 = axes[1].imshow(
        model_image, origin="lower", extent=ext_i, cmap=CMAP_VIRIDIS, vmin=0, vmax=vmax,
    )
    axes[1].set_title("Pix-src model image", fontsize=11)
    axes[1].set_xlabel("arcsec")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: normalised residual
    resid_display = np.where(mask, np.nan, resid_norm)
    res_title = "Norm. residual (σ)"
    if chi2_nu is not None:
        res_title += f"\nχ²/ν = {chi2_nu:.3f}"
    im2 = axes[2].imshow(
        resid_display, origin="lower", extent=ext_i,
        cmap=CMAP_VIRIDIS_RESIDUALS, vmin=-5, vmax=5,
    )
    axes[2].set_title(res_title, fontsize=11)
    axes[2].set_xlabel("arcsec")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: source reconstruction
    im3 = axes[3].imshow(
        source_image, origin="lower", extent=ext_s, cmap=CMAP_VIRIDIS,
    )
    src_title = "Source reconstruction"
    if lambda_reg is not None:
        src_title += f"\n(λ={lambda_reg:.2e})"
    axes[3].set_title(src_title, fontsize=11)
    axes[3].set_xlabel("arcsec")
    axes[3].set_ylabel("arcsec")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # Overall title
    if title:
        plt.suptitle(title, fontsize=12)

    # -------- critical lines / caustics ----------------------------------
    if (show_critical_lines or show_caustics) and lens_mass is not None:
        image_panels = [axes[0], axes[1], axes[2]]
        overlay_critical_and_caustics(
            image_axes=image_panels,
            source_ax=axes[3],
            lens_mass=lens_mass,
            x_range=crit_x_range,
            y_range=crit_y_range,
            n_grid=crit_n_grid,
        )
    elif show_critical_lines and lens_mass is None:
        raise ValueError(
            "lens_mass must be provided when show_critical_lines=True"
        )
    elif show_caustics and lens_mass is None:
        raise ValueError(
            "lens_mass must be provided when show_caustics=True"
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig, axes
