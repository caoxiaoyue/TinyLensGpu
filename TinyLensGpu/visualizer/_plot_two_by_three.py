"""
2×3 diagnostic plot for parametric lens models.

Provides :func:`plot_model_results` — the enhanced version of the
original ``visualizer.plot_model_results`` with optional critical-line
and caustic overlays.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ._helpers import (
    CMAP_IMAGE,
    CMAP_RESIDUALS,
    apply_bounding_box,
    apply_mask,
    compute_residuals,
    image_extent,
    make_source_grid,
)
from ._overlays import overlay_critical_and_caustics


def plot_model_results(
    likelihood_obj,
    theta,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show_critical_lines: bool = False,
    show_caustics: bool = False,
    crit_x_range: Tuple[float, float] = (-3.0, 3.0),
    crit_y_range: Tuple[float, float] = (-3.0, 3.0),
    crit_n_grid: int = 512,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot model results in a 2×3 grid.

    Panels
    ------
    1. ``(0, 0)`` — Observed data
    2. ``(0, 1)`` — Lens light model
    3. ``(0, 2)`` — Data − lens light
    4. ``(1, 0)`` — Lensed image model
    5. ``(1, 1)`` — Normalised residuals
    6. ``(1, 2)`` — Source-plane reconstruction

    Parameters
    ----------
    likelihood_obj :
        A likelihood object providing ``forward_model(...)`` and
        ``sim_obj`` (typically :class:`ImageProbModel`).
    theta : array-like
        Non-linear parameter values (list, ``np.ndarray``, or
        ``jnp.ndarray``).
    save_path : str, optional
        If given, the figure is saved to this path.
    title : str, optional
        Suptitle for the figure.
    show_critical_lines : bool
        When ``True``, overlay critical lines on image-plane panels
        (0, 0) through (1, 1).
    show_caustics : bool
        When ``True``, overlay caustics on the source-plane panel
        (1, 2).
    crit_x_range, crit_y_range : tuple
        Image-plane coordinate range for critical-line search (arcsec).
    crit_n_grid : int
        Grid resolution for critical-line contour extraction.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
        The 2×3 Axes array.
    """
    # -------- parameter setup -------------------------------------------
    if isinstance(theta, (np.ndarray, jnp.ndarray)):
        theta = theta.tolist()
    likelihood_obj.set_values(theta)

    data = np.array(likelihood_obj.image_data)
    noise = np.array(likelihood_obj.noise_map)

    if not (
        hasattr(likelihood_obj, "forward_model")
        and hasattr(likelihood_obj, "sim_obj")
    ):
        raise TypeError(
            "likelihood_obj must provide forward_model(...) and sim_obj."
        )

    fwd_result = likelihood_obj.forward_model(
        use_linear=likelihood_obj.use_linear,
        return_intensity=True,
        ret_each_plane=True,
        image_map=likelihood_obj.image_data,
        noise_map=likelihood_obj.noise_map,
    )

    linear_intensities = None
    if len(fwd_result) == 3:
        lensed_image_model, lens_light_model, linear_intensities = fwd_result
    else:
        lensed_image_model, lens_light_model = fwd_result

    sim_config = likelihood_obj.sim_obj.sim_config
    lensed_image_model = np.array(lensed_image_model)
    lens_light_model = np.array(lens_light_model)
    total_model = lensed_image_model + lens_light_model

    # -------- source plane -----------------------------------------------
    cx, cy = 0.0, 0.0
    sx, sy, s_npix, s_dpix = make_source_grid(sim_config.dpix, cx, cy, scale=3.0)

    source_light = getattr(
        likelihood_obj.sim_obj.phys_model, "source_light", []
    )
    if len(source_light) > 0:
        source_planes = []
        for i, m in enumerate(source_light):
            kwargs = {}
            if linear_intensities is not None:
                for name in ["flux", "Ie", "amp", "intensity", "I0"]:
                    if hasattr(m, name):
                        kwargs[name] = linear_intensities[i]
                        break
            source_planes.append(m.light(x=sx, y=sy, **kwargs))
        source_plane_image = jnp.stack(source_planes, axis=-1)
        source_plane_image = np.asarray(source_plane_image)
    else:
        source_plane_image = np.asarray(jnp.zeros_like(sx))

    # -------- mask -------------------------------------------------------
    mask = None
    if hasattr(sim_config, "mask") and sim_config.mask is not None:
        mask = np.asarray(sim_config.mask)

    # -------- masked arrays ----------------------------------------------
    if mask is not None:
        data_m = np.ma.masked_array(data, mask=mask)
        ll_m = np.ma.masked_array(lens_light_model, mask=mask)
        li_m = np.ma.masked_array(lensed_image_model, mask=mask)
        total_m = li_m + ll_m
    else:
        data_m = data
        ll_m = lens_light_model
        li_m = lensed_image_model
        total_m = total_model

    residuals = compute_residuals(data, total_model, noise, mask)

    # -------- extent & bounding box --------------------------------------
    extent = image_extent(sim_config.npix, sim_config.dpix)
    s_extent = (
        float(cx - s_npix * s_dpix / 2),
        float(cx + s_npix * s_dpix / 2),
        float(cy - s_npix * s_dpix / 2),
        float(cy + s_npix * s_dpix / 2),
    )

    # -------- figure -----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    if title:
        fig.suptitle(title, fontsize=16)

    # (0, 0): Data
    im00 = axes[0, 0].imshow(
        data_m, origin="lower", extent=extent, cmap=CMAP_IMAGE
    )
    axes[0, 0].set_title("Observed Data")
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # (0, 1): Lens Light Model
    im01 = axes[0, 1].imshow(
        ll_m, origin="lower", extent=extent, cmap=CMAP_IMAGE
    )
    axes[0, 1].set_title("Lens Light Model")
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (0, 2): Data - Lens Light
    diff = data_m - ll_m if mask is None else np.ma.masked_array(
        data - lens_light_model, mask=mask
    )
    im02 = axes[0, 2].imshow(
        diff, origin="lower", extent=extent, cmap=CMAP_IMAGE
    )
    axes[0, 2].set_title("Data − Lens Light")
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # (1, 0): Lensed Image Model
    im10 = axes[1, 0].imshow(
        li_m, origin="lower", extent=extent, cmap=CMAP_IMAGE
    )
    axes[1, 0].set_title("Lensed Image Model")
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # (1, 1): Normalised Residuals
    if mask is not None:
        res_display = np.ma.masked_array(residuals, mask=mask)
    else:
        res_display = residuals
    im11 = axes[1, 1].imshow(
        res_display,
        origin="lower",
        extent=extent,
        cmap=CMAP_RESIDUALS,
        vmin=-5,
        vmax=5,
    )
    axes[1, 1].set_title("Normalised Residuals")
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # -------- bounding-box crop for image-plane panels -------------------
    image_plane_panels = [
        axes[0, 0],
        axes[0, 1],
        axes[0, 2],
        axes[1, 0],
        axes[1, 1],
    ]
    apply_bounding_box(image_plane_panels, mask, sim_config.npix, sim_config.dpix)

    # -------- (1, 2): Source Plane ---------------------------------------
    src_img = (
        source_plane_image.sum(axis=-1)
        if source_plane_image.ndim == 3
        else source_plane_image
    )
    im12 = axes[1, 2].imshow(
        src_img, origin="lower", extent=s_extent, cmap=CMAP_IMAGE
    )
    axes[1, 2].set_title("Source (Source Plane)")
    plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # -------- axes labels ------------------------------------------------
    for ax in axes.flat:
        ax.set_xlabel("Arcsec")
        ax.set_ylabel("Arcsec")

    # -------- critical lines / caustics ----------------------------------
    if show_critical_lines or show_caustics:
        phys_model = likelihood_obj.sim_obj.phys_model
        if show_critical_lines and show_caustics:
            overlay_critical_and_caustics(
                image_axes=image_plane_panels,
                source_ax=axes[1, 2],
                lens_mass=phys_model,
                x_range=crit_x_range,
                y_range=crit_y_range,
                n_grid=crit_n_grid,
            )
        elif show_critical_lines:
            from ._overlays import overlay_critical_lines

            overlay_critical_lines(
                image_plane_panels[0],  # dummy — we overlay on each below
                phys_model,
                x_range=crit_x_range,
                y_range=crit_y_range,
                n_grid=crit_n_grid,
            )
            # overlay on each image-plane panel
            from TinyLensGpu.utils.lensing.critical_line import (
                find_critical_lines,
            )

            paths = find_critical_lines(
                phys_model, crit_x_range, crit_y_range, crit_n_grid
            )
            for ax in image_plane_panels:
                for xp, yp in paths:
                    ax.plot(
                        xp, yp, color="white", linewidth=1.2, linestyle="--", alpha=0.85,
                    )
        elif show_caustics:
            from ._overlays import overlay_caustics

            overlay_caustics(
                axes[1, 2],
                phys_model,
                x_range=crit_x_range,
                y_range=crit_y_range,
                n_grid=crit_n_grid,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Plot saved to {save_path}")

    return fig, axes
