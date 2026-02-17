"""
Visualization module for lens model fitting results.
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d
from scipy.interpolate import griddata
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _plot_irregular_source_interpolate(
    ax,
    points_xy,
    values,
    enlarge_factor=1.1,
    npixels=150,
    cmap="inferno",
):
    """
    Render irregular source samples on a regular grid via interpolation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis used for drawing the interpolated source image.
    points_xy : array_like
        Source-node coordinates with shape ``(n_points, 2)`` in source-plane units.
    values : array_like
        Source intensities at ``points_xy`` with shape ``(n_points,)``.
    enlarge_factor : float, optional
        Multiplicative padding applied to the plotting extent.
    npixels : int, optional
        Resolution of the interpolation grid along each axis.
    cmap : str, optional
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.image.AxesImage
        Image artist returned by ``ax.imshow``.
    """
    points_xy = np.asarray(points_xy)
    values = np.asarray(values)

    half_width = float(np.max(np.abs(points_xy))) if points_xy.size else 1.0
    half_width *= float(enlarge_factor)

    coord_1d, dpix = np.linspace(-half_width, half_width, int(npixels), endpoint=True, retstep=True)
    xgrid, ygrid = np.meshgrid(coord_1d, coord_1d)
    extent = [-half_width - 0.5 * dpix, half_width + 0.5 * dpix, -half_width - 0.5 * dpix, half_width + 0.5 * dpix]

    img = griddata(points_xy, values, (xgrid, ygrid), method="linear", fill_value=0.0)
    im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect("equal", adjustable="box")
    return im


def _plot_irregular_source_voronoi(
    ax,
    points_xy,
    values,
    enlarge_factor=1.1,
    cmap="inferno",
    minima=None,
    maxima=None,
):
    """
    Render irregular source samples using Voronoi cells.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis used for drawing Voronoi polygons.
    points_xy : array_like
        Source-node coordinates with shape ``(n_points, 2)``.
    values : array_like
        Source intensities at ``points_xy``.
    enlarge_factor : float, optional
        Multiplicative padding applied to the displayed extent.
    cmap : str, optional
        Matplotlib colormap name.
    minima : float, optional
        Lower bound for colormap normalization. If ``None``, uses ``values.min()``.
    maxima : float, optional
        Upper bound for colormap normalization. If ``None``, uses ``values.max()``.

    Returns
    -------
    None
        The function draws directly on ``ax`` and adds a colorbar.
    """
    points_xy = np.asarray(points_xy)
    values = np.asarray(values)

    half_width = float(np.max(np.abs(points_xy)))
    half_width *= float(enlarge_factor)

    extra_points = np.array([[999, 999], [-999, 999], [999, -999], [-999, -999]], dtype=float)
    points_aug = np.append(points_xy, extra_points, axis=0)
    vor = Voronoi(points_aug)

    if minima is None:
        minima = float(np.min(values))
    if maxima is None:
        maxima = float(np.max(values))

    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])

    voronoi_plot_2d(
        vor,
        ax=ax,
        show_points=False,
        show_vertices=False,
        line_width=0.05,
        point_size=1,
        line_colors="k",
        line_alpha=0.2,
    )
    n_data = len(values)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=mapper.to_rgba(values[r]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mapper, cax=cax)
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_aspect('equal', adjustable='box')


def plot_model_results(
    likelihood_obj,
    theta,
    save_path=None,
    title=None,
    pix_src_render="interpolate",
    pix_src_show_grid=False,
    pix_src_npixels=150,
):
    """
    Plot model results in a 2x3 grid.
    
    1. (0,0): Observed data
    2. (0,1): Lens light model
    3. (0,2): Data - Lens light
    4. (1,0): Lensed image model
    5. (1,1): Normalized residuals
    6. (1,2): Source in source plane
    """
    # Ensure theta is a Sequence for caskade (numpy arrays are rejected)
    if isinstance(theta, (np.ndarray, jnp.ndarray)):
        theta = theta.tolist()
    likelihood_obj.set_values(theta) 

    data = np.asarray(likelihood_obj.image_data)
    noise = np.asarray(likelihood_obj.noise_map)

    is_parametric = hasattr(likelihood_obj, "forward_model") and hasattr(likelihood_obj, "sim_obj")
    is_pixelized = hasattr(likelihood_obj, "reconstruct_source")

    if is_parametric:
        lensed_image_model, lens_light_model = likelihood_obj.forward_model(
            use_linear=likelihood_obj.use_linear,
            return_intensity=False,
            ret_each_plane=True,
            image_map=likelihood_obj.image_data,
            noise_map=likelihood_obj.noise_map,
        )
        sim_config = likelihood_obj.sim_obj.sim_config
        lensed_image_model = np.asarray(lensed_image_model)
        lens_light_model = np.asarray(lens_light_model)
        total_model = lensed_image_model + lens_light_model

        cx, cy = 0.0, 0.0
        s_dpix = sim_config.dpix / 2.0
        s_npix = int(np.ceil(3.0 / s_dpix))
        sx, sy = make_grid_2d(s_npix, s_dpix, 1)
        sx = jnp.array(sx) + cx
        sy = jnp.array(sy) + cy

        source_light = getattr(likelihood_obj.sim_obj.phys_model, "source_light", [])
        if len(source_light) > 0:
            source_plane_image = jnp.stack([m.light(x=sx, y=sy) for m in source_light], axis=-1)
            source_plane_image = np.asarray(source_plane_image)
        else:
            source_plane_image = np.asarray(jnp.zeros_like(sx))
    elif is_pixelized:
        source_intensities, source_mesh_beta, model_image = likelihood_obj.reconstruct_source(return_2d=True)
        lensed_image_model = np.asarray(model_image)
        lens_light_model = np.zeros_like(lensed_image_model)
        total_model = lensed_image_model
        sim_config = type("SimConfig", (), {"npix": int(lensed_image_model.shape[0]), "dpix": float(likelihood_obj.dpix)})()
        source_intensities = np.asarray(source_intensities)
        source_mesh_beta = np.asarray(source_mesh_beta)
    else:
        raise TypeError("likelihood_obj must provide forward_model(...) or reconstruct_source().")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    if title:
        fig.suptitle(title, fontsize=16)

    extent = [
        -sim_config.npix * sim_config.dpix / 2, sim_config.npix * sim_config.dpix / 2,
        -sim_config.npix * sim_config.dpix / 2, sim_config.npix * sim_config.dpix / 2
    ]
    
    # (0,0): Data
    im0 = axes[0, 0].imshow(data, origin='lower', extent=extent, cmap='inferno')
    axes[0, 0].set_title("Observed Data")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # (0,1): Lens Light Model
    im1 = axes[0, 1].imshow(lens_light_model, origin='lower', extent=extent, cmap='inferno')
    axes[0, 1].set_title("Lens Light Model")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # (0,2): Data - Lens Light
    im2 = axes[0, 2].imshow(data - lens_light_model, origin='lower', extent=extent, cmap='inferno')
    axes[0, 2].set_title("Data - Lens Light")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # (1,0): Lensed Image Model
    im3 = axes[1, 0].imshow(lensed_image_model, origin='lower', extent=extent, cmap='inferno')
    axes[1, 0].set_title("Lensed Image Model")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # (1,1): Normalized Residuals
    # Residuals = (Data - (Lens + Source)) / Noise
    res = (data - total_model) / noise
    im4 = axes[1, 1].imshow(res, origin='lower', extent=extent, cmap='RdBu_r', vmin=-5, vmax=5)
    axes[1, 1].set_title("Normalized Residuals")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # (1,2): Source Plane
    if is_parametric:
        s_extent = [
            cx - s_npix * s_dpix / 2,
            cx + s_npix * s_dpix / 2,
            cy - s_npix * s_dpix / 2,
            cy + s_npix * s_dpix / 2,
        ]
        src_img = source_plane_image.sum(axis=-1) if source_plane_image.ndim == 3 else source_plane_image
        im5 = axes[1, 2].imshow(src_img, origin="lower", extent=s_extent, cmap="inferno")
        axes[1, 2].set_title("Source (Source Plane)")
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
    else:
        axes[1, 2].set_title("Source (Source Plane)")
        if str(pix_src_render).lower() == "voronoi":
            _plot_irregular_source_voronoi(axes[1, 2], source_mesh_beta, source_intensities, cmap="inferno")
        else:
            _plot_irregular_source_interpolate(
                axes[1, 2],
                source_mesh_beta,
                source_intensities,
                cmap="inferno",
                npixels=pix_src_npixels,
            )
        if pix_src_show_grid:
            axes[1, 2].scatter(
                source_mesh_beta[:, 0],
                source_mesh_beta[:, 1],
                c="black",
                s=0.2,
                alpha=0.3,
            )
    
    for ax in axes.flat:
        ax.set_xlabel("Arcsec")
        ax.set_ylabel("Arcsec")
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Plot saved to {save_path}")
        
    return fig, axes
