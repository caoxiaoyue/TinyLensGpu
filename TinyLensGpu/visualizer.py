"""
Visualization module for lens model fitting results.
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d
from TinyLensGpu.utils.misc import get_mask_bounding_box


def plot_model_results(
    likelihood_obj,
    theta,
    save_path=None,
    title=None,
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

    data = np.array(likelihood_obj.image_data)
    noise = np.array(likelihood_obj.noise_map)

    if not (hasattr(likelihood_obj, "forward_model") and hasattr(likelihood_obj, "sim_obj")):
        raise TypeError("likelihood_obj must provide forward_model(...) and sim_obj.")

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

    cx, cy = 0.0, 0.0
    s_dpix = sim_config.dpix / 2.0
    s_npix = int(np.ceil(3.0 / s_dpix))
    sx, sy = make_grid_2d(s_npix, s_dpix, 1)
    sx = jnp.array(sx) + cx
    sy = jnp.array(sy) + cy

    source_light = getattr(likelihood_obj.sim_obj.phys_model, "source_light", [])
    if len(source_light) > 0:
        source_planes = []
        for i, m in enumerate(source_light):
            kwargs = {}
            if linear_intensities is not None:
                # Find linear parameter name
                for name in ['flux', 'Ie', 'amp', 'intensity', 'I0']:
                    if hasattr(m, name):
                        kwargs[name] = linear_intensities[i]
                        break
            source_planes.append(m.light(x=sx, y=sy, **kwargs))

        source_plane_image = jnp.stack(source_planes, axis=-1)
        source_plane_image = np.asarray(source_plane_image)
    else:
        source_plane_image = np.asarray(jnp.zeros_like(sx))

    # Apply mask if available
    mask = None
    if hasattr(sim_config, "mask") and sim_config.mask is not None:
        mask = np.asarray(sim_config.mask)

    if mask is not None:
        # Create masked arrays for visualization
        data = np.ma.masked_array(data, mask=mask)
        lens_light_model = np.ma.masked_array(lens_light_model, mask=mask)
        lensed_image_model = np.ma.masked_array(lensed_image_model, mask=mask)
        total_model = lensed_image_model + lens_light_model
        # Recompute residuals and mask them
        res = (data - total_model) / noise
        res = np.ma.masked_array(res, mask=mask)
    else:
        # If no mask, still compute residuals normally
        res = (data - total_model) / noise

    # Calculate bounding box for unmasked pixels
    xlim, ylim = None, None
    if mask is not None:
        xlim, ylim = get_mask_bounding_box(mask, sim_config.npix, sim_config.dpix)

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    if title:
        fig.suptitle(title, fontsize=16)

    # Configure colormaps to show masked areas as white
    cmap_main = plt.get_cmap('inferno').copy()
    cmap_main.set_bad(color='white')
    cmap_res = plt.get_cmap('RdBu_r').copy()
    cmap_res.set_bad(color='white')

    extent = [
        -sim_config.npix * sim_config.dpix / 2, sim_config.npix * sim_config.dpix / 2,
        -sim_config.npix * sim_config.dpix / 2, sim_config.npix * sim_config.dpix / 2
    ]

    # (0,0): Data
    im0 = axes[0, 0].imshow(data, origin='lower', extent=extent, cmap=cmap_main)
    axes[0, 0].set_title("Observed Data")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # (0,1): Lens Light Model
    im1 = axes[0, 1].imshow(lens_light_model, origin='lower', extent=extent, cmap=cmap_main)
    axes[0, 1].set_title("Lens Light Model")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (0,2): Data - Lens Light
    diff = data - lens_light_model
    if mask is not None:
        diff = np.ma.masked_array(diff, mask=mask)
    im2 = axes[0, 2].imshow(diff, origin='lower', extent=extent, cmap=cmap_main)
    axes[0, 2].set_title("Data - Lens Light")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # (1,0): Lensed Image Model
    im3 = axes[1, 0].imshow(lensed_image_model, origin='lower', extent=extent, cmap=cmap_main)
    axes[1, 0].set_title("Lensed Image Model")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # (1,1): Normalized Residuals
    im4 = axes[1, 1].imshow(res, origin='lower', extent=extent, cmap=cmap_res, vmin=-5, vmax=5)
    axes[1, 1].set_title("Normalized Residuals")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Apply square bounding box to image plane subplots
    if xlim is not None and ylim is not None:
        for i in range(2):
            for j in range(3):
                # Skip the source plane plot (bottom right) as it uses different coordinates
                if i == 1 and j == 2:
                    continue
                axes[i, j].set_xlim(*xlim)
                axes[i, j].set_ylim(*ylim)

    # (1,2): Source Plane
    s_extent = [
        cx - s_npix * s_dpix / 2,
        cx + s_npix * s_dpix / 2,
        cy - s_npix * s_dpix / 2,
        cy + s_npix * s_dpix / 2,
    ]
    src_img = source_plane_image.sum(axis=-1) if source_plane_image.ndim == 3 else source_plane_image
    im5 = axes[1, 2].imshow(src_img, origin="lower", extent=s_extent, cmap=cmap_main)
    axes[1, 2].set_title("Source (Source Plane)")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xlabel("Arcsec")
        ax.set_ylabel("Arcsec")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Plot saved to {save_path}")

    return fig, axes
