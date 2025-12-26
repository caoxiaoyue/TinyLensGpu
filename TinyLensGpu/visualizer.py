"""
Visualization module for lens model fitting results.
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from TinyLensGpu.Simulator.config import make_grid_2d
import copy

def plot_model_results(likelihood_obj, theta, save_path=None, title=None):
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

    # Extract everything in one go via the @ck.forward mechanism
    lensed_image_model, lens_light_model = likelihood_obj.forward_model(
        theta,
        use_linear=likelihood_obj.use_linear,
        return_intensity=False,
        ret_each_plane=True,
        image_map=likelihood_obj.image_data,
        noise_map=likelihood_obj.noise_map,
    )
    
    # Extract components for easier access
    sim_config = likelihood_obj.sim_obj.sim_config
    
    # Data components
    data = np.asarray(likelihood_obj.image_data)
    noise = np.asarray(likelihood_obj.noise_map)
    total_model = np.asarray(lensed_image_model+lens_light_model)
    
    # Source plane visualization
    # Create a grid in the source plane
    # Find the extent of the source
    # Note: since we use @ck.forward, we can't easily access the values in the modules 
    # if they were set via theta temporarily. 
    # But usually, it's fine to just center it.
    cx, cy = 0.0, 0.0
    
    # Use a smaller grid for source plane to see the source better
    s_dpix = sim_config.dpix / 2.0 # Higher resolution
    s_npix = 3.0/s_dpix 
    sx, sy = make_grid_2d(s_npix, s_dpix, 1)
    sx = jnp.array(sx) + cx
    sy = jnp.array(sy) + cy
    
    # Get source plane (per component, theta-aware via @ck.forward)
    likelihood_obj_tmp = copy.deepcopy(likelihood_obj)
    n_src = len(likelihood_obj_tmp.sim_obj.phys_model.source_light)
    source_plane_image = jnp.zeros_like(sx)
    source_plane_image = jnp.repeat(source_plane_image[..., jnp.newaxis], n_src, axis=-1)
    for i, light_model in enumerate(likelihood_obj_tmp.sim_obj.phys_model.source_light):
        source_plane_image = source_plane_image.at[..., i].set(light_model.light(theta, x=sx, y=sy))
    source_plane_image = np.asarray(source_plane_image)

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
    s_extent = [
        cx - s_npix * s_dpix / 2, cx + s_npix * s_dpix / 2,
        cy - s_npix * s_dpix / 2, cy + s_npix * s_dpix / 2
    ]
    im5 = axes[1, 2].imshow(source_plane_image.sum(axis=-1), origin='lower', extent=s_extent, cmap='inferno')
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
