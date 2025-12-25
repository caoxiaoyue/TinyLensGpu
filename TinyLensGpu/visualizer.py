"""
Visualization module for lens model fitting results.
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import os
from TinyLensGpu.Simulator.lens_simulator import bin_image_general

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
    return #disable this module temporarily for debugging
    # Extract everything in one go via the @ck.forward mechanism
    image_model, intensity_list, img_lens_sub, img_arc_sub = likelihood_obj.get_all_components(theta)
    
    # Extract components for easier access
    prob_model = likelihood_obj.prob_model
    sim_obj = prob_model.sim_obj
    phys_model = sim_obj.phys_model
    sim_config = sim_obj.sim_config
    
    # Extract individual components
    n_src = len(phys_model.source_light)
    n_lens_light = len(phys_model.lens_light)
    
    psf_kernel = sim_obj.psf_kernel
    
    # Bin and convolve components
    img_lens = bin_image_general(img_lens_sub, sim_config.nsub)
    img_arc = bin_image_general(img_arc_sub, sim_config.nsub)
    
    # If linear solver was used, we need to apply the intensities
    if intensity_list is not None:
        # intensity_list contains [source1, source2, ..., lens1, lens2, ...]
        # based on LinearSolver/linear_solver.py and LensSimulator._simulate_linear
        # X_vec = [I_src, I_lens]
        # Actually, let's check prepare_linear_system
        # D_vec = [img_arc0_conv, img_arc1_conv, ..., img_lens0_conv, img_lens1_conv, ...]
        # In _simulate_linear: img_components = jnp.concatenate([img_arc_convolved, img_lens_convolved], axis=-1)
        # So X_vec order is [sources..., lenses...]
        
        # Convolve each component
        img_lens_conv = np.zeros_like(img_lens)
        for i in range(n_lens_light):
            conv = jsp.signal.fftconvolve(img_lens[..., i], psf_kernel, mode='same')
            img_lens_conv[..., i] = conv * intensity_list[n_src + i]
            
        img_arc_conv = np.zeros_like(img_arc)
        for i in range(n_src):
            conv = jsp.signal.fftconvolve(img_arc[..., i], psf_kernel, mode='same')
            img_arc_conv[..., i] = conv * intensity_list[i]
            
        lens_light_model = np.sum(img_lens_conv, axis=-1)
        lensed_image_model = np.sum(img_arc_conv, axis=-1)
    else:
        # Nonlinear case (intensities are already in the light models or set via parameters)
        # Convolve components
        lens_light_model = np.zeros((sim_config.npix, sim_config.npix))
        for i in range(n_lens_light):
            lens_light_model += jsp.signal.fftconvolve(img_lens[..., i], psf_kernel, mode='same')
            
        lensed_image_model = np.zeros((sim_config.npix, sim_config.npix))
        for i in range(n_src):
            lensed_image_model += jsp.signal.fftconvolve(img_arc[..., i], psf_kernel, mode='same')

    # Data components
    data = np.asarray(prob_model.image_data)
    noise = np.asarray(prob_model.noise_map)
    total_model = np.asarray(image_model)
    
    # Source plane visualization
    # Create a grid in the source plane
    from TinyLensGpu.Simulator.config import make_grid_2d
    # Find the extent of the source
    # Note: since we use @ck.forward, we can't easily access the values in the modules 
    # if they were set via theta temporarily. 
    # But usually, it's fine to just center it.
    cx, cy = 0.0, 0.0
    
    # Use a smaller grid for source plane to see the source better
    s_npix = 100
    s_dpix = sim_config.dpix / 2.0 # Higher resolution
    sx, sy = make_grid_2d(s_npix, s_dpix, 1)
    sx = jnp.array(sx) + cx
    sy = jnp.array(sy) + cy
    
    # Get source plane via @ck.forward
    source_plane = likelihood_obj.get_source_plane(sx, sy, theta)
    source_plane = np.asarray(source_plane)

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
    fig.colorbar(im0, ax=axes[0,0])
    
    # (0,1): Lens Light Model
    im1 = axes[0, 1].imshow(lens_light_model, origin='lower', extent=extent, cmap='inferno')
    axes[0, 1].set_title("Lens Light Model")
    fig.colorbar(im1, ax=axes[0,1])
    
    # (0,2): Data - Lens Light
    im2 = axes[0, 2].imshow(data - lens_light_model, origin='lower', extent=extent, cmap='inferno')
    axes[0, 2].set_title("Data - Lens Light")
    fig.colorbar(im2, ax=axes[0,2])
    
    # (1,0): Lensed Image Model
    im3 = axes[1, 0].imshow(lensed_image_model, origin='lower', extent=extent, cmap='inferno')
    axes[1, 0].set_title("Lensed Image Model")
    fig.colorbar(im3, ax=axes[1,0])
    
    # (1,1): Normalized Residuals
    # Residuals = (Data - (Lens + Source)) / Noise
    res = (data - total_model) / noise
    im4 = axes[1, 1].imshow(res, origin='lower', extent=extent, cmap='RdBu_r', vmin=-5, vmax=5)
    axes[1, 1].set_title("Normalized Residuals")
    fig.colorbar(im4, ax=axes[1,1])
    
    # (1,2): Source Plane
    s_extent = [
        cx - s_npix * s_dpix / 2, cx + s_npix * s_dpix / 2,
        cy - s_npix * s_dpix / 2, cy + s_npix * s_dpix / 2
    ]
    im5 = axes[1, 2].imshow(source_plane, origin='lower', extent=s_extent, cmap='inferno')
    axes[1, 2].set_title("Source (Source Plane)")
    fig.colorbar(im5, ax=axes[1,2])
    
    for ax in axes.flat:
        ax.set_xlabel("Arcsec")
        ax.set_ylabel("Arcsec")
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Plot saved to {save_path}")
        
    return fig, axes
