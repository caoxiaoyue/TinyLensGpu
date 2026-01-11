"""
Demo: Pixelized Source Reconstruction with TinyLensGpu

This demo shows how to use pixelized source modeling for gravitational lensing
reconstruction. Unlike parametric source models, pixelized source reconstruction
uses discrete pixels in the source plane with Gaussian Process regularization.

The log evidence is computed instead of log likelihood, allowing for:
1. Hyperparameter optimization (regularization scale/coefficient)
2. Mass model parameter inference
3. Full Bayesian inference via nested sampling
"""

import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt

from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse, GaussianEllipse
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE, Shear
from TinyLensGpu.utils.geometry import phi_q2_ellipticity
from TinyLensGpu.ForwardModel import SimulatorConfig, LensSimulator, make_grid_2d

from TinyLensGpu.PhysicalModel.LensImage.Pixelized import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
from TinyLensGpu.visualizer import _plot_irregular_source_voronoi


def simulate_lensing_data():
    """Simulate a gravitational lensing observation."""
    print("=" * 60)
    print("Step 1: Simulating Lensing Data")
    print("=" * 60)
    
    e1_l, e2_l = phi_q2_ellipticity(90*np.pi/180, 0.9)
    
    phy_model = PhysicalModel(
        lens_mass=[
            SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0),
        ],
        source_light=[
            SersicEllipse(R_sersic=0.3, n_sersic=1.0, e1=0.05, e2=0.05, center_x=0.0, center_y=0.3, Ie=1.0)
        ],
        lens_light=[],
    )
    
    npix = 200
    image_size = 10.0
    dpix = image_size / npix
    
    x_psf, y_psf = make_grid_2d(21, dpix)
    psf_kernel = GaussianEllipse(flux=1.0, sigma=0.05, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0).light(x=x_psf, y=y_psf)
    psf_kernel /= psf_kernel.sum()
    psf_kernel = np.asarray(psf_kernel)
    
    sim_config = SimulatorConfig(
        dpix=dpix,
        npix=npix,
        psf_kernel=psf_kernel,
        nsub=16,
    )
    
    sim_obj = LensSimulator(phy_model, sim_config)
    
    img_2d = sim_obj.simulate()
    
    def mock_lens(ideal_image, back_rms, exp_time):
        noise_map = np.sqrt(ideal_image/exp_time + back_rms**2)
        noisy_image = ideal_image + np.random.normal(0, noise_map)
        return noisy_image, noise_map
    
    noisy_image, noise_map = mock_lens(img_2d, 0.1, 300)
    
    xgrid_image, ygrid_image = make_grid_2d(npix, dpix)
    rgrid_image = np.sqrt(xgrid_image**2 + ygrid_image**2)
    mask = (rgrid_image > 2.7)
    
    print(f"  Image size: {npix}x{npix} pixels")
    print(f"  Pixel scale: {dpix:.4f} arcsec/pixel")
    print(f"  Valid pixels: {np.sum(~mask)}")
    print(f"  SNR (median): {np.median(noisy_image[~mask] / noise_map[~mask]):.2f}")
    
    return {
        'noisy_image': noisy_image,
        'noise_map': noise_map,
        'psf_kernel': psf_kernel,
        'mask': mask,
        'dpix': dpix,
    }


def setup_pixelized_model(data_dict):
    """Set up pixelized source model with known mass parameters."""
    print("\n" + "=" * 60)
    print("Step 2: Setting Up Pixelized Source Model")
    print("=" * 60)
    
    e1_l, e2_l = phi_q2_ellipticity(90*np.pi/180, 0.9)
    
    sie = SIE(
        theta_E=1.5,
        e1=e1_l,
        e2=e2_l,
        center_x=0.0,
        center_y=0.0,
    )
    
    phys_model = PhysicalModel(lens_mass=[sie])
    
    pix_src_model = PixelizedSourceModel(
        reg_scale=0.05,
        reg_coefficient=1.0,
        reg_type='exp',
        n_source_points=1500,
        mesh_alpha=1.5,
        mesh_blur_sigma=0.0,
        mesh_method='random',
        mesh_seed=42,
        k_neighbors=5,
        interp_kernel='wendland_c4',
        radius_scale=1.5,
    )
    
    prob_model = PixelizedImageProbModel(
        image_data=data_dict['noisy_image'],
        noise_map=data_dict['noise_map'],
        psf_kernel=data_dict['psf_kernel'],
        dpix=data_dict['dpix'],
        phys_model=phys_model,
        pix_src_model=pix_src_model,
        mask=data_dict['mask'],
    )
    
    print(f"  Physical model: {phys_model.get_component_counts()}")
    print(f"  Pixelized source: {pix_src_model}")
    print(f"  Source mesh points: {pix_src_model.n_source_points}")
    print(f"  Regularization: scale={pix_src_model.reg_scale.value}, coeff={pix_src_model.reg_coefficient.value}")
    
    return prob_model


def reconstruct_source(prob_model):
    """Reconstruct source and compute log evidence."""
    print("\n" + "=" * 60)
    print("Step 3: Source Reconstruction")
    print("=" * 60)
    
    log_ev = prob_model.log_evidence()
    print(f"  Log evidence: {log_ev:.2f}")
    
    source_intensities, source_mesh_beta, model_image = prob_model.reconstruct_source()
    
    print(f"  Source intensities shape: {source_intensities.shape}")
    print(f"  Source mesh beta shape: {source_mesh_beta.shape}")
    print(f"  Model image shape: {model_image.shape}")
    
    return {
        'log_evidence': log_ev,
        'source_intensities': np.array(source_intensities),
        'source_mesh_beta': np.array(source_mesh_beta),
        'model_image': np.array(model_image),
    }


def visualize_results(data_dict, results):
    """Visualize reconstruction results."""
    print("\n" + "=" * 60)
    print("Step 4: Visualizing Results")
    print("=" * 60)
    
    noisy_image = data_dict['noisy_image']
    noise_map = data_dict['noise_map']
    mask = data_dict['mask']
    
    model_image = results['model_image']
    source_intensities = results['source_intensities']
    source_mesh_beta = results['source_mesh_beta']
    log_evidence = results['log_evidence']
    
    fig = plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    img_obs = noisy_image * (~mask).astype(float)
    im1 = plt.imshow(img_obs, origin='lower', cmap='viridis')
    plt.title('Observed Noisy Image', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.xlabel('x [pixels]', fontsize=10)
    plt.ylabel('y [pixels]', fontsize=10)
    
    ax2 = plt.subplot(2, 3, 2)
    im2 = plt.imshow(model_image, origin='lower', cmap='viridis')
    plt.title(f'Model Image\nLog Evidence = {log_evidence:.2f}', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    plt.xlabel('x [pixels]', fontsize=10)
    plt.ylabel('y [pixels]', fontsize=10)
    
    ax3 = plt.subplot(2, 3, 3)
    residual_image = np.zeros_like(noisy_image)
    residual_image[~mask] = noisy_image[~mask] - model_image[~mask]
    vmax_res = np.max(np.abs(residual_image))
    im3 = plt.imshow(residual_image, origin='lower', cmap='RdBu_r', 
                     vmin=-vmax_res, vmax=vmax_res)
    plt.title('Residual (Data - Model)', fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    plt.xlabel('x [pixels]', fontsize=10)
    plt.ylabel('y [pixels]', fontsize=10)
    
    ax4 = plt.subplot(2, 3, 4)
    normalized_residual = np.zeros_like(noisy_image)
    normalized_residual[~mask] = (noisy_image[~mask] - model_image[~mask]) / noise_map[~mask]
    vmax_norm = 3.0
    im4 = plt.imshow(normalized_residual, origin='lower', cmap='RdBu_r',
                     vmin=-vmax_norm, vmax=vmax_norm)
    plt.title('Normalized Residual (σ units)', fontsize=13, fontweight='bold')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='σ')
    plt.xlabel('x [pixels]', fontsize=10)
    plt.ylabel('y [pixels]', fontsize=10)
    
    ax5 = plt.subplot(2, 3, 5)
    _plot_irregular_source_voronoi(
        ax5,
        source_mesh_beta,
        source_intensities,
        cmap='viridis'
    )
    ax5.set_title('Source Reconstruction', fontsize=13, fontweight='bold')
    ax5.set_xlabel('β₁ [arcsec]', fontsize=10)
    ax5.set_ylabel('β₂ [arcsec]', fontsize=10)
    plt.grid(True, alpha=0.2)
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    Reconstruction Statistics:
    
    Log Evidence: {log_evidence:.2f}
    
    Source Points: {len(source_intensities)}
    Valid Pixels: {np.sum(~mask)}
    
    Chi-squared: {np.sum(((noisy_image[~mask] - model_image[~mask]) / noise_map[~mask])**2):.2f}
    DOF: {np.sum(~mask) - len(source_intensities)}
    
    Residual RMS: {np.std(residual_image[~mask]):.4f}
    Normalized RMS: {np.std(normalized_residual[~mask]):.4f}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             family='monospace')
    
    plt.tight_layout()
    plt.savefig('pixelized_source_reconstruction.png', dpi=300, bbox_inches='tight')
    print("  Saved figure: pixelized_source_reconstruction.png")
    plt.show()


def main():
    """Main demo function."""
    print("\n" + "=" * 60)
    print("Pixelized Source Reconstruction Demo")
    print("=" * 60)
    
    data_dict = simulate_lensing_data()
    
    prob_model = setup_pixelized_model(data_dict)
    
    results = reconstruct_source(prob_model)
    
    visualize_results(data_dict, results)
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Pixelized source uses discrete pixels instead of parametric profiles")
    print("  2. Log evidence (not likelihood) is computed for Bayesian inference")
    print("  3. Regularization hyperparameters control source smoothness")
    print("  4. Source mesh is adaptively generated based on image brightness")
    print("  5. Can be used for hyperparameter optimization and mass model inference")


if __name__ == "__main__":
    main()
