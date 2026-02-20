"""
Pixelized Source Regularization Optimization Demo

This demo shows how to optimize the regularization hyperparameters (scale and coefficient)
for pixelized source reconstruction with a FIXED mass model.

The optimization uses Nautilus nested sampling with Bayesian evidence (log evidence).
"""

import os
import pickle
import gzip
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
from nautilus import Sampler
from TinyLensGpu.visualizer import _plot_irregular_source_voronoi

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.PhysicalModel import PhysicalModel, SersicEllipse, SIE, GaussianEllipse
from TinyLensGpu.utils.geometry import phi_q2_ellipticity
from TinyLensGpu.ForwardSimulation import SimulatorConfig, LensSimulator, make_grid_2d
from TinyLensGpu.PhysicalModel import (
    IrregularGridConfig,
    MappingConfig,
    PixelizedSourceConfig,
    PixelizedSourceModel,
    RegularizationConfig,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import PixelizedImageProbModel
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood


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
    
    sim_config = SimulatorConfig(dpix=dpix, npix=npix, psf_kernel=psf_kernel, nsub=16)
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
    print(f"  Valid pixels: {np.sum(~mask)}")
    
    return {
        'noisy_image': noisy_image,
        'noise_map': noise_map,
        'psf_kernel': psf_kernel,
        'mask': mask,
        'dpix': dpix,
    }


def build_model(data_dict):
    """Build lens model with pixelized source and ParamU hyperparameters."""
    
    print("\nBuilding model components...")
    
    # Define regularization hyperparameters with ParamU
    reg_scale = ParamU(
        "reg_scale", 0.05, prior_type="log_uniform",
        prior_settings=[0.01, 0.5], limits=[0.001, 1.0]
    )
    reg_coefficient = ParamU(
        "reg_coefficient", 1.0, prior_type="log_uniform",
        prior_settings=[0.1, 10.0], limits=[0.01, 100.0]
    )
    
    pix_config = PixelizedSourceConfig(
        grid=IrregularGridConfig(
            n_source_points=1500,
            mesh_alpha=1.5,
            mesh_seed=42,
        ),
        mapping=MappingConfig(
            k_neighbors=5,
            interp_kernel="wendland_c4",
            radius_scale=1.5,
        ),
        regularization=RegularizationConfig(
            scheme="irregular_gp_exp",
        ),
    )
    pix_src_model = PixelizedSourceModel(
        config=pix_config,
        reg_scale=reg_scale,
        reg_coefficient=reg_coefficient,
    )

    e1_l, e2_l = phi_q2_ellipticity(90*np.pi/180, 0.9)
    phys_model = PhysicalModel(
        lens_mass=[
            SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0),
        ],
        source_light=[pix_src_model],
        lens_light=[],
    )
    
    # Set dynamic parameters for sampling
    pix_src_model.reg_scale.to_dynamic()
    pix_src_model.reg_coefficient.to_dynamic()
    
    sim_config = SimulatorConfig(
        dpix=data_dict["dpix"],
        npix=data_dict["noisy_image"].shape[0],
        psf_kernel=data_dict["psf_kernel"],
        mask=data_dict["mask"],
    )

    prob_model = PixelizedImageProbModel(
        image_data=data_dict['noisy_image'],
        noise_map=data_dict['noise_map'],
        sim_config=sim_config,
        phys_model=phys_model,
    )
    
    return prob_model, phys_model


def run_sampling():
    """Run Nautilus sampling for regularization optimization."""
    
    print("="*60)
    print("Regularization Optimization (Nautilus)")
    print("="*60)
    
    # Step 1: Simulate data
    data_dict = simulate_lensing_data()
    
    # Step 2: Build model
    prob_model, phys_model = build_model(data_dict)
    
    # Extract prior transformation and likelihood
    print("\nExtracting prior specifications...")
    prior, prior_specs = make_prior_transformation(prob_model)
    param_names = [spec.name for spec in prior_specs]
    
    print(f"\nModel has {len(param_names)} dynamic parameters:")
    for spec in prior_specs:
        print(f"  {spec.name}: {spec.describe()}")
    
    print("\nCreating likelihood function (log evidence)...")
    loglike = make_likelihood(prob_model, vectorized=False)
    
    # Run sampler
    print("\nRunning Nautilus sampler...")
    sampler = Sampler(
        prior,
        loglike,
        n_dim=len(param_names),
        n_live=100,
        vectorized=False,
    )
    
    sampler.run(verbose=True, n_eff=400)
    
    # Get results
    samples, log_w, log_l = sampler.posterior()
    log_z = float(np.asarray(sampler.log_z))
    weights = np.exp(log_w - np.max(log_w))
    weights /= weights.sum()
    
    return {
        'samples': samples,
        'weights': weights,
        'log_z': log_z,
        'param_names': param_names,
        'sampler': sampler,
        'prob_model': prob_model,
        'phys_model': phys_model,
        'data_dict': data_dict
    }


def summarize_results(results):
    """Print posterior summary."""
    samples = results['samples']
    weights = results['weights']
    param_names = results['param_names']
    
    print("\n" + "="*60)
    print("Posterior Summary")
    print("="*60)
    
    for i, name in enumerate(param_names):
        # We sampled in log space for reg parameters if they are log_uniform
        # ParamU handles the transform, so samples are in the physical space
        sorted_idx = np.argsort(samples[:, i])
        sorted_samples = samples[sorted_idx, i]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        cumsum /= cumsum[-1]
        
        q16 = np.interp(0.16, cumsum, sorted_samples)
        q50 = np.interp(0.50, cumsum, sorted_samples)
        q84 = np.interp(0.84, cumsum, sorted_samples)
        
        print(f"  {name:15s} = {q50:.4f} ({q16-q50:+.4f}, {q84-q50:+.4f})")
    
    print(f"\nlog(Z) = {results['log_z']:.2f}")


def visualize_results(data_dict, results):
    """Visualize reconstruction results with best-fit parameters."""
    print("\n" + "=" * 60)
    print("Step 4: Visualizing Best-Fit Results")
    print("=" * 60)
    
    # 1. Get best-fit parameters (median)
    samples = results['samples']
    weights = results['weights']
    prob_model = results['prob_model']
    
    # Re-get dynamic parameters
    if hasattr(prob_model, 'get_dynamic_params'):
        dynamic_params = prob_model.get_dynamic_params()
    else:
        dynamic_params = prob_model.dynamic_params
    
    print("Setting model to best-fit parameters:")
    for i, param in enumerate(dynamic_params):
        sorted_idx = np.argsort(samples[:, i])
        sorted_samples = samples[sorted_idx, i]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        cumsum /= cumsum[-1]
        median_val = np.interp(0.50, cumsum, sorted_samples)
        
        # Update model parameter
        param.value = median_val
        print(f"  {param.name}: {median_val:.4f}")
        
    # 2. Reconstruct source with best parameters
    # Updated API usage: Call simulator directly for reconstruction
    data_vector = prob_model.image_data[~prob_model.mask]
    noise_variance = prob_model.noise_map[~prob_model.mask] ** 2
    reg_scale = prob_model.pix_src_model.reg_scale.value
    reg_coefficient = prob_model.pix_src_model.reg_coefficient.value
    
    source_intensities, source_mesh_beta, model_image, _ = prob_model.simulator.reconstruct_source(
        data_vector=data_vector,
        noise_variance=noise_variance,
        reg_scale=reg_scale,
        reg_coefficient=reg_coefficient,
        return_2d=True
    )
    
    log_evidence = prob_model.log_evidence()
    print(f"  Log evidence (best fit): {log_evidence:.2f}")
    
    # 3. Plot
    noisy_image = data_dict['noisy_image']
    noise_map = data_dict['noise_map']
    mask = data_dict['mask']
    
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
    
    Log Evidence: {{log_evidence:.2f}}
    
    Source Points: {{len(source_intensities)}}
    Valid Pixels: {{np.sum(~mask)}}
    
    Chi-squared: {{np.sum(((noisy_image[~mask] - model_image[~mask]) / noise_map[~mask])**2):.2f}}
    DOF: {{np.sum(~mask) - len(source_intensities)}}
    
    Residual RMS: {{np.std(residual_image[~mask]):.4f}}
    Normalized RMS: {{np.std(normalized_residual[~mask]):.4f}}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             family='monospace')
    
    plt.tight_layout()
    plt.savefig('optimization_reg_results.png', dpi=300, bbox_inches='tight')
    print("  Saved figure: optimization_results.png")
    plt.show()


def save_results(results):
    """Save results to output directory."""
    os.makedirs('output', exist_ok=True)
    
    print("\nSaving results...")
    
    # Save samples
    np.savetxt('output/result_samples.csv', 
               results['samples'], 
               delimiter=',',
               header=','.join(results['param_names']))
    
    # Save summary
    samples = results['samples']
    weights = results['weights']
    param_names = results['param_names']
    
    with open('output/result_summary.csv', 'w') as f:
        f.write('parameter,median,lower,upper\n')
        for i, name in enumerate(param_names):
            sorted_idx = np.argsort(samples[:, i])
            sorted_samples = samples[sorted_idx, i]
            sorted_weights = weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights)
            cumsum /= cumsum[-1]
            
            q16 = np.interp(0.16, cumsum, sorted_samples)
            q50 = np.interp(0.50, cumsum, sorted_samples)
            q84 = np.interp(0.84, cumsum, sorted_samples)
            
            f.write(f'{name},{q50:.6f},{q16:.6f},{q84:.6f}\n')
    
    # Save full results as pickle
    save_dict = {
        'samples': results['samples'],
        'weights': results['weights'],
        'log_z': results['log_z'],
        'param_names': results['param_names']
    }
    with gzip.open('output/results.pkl.gz', 'wb') as f:
        pickle.dump(save_dict, f)
    
    print("Results saved to output/")


if __name__ == "__main__":
    results = run_sampling()
    summarize_results(results)
    save_results(results)
    
    # Visualize results
    visualize_results(results['data_dict'], results)
    
    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)
