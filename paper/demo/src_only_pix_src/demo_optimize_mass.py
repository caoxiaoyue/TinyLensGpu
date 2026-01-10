"""
Pixelized Source Mass Model Optimization Demo

This demo shows how to optimize the mass model parameters (e.g., Einstein radius, ellipticity)
for pixelized source reconstruction with FIXED regularization hyperparameters.

The optimization uses dynesty nested sampling with Bayesian evidence (log evidence) 
from pixelized source inversion for mass model inference.
"""

import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
import dynesty
from dynesty import plotting as dyplot

from TinyLensGpu.PhysicalModel import PhysicalModel, SersicEllipse, SIE, Shear, GaussianEllipse
from TinyLensGpu.PhysicalModel.LensImage.Parametric.utils import phi_q2_ellipticity
from TinyLensGpu.Simulator import SimulatorConfig, LensSimulator
from TinyLensGpu.Simulator.config import make_grid_2d

from TinyLensGpu.PhysicalModel import PixelizedSourceModel, PixelizedSourceConfig
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import PixelizedImageProbModel


def simulate_lensing_data():
    """Simulate a gravitational lensing observation with known mass parameters."""
    print("=" * 60)
    print("Step 1: Simulating Lensing Data")
    print("=" * 60)
    
    # True mass model parameters
    true_theta_E = 1.5
    true_e1, true_e2 = phi_q2_ellipticity(90*np.pi/180, 0.9)
    
    phy_model = PhysicalModel(
        lens_mass=[
            SIE(theta_E=true_theta_E, e1=true_e1, e2=true_e2, center_x=0.0, center_y=0.0),
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
    print(f"  True theta_E: {true_theta_E:.3f}")
    print(f"  True e1, e2: {true_e1:.3f}, {true_e2:.3f}")
    
    return {
        'noisy_image': noisy_image,
        'noise_map': noise_map,
        'psf_kernel': psf_kernel,
        'mask': mask,
        'dpix': dpix,
        'true_params': {
            'theta_E': true_theta_E,
            'e1': true_e1,
            'e2': true_e2,
        }
    }


def create_prob_model(data_dict, theta_E, e1, e2):
    """Create probability model with given mass parameters and fixed regularization."""
    
    phys_model = PhysicalModel(lens_mass=[
        SIE(theta_E=theta_E, e1=e1, e2=e2, center_x=0.0, center_y=0.0),
    ])
    
    # Fixed regularization parameters
    pix_src_config = PixelizedSourceConfig(
        reg_scale=0.05,
        reg_coefficient=1.0,
        reg_type='exp',
        n_source_points=1500,
        mesh_alpha=1.5,
        mesh_seed=42,
    )
    
    pix_src_model = PixelizedSourceModel(config=pix_src_config)
    
    prob_model = PixelizedImageProbModel(
        image_data=data_dict['noisy_image'],
        noise_map=data_dict['noise_map'],
        psf_kernel=data_dict['psf_kernel'],
        dpix=data_dict['dpix'],
        phys_model=phys_model,
        pix_src_model=pix_src_model,
        mask=data_dict['mask'],
    )
    
    return prob_model


def optimize_mass_model(data_dict):
    """Optimize mass model parameters using dynesty nested sampling."""
    print("\n" + "=" * 60)
    print("Step 2: Optimizing Mass Model Parameters")
    print("=" * 60)
    
    # Define prior transform (uniform priors)
    def prior_transform(u):
        """Transform unit cube to parameter space."""
        # u[0]: theta_E in range [0.8, 2.5]
        # u[1]: e1 in range [-0.4, 0.4]
        # u[2]: e2 in range [-0.4, 0.4]
        theta_E = u[0] * (2.5 - 0.8) + 0.8
        e1 = u[1] * (0.4 - (-0.4)) + (-0.4)
        e2 = u[2] * (0.4 - (-0.4)) + (-0.4)
        return np.array([theta_E, e1, e2])
    
    # Define log likelihood (which is the log evidence from pixelized source)
    def log_likelihood(params):
        """Log likelihood function for dynesty."""
        theta_E, e1, e2 = params
        
        try:
            prob_model = create_prob_model(data_dict, theta_E, e1, e2)
            log_ev = prob_model.log_evidence()
            
            if not np.isfinite(log_ev):
                return -1e10
            
            return log_ev
        except Exception as e:
            print(f"    Error with theta_E={theta_E:.3f}, e1={e1:.3f}, e2={e2:.3f}: {e}")
            return -1e10
    
    print(f"  True values: theta_E={data_dict['true_params']['theta_E']:.3f}, "
          f"e1={data_dict['true_params']['e1']:.3f}, e2={data_dict['true_params']['e2']:.3f}")
    print("  Starting dynesty nested sampling...")
    print("  Prior ranges:")
    print("    theta_E: [0.8, 2.5]")
    print("    e1: [-0.4, 0.4]")
    print("    e2: [-0.4, 0.4]")
    
    # Run dynesty sampler
    sampler = dynesty.NestedSampler(
        log_likelihood,
        prior_transform,
        ndim=3,
        nlive=100,
        bound='multi',
        sample='rwalk',
    )
    
    sampler.run_nested(dlogz=0.5, print_progress=True)
    results = sampler.results
    
    # Extract results
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    
    # Get maximum likelihood parameters
    max_idx = np.argmax(results.logl)
    optimal_params = samples[max_idx]
    optimal_theta_E, optimal_e1, optimal_e2 = optimal_params
    optimal_log_ev = results.logl[max_idx]
    
    # Get weighted mean parameters
    mean_params = np.average(samples, weights=weights, axis=0)
    mean_theta_E, mean_e1, mean_e2 = mean_params
    
    # Get parameter uncertainties (standard deviation)
    std_params = np.sqrt(np.average((samples - mean_params)**2, weights=weights, axis=0))
    std_theta_E, std_e1, std_e2 = std_params
    
    print(f"\n  Sampling complete!")
    print(f"  Log evidence (from nested sampling): {results.logz[-1]:.2f} ± {results.logzerr[-1]:.2f}")
    print(f"\n  Maximum likelihood parameters:")
    print(f"    theta_E: {optimal_theta_E:.4f} (true: {data_dict['true_params']['theta_E']:.4f})")
    print(f"    e1: {optimal_e1:.4f} (true: {data_dict['true_params']['e1']:.4f})")
    print(f"    e2: {optimal_e2:.4f} (true: {data_dict['true_params']['e2']:.4f})")
    print(f"    log evidence: {optimal_log_ev:.2f}")
    print(f"\n  Weighted mean parameters:")
    print(f"    theta_E: {mean_theta_E:.4f} ± {std_theta_E:.4f}")
    print(f"    e1: {mean_e1:.4f} ± {std_e1:.4f}")
    print(f"    e2: {mean_e2:.4f} ± {std_e2:.4f}")
    
    return {
        'optimal_theta_E': optimal_theta_E,
        'optimal_e1': optimal_e1,
        'optimal_e2': optimal_e2,
        'optimal_log_ev': optimal_log_ev,
        'mean_theta_E': mean_theta_E,
        'mean_e1': mean_e1,
        'mean_e2': mean_e2,
        'std_theta_E': std_theta_E,
        'std_e1': std_e1,
        'std_e2': std_e2,
        'samples': samples,
        'weights': weights,
        'results': results,
    }


def reconstruct_with_optimal_mass(data_dict, opt_results):
    """Reconstruct source with optimal mass model."""
    print("\n" + "=" * 60)
    print("Step 3: Source Reconstruction with Optimal Mass Model")
    print("=" * 60)
    
    prob_model = create_prob_model(
        data_dict,
        opt_results['optimal_theta_E'],
        opt_results['optimal_e1'],
        opt_results['optimal_e2']
    )
    
    source_intensities, source_mesh_beta, model_image = prob_model.reconstruct_source()
    
    print(f"  Reconstructed {len(source_intensities)} source pixels")
    print(f"  Model image shape: {model_image.shape}")
    
    return {
        'source_intensities': np.array(source_intensities),
        'source_mesh_beta': np.array(source_mesh_beta),
        'model_image': np.array(model_image),
    }


def visualize_results(data_dict, opt_results, recon_results):
    """Visualize optimization results."""
    print("\n" + "=" * 60)
    print("Step 4: Visualizing Results")
    print("=" * 60)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Observed image
    ax1 = plt.subplot(3, 4, 1)
    img_obs = data_dict['noisy_image'] * (~data_dict['mask']).astype(float)
    im1 = plt.imshow(img_obs, origin='lower', cmap='viridis')
    plt.title('Observed Image', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot 2: Model image
    ax2 = plt.subplot(3, 4, 2)
    im2 = plt.imshow(recon_results['model_image'], origin='lower', cmap='viridis')
    plt.title(f'Model Image\nLog Ev = {opt_results["optimal_log_ev"]:.2f}', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Plot 3: Residuals
    ax3 = plt.subplot(3, 4, 3)
    residuals = (data_dict['noisy_image'] - recon_results['model_image']) * (~data_dict['mask']).astype(float)
    im3 = plt.imshow(residuals, origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    plt.title('Residuals', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Plot 4: Reconstructed source
    ax4 = plt.subplot(3, 4, 4)
    plt.scatter(recon_results['source_mesh_beta'][:, 0], 
                recon_results['source_mesh_beta'][:, 1],
                c=recon_results['source_intensities'], 
                s=20, cmap='hot', marker='o')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Reconstructed Source', fontsize=11, fontweight='bold')
    plt.xlabel('β_x [arcsec]')
    plt.ylabel('β_y [arcsec]')
    plt.axis('equal')
    
    # Plot 5-7: Corner plot for posterior samples
    samples = opt_results['samples']
    weights = opt_results['weights']
    
    # theta_E vs e1
    ax5 = plt.subplot(3, 4, 5)
    plt.scatter(samples[:, 0], samples[:, 1], c=weights, s=5, alpha=0.5, cmap='viridis')
    plt.axvline(data_dict['true_params']['theta_E'], color='r', linestyle='--', label='True')
    plt.axhline(data_dict['true_params']['e1'], color='r', linestyle='--')
    plt.xlabel('theta_E', fontsize=10)
    plt.ylabel('e1', fontsize=10)
    plt.title('theta_E vs e1', fontsize=11, fontweight='bold')
    plt.legend()
    
    # theta_E vs e2
    ax6 = plt.subplot(3, 4, 6)
    plt.scatter(samples[:, 0], samples[:, 2], c=weights, s=5, alpha=0.5, cmap='viridis')
    plt.axvline(data_dict['true_params']['theta_E'], color='r', linestyle='--', label='True')
    plt.axhline(data_dict['true_params']['e2'], color='r', linestyle='--')
    plt.xlabel('theta_E', fontsize=10)
    plt.ylabel('e2', fontsize=10)
    plt.title('theta_E vs e2', fontsize=11, fontweight='bold')
    plt.legend()
    
    # e1 vs e2
    ax7 = plt.subplot(3, 4, 7)
    plt.scatter(samples[:, 1], samples[:, 2], c=weights, s=5, alpha=0.5, cmap='viridis')
    plt.axvline(data_dict['true_params']['e1'], color='r', linestyle='--', label='True')
    plt.axhline(data_dict['true_params']['e2'], color='r', linestyle='--')
    plt.xlabel('e1', fontsize=10)
    plt.ylabel('e2', fontsize=10)
    plt.title('e1 vs e2', fontsize=11, fontweight='bold')
    plt.legend()
    
    # Plot 8: Parameter comparison
    ax8 = plt.subplot(3, 4, 8)
    ax8.axis('off')
    
    true_params = data_dict['true_params']
    stats_text = f"""Parameter Comparison:

theta_E:
  True:  {true_params['theta_E']:.4f}
  Mean:  {opt_results['mean_theta_E']:.4f}±{opt_results['std_theta_E']:.4f}
  ML:    {opt_results['optimal_theta_E']:.4f}

e1:
  True:  {true_params['e1']:.4f}
  Mean:  {opt_results['mean_e1']:.4f}±{opt_results['std_e1']:.4f}
  ML:    {opt_results['optimal_e1']:.4f}

e2:
  True:  {true_params['e2']:.4f}
  Mean:  {opt_results['mean_e2']:.4f}±{opt_results['std_e2']:.4f}
  ML:    {opt_results['optimal_e2']:.4f}
"""
    
    ax8.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center', family='monospace')
    
    # Plot 9-11: 1D posterior histograms
    ax9 = plt.subplot(3, 4, 9)
    plt.hist(samples[:, 0], bins=30, weights=weights, alpha=0.7, color='blue', density=True)
    plt.axvline(data_dict['true_params']['theta_E'], color='r', linestyle='--', linewidth=2, label='True')
    plt.axvline(opt_results['mean_theta_E'], color='g', linestyle='-', linewidth=2, label='Mean')
    plt.xlabel('theta_E', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.title('theta_E Posterior', fontsize=11, fontweight='bold')
    plt.legend(fontsize=8)
    
    ax10 = plt.subplot(3, 4, 10)
    plt.hist(samples[:, 1], bins=30, weights=weights, alpha=0.7, color='blue', density=True)
    plt.axvline(data_dict['true_params']['e1'], color='r', linestyle='--', linewidth=2, label='True')
    plt.axvline(opt_results['mean_e1'], color='g', linestyle='-', linewidth=2, label='Mean')
    plt.xlabel('e1', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.title('e1 Posterior', fontsize=11, fontweight='bold')
    plt.legend(fontsize=8)
    
    ax11 = plt.subplot(3, 4, 11)
    plt.hist(samples[:, 2], bins=30, weights=weights, alpha=0.7, color='blue', density=True)
    plt.axvline(data_dict['true_params']['e2'], color='r', linestyle='--', linewidth=2, label='True')
    plt.axvline(opt_results['mean_e2'], color='g', linestyle='-', linewidth=2, label='Mean')
    plt.xlabel('e2', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.title('e2 Posterior', fontsize=11, fontweight='bold')
    plt.legend(fontsize=8)
    
    # Plot 12: Nested sampling run
    ax12 = plt.subplot(3, 4, 12)
    plt.plot(opt_results['results'].logl, 'b-', alpha=0.5, linewidth=1)
    plt.xlabel('Sample', fontsize=10)
    plt.ylabel('Log Likelihood', fontsize=10)
    plt.title('Nested Sampling Progress', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mass_optimization_results.png', dpi=300, bbox_inches='tight')
    print("  Saved figure: mass_optimization_results.png")
    plt.show()


def main():
    """Main demo function."""
    print("\n" + "=" * 60)
    print("Mass Model Optimization Demo")
    print("=" * 60)
    
    data_dict = simulate_lensing_data()
    opt_results = optimize_mass_model(data_dict)
    recon_results = reconstruct_with_optimal_mass(data_dict, opt_results)
    visualize_results(data_dict, opt_results, recon_results)
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
