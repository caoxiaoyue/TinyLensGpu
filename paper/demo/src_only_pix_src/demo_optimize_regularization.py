"""
Pixelized Source Regularization Optimization Demo

This demo shows how to optimize the regularization hyperparameters (scale and coefficient)
for pixelized source reconstruction with a FIXED mass model.

The optimization uses dynesty nested sampling with Bayesian evidence (log evidence).
"""

import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
import dynesty
from dynesty import plotting as dyplot

from TinyLensGpu.PhysicalModel import PhysicalModel, SersicEllipse, SIE, Shear, GaussianEllipse
from TinyLensGpu.utils.geometry import phi_q2_ellipticity
from TinyLensGpu.ForwardSimulation import SimulatorConfig, LensSimulator, make_grid_2d

from TinyLensGpu.PhysicalModel import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import PixelizedImageProbModel


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


def create_prob_model(data_dict, reg_scale, reg_coefficient):
    """Create probability model with given regularization parameters."""
    e1_l, e2_l = phi_q2_ellipticity(90*np.pi/180, 0.9)

    pix_src_model = PixelizedSourceModel(
        reg_scale=reg_scale,
        reg_coefficient=reg_coefficient,
        reg_type='exp',
        n_source_points=1500,
        mesh_alpha=1.5,
        mesh_seed=42,
    )

    phys_model = PhysicalModel(
        lens_mass=[
            SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0),
        ],
        source_light=[pix_src_model],
        lens_light=[],
    )
    
    prob_model = PixelizedImageProbModel(
        image_data=data_dict['noisy_image'],
        noise_map=data_dict['noise_map'],
        psf_kernel=data_dict['psf_kernel'],
        dpix=data_dict['dpix'],
        phys_model=phys_model,
        mask=data_dict['mask'],
    )
    
    return prob_model


def optimize_regularization(data_dict):
    """Optimize regularization hyperparameters using dynesty nested sampling."""
    print("\n" + "=" * 60)
    print("Step 2: Optimizing Regularization Hyperparameters")
    print("=" * 60)
    
    # Define prior transform (uniform priors in log space)
    def prior_transform(u):
        """Transform unit cube to parameter space."""
        # u[0]: log(reg_scale) in range [log(0.01), log(0.5)]
        # u[1]: log(reg_coeff) in range [log(0.1), log(10.0)]
        log_reg_scale = u[0] * (np.log(0.5) - np.log(0.01)) + np.log(0.01)
        log_reg_coeff = u[1] * (np.log(10.0) - np.log(0.1)) + np.log(0.1)
        return np.array([log_reg_scale, log_reg_coeff])
    
    # Define log likelihood (which is the log evidence from pixelized source)
    def log_likelihood(params):
        """Log likelihood function for dynesty."""
        log_reg_scale, log_reg_coeff = params
        reg_scale = np.exp(log_reg_scale)
        reg_coefficient = np.exp(log_reg_coeff)
        
        try:
            prob_model = create_prob_model(data_dict, reg_scale, reg_coefficient)
            log_ev = prob_model.log_evidence()
            
            if not np.isfinite(log_ev):
                return -1e10
            
            return log_ev
        except Exception as e:
            print(f"    Error with reg_scale={reg_scale:.4f}, reg_coeff={reg_coefficient:.4f}: {e}")
            return -1e10
    
    print("  Starting dynesty nested sampling...")
    print("  Prior ranges:")
    print("    reg_scale: [0.01, 0.5]")
    print("    reg_coefficient: [0.1, 10.0]")
    
    # Run dynesty sampler
    sampler = dynesty.NestedSampler(
        log_likelihood,
        prior_transform,
        ndim=2,
        nlive=50,
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
    optimal_log_params = samples[max_idx]
    optimal_reg_scale = np.exp(optimal_log_params[0])
    optimal_reg_coeff = np.exp(optimal_log_params[1])
    optimal_log_ev = results.logl[max_idx]
    
    # Get weighted mean parameters
    mean_log_params = np.average(samples, weights=weights, axis=0)
    mean_reg_scale = np.exp(mean_log_params[0])
    mean_reg_coeff = np.exp(mean_log_params[1])
    
    # Get parameter uncertainties
    std_log_params = np.sqrt(np.average((samples - mean_log_params)**2, weights=weights, axis=0))
    
    print(f"\n  Sampling complete!")
    print(f"  Log evidence (from nested sampling): {results.logz[-1]:.2f} ± {results.logzerr[-1]:.2f}")
    print(f"\n  Maximum likelihood parameters:")
    print(f"    reg_scale: {optimal_reg_scale:.4f}")
    print(f"    reg_coefficient: {optimal_reg_coeff:.4f}")
    print(f"    log evidence: {optimal_log_ev:.2f}")
    print(f"\n  Weighted mean parameters:")
    print(f"    reg_scale: {mean_reg_scale:.4f} ± {np.exp(mean_log_params[0] + std_log_params[0]) - mean_reg_scale:.4f}")
    print(f"    reg_coefficient: {mean_reg_coeff:.4f} ± {np.exp(mean_log_params[1] + std_log_params[1]) - mean_reg_coeff:.4f}")
    
    return {
        'optimal_reg_scale': optimal_reg_scale,
        'optimal_reg_coeff': optimal_reg_coeff,
        'optimal_log_ev': optimal_log_ev,
        'mean_reg_scale': mean_reg_scale,
        'mean_reg_coeff': mean_reg_coeff,
        'samples': samples,
        'weights': weights,
        'results': results,
    }


def main():
    """Main demo function."""
    print("\n" + "=" * 60)
    print("Regularization Optimization Demo")
    print("=" * 60)
    
    data_dict = simulate_lensing_data()
    opt_results = optimize_regularization(data_dict)
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
