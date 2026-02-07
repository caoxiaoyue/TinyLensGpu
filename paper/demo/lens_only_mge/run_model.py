"""
MGE (Multi-Gaussian Expansion) lens light modeling example.

This example demonstrates how to build an MGE model programmatically,
where multiple Gaussian components share common geometric parameters.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.PhysicalModel import PhysicalModel, GaussianEllipse
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from nautilus import Sampler
import jax.numpy as jnp
from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import ImageProbModel
from TinyLensGpu.visualizer import plot_model_results


if __name__ == "__main__":
    print("="*60)
    print("MGE Lens Light Model Inference")
    print("="*60)

    # 1. Load data
    print("\n[Stage 1] Loading data...")
    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path='data/image.fits',
        noise_path='data/noise.fits',
        psf_path='data/psf.fits',
    )
    
    # 2. Build MGE model components
    print("\n[Stage 2] Building MGE model...")
    N_gaussians = 15
    sigma_list = 10**(np.linspace(-2.0, np.log10(3.0), N_gaussians))
    
    # Create shared geometric parameters
    center_x_shared = ParamU("center_x", 0.0,
                             prior_type="gaussian",
                             prior_settings=[0.0, 0.1],
                             limits=[-3.0, 3.0])
    center_y_shared = ParamU("center_y", 0.0,
                             prior_type="gaussian",
                             prior_settings=[0.0, 0.1],
                             limits=[-3.0, 3.0])
    e1_shared = ParamU("e1", 0.0,
                       prior_type="gaussian",
                       prior_settings=[0.0, 0.3],
                       limits=[-1.0, 1.0])
    e2_shared = ParamU("e2", 0.0,
                       prior_type="gaussian",
                       prior_settings=[0.0, 0.3],
                       limits=[-1.0, 1.0])
    
    # Create list of Gaussian components
    gaussians = []
    for i, sigma in enumerate(sigma_list):
        gauss = GaussianEllipse(
            sigma=ParamU(f"sigma_{i}", float(sigma)),
            center_x=center_x_shared,
            center_y=center_y_shared,
            e1=e1_shared,
            e2=e2_shared,
            flux=ParamU(f"flux_{i}", 1.0),
        )
        gauss.sigma.to_static(float(sigma))
        gauss.flux.to_static(1.0)
        gaussians.append(gauss)
    
    # Set shared parameters to dynamic
    center_x_shared.to_dynamic()
    center_y_shared.to_dynamic()
    e1_shared.to_dynamic()
    e2_shared.to_dynamic()
    
    print(f"Created {N_gaussians} Gaussian components")
    print(f"Sigma range: {sigma_list[0]:.4f} to {sigma_list[-1]:.4f} arcsec")
    
    # Build physical model directly
    phys_model = PhysicalModel(lens_light=gaussians)
    
    # Build likelihood (ImageProbModel)
    likelihood = ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=0.074,
        nsub=4,
        phys_model=phys_model,
        use_linear=True,
        mask=mask,
        solver_type='nnls'
    )

    # 3. Extract prior and setup likelihood
    print("\n[Stage 3] Extracting prior specifications...")
    prior, prior_specs = make_prior_transformation(likelihood)
    param_names = [spec.name for spec in prior_specs]
    
    print(f"Model has {len(param_names)} dynamic parameters:")
    for spec in prior_specs:
        print(f"  {spec.name}: {spec.describe()}")
    
    print("\nCreating likelihood function...")
    loglike = make_likelihood(likelihood, vectorized=True)

    # 4. Run sampling
    print("\n[Stage 4] Running Nautilus sampler...")
    sampler = Sampler(
        prior, 
        loglike, 
        n_dim=len(param_names), 
        n_live=200, 
        vectorized=True, 
        n_batch=200
    )
    sampler.run(verbose=True, n_eff=800)
    
    # 5. Process results and summary
    print("\n[Stage 5] Processing results...")
    samples, log_w, _ = sampler.posterior()
    samples = jnp.asarray(samples, dtype=jnp.float32)
    weights = jnp.exp(log_w)
    weights /= weights.sum()
    
    print("\n" + "="*60)
    print("Posterior Summary")
    print("="*60)
    q16_list, q50_list, q84_list = [], [], []
    for i, name in enumerate(param_names):
        q16, q50, q84 = jnp.percentile(samples[:, i], jnp.array([16, 50, 84]))
        q16_list.append(float(q16))
        q50_list.append(float(q50))
        q84_list.append(float(q84))
        print(f"  {name:12s} = {q50:.4f} ({q16-q50:+.4f}, {q84-q50:+.4f})")
    
    # 6. Save results
    print("\n[Stage 6] Saving results...")
    os.makedirs('output', exist_ok=True)
    
    np.savetxt('output/result_samples.csv', 
               samples, 
               delimiter=',',
               header=','.join(param_names))
    
    with open('output/result_summary.csv', 'w') as f:
        f.write('parameter,median,lower,upper\n')
        for i, name in enumerate(param_names):
            f.write(f'{name},{q50_list[i]:.6f},{q16_list[i]:.6f},{q84_list[i]:.6f}\n')
    
    print("Results saved to output/")

    # 7. Visualization
    print("\n[Stage 7] Generating visualization...")
    plot_model_results(
        likelihood, 
        jnp.array(q50_list), 
        save_path='output/model_visualization.png',
        title="Lens Model Fit Results"
    ) 

    # 8. Model median (including linear parameters)
    model_median = likelihood.get_linear_solved_params(q50_list)
    print("\nPosterior median with linear-solved light amplitude:")
    print(model_median)

    print(f"\nLog evidence: {sampler.evidence:.4f}")

    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)
