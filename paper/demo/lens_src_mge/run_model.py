"""
MGE (Multi-Gaussian Expansion) lens light + source light modeling example.

This example demonstrates how to build an MGE model for lens light
combined with a Sersic source light profile, programmatically.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import GaussianEllipse
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE, Shear
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from nautilus import Sampler
import jax.numpy as jnp
from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import ImageProbModel
from TinyLensGpu.visualizer import plot_model_results


if __name__ == "__main__":
    print("="*60)
    print("MGE Lens Light + Source Light Model Inference")
    print("="*60)

    # 1. Load data
    print("\n[Stage 1] Loading data...")
    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path='data/image.fits',
        noise_path='data/noise.fits',
        psf_path='data/psf.fits',
    )
    
    # 2. Build model components
    print("\n[Stage 2] Building model components...")
    
    # SIE mass profile
    sie = SIE(
        theta_E=ParamU("theta_E", 1.5, prior_type="uniform", 
                       prior_settings=[0.001, 3.001], limits=[0.0, 10.0]),
        e1=ParamU("e1_mass", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        e2=ParamU("e2_mass", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        center_x=ParamU("center_x_mass", 0.0),
        center_y=ParamU("center_y_mass", 0.0),
    )
    
    # External shear
    shear = Shear(
        gamma1=ParamU("gamma1", 0.0, prior_type="uniform", 
                      prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
        gamma2=ParamU("gamma2", 0.0, prior_type="uniform", 
                      prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
    )
    
    # Set mass parameters to dynamic
    sie.theta_E.to_dynamic()
    sie.e1.to_dynamic()
    sie.e2.to_dynamic()
    shear.gamma1.to_dynamic()
    shear.gamma2.to_dynamic()
    
    # Build MGE source light model
    print("Building MGE source light model...")
    N_gaussians_src = 10
    sigma_list_src = 10**(np.linspace(-2.0, np.log10(1.0), N_gaussians_src))
    
    # Create shared geometric parameters for source MGE
    center_x_src = ParamU("center_x_src", 0.0,
                          prior_type="gaussian",
                          prior_settings=[0.0, 0.5],
                          limits=[-3.0, 3.0])
    center_y_src = ParamU("center_y_src", 0.0,
                          prior_type="gaussian",
                          prior_settings=[0.0, 0.5],
                          limits=[-3.0, 3.0])
    e1_src = ParamU("e1_src", 0.0,
                    prior_type="gaussian",
                    prior_settings=[0.0, 0.3],
                    limits=[-1.0, 1.0])
    e2_src = ParamU("e2_src", 0.0,
                    prior_type="gaussian",
                    prior_settings=[0.0, 0.3],
                    limits=[-1.0, 1.0])
    
    # Create list of Gaussian components for source light
    source_gaussians = []
    for i, sigma in enumerate(sigma_list_src):
        gauss = GaussianEllipse(
            sigma=ParamU(f"sigma_src_{i}", float(sigma)),  # Fixed sigma
            center_x=center_x_src,  # Shared parameter
            center_y=center_y_src,  # Shared parameter
            e1=e1_src,  # Shared parameter
            e2=e2_src,  # Shared parameter
            flux=ParamU(f"flux_src_{i}", 1.0),  # Linear parameter
        )
        
        # Set sigma to static (fixed)
        gauss.sigma.to_static(float(sigma))
        
        # Set flux to static (will be solved linearly)
        gauss.flux.to_static(1.0)
        
        source_gaussians.append(gauss)
    
    # Set shared geometric parameters to dynamic (only once)
    center_x_src.to_dynamic()
    center_y_src.to_dynamic()
    e1_src.to_dynamic()
    e2_src.to_dynamic()
    
    print(f"Created {N_gaussians_src} Gaussian components for source light")
    
    # Build MGE lens light model
    print("Building MGE lens light model...")
    N_gaussians_lens = 10
    sigma_list_lens = 10**(np.linspace(-2.0, np.log10(3.0), N_gaussians_lens))
    
    # Create shared geometric parameters for MGE
    center_x_lens = ParamU("center_x_lens", 0.0,
                           prior_type="gaussian",
                           prior_settings=[0.0, 0.1],
                           limits=[-3.0, 3.0])
    center_y_lens = ParamU("center_y_lens", 0.0,
                           prior_type="gaussian",
                           prior_settings=[0.0, 0.1],
                           limits=[-3.0, 3.0])
    e1_lens = ParamU("e1_lens", 0.0,
                     prior_type="gaussian",
                     prior_settings=[0.0, 0.3],
                     limits=[-1.0, 1.0])
    e2_lens = ParamU("e2_lens", 0.0,
                     prior_type="gaussian",
                     prior_settings=[0.0, 0.3],
                     limits=[-1.0, 1.0])
    
    # Create list of Gaussian components for lens light
    lens_gaussians = []
    for i, sigma in enumerate(sigma_list_lens):
        gauss = GaussianEllipse(
            sigma=ParamU(f"sigma_lens_{i}", float(sigma)),  # Fixed sigma
            center_x=center_x_lens,  # Shared parameter
            center_y=center_y_lens,  # Shared parameter
            e1=e1_lens,  # Shared parameter
            e2=e2_lens,  # Shared parameter
            flux=ParamU(f"flux_lens_{i}", 1.0),  # Linear parameter
        )
        
        # Set sigma to static (fixed)
        gauss.sigma.to_static(float(sigma))
        
        # Set flux to static (will be solved linearly)
        gauss.flux.to_static(1.0)
        
        lens_gaussians.append(gauss)
    
    # Set shared geometric parameters to dynamic (only once)
    center_x_lens.to_dynamic()
    center_y_lens.to_dynamic()
    e1_lens.to_dynamic()
    e2_lens.to_dynamic()
    
    print(f"Created {N_gaussians_lens} Gaussian components for lens light")
    
    # Build Physical Model
    phys_model = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=source_gaussians,
        lens_light=lens_gaussians
    )
    
    # Build likelihood
    likelihood = ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=0.074,
        nsub=4,
        phys_model=phys_model,
        use_linear=True,  # Use linear solver for flux parameters
        mask=mask,
        solver_type='nnls'  # Non-negative least squares
    )

    # 3. Extract prior and setup likelihood
    print("\n[Stage 3] Extracting prior specifications...")
    prior, prior_specs = make_prior_transformation(likelihood)
    param_names = [spec.name for spec in prior_specs]
    
    print(f"\nModel has {len(param_names)} dynamic parameters:")
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
        print(f"  {name:15s} = {q50:.4f} ({q16-q50:+.4f}, {q84-q50:+.4f})")
    
    # 6. Save results
    print("\n[Stage 6] Saving results...")
    if not os.path.exists('output'):
        os.makedirs('output')
        
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

    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)
