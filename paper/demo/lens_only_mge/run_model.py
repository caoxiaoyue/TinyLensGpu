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
from TinyLensGpu.ObservationModel.LensImage.image_model import ImageProbModel


def build_problem():
    """Build the MGE lens model problem programmatically."""
    
    # Load data
    print("Loading data...")
    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path='data/image.fits',
        noise_path='data/noise.fits',
        psf_path='data/psf.fits',
    )
    
    # MGE configuration: 15 Gaussian components with logarithmically spaced sigmas
    print("Building MGE model...")
    N_gaussians = 15
    sigma_list = 10**(np.linspace(-2.0, np.log10(3.0), N_gaussians))
    
    # Create shared geometric parameters (only for the first Gaussian)
    # All other Gaussians will reference these
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
            sigma=ParamU(f"sigma_{i}", float(sigma)),  # Fixed sigma
            center_x=center_x_shared,  # Shared parameter
            center_y=center_y_shared,  # Shared parameter
            e1=e1_shared,  # Shared parameter
            e2=e2_shared,  # Shared parameter
            flux=ParamU(f"flux_{i}", 1.0),  # Linear parameter
        )
        
        # Set sigma to static (fixed)
        gauss.sigma.to_static(float(sigma))
        
        # Set flux to static (will be solved linearly)
        gauss.flux.to_static(1.0)
        
        gaussians.append(gauss)
    
    # Set shared geometric parameters to dynamic (only once)
    center_x_shared.to_dynamic()
    center_y_shared.to_dynamic()
    e1_shared.to_dynamic()
    e2_shared.to_dynamic()
    
    print(f"Created {N_gaussians} Gaussian components")
    print(f"Sigma range: {sigma_list[0]:.4f} to {sigma_list[-1]:.4f} arcsec")
    
    # Build physical model directly
    phys_model = PhysicalModel(lens_light=gaussians)
    
    # Build likelihood
    prob_model = ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=0.074,
        nsub=4,
        phys_model=phys_model,
        use_linear=True,  # Use linear solver for flux parameters
        mask=mask,
        solver_type='nnls'  # Non-negative least squares (recommended for MGE)
    )
    
    return prob_model


def run_sampling():
    """Run sampling for MGE model."""
    
    print("="*60)
    print("MGE Lens Light Model Inference")
    print("="*60)
    
    # Build problem
    likelihood = build_problem()
    
    # Extract prior transformation
    print("\nExtracting prior specifications...")
    prior, prior_specs = make_prior_transformation(likelihood)
    param_names = [spec.name for spec in prior_specs]
    
    print(f"\nModel has {len(param_names)} dynamic parameters:")
    for spec in prior_specs:
        print(f"  {spec.name}: {spec.describe()}")
    
    # Create likelihood function
    print("\nCreating likelihood function...")
    loglike = make_likelihood(likelihood, vectorized=True)
    
    # Run sampler
    print("\nRunning Nautilus sampler...")
    sampler = Sampler(
        prior, 
        loglike, 
        n_dim=len(param_names), 
        n_live=200, 
        vectorized=True, 
        n_batch=200
    )
    sampler.run(verbose=True, n_eff=800)
    
    # Process results
    print("\nProcessing results...")
    samples, log_w, _ = sampler.posterior()
    samples = jnp.asarray(samples, dtype=jnp.float32)
    weights = jnp.exp(log_w)
    weights /= weights.sum()
    
    # Print summary
    print("\n" + "="*60)
    print("Posterior Summary")
    print("="*60)
    for i, name in enumerate(param_names):
        q16, q50, q84 = jnp.percentile(samples[:, i], jnp.array([16, 50, 84]))
        print(f"  {name:12s} = {q50:.3f} (-{q50-q16:.3f}, +{q84-q50:.3f})")
    
    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)
    
    return samples, weights, param_names


if __name__ == "__main__":
    samples, weights, param_names = run_sampling()
