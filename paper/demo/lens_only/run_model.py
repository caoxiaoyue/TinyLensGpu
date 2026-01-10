"""
Programmatic lens model inference example (no YAML).

This example demonstrates how to build and run lens model inference
programmatically, following the example_v4.py style.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.util import load_lens_data
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from nautilus import Sampler
import jax.numpy as jnp
from TinyLensGpu.visualizer import plot_model_results
from TinyLensGpu.ObservationModel.LensImage import ImageProbModel


def build_problem():
    """Build the lens model problem programmatically."""
    
    # Load data
    print("Loading data...")
    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path='data/image.fits',
        noise_path='data/noise.fits',
        psf_path='data/psf.fits',
        mask_path='data/mask.fits',
    )
    
    # Create lens light component with ParamU parameters
    print("Building model...")
    lens_light = SersicEllipse(
        R_sersic=ParamU("R_sersic", 1.0, 
                        prior_type="uniform", 
                        prior_settings=[0.001, 2.001],
                        limits=[0.0, 5.0]),
        n_sersic=ParamU("n_sersic", 4.0,
                        prior_type="gaussian",
                        prior_settings=[4.0, 0.5],
                        limits=[0.3, 6.0]),
        e1=ParamU("e1", 0.0,
                  prior_type="gaussian",
                  prior_settings=[0.0, 0.3],
                  limits=[-1.0, 1.0]),
        e2=ParamU("e2", 0.0,
                  prior_type="gaussian",
                  prior_settings=[0.0, 0.3],
                  limits=[-1.0, 1.0]),
        center_x=ParamU("center_x", 0.0),
        center_y=ParamU("center_y", 0.0),
        Ie=ParamU("Ie", 1.0),
    )
    
    # Set parameters to dynamic for sampling
    lens_light.R_sersic.to_dynamic()
    lens_light.n_sersic.to_dynamic()
    lens_light.e1.to_dynamic()
    lens_light.e2.to_dynamic()
    
    # Set fixed parameters to static
    lens_light.center_x.to_static(0.0)
    lens_light.center_y.to_static(0.0)
    lens_light.Ie.to_static(1.0)  # Will be solved linearly
    
    # Build physical model directly
    phys_model = PhysicalModel(lens_light=[lens_light])
    
    # Build likelihood (ImageProbModel)
    prob_model = ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=0.074,
        nsub=4,
        phys_model=phys_model,
        use_linear=True,  # Use linear solver for Ie
        mask=mask,
        solver_type='nnls',
    )
    
    return prob_model


def run_sampling():
    """Run sampling following example_v4.py style (lines 292-305)."""
    
    print("="*60)
    print("Programmatic Lens Model Inference (No YAML)")
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
    
    # Run sampler (following example_v4.py lines 303-304)
    print("\nRunning Nautilus sampler...")
    import time
    start_time = time.time()
    sampler = Sampler(
        prior, 
        loglike, 
        n_dim=len(param_names), 
        n_live=200, 
        vectorized=True, 
        n_batch=200
    )
    sampler.run(verbose=True, n_eff=800)
    end_time = time.time()
    print(f"\nSampling completed in {end_time - start_time:.2f} seconds")
    
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

    # Plot results
    print("\nGenerating visualization...")
    # Get median parameters
    q50 = []
    for i in range(len(param_names)):
        sorted_idx = jnp.argsort(samples[:, i])
        sorted_samples = samples[sorted_idx, i]
        sorted_weights = weights[sorted_idx]
        cumsum = jnp.cumsum(sorted_weights)
        cumsum /= cumsum[-1]
        q50.append(jnp.interp(0.50, cumsum, sorted_samples))
    
    plot_model_results(
        likelihood, 
        q50, 
        save_path='output/model_visualization.png',
        title="Lens Model Fit Results"
    ) 
    
    return samples, weights, param_names


if __name__ == "__main__":
    samples, weights, param_names = run_sampling()
