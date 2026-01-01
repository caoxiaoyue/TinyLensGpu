"""
Source-only model with position likelihood constraint (Programmatic API).

This demo shows how to model a lensed source with position likelihood constraints.
"""

import os
import pickle
import gzip
import numpy as np
from nautilus import Sampler
import time 

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from TinyLensGpu.Models import ParamU, SersicEllipse, PhysicalModel
from TinyLensGpu.Models.mass import SIE, Shear
from TinyLensGpu.util import load_lens_data
from TinyLensGpu.Models.prior_spec import make_prior_transformation
from TinyLensGpu.Models.likelihood import make_likelihood
from TinyLensGpu.visualizer import plot_model_results
from TinyLensGpu.ProbModel.Image.image_model import ImageProbModel


def build_model():
    """Build lens model with source light and position likelihood."""
    
    print("Loading data...")
    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path="data/image.fits",
        noise_path="data/noise.fits",
        psf_path="data/psf.fits",
    )
    
    print("Building model components...")
    
    # SIE mass profile
    sie = SIE(
        theta_E=ParamU("theta_E", 1.5, prior_type="uniform", 
                       prior_settings=[0.001, 3.001], limits=[0.0, 10.0]),
        e1=ParamU("e1", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        e2=ParamU("e2", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        center_x=ParamU("center_x", 0.0),  # Fixed
        center_y=ParamU("center_y", 0.0),  # Fixed
    )
    
    # External shear
    shear = Shear(
        gamma1=ParamU("gamma1", 0.0, prior_type="uniform", 
                      prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
        gamma2=ParamU("gamma2", 0.0, prior_type="uniform", 
                      prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
    )
    
    # Source Sersic profile
    source = SersicEllipse(
        R_sersic=ParamU("R_sersic", 1.0, prior_type="uniform", 
                        prior_settings=[0.001, 2.001], limits=[0.0, 5.0]),
        n_sersic=ParamU("n_sersic", 1.0, prior_type="uniform", 
                        prior_settings=[0.3, 2.3], limits=[0.3, 6.0]),
        e1=ParamU("e1_src", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        e2=ParamU("e2_src", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        center_x=ParamU("center_x_src", 0.0, prior_type="gaussian", 
                        prior_settings=[0.0, 0.5], limits=[-3.0, 3.0]),
        center_y=ParamU("center_y_src", 0.0, prior_type="gaussian", 
                        prior_settings=[0.0, 0.5], limits=[-3.0, 3.0]),
        Ie=ParamU("Ie", 1.0),  # Linear parameter
    )
    
    # Build physical model directly
    phys_model = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=[source],
        lens_light=[]
    )
    
    # Set dynamic parameters
    sie.theta_E.to_dynamic()
    sie.e1.to_dynamic()
    sie.e2.to_dynamic()
    shear.gamma1.to_dynamic()
    shear.gamma2.to_dynamic()
    source.R_sersic.to_dynamic()
    source.n_sersic.to_dynamic()
    source.e1.to_dynamic()
    source.e2.to_dynamic()
    source.center_x.to_dynamic()
    source.center_y.to_dynamic()
    
    # Position likelihood configuration
    position_likelihood = {
        'positions': [[0.115, -1.071], [0.926, 1.521]],
        'threshold_arcsec': 0.3,
        'min_log_like': -1.0e10
    }
    
    # Build likelihood model
    prob_model = ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=0.074,
        nsub=4,
        phys_model=phys_model,
        use_linear=True,
        mask=mask,
        solver_type='nnls',
        position_likelihood=position_likelihood
    )
    
    return prob_model, phys_model


def run_sampling():
    """Run Nautilus sampling."""
    
    print("="*60)
    print("Source-Only Model with Position Likelihood")
    print("="*60)
    
    # Build model
    likelihood, phys_model = build_model()
    
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
        n_batch=200,
    )
    
    start_time = time.time()
    sampler.run(verbose=True, n_eff=800)
    end_time = time.time()
    
    print(f"\nSampling completed in {(end_time - start_time):.2f} seconds")
    
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
        'likelihood': likelihood,
        'phys_model': phys_model
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
        sorted_idx = np.argsort(samples[:, i])
        sorted_samples = samples[sorted_idx, i]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        cumsum /= cumsum[-1]
        
        q16 = np.interp(0.16, cumsum, sorted_samples)
        q50 = np.interp(0.50, cumsum, sorted_samples)
        q84 = np.interp(0.84, cumsum, sorted_samples)
        
        print(f"  {name:15s} = {q50:.4f} ({q16-q50:+.4f}, {q84-q50:+.4f})")
    
    log_z = results['log_z']
    if isinstance(log_z, (list, tuple, np.ndarray)):
        log_z = log_z[0] if len(log_z) > 0 else 0.0
    print(f"\nlog(Z) = {log_z:.2f}")


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
    
    # Save full results as pickle (exclude non-serializable objects)
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
      
    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)

    # Plot results
    print("\nGenerating visualization...")
    # Get median parameters
    samples = results['samples']
    weights = results['weights']
    param_names = results['param_names']

    q50 = []
    for i in range(len(param_names)):
        sorted_idx = np.argsort(samples[:, i])
        sorted_samples = samples[sorted_idx, i]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        cumsum /= cumsum[-1]
        q50.append(np.interp(0.50, cumsum, sorted_samples))
    
    plot_model_results(
        results['likelihood'], 
        q50, 
        save_path='output/model_visualization.png',
        title="Lens Model Fit Results"
    ) 
