"""
Source-only model with position likelihood constraint (Programmatic API).

This demo shows how to model a lensed source with position likelihood constraints.
"""

import os
import pickle
import gzip
import numpy as np
import jax.numpy as jnp
from nautilus import Sampler
import time 

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from TinyLensGpu.Inference import ParamU, nautilus_posterior_summary
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE, Shear
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.visualizer import plot_model_results
from TinyLensGpu.ObservationModel.LensImage import ImageProbModel


if __name__ == "__main__":
    print("="*60)
    print("Source-Only Model with Position Likelihood")
    print("="*60)

    # 1. Load data
    print("\n[Stage 1] Loading data...")
    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path="data/image.fits",
        noise_path="data/noise.fits",
        psf_path="data/psf.fits",
    )
    
    # 2. Build model components
    print("\n[Stage 2] Building model components...")
    
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
    likelihood = ImageProbModel(
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
        n_batch=200,
    )
    
    start_time = time.time()
    sampler.run(verbose=True, n_eff=800)
    end_time = time.time()
    print(f"\nSampling completed in {(end_time - start_time):.2f} seconds")

    # 5. Process results and summary
    print("\n[Stage 5] Processing results...")
    samples, weights, quantiles, log_z = nautilus_posterior_summary(sampler, param_names)
    q16_list = [float(qs[0]) for qs in quantiles.values()]
    q50_list = [float(qs[1]) for qs in quantiles.values()]
    q84_list = [float(qs[2]) for qs in quantiles.values()]

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
    
    save_dict = {
        'samples': samples,
        'weights': weights,
        'log_z': log_z,
        'param_names': param_names
    }
    with gzip.open('output/results.pkl.gz', 'wb') as f:
        pickle.dump(save_dict, f)
    
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
