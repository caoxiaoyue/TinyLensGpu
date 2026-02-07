"""
Lens + Source model (Programmatic API).

This demo shows how to model both lens light and lensed source.
"""

import os
import pickle
import gzip
import numpy as np
import jax.numpy as jnp
from nautilus import Sampler

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from TinyLensGpu.Inference import ParamU
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
    print("Lens + Source Model Inference")
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
        e1=ParamU("e1_mass", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        e2=ParamU("e2_mass", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        center_x=ParamU("center_x", 0.0),
        center_y=ParamU("center_y", 0.0),
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
        R_sersic=ParamU("R_sersic_src", 1.0, prior_type="uniform", 
                        prior_settings=[0.001, 2.001], limits=[0.0, 5.0]),
        n_sersic=ParamU("n_sersic_src", 1.0, prior_type="uniform", 
                        prior_settings=[0.3, 2.3], limits=[0.3, 6.0]),
        e1=ParamU("e1_src", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        e2=ParamU("e2_src", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        center_x=ParamU("center_x_src", 0.0, prior_type="gaussian", 
                        prior_settings=[0.0, 0.5], limits=[-3.0, 3.0]),
        center_y=ParamU("center_y_src", 0.0, prior_type="gaussian", 
                        prior_settings=[0.0, 0.5], limits=[-3.0, 3.0]),
        Ie=ParamU("Ie_src", 1.0),
    )
    
    # Lens Sersic profile
    lens = SersicEllipse(
        R_sersic=ParamU("R_sersic_lens", 1.0, prior_type="uniform", 
                        prior_settings=[0.001, 2.001], limits=[0.0, 5.0]),
        n_sersic=ParamU("n_sersic_lens", 4.0, prior_type="gaussian", 
                        prior_settings=[4.0, 0.5], limits=[0.3, 6.0]),
        e1=ParamU("e1_lens", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        e2=ParamU("e2_lens", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        center_x=ParamU("center_x_lens", 0.0),
        center_y=ParamU("center_y_lens", 0.0),
        Ie=ParamU("Ie_lens", 1.0),
    )
    
    # Build physical model
    phys_model = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=[source],
        lens_light=[lens]
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
    lens.R_sersic.to_dynamic()
    lens.n_sersic.to_dynamic()
    lens.e1.to_dynamic()
    lens.e2.to_dynamic()
    
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
        n_batch=800,
    )
    import time
    start_time = time.time()
    sampler.run(verbose=True, n_eff=800)
    end_time = time.time()
    print(f"\nSampling completed in {end_time - start_time:.2f} seconds")

    # 5. Process results and summary
    print("\n[Stage 5] Processing results...")
    samples, log_w, _ = sampler.posterior()
    weights = np.exp(log_w - np.max(log_w))
    weights /= weights.sum()
    
    print("\n" + "="*60)
    print("Posterior Summary")
    print("="*60)
    
    q16_list, q50_list, q84_list = [], [], []
    for i, name in enumerate(param_names):
        sorted_idx = np.argsort(samples[:, i])
        sorted_samples = samples[sorted_idx, i]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        cumsum /= cumsum[-1]
        
        q16 = np.interp(0.16, cumsum, sorted_samples)
        q50 = np.interp(0.50, cumsum, sorted_samples)
        q84 = np.interp(0.84, cumsum, sorted_samples)
        
        q16_list.append(q16)
        q50_list.append(q50)
        q84_list.append(q84)
        
        print(f"  {name:20s} = {q50:.4f} ({q16-q50:+.4f}, {q84-q50:+.4f})")
    
    log_z = float(np.asarray(sampler.log_z))
    print(f"\nlog(Z) = {log_z:.2f}")

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