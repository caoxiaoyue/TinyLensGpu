"""
B-spline lens light + source light modeling example.

This example demonstrates how to build a B-spline model for lens light
combined with a B-spline source light profile, programmatically.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from TinyLensGpu.Inference import ParamU, nautilus_posterior_summary
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import build_bspline_multipole_set
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
    print("B-spline Lens Light + Source Light Model Inference")
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
    
    # Build B-spline source light model
    print("Building B-spline source light model...")
    
    # Create shared geometric parameters for source B-spline
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
    
    source_components = build_bspline_multipole_set(
        dpix=0.074,
        r_min=0.01,
        r_max=3.0,
        n_radial=20,
        ntheta=[0],
        degree=3,
        center_x=center_x_src,
        center_y=center_y_src,
        e1=e1_src,
        e2=e2_src,
        mask=None,
    )
    
    # Set shared geometric parameters to dynamic (only once)
    center_x_src.to_dynamic()
    center_y_src.to_dynamic()
    e1_src.to_dynamic()
    e2_src.to_dynamic()
    
    print(f"Created {len(source_components)} B-spline components for source light")
    
    # Build B-spline lens light model
    print("Building B-spline lens light model...")
    
    # Create shared geometric parameters for B-spline
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
    
    lens_components = build_bspline_multipole_set(
        dpix=0.074,
        r_min=0.01,
        r_max=5.0,
        n_radial=20,
        ntheta=[0],
        degree=3,
        center_x=center_x_lens,
        center_y=center_y_lens,
        e1=e1_lens,
        e2=e2_lens,
        mask=None
    )
    
    # Set shared geometric parameters to dynamic (only once)
    center_x_lens.to_dynamic()
    center_y_lens.to_dynamic()
    e1_lens.to_dynamic()
    e2_lens.to_dynamic()
    
    print(f"Created {len(lens_components)} B-spline components for lens light")
    
    # Build Physical Model
    phys_model = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=source_components,
        lens_light=lens_components
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
        solver_type='normal'  # Normal equation linear solver
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
    import time
    start_time = time.time()
    sampler = Sampler(
        prior, 
        loglike, 
        n_dim=len(param_names), 
        n_live=200, 
        vectorized=True, 
        n_batch=10  # Reduced batch size to prevent OOM with B-splines
    )
    sampler.run(verbose=True, n_eff=800)
    end_time = time.time()
    print(f"\nSampling completed in {end_time - start_time:.2f} seconds")

    # 5. Process results and summary
    print("\n[Stage 5] Processing results...")
    samples, weights, quantiles, log_z = nautilus_posterior_summary(sampler, param_names)
    q16_list = [float(qs[0]) for qs in quantiles.values()]
    q50_list = [float(qs[1]) for qs in quantiles.values()]
    q84_list = [float(qs[2]) for qs in quantiles.values()]

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
