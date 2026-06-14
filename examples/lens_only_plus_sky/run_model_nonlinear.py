"""
This script performs a lens-light fit where both the Sersic amplitude and the
constant sky background amplitude are sampled non-linearly using Nautilus.

Key Differences from the linear demo:
- Optimization: Neither the Sersic light amplitude (Ie) nor the sky background
  amplitude is solved linearly.
- Priors: Both amplitudes are sampled directly alongside the Sersic shape
  parameters.
- Inference: Evaluates the sampler performance when the lens light and sky
  background normalizations are treated as nonlinear parameters.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from TinyLensGpu.Inference import ParamU, nautilus_posterior_summary
from TinyLensGpu.PhysicalModel import ConstantBackground
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from nautilus import Sampler
import jax.numpy as jnp
import numpy as np
from TinyLensGpu.visualizer import plot_model_results
from TinyLensGpu.ObservationModel.LensImage import ImageProbModel


if __name__ == "__main__":
    print("=" * 60)
    print("=" * 60)

    # 1. Load data
    print("\n[Stage 1] Loading data...")
    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path='data/image.fits',
        noise_path='data/noise.fits',
        psf_path='data/psf.fits',
        mask_path='data/mask.fits',
    )

    # 2. Build model components
    print("\n[Stage 2] Building model...")
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
        Ie=ParamU(
            "Ie", 1.0,
            prior_type="log_uniform",
            prior_settings=[1e-5, 1e5],
            limits=[0.0, 1e6],
        ),
    )
    sky_background = ConstantBackground(
        intensity=ParamU(
            "sky_intensity", 0.5,
            prior_type="uniform",
            prior_settings=[0.0, 1.5],
            limits=[0.0, 5.0],
        )
    )

    # Set parameters to dynamic for sampling
    lens_light.R_sersic.to_dynamic()
    lens_light.n_sersic.to_dynamic()
    lens_light.e1.to_dynamic()
    lens_light.e2.to_dynamic()
    lens_light.Ie.to_dynamic()
    sky_background.intensity.to_dynamic()

    # Set fixed parameters to static
    lens_light.center_x.to_static(0.0)
    lens_light.center_y.to_static(0.0)

    # Build physical model directly
    phys_model = PhysicalModel(lens_light=[lens_light, sky_background])

    # Build likelihood (ImageProbModel)
    likelihood = ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=0.074,
        nsub=4,
        phys_model=phys_model,
        use_linear=False,  # do not solve Ie linearly
        mask=mask,
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
    import time
    start_time = time.time()
    sampler = Sampler(
        prior,
        loglike,
        n_dim=len(param_names),
        n_live=200,
        vectorized=True,
        n_batch=200,
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
        title="Lens Model Fit Results",
    )

    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
