"""
Benchmark log-likelihood calculation for matrix vs operator backends (Rectangular Grid).
Reference: bench_inversion_loglike.py (Irregular Grid)
"""

import os
import time
import numpy as np
import jax.numpy as jnp
import jax
from TinyLensGpu.PhysicalModel import PhysicalModel, SersicEllipse, SIE, GaussianEllipse
from TinyLensGpu.utils.geometry import phi_q2_ellipticity
from TinyLensGpu.ForwardSimulation import SimulatorConfig, LensSimulator, make_grid_2d
from TinyLensGpu.PhysicalModel import (
    RectangularGridConfig,
    MappingConfig,
    PixelizedSourceConfig,
    PixelizedSourceModel,
    RegularizationConfig,
    SolverConfig,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import PixelizedImageProbModel
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference import ParamU

# Set environment variables for JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def simulate_lensing_data():
    """Simulate a gravitational lensing observation (same as sim_data.py)."""
    # 1. Define physical model
    e1_l, e2_l = phi_q2_ellipticity(90 * np.pi / 180, 0.9)
    phy_model = PhysicalModel(
        lens_mass=[SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0)],
        source_light=[
            SersicEllipse(
                R_sersic=0.3,
                n_sersic=1.0,
                e1=0.05,
                e2=0.05,
                center_x=0.0,
                center_y=0.3,
                Ie=1.0,
            )
        ],
        lens_light=[],
    )

    # 2. Simulation configuration
    npix = 200
    image_size = 10.0
    dpix = image_size / npix

    # PSF
    x_psf, y_psf = make_grid_2d(21, dpix)
    psf_kernel = GaussianEllipse(
        flux=1.0,
        sigma=0.05,
        e1=0.0,
        e2=0.0,
        center_x=0.0,
        center_y=0.0,
    ).light(x=x_psf, y=y_psf)
    psf_kernel /= psf_kernel.sum()
    psf_kernel = np.asarray(psf_kernel)

    sim_config = SimulatorConfig(dpix=dpix, npix=npix, psf_kernel=psf_kernel, nsub=16)
    sim_obj = LensSimulator(phy_model, sim_config)
    img_2d = sim_obj.simulate()

    # 3. Add noise
    def mock_lens(ideal_image, back_rms, exp_time):
        noise_map = np.sqrt(ideal_image / exp_time + back_rms**2)
        noisy_image = ideal_image + np.random.normal(0, noise_map)
        return noisy_image, noise_map

    noisy_image, noise_map = mock_lens(img_2d, 0.1, 300)

    # 4. Create mask
    xgrid_image, ygrid_image = make_grid_2d(npix, dpix)
    rgrid_image = np.sqrt(xgrid_image**2 + ygrid_image**2)
    mask = rgrid_image > 2.7  # Using 2.7 to match bench_inversion_loglike.py, sim_data uses 2.7 too

    return {
        'noisy_image': noisy_image,
        'noise_map': noise_map,
        'psf_kernel': psf_kernel,
        'mask': mask,
        'dpix': dpix,
        'true_params': {
            'theta_E': 1.5,
            'e1': e1_l,
            'e2': e2_l,
        }
    }

def build_model(data_dict, backend="matrix"):
    """Build lens model with specified backend."""
    
    # Define mass parameters
    theta_E = ParamU("theta_E", 1.5, prior_type="uniform", prior_settings=[0.8, 2.5], limits=[0.0, 5.0])
    e1 = ParamU("e1", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2 = ParamU("e2", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    
    sie = SIE(theta_E=theta_E, e1=e1, e2=e2, center_x=0.0, center_y=0.0)
    
    # Solver config based on backend
    if backend == "matrix":
        solver_config = SolverConfig(inversion_backend="matrix")
    else:
        # Operator backend often benefits from specific solver settings
        solver_config = SolverConfig(
            inversion_backend="operator",
        )

    # Rectangular Grid Config
    pix_config = PixelizedSourceConfig(
        grid=RectangularGridConfig(
            nx=60,
            ny=60,
            margin_frac=0.1,
        ),
        mapping=MappingConfig(
            # Rectangular grid usually uses bilinear interpolation which is handled internally,
            # but we keep default values here or explicit ones if needed.
            # Usually 'wendland_c4' is for irregular, but MappingConfig is generic.
        ),
        regularization=RegularizationConfig(
            scheme="rectangular_first", # 1st order (gradient) regularization
        ),
        solver=solver_config,
    )
    
    pix_src_model = PixelizedSourceModel(
        config=pix_config,
        reg_scale=0.05,
        reg_coefficient=1.0,
    )

    phys_model = PhysicalModel(
        lens_mass=[sie],
        source_light=[pix_src_model],
        lens_light=[],
    )
    
    # Set dynamic parameters for sampling
    sie.theta_E.to_dynamic()
    sie.e1.to_dynamic()
    sie.e2.to_dynamic()
    
    sim_config = SimulatorConfig(
        dpix=data_dict["dpix"],
        npix=data_dict["noisy_image"].shape[0],
        psf_kernel=data_dict["psf_kernel"],
        mask=data_dict["mask"],
    )

    prob_model = PixelizedImageProbModel(
        image_data=data_dict['noisy_image'],
        noise_map=data_dict['noise_map'],
        sim_config=sim_config,
        phys_model=phys_model,
    )
    
    return prob_model

def benchmark_backend(backend_name, data_dict, theta_test, n_runs=10):
    """Benchmark a specific backend."""
    print(f"\nBenchmarking {backend_name} backend...")
    
    # Build model and likelihood
    prob_model = build_model(data_dict, backend=backend_name)
    loglike = make_likelihood(prob_model, vectorized=False)
    
    # JIT Warm-up
    print("  Warming up (JIT compilation)...")
    start_warmup = time.time()
    val = loglike(theta_test)
    # Ensure JAX finishes the computation
    if hasattr(val, 'block_until_ready'):
        val.block_until_ready()
    warmup_time = time.time() - start_warmup
    print(f"  Warm-up time: {warmup_time:.4f} s")
    print(f"  Initial log-likelihood value: {float(val):.4f}")
    
    # Benchmark runs
    times = []
    for i in range(n_runs):
        start = time.time()
        val = loglike(theta_test)
        if hasattr(val, 'block_until_ready'):
            val.block_until_ready()
        end = time.time()
        times.append(end - start)
        
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"  Benchmark results ({n_runs} runs):")
    print(f"    Average time: {avg_time*1000:.2f} ms")
    print(f"    Std dev:      {std_time*1000:.2f} ms")
    
    return avg_time, val

if __name__ == "__main__":
    print("="*60)
    print("Log-Likelihood Benchmark: Matrix vs Operator (Rectangular Grid)")
    print("="*60)
    
    # Step 1: Simulate data
    data_dict = simulate_lensing_data()
    
    # Test parameters (close to true values)
    # e1_l, e2_l from sim_data are approx -0.0, 0.0 for q=0.9, phi=90?
    # phi=90, q=0.9 => e = (1-0.9)/(1+0.9) = 0.1/1.9 ~ 0.0526. 
    # phi=90 means along y-axis. e1 = e*cos(2phi) = -e, e2 = e*sin(2phi) = 0.
    # So e1 approx -0.05, e2 approx 0.0.
    
    # Using the values from data_dict['true_params'] would be better if we wanted exact match, 
    # but for benchmark any valid params work.
    # Let's use the same test values as bench_inversion_loglike.py but adjusted for the lens model if needed.
    # bench_inversion_loglike.py used [1.5, 0.05, 0.05].
    # Here our lens is SIE.
    
    theta_test = jnp.array([1.5, data_dict['true_params']['e1'], data_dict['true_params']['e2']])
    print(f"Test parameters: {theta_test}")

    # Benchmark Matrix Backend
    time_matrix, val_matrix = benchmark_backend("matrix", data_dict, theta_test)
    
    # Benchmark Operator Backend
    time_operator, val_operator = benchmark_backend("operator", data_dict, theta_test)
    
    print("\n" + "="*60)
    print("Final Comparison")
    print("="*60)
    print(f"Matrix Backend:   {time_matrix*1000:8.2f} ms")
    print(f"Operator Backend: {time_operator*1000:8.2f} ms")
    print(f"Speedup:          {time_matrix/time_operator:.2f}x" if time_operator < time_matrix else f"Slowdown: {time_operator/time_matrix:.2f}x")
    
    # Accuracy check
    diff = jnp.abs(val_matrix - val_operator)
    print(f"\nDifference in log-likelihood: {diff:.6f}")
    if diff < 1.0: # Allow some tolerance due to SLQ stochasticity in operator backend
        print("Backends are consistent (within tolerance).")
    else:
        print("WARNING: Backends show significant difference.")
    
    print("\nBenchmark Complete!")
    print("="*60)
