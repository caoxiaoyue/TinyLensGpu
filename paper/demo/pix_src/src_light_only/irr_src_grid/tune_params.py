
import os
import time
import numpy as np
import jax.numpy as jnp
import jax
import argparse
from TinyLensGpu.PhysicalModel import PhysicalModel, SersicEllipse, SIE, GaussianEllipse
from TinyLensGpu.utils.geometry import phi_q2_ellipticity
from TinyLensGpu.ForwardSimulation import SimulatorConfig, LensSimulator, make_grid_2d
from TinyLensGpu.PhysicalModel import (
    IrregularGridConfig,
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
    """Simulate a gravitational lensing observation (same as demo_optimize_mass.py)."""
    true_theta_E = 1.5
    true_e1, true_e2 = phi_q2_ellipticity(90*np.pi/180, 0.9)
    
    phy_model = PhysicalModel(
        lens_mass=[
            SIE(theta_E=true_theta_E, e1=true_e1, e2=true_e2, center_x=0.0, center_y=0.0),
        ],
        source_light=[
            SersicEllipse(R_sersic=0.3, n_sersic=1.0, e1=0.05, e2=0.05, center_x=0.0, center_y=0.3, Ie=1.0)
        ],
        lens_light=[],
    )
    
    npix = 200
    image_size = 10.0
    dpix = image_size / npix
    
    x_psf, y_psf = make_grid_2d(21, dpix)
    psf_kernel = GaussianEllipse(flux=1.0, sigma=0.05, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0).light(x=x_psf, y=y_psf)
    psf_kernel /= psf_kernel.sum()
    psf_kernel = np.asarray(psf_kernel)
    
    sim_config = SimulatorConfig(dpix=dpix, npix=npix, psf_kernel=psf_kernel, nsub=16)
    sim_obj = LensSimulator(phy_model, sim_config)
    img_2d = sim_obj.simulate()
    
    def mock_lens(ideal_image, back_rms, exp_time):
        noise_map = np.sqrt(ideal_image/exp_time + back_rms**2)
        noisy_image = ideal_image + np.random.normal(0, noise_map)
        return noisy_image, noise_map
    
    noisy_image, noise_map = mock_lens(img_2d, 0.1, 300)
    
    xgrid_image, ygrid_image = make_grid_2d(npix, dpix)
    rgrid_image = np.sqrt(xgrid_image**2 + ygrid_image**2)
    mask = (rgrid_image > 2.7)
    
    return {
        'noisy_image': noisy_image,
        'noise_map': noise_map,
        'psf_kernel': psf_kernel,
        'mask': mask,
        'dpix': dpix,
        'true_params': {
            'theta_E': true_theta_E,
            'e1': true_e1,
            'e2': true_e2,
        }
    }

def build_model(data_dict, backend="matrix", solver_args=None):
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
        # Operator backend
        solver_config = SolverConfig(
            inversion_backend="operator",
            cg_tol=solver_args.get('cg_tol', 1e-4),
            cg_maxiter=solver_args.get('cg_maxiter', 200),
            slq_probes=solver_args.get('slq_probes', 16),
            slq_steps=solver_args.get('slq_steps', 40),
        )

    pix_config = PixelizedSourceConfig(
        grid=IrregularGridConfig(
            n_source_points=1500,
            mesh_alpha=1.5,
            mesh_seed=42,
        ),
        mapping=MappingConfig(
            k_neighbors=5,
            interp_kernel="wendland_c4",
            radius_scale=1.5,
        ),
        regularization=RegularizationConfig(
            scheme="irregular_gp_exp",
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
    
    # Set dynamic parameters for sampling (to match demo_optimize_mass.py)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cg_tol", type=float, default=1e-4)
    parser.add_argument("--cg_maxiter", type=int, default=200)
    parser.add_argument("--slq_probes", type=int, default=16)
    parser.add_argument("--slq_steps", type=int, default=40)
    args = parser.parse_args()
    
    solver_args = {
        'cg_tol': args.cg_tol,
        'cg_maxiter': args.cg_maxiter,
        'slq_probes': args.slq_probes,
        'slq_steps': args.slq_steps,
    }
    
    print(f"Testing with parameters: {solver_args}")
    
    # Step 1: Simulate data
    np.random.seed(42) # Ensure consistent data
    data_dict = simulate_lensing_data()
    
    # Test parameters (close to true values)
    theta_test = jnp.array([1.5, 0.05, 0.05])
    
    # Benchmark Matrix Backend
    print("Running Matrix Backend...")
    prob_model_matrix = build_model(data_dict, backend="matrix")
    loglike_matrix_fn = make_likelihood(prob_model_matrix, vectorized=False)
    val_matrix = loglike_matrix_fn(theta_test)
    if hasattr(val_matrix, 'block_until_ready'):
        val_matrix.block_until_ready()
    print(f"Matrix LogLike: {float(val_matrix):.6f}")
    
    # Benchmark Operator Backend
    print("Running Operator Backend...")
    prob_model_operator = build_model(data_dict, backend="operator", solver_args=solver_args)
    loglike_operator_fn = make_likelihood(prob_model_operator, vectorized=False)
    val_operator = loglike_operator_fn(theta_test)
    if hasattr(val_operator, 'block_until_ready'):
        val_operator.block_until_ready()
    print(f"Operator LogLike: {float(val_operator):.6f}")
    
    diff = abs(float(val_matrix) - float(val_operator))
    print(f"Difference: {diff:.6f}")

if __name__ == "__main__":
    main()
