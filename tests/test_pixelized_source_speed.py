"""
Performance tests for Pixelized Source Model.

This module benchmarks the critical components of the pixelized source reconstruction pipeline:
- Regularization matrix construction
- Lens mapping matrix construction
- PSF matrix construction and application
- Linear inversion
- End-to-end probability model evaluation
"""

import numpy as np
import time
import pytest
import jax
import jax.numpy as jnp
from jax import random

from TinyLensGpu.utils.lensing import (
    exp_cov_matrix_from,
    gauss_cov_matrix_from,
    matern32_cov_matrix_from,
    matern52_cov_matrix_from,
    dense_mapping_from_weights_indices,
    build_psf_matrix_dense,
    apply_psf_to_mapping_matrix,
)
from TinyLensGpu.utils.interpolation.kernels import get_interpolation_weights
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import LinearInversion
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.pixelized_source import (
    PixelizedSourceModel,
)
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import (
    IrregularGridConfig,
    PixelizedSourceConfig,
)
from tests.pixelized_test_factory import build_pixelized_source_model
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
    PixelizedImageProbModel,
)

# =============================================================================
# Fixtures and Helpers
# =============================================================================

@pytest.fixture
def benchmark_data():
    """Generate data for benchmarking."""
    np.random.seed(42)
    # Use a moderately sized image for speed tests
    size = 60  
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Image with some structure
    image = np.exp(-(X**2 + Y**2) / 0.3)
    
    # Circular mask
    R = np.sqrt(X**2 + Y**2)
    mask = R > 0.95  # True means masked out
    
    # PSF
    psf_size = 11
    psf_sigma = 1.5
    px = np.arange(psf_size) - psf_size // 2
    py = np.arange(psf_size) - psf_size // 2
    PX, PY = np.meshgrid(px, py)
    psf = np.exp(-(PX**2 + PY**2) / (2 * psf_sigma**2))
    psf = psf / np.sum(psf)
    
    noise_map = np.ones_like(image) * 0.05
    
    return {
        'image': image,
        'mask': mask,
        'psf': psf,
        'noise_map': noise_map,
        'dpix': 0.05,
        'sim_config': SimulatorConfig(
            dpix=0.05,
            npix=size,
            psf_kernel=psf,
            mask=mask,
        ),
    }

def time_function(func, *args, n_repeats=5, warmup=True, name="Function", **kwargs):
    """Helper to time a JAX function."""
    # Warmup
    if warmup:
        _ = func(*args, **kwargs)
        _.block_until_ready() if hasattr(_, 'block_until_ready') else None
    
    start = time.time()
    for _ in range(n_repeats):
        res = func(*args, **kwargs)
        res.block_until_ready() if hasattr(res, 'block_until_ready') else None
    end = time.time()
    
    avg_time = (end - start) / n_repeats
    print(f"\n[Speed] {name}: {avg_time*1000:.4f} ms (avg of {n_repeats} runs)")
    return avg_time

# =============================================================================
# Benchmarks
# =============================================================================

class TestPixelizedSpeed:
    
    def test_speed_regularization_matrix(self):
        """Benchmark regularization matrix construction."""
        print("\n--- Regularization Matrix Speed ---")
        n_points_list = [500, 1000]
        
        for n_points in n_points_list:
            key = random.PRNGKey(0)
            points = random.normal(key, (n_points, 2))
            scale = 0.1
            
            print(f"N_points = {n_points}")
            
            time_function(exp_cov_matrix_from, scale, points, name="Exp Cov")
            time_function(gauss_cov_matrix_from, scale, points, name="Gauss Cov")
            time_function(matern32_cov_matrix_from, scale, points, name="Matern32 Cov")
            time_function(matern52_cov_matrix_from, scale, points, name="Matern52 Cov")

    def test_speed_lens_mapping_matrix(self):
        """Benchmark lens mapping matrix construction."""
        print("\n--- Lens Mapping Matrix Speed ---")
        n_source_list = [500, 1000]
        n_data = 2000 # Approx 45x45 image unmasked
        
        for n_source in n_source_list:
            key = random.PRNGKey(1)
            k1, k2 = random.split(key)
            source_mesh = random.normal(k1, (n_source, 2))
            data_mesh = random.normal(k2, (n_data, 2))
            
            print(f"N_source = {n_source}, N_data = {n_data}")
            
            @jax.jit
            def build_mapping(source, data):
                weights, indices, _ = get_interpolation_weights(
                    points=source,
                    query_points=data,
                    k_neighbors=5,
                    kernel='wendland_c4',
                    radius_scale=1.5
                )
                return dense_mapping_from_weights_indices(weights, indices, n_source)

            time_function(
                build_mapping, 
                source_mesh, 
                data_mesh, 
                name="Lens Mapping Matrix"
            )

    def test_speed_psf_matrix(self, benchmark_data):
        """Benchmark PSF matrix construction."""
        print("\n--- PSF Matrix Speed ---")
        mask = benchmark_data['mask']
        psf = benchmark_data['psf']
        
        # Count unmasked pixels
        n_unmasked = np.sum(~mask)
        print(f"Image shape: {mask.shape}, Unmasked pixels: {n_unmasked}")
        print(f"PSF shape: {psf.shape}")
        
        # Dense PSF Matrix
        # Note: build_psf_matrix_dense uses Numba, so no jax.jit here, but result is jax array
        # It handles JIT internally via Numba
        start = time.time()
        res = build_psf_matrix_dense(mask, psf)
        end = time.time()
        print(f"\n[Speed] Dense PSF Matrix (First run/Compile): {(end-start)*1000:.4f} ms")
        
        start = time.time()
        res = build_psf_matrix_dense(mask, psf)
        res.block_until_ready()
        end = time.time()
        print(f"[Speed] Dense PSF Matrix (Second run): {(end-start)*1000:.4f} ms")

    def test_speed_apply_psf_to_mapping(self):
        """Benchmark applying PSF to mapping matrix (convolution approach)."""
        print("\n--- Apply PSF to Mapping Matrix Speed ---")
        
        # Setup
        image_shape = (60, 60)
        n_data = image_shape[0] * image_shape[1]
        n_source = 500
        
        # Create dummy mapping matrix (n_unmasked, n_source)
        # Assume full image is unmasked for simplicity or create mask
        mask = np.zeros(image_shape, dtype=bool) # All valid
        y_ind, x_ind = np.where(~mask)
        unmasked_indices = (jnp.array(y_ind), jnp.array(x_ind))
        
        key = random.PRNGKey(2)
        mapping_matrix = random.normal(key, (n_data, n_source))
        psf_kernel = random.normal(key, (11, 11))
        
        print(f"Mapping Matrix: {mapping_matrix.shape}, Image: {image_shape}")
        time_function(
            apply_psf_to_mapping_matrix,
            mapping_matrix,
            psf_kernel,
            image_shape,
            unmasked_indices,
            name="Apply PSF (Convolution)"
        )

    def test_speed_linear_inversion(self):
        """Benchmark Linear Inversion solver."""
        print("\n--- Linear Inversion Speed ---")
        n_data = 2000
        n_source = 1000
        
        key = random.PRNGKey(3)
        k1, k2, k3, k4 = random.split(key, 4)
        
        # Mock matrices
        data_vector = random.normal(k1, (n_data,))
        noise_variance = jnp.ones((n_data,)) * 0.01
        
        # L_matrix: (n_data, n_source)
        L_matrix = random.normal(k2, (n_data, n_source)) * 0.01
        
        # R_matrix: (n_source, n_source) - make it pos def
        A = random.normal(k3, (n_source, n_source))
        R_matrix = A @ A.T + jnp.eye(n_source) * 0.1
        
        reg_strength = 1.0
        
        # Define function to benchmark
        @jax.jit
        def solve_inversion(d, n, L, R, lam):
            # H matrix should include lambda (regularization strength)
            H = R * lam
            inv = LinearInversion(
                d=d,
                F=L,
                noise_cov=n,
                H=H
            )
            # Access property instead of attribute if it's a property, 
            # but LinearInversion has solve() method, let's use that.
            # Reading code: solve() returns 's'. 
            # source_intensities is NOT a property on LinearInversion class in linear_solver.py
            # So we use inv.solve()
            return inv.solve(), inv.log_evidence()
            
        print(f"Data: {n_data}, Source: {n_source}")
        time_function(
            solve_inversion,
            data_vector,
            noise_variance,
            L_matrix,
            R_matrix,
            reg_strength,
            name="Linear Inversion Solve + Evidence"
        )

    def test_speed_end_to_end(self, benchmark_data):
        """Benchmark end-to-end PixelizedImageProbModel."""
        print("\n--- End-to-End Prob Model Speed ---")
        
        # Setup Model
        sie = SIE(theta_E=1.5, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
        pix_src = build_pixelized_source_model(
            config=PixelizedSourceConfig(grid=IrregularGridConfig(n_source_points=500)),
            reg_scale=0.1,
            reg_coefficient=1.0,
        )
        phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src])
        
        prob_model = PixelizedImageProbModel(
            image_data=benchmark_data['image'],
            noise_map=benchmark_data['noise_map'],
            sim_config=benchmark_data['sim_config'],
            phys_model=phys_model,
        )
        
        # Benchmark __call__
        # Note: __call__ is @ck.forward, usually not fully JIT-able at top level due to caching logic 
        # but the heavy lifting inside is JIT-ed.
        # We'll time the python call which includes the JIT-ed internals.
        
        print("First call (Compiling/Caching)...")
        start = time.time()
        res = prob_model()
        # Force wait
        res.block_until_ready() if hasattr(res, 'block_until_ready') else None
        end = time.time()
        print(f"First call time: {(end-start)*1000:.4f} ms")
        
        print("Subsequent calls (Cached)...")
        # Note: If parameters don't change, it returns cached result instantly.
        # To test speed, we should change parameters or force re-computation?
        # Ideally we want to test the speed of the computation when parameters change.
        
        # Let's change a parameter slightly to force re-computation
        prob_model.pix_src_model.reg_coefficient.value = 1.1
        
        start = time.time()
        res = prob_model()
        res.block_until_ready() if hasattr(res, 'block_until_ready') else None
        end = time.time()
        print(f"Re-computation time (Param change): {(end-start)*1000:.4f} ms")
        
        # Benchmark just the inverter part if accessible or by wrapping
        # The prob_model() call effectively measures the "likelihood evaluation" time
        # which is what samplers care about.
