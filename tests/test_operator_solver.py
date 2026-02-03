"""
Tests for OperatorInversion solver and comparison with LinearInversion.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_allclose

from TinyLensGpu.utils.inversion import LinearInversion, OperatorInversion
from TinyLensGpu.utils.inversion.operator_solver import (
    _apply_psf_unmasked_to_unmasked,
    _apply_mapping,
    _apply_mapping_transpose
)
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.pixelized_source import (
    PixelizedSourceModel,
)
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
    PixelizedImageProbModel,
)


class TestOperatorSolverComponents:
    """Test individual components of the operator solver."""

    @pytest.fixture
    def component_setup(self):
        """Setup for component testing."""
        np.random.seed(42)
        h, w = 20, 20
        psf_h, psf_w = 5, 5
        
        # Create mask (circle in center)
        y, x = np.mgrid[:h, :w]
        center = h // 2
        r = np.sqrt((y - center)**2 + (x - center)**2)
        mask = r > 8  # True means masked out
        
        y_indices, x_indices = np.where(~mask)
        unmasked_indices = (jnp.array(y_indices), jnp.array(x_indices))
        
        # PSF
        psf = np.random.rand(psf_h, psf_w)
        psf /= psf.sum()
        fft_shape = (h + psf_h - 1, w + psf_w - 1)
        psf_fft = jnp.fft.rfft2(psf, s=fft_shape)
        
        return {
            'image_shape': (h, w),
            'psf_shape': (psf_h, psf_w),
            'unmasked_indices': unmasked_indices,
            'psf_fft': psf_fft,
            'psf': psf,
            'mask': mask
        }

    def test_apply_psf_unmasked_consistency(self, component_setup):
        """Test that _apply_psf_unmasked_to_unmasked matches dense matrix multiplication."""
        setup = component_setup
        
        # Create a random unmasked image
        n_unmasked = len(setup['unmasked_indices'][0])
        x_unmasked = jnp.array(np.random.randn(n_unmasked).astype(np.float32))
        
        # 1. Operator application
        y_op = _apply_psf_unmasked_to_unmasked(
            x_unmasked,
            setup['psf_fft'],
            setup['image_shape'],
            setup['psf_shape'],
            setup['unmasked_indices'],
            adjoint=False
        )
        
        # 2. Dense matrix application
        from TinyLensGpu.utils.lensing.psf import build_psf_matrix_dense
        psf_mat = build_psf_matrix_dense(setup['mask'], setup['psf'])
        y_dense = psf_mat @ x_unmasked
        
        assert_allclose(y_op, y_dense, rtol=1e-4, atol=1e-4, 
                       err_msg="Forward PSF operator mismatch")
        
        # 3. Adjoint operator application
        y_adj_op = _apply_psf_unmasked_to_unmasked(
            x_unmasked,
            setup['psf_fft'],
            setup['image_shape'],
            setup['psf_shape'],
            setup['unmasked_indices'],
            adjoint=True
        )
        
        # 4. Dense matrix adjoint
        y_adj_dense = psf_mat.T @ x_unmasked
        
        assert_allclose(y_adj_op, y_adj_dense, rtol=1e-4, atol=1e-4,
                       err_msg="Adjoint PSF operator mismatch")

    def test_mapping_consistency(self, component_setup):
        """Test that _apply_mapping matches lens_mapping_matrix_from."""
        # Need some source points and data points
        n_source = 50
        n_data = 100
        source_mesh = jnp.array(np.random.randn(n_source, 2).astype(np.float32))
        data_mesh = jnp.array(np.random.randn(n_data, 2).astype(np.float32))
        
        from TinyLensGpu.utils.interpolation.kernels import get_interpolation_weights
        from TinyLensGpu.utils.lensing import lens_mapping_matrix_from
        
        # 1. Operator weights
        weights, indices, _ = get_interpolation_weights(
            points=source_mesh,
            query_points=data_mesh,
            k_neighbors=5,
            kernel='wendland_c2',
            radius_scale=2.0
        )
        
        # 2. Dense Matrix
        mapping_matrix = lens_mapping_matrix_from(
            source_mesh_beta=source_mesh,
            data_mesh_beta=data_mesh,
            k_neighbors=5,
            kernel='wendland_c2',
            radius_scale=2.0
        )
        
        # Test forward mapping
        source_vals = jnp.array(np.random.randn(n_source).astype(np.float32))
        
        res_op = _apply_mapping(source_vals, weights, indices)
        res_mat = mapping_matrix @ source_vals
        
        assert_allclose(res_op, res_mat, rtol=1e-5, atol=1e-5,
                       err_msg="Forward mapping operator mismatch")
                       
        # Test adjoint mapping
        data_vals = jnp.array(np.random.randn(n_data).astype(np.float32))
        
        res_adj_op = _apply_mapping_transpose(data_vals, weights, indices, n_source)
        res_adj_mat = mapping_matrix.T @ data_vals
        
        assert_allclose(res_adj_op, res_adj_mat, rtol=1e-5, atol=1e-5,
                       err_msg="Adjoint mapping operator mismatch")


class TestSolverComparison:
    """Compare OperatorInversion with LinearInversion."""
    
    @pytest.fixture
    def prob_model_setup(self):
        """Setup a complete model for comparison."""
        np.random.seed(42)
        size = 30
        
        # Image data
        image = np.random.randn(size, size) * 0.1 + 1.0
        noise = np.ones_like(image) * 0.1
        
        # Mask
        y, x = np.mgrid[:size, :size]
        r = np.sqrt((y - size//2)**2 + (x - size//2)**2)
        mask = r > 12
        
        # PSF
        psf = np.zeros((5, 5))
        psf[2, 2] = 1.0
        from scipy.ndimage import gaussian_filter
        psf = gaussian_filter(psf, 1.0)
        psf /= psf.sum()
        
        # Model components
        sie = SIE(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
        # Fix parameters to avoid re-jitting issues if any
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
            
        pix_src = PixelizedSourceModel(
            reg_scale=0.1,
            reg_coefficient=1.0,
            n_source_points=100,
            mesh_seed=123
        )
        
        phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src])
        
        return {
            'image': image,
            'noise': noise,
            'psf': psf,
            'mask': mask,
            'phys_model': phys_model,
            'dpix': 0.05
        }

    def test_solve_comparison(self, prob_model_setup):
        """Compare reconstructed source between Exact and Fast backends."""
        setup = prob_model_setup
        
        # 1. Exact Inversion
        model_exact = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            mask=setup['mask'],
            inversion_backend='exact'
        )
        
        # Access simulator directly to get inverter
        data_vector = model_exact.image_data[~model_exact.mask]
        noise_variance = model_exact.noise_map[~model_exact.mask] ** 2
        
        inverter_exact = model_exact.simulator.build_inverter(
            data_vector, noise_variance, 
            model_exact.pix_src_model.reg_scale.value,
            model_exact.pix_src_model.reg_coefficient.value,
            inversion_backend='exact'
        )
        s_exact = inverter_exact.solve()
        
        # 2. Fast Operator Inversion
        inverter_fast = model_exact.simulator.build_inverter(
            data_vector, noise_variance,
            model_exact.pix_src_model.reg_scale.value,
            model_exact.pix_src_model.reg_coefficient.value,
            inversion_backend='fast',
            cg_tol=1e-8, # High precision for comparison
            cg_maxiter=200
        )
        s_fast = inverter_fast.solve()
        
        # Compare
        # Note: CG might not converge exactly to direct solve result, but should be close
        # Relaxed tolerance for CG approximation
        assert_allclose(s_fast, s_exact, rtol=0.02, atol=1e-3,
                       err_msg="Operator solver result should match LinearInversion")

    def test_log_evidence_comparison(self, prob_model_setup):
        """Compare log evidence between Exact and Fast backends."""
        setup = prob_model_setup
        
        # 1. Exact
        model_exact = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            mask=setup['mask'],
            inversion_backend='exact'
        )
        log_ev_exact = model_exact.log_evidence()
        
        # 2. Fast
        model_fast = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            mask=setup['mask'],
            inversion_backend='fast',
            cg_tol=1e-6,
            slq_probes=10, # More probes for better estimation
            slq_steps=30
        )
        log_ev_fast = model_fast.log_evidence()
        
        # SLQ is stochastic, so we check if it's in the ballpark
        # The log evidence values are usually large, so relative error is small
        # But for small problems, they should be relatively close
        print(f"Log Evidence: Exact={log_ev_exact}, Fast={log_ev_fast}")
        
        # Allow some deviation due to stochastic nature of SLQ
        # We just want to ensure it's not completely off (e.g. sign error or order of magnitude)
        diff = abs(log_ev_exact - log_ev_fast)
        assert diff < 20.0 or diff / abs(log_ev_exact) < 0.05, \
            f"Log evidence mismatch too large: Exact={log_ev_exact}, Fast={log_ev_fast}"

    def test_reconstruct_source_works_with_fast_backend(self, prob_model_setup):
        """
        Check if reconstruct_source works with fast backend.
        """
        setup = prob_model_setup
        model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            mask=setup['mask'],
            inversion_backend='fast'
        )
        
        # This calls reconstruct_source internally via prob_model (if we had a method)
        # But let's call simulator directly as in the usage pattern
        data_vector = model.image_data[~model.mask]
        noise_variance = model.noise_map[~model.mask] ** 2
        
        # Should run without error now
        source_intensities, source_mesh_beta, model_image, inverter = model.simulator.reconstruct_source(
            data_vector, noise_variance,
            model.pix_src_model.reg_scale.value,
            model.pix_src_model.reg_coefficient.value,
            inversion_backend='fast'
        )
        
        assert isinstance(inverter, OperatorInversion)
        assert not jnp.any(jnp.isnan(source_intensities))
        assert not jnp.any(jnp.isnan(model_image))
