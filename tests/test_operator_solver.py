"""
Tests for OperatorInversion solver and comparison with LinearInversion.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_allclose

from TinyLensGpu.utils.inversion import (
    LinearInversion,
    NNLSInversion,
    OperatorInversion,
    OperatorNNLSInversion,
)
from TinyLensGpu.utils.inversion.operator_solver import (
    _apply_psf_unmasked_to_unmasked,
    _apply_mapping,
    _apply_mapping_transpose
)
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import (
    IrregularGridConfig,
    PixelizedSourceConfig,
    RectangularGridConfig,
    RegularizationConfig,
)
from tests.pixelized_test_factory import build_pixelized_source_model
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
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
            
        pix_src = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=IrregularGridConfig(n_source_points=100, mesh_seed=123),
            ),
            reg_scale=0.1,
            reg_coefficient=1.0,
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
        """Compare reconstructed source between matrix and operator backends."""
        setup = prob_model_setup
        
        # 1. Matrix inversion
        model_exact = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            mask=setup['mask'],
            inversion_backend='matrix'
        )
        
        # Access simulator directly to get inverter
        data_vector = model_exact.image_data[~model_exact.mask]
        noise_variance = model_exact.noise_map[~model_exact.mask] ** 2
        
        inverter_exact = model_exact.simulator.build_inverter(
            data_vector, noise_variance, 
            model_exact.pix_src_model.reg_scale.value,
            model_exact.pix_src_model.reg_coefficient.value,
            inversion_backend='matrix'
        )
        s_exact = inverter_exact.solve()
        
        # 2. Operator inversion
        inverter_fast = model_exact.simulator.build_inverter(
            data_vector, noise_variance,
            model_exact.pix_src_model.reg_scale.value,
            model_exact.pix_src_model.reg_coefficient.value,
            inversion_backend='operator',
            cg_tol=1e-8, # High precision for comparison
            cg_maxiter=400
        )
        s_fast = inverter_fast.solve()
        
        # Compare
        # Note: CG might not converge exactly to direct solve result, but should be close
        assert_allclose(s_fast, s_exact, rtol=1.5e-2, atol=1e-3,
                       err_msg="Operator solver result should match LinearInversion")

    def test_log_evidence_comparison(self, prob_model_setup):
        """Compare log evidence between matrix and operator backends."""
        setup = prob_model_setup
        
        # 1. Exact
        model_exact = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            mask=setup['mask'],
            inversion_backend='matrix'
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
            inversion_backend='operator',
            cg_tol=1e-6,
            cg_maxiter=300,
            slq_probes=32,
            slq_steps=60,
        )
        log_ev_fast = model_fast.log_evidence()
        
        # SLQ is stochastic, so we check if it's in the ballpark
        # The log evidence values are usually large, so relative error is small
        # But for small problems, they should be relatively close
        print(f"Log Evidence: Exact={log_ev_exact}, Fast={log_ev_fast}")
        
        diff = abs(log_ev_exact - log_ev_fast)
        assert diff < 0.25 or diff / abs(log_ev_exact) < 2e-3, \
            f"Log evidence mismatch too large: Exact={log_ev_exact}, Fast={log_ev_fast}"

    def test_reconstruct_source_works_with_operator_backend(self, prob_model_setup):
        """Check if reconstruct_source works with operator backend."""
        setup = prob_model_setup
        model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            mask=setup['mask'],
            inversion_backend='operator'
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
            inversion_backend='operator'
        )
        
        assert isinstance(inverter, OperatorInversion)
        assert not jnp.any(jnp.isnan(source_intensities))
        assert not jnp.any(jnp.isnan(model_image))

    def test_backend_rejects_removed_aliases(self, prob_model_setup):
        """Removed inversion-backend aliases should raise ValueError."""
        setup = prob_model_setup
        model_exact_alias = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            mask=setup['mask'],
            inversion_backend='exact',
        )
        with pytest.raises(ValueError, match="Unknown inversion_backend"):
            _ = model_exact_alias.log_evidence()

        model_fast_alias = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            mask=setup['mask'],
            inversion_backend='fast',
        )
        with pytest.raises(ValueError, match="Unknown inversion_backend"):
            _ = model_fast_alias.log_evidence()

    def test_operator_nonnegative_matches_matrix_nnls(self, prob_model_setup):
        """Operator NNLS (FISTA) should match matrix NNLS backend."""
        setup = prob_model_setup

        model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            mask=setup['mask'],
            inversion_backend='matrix',
            nonnegative=True,
        )

        data_vector = model.image_data[~model.mask]
        noise_variance = model.noise_map[~model.mask] ** 2
        reg_scale = model.pix_src_model.reg_scale.value
        reg_coeff = model.pix_src_model.reg_coefficient.value

        inv_matrix = model.simulator.build_inverter(
            data_vector,
            noise_variance,
            reg_scale,
            reg_coeff,
            inversion_backend='matrix',
            nonnegative=True,
        )
        inv_operator = model.simulator.build_inverter(
            data_vector,
            noise_variance,
            reg_scale,
            reg_coeff,
            inversion_backend='operator',
            nonnegative=True,
            nnls_maxiter=900,
            nnls_tol=1e-7,
        )

        assert isinstance(inv_matrix, NNLSInversion)
        assert isinstance(inv_operator, OperatorNNLSInversion)

        x_matrix = inv_matrix.solve()
        x_operator = inv_operator.solve()

        assert jnp.all(x_operator >= -1e-7)
        # Different NNLS solvers can reach slightly different active-set solutions
        # with near-identical objective values. We keep a loose coefficient check
        # and strict objective/model checks below.
        assert_allclose(x_operator, x_matrix, rtol=0.10, atol=5e-2)

        model_matrix = inv_matrix.model_predict(x_matrix)
        model_operator = inv_operator.model_predict(x_operator)
        assert_allclose(model_operator, model_matrix, rtol=0.02, atol=2e-3)

        obj_matrix = float(np.asarray(inv_matrix.objective_value(x_matrix)))
        obj_operator = float(np.asarray(inv_operator.objective_value(x_operator)))
        assert abs(obj_operator - obj_matrix) <= max(0.5, 0.01 * abs(obj_matrix))

    def test_operator_nonnegative_log_evidence_finite(self, prob_model_setup):
        """Nonnegative operator backend returns a finite approximate evidence."""
        setup = prob_model_setup
        model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            mask=setup['mask'],
            inversion_backend='operator',
            nonnegative=True,
        )
        log_ev = model.log_evidence()
        assert np.isfinite(log_ev)

    def test_sparse_knn_operator_matches_matrix_solution(self, prob_model_setup):
        """Sparse regularization mode should agree between matrix and operator solves."""
        setup = prob_model_setup
        pix_src = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=IrregularGridConfig(n_source_points=100, mesh_seed=123),
                regularization=RegularizationConfig(mode='sparse_knn', sparse_k_neighbors=16),
            ),
            reg_scale=0.1,
            reg_coefficient=1.0,
        )
        phys_model = PhysicalModel(
            lens_mass=setup['phys_model'].lens_mass,
            source_light=[pix_src],
            lens_light=setup['phys_model'].lens_light,
        )

        model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=phys_model,
            mask=setup['mask'],
            inversion_backend='matrix',
        )

        data_vector = model.image_data[~model.mask]
        noise_variance = model.noise_map[~model.mask] ** 2
        reg_scale = model.pix_src_model.reg_scale.value
        reg_coeff = model.pix_src_model.reg_coefficient.value

        inv_matrix = model.simulator.build_inverter(
            data_vector,
            noise_variance,
            reg_scale,
            reg_coeff,
            inversion_backend='matrix',
        )
        inv_operator = model.simulator.build_inverter(
            data_vector,
            noise_variance,
            reg_scale,
            reg_coeff,
            inversion_backend='operator',
            cg_tol=1e-7,
            cg_maxiter=500,
            evidence_mode='accurate',
            slq_probes=32,
            slq_steps=60,
        )

        assert isinstance(inv_operator, OperatorInversion)
        assert inv_operator.reg_operator_mode == 'sparse_knn'
        assert inv_operator.H_sparse_values.shape[0] > 0
        assert inv_operator.H.ndim == 1

        s_matrix = inv_matrix.solve()
        s_operator = inv_operator.solve()

        assert_allclose(s_operator, s_matrix, rtol=1e-2, atol=1e-3)

        m_matrix = inv_matrix.model_predict(s_matrix)
        m_operator = inv_operator.model_predict(s_operator)
        assert_allclose(m_operator, m_matrix, rtol=2e-3, atol=2e-4)

    def test_sparse_knn_operator_log_evidence_close_to_matrix(self, prob_model_setup):
        """Sparse regularization mode keeps evidence close across backends."""
        setup = prob_model_setup
        pix_src = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=IrregularGridConfig(n_source_points=100, mesh_seed=123),
                regularization=RegularizationConfig(mode='sparse_knn', sparse_k_neighbors=16),
            ),
            reg_scale=0.1,
            reg_coefficient=1.0,
        )
        phys_model = PhysicalModel(
            lens_mass=setup['phys_model'].lens_mass,
            source_light=[pix_src],
            lens_light=setup['phys_model'].lens_light,
        )

        model_matrix = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=phys_model,
            mask=setup['mask'],
            inversion_backend='matrix',
        )
        model_operator = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=phys_model,
            mask=setup['mask'],
            inversion_backend='operator',
            evidence_mode='accurate',
            cg_tol=1e-7,
            cg_maxiter=500,
            slq_probes=32,
            slq_steps=60,
        )

        log_ev_matrix = model_matrix.log_evidence()
        log_ev_operator = model_operator.log_evidence()

        diff = abs(log_ev_matrix - log_ev_operator)
        assert diff < 0.8 or diff / max(abs(log_ev_matrix), 1.0) < 1e-3

    def test_rectangular_bilinear_operator_mode(self, prob_model_setup):
        """Rectangular bilinear source-grid runs in sparse operator mode."""
        setup = prob_model_setup

        pix_src = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=RectangularGridConfig(nx=20, ny=18),
                regularization=RegularizationConfig(mode='sparse_rectangular', rect_scheme='curvature'),
            ),
            reg_scale=0.1,
            reg_coefficient=1.0,
        )
        phys_model = PhysicalModel(lens_mass=setup['phys_model'].lens_mass, source_light=[pix_src])

        model_operator = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=phys_model,
            mask=setup['mask'],
            inversion_backend='operator',
            evidence_mode='fast',
            cg_tol=1e-5,
            cg_maxiter=200,
            slq_probes=8,
            slq_steps=20,
        )

        data_vector = model_operator.image_data[~model_operator.mask]
        noise_variance = model_operator.noise_map[~model_operator.mask] ** 2

        inv_operator = model_operator.simulator.build_inverter(
            data_vector,
            noise_variance,
            model_operator.pix_src_model.reg_scale.value,
            model_operator.pix_src_model.reg_coefficient.value,
            inversion_backend='operator',
            evidence_mode='fast',
            cg_tol=1e-5,
            cg_maxiter=200,
            slq_probes=8,
            slq_steps=20,
        )

        assert isinstance(inv_operator, OperatorInversion)
        assert inv_operator.reg_operator_mode == 'sparse_rectangular'
        assert inv_operator.H_sparse_values.shape[0] > 0

        s_operator = inv_operator.solve()
        assert s_operator.shape[0] == 20 * 18
        assert not jnp.any(jnp.isnan(s_operator))

    def test_rectangular_bilinear_matrix_backend(self, prob_model_setup):
        """Rectangular bilinear source-grid also supports the matrix backend."""
        setup = prob_model_setup

        pix_src = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=RectangularGridConfig(nx=10, ny=10),
                regularization=RegularizationConfig(mode='sparse_rectangular', rect_scheme='gradient'),
            ),
            reg_scale=0.1,
            reg_coefficient=1.0,
        )
        phys_model = PhysicalModel(lens_mass=setup['phys_model'].lens_mass, source_light=[pix_src])

        model_matrix = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=phys_model,
            mask=setup['mask'],
            inversion_backend='matrix',
        )

        model_operator = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=phys_model,
            mask=setup['mask'],
            inversion_backend='operator',
            evidence_mode='fast',
            cg_tol=1e-5,
            cg_maxiter=240,
            slq_probes=8,
            slq_steps=20,
        )

        data_vector = model_matrix.image_data[~model_matrix.mask]
        noise_variance = model_matrix.noise_map[~model_matrix.mask] ** 2

        inv_matrix = model_matrix.simulator.build_inverter(
            data_vector,
            noise_variance,
            model_matrix.pix_src_model.reg_scale.value,
            model_matrix.pix_src_model.reg_coefficient.value,
            inversion_backend='matrix',
        )

        inv_operator = model_operator.simulator.build_inverter(
            data_vector,
            noise_variance,
            model_operator.pix_src_model.reg_scale.value,
            model_operator.pix_src_model.reg_coefficient.value,
            inversion_backend='operator',
            evidence_mode='fast',
            cg_tol=1e-5,
            cg_maxiter=240,
            slq_probes=8,
            slq_steps=20,
        )

        s_matrix = inv_matrix.solve()
        s_operator = inv_operator.solve()

        assert isinstance(inv_matrix, LinearInversion)
        assert inv_matrix.H.shape == (10 * 10, 10 * 10)
        assert not jnp.any(jnp.isnan(s_matrix))
        assert_allclose(s_operator, s_matrix, rtol=6e-2, atol=4e-3)

        m_matrix = inv_matrix.model_predict(s_matrix)
        m_operator = inv_operator.model_predict(s_operator)
        assert_allclose(m_operator, m_matrix, rtol=2e-2, atol=2e-3)

    def test_rectangular_bilinear_matrix_backend_with_lens_light(self, prob_model_setup):
        """Rectangular matrix backend supports joint source+lens-light inversion."""
        setup = prob_model_setup

        pix_src = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=RectangularGridConfig(nx=12, ny=9),
                regularization=RegularizationConfig(mode='sparse_rectangular', rect_scheme='gradient'),
            ),
            reg_scale=0.1,
            reg_coefficient=1.0,
        )
        lens_light = SersicEllipse(
            R_sersic=0.8,
            n_sersic=2.0,
            e1=0.02,
            e2=-0.01,
            center_x=0.03,
            center_y=-0.02,
            Ie=0.5,
        )
        lens_light.R_sersic.to_static()
        lens_light.n_sersic.to_static()
        lens_light.e1.to_static()
        lens_light.e2.to_static()
        lens_light.center_x.to_static()
        lens_light.center_y.to_static()
        lens_light.Ie.to_static()

        phys_model = PhysicalModel(
            lens_mass=setup['phys_model'].lens_mass,
            source_light=[pix_src],
            lens_light=[lens_light],
        )

        model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=phys_model,
            mask=setup['mask'],
            inversion_backend='matrix',
            include_lens_light=True,
            nonnegative=False,
            lens_light_ridge=1e-6,
        )

        data_vector = model.image_data[~model.mask]
        noise_variance = model.noise_map[~model.mask] ** 2
        source_i, lens_i, beta, model_image, inverter = model.simulator.reconstruct_source_and_lens_light(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=model.pix_src_model.reg_scale.value,
            reg_coefficient=model.pix_src_model.reg_coefficient.value,
            lens_light_ridge=model.lens_light_ridge,
            nonnegative=model.nonnegative,
            return_2d=True,
        )

        assert isinstance(inverter, LinearInversion)
        assert source_i.shape[0] == 12 * 9
        assert lens_i.shape[0] == 1
        assert beta.shape[0] == 12 * 9
        assert model_image.shape == setup['image'].shape
        assert np.isfinite(float(np.asarray(inverter.log_evidence())))
        assert not jnp.any(jnp.isnan(source_i))
        assert not jnp.any(jnp.isnan(lens_i))
        assert not jnp.any(jnp.isnan(model_image))

    def test_rectangular_bilinear_operator_backend_with_lens_light(self, prob_model_setup):
        """Rectangular operator backend supports joint source+lens-light inversion."""
        setup = prob_model_setup

        pix_src = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=RectangularGridConfig(nx=12, ny=9),
                regularization=RegularizationConfig(mode='sparse_rectangular', rect_scheme='gradient'),
            ),
            reg_scale=0.1,
            reg_coefficient=1.0,
        )
        lens_light = SersicEllipse(
            R_sersic=0.8,
            n_sersic=2.0,
            e1=0.02,
            e2=-0.01,
            center_x=0.03,
            center_y=-0.02,
            Ie=0.5,
        )
        lens_light.R_sersic.to_static()
        lens_light.n_sersic.to_static()
        lens_light.e1.to_static()
        lens_light.e2.to_static()
        lens_light.center_x.to_static()
        lens_light.center_y.to_static()
        lens_light.Ie.to_static()

        phys_model = PhysicalModel(
            lens_mass=setup['phys_model'].lens_mass,
            source_light=[pix_src],
            lens_light=[lens_light],
        )

        model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=phys_model,
            mask=setup['mask'],
            inversion_backend='operator',
            include_lens_light=True,
            nonnegative=False,
            lens_light_ridge=1e-6,
            evidence_mode='fast',
            cg_tol=1e-6,
            cg_maxiter=300,
            slq_probes=8,
            slq_steps=20,
        )

        data_vector = model.image_data[~model.mask]
        noise_variance = model.noise_map[~model.mask] ** 2
        source_i, lens_i, beta, model_image, inverter = model.simulator.reconstruct_source_and_lens_light(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=model.pix_src_model.reg_scale.value,
            reg_coefficient=model.pix_src_model.reg_coefficient.value,
            lens_light_ridge=model.lens_light_ridge,
            nonnegative=model.nonnegative,
            inversion_backend='operator',
            evidence_mode='fast',
            cg_tol=1e-6,
            cg_maxiter=300,
            slq_probes=8,
            slq_steps=20,
            return_2d=True,
        )

        assert isinstance(inverter, OperatorInversion)
        assert source_i.shape[0] == 12 * 9
        assert lens_i.shape[0] == 1
        assert beta.shape[0] == 12 * 9
        assert model_image.shape == setup['image'].shape
        assert np.isfinite(float(np.asarray(inverter.log_evidence())))
        assert not jnp.any(jnp.isnan(source_i))
        assert not jnp.any(jnp.isnan(lens_i))
        assert not jnp.any(jnp.isnan(model_image))

    def test_rectangular_joint_operator_matches_matrix(self, prob_model_setup):
        """Joint source+lens-light operator solution should match matrix backend."""
        setup = prob_model_setup

        pix_src = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=RectangularGridConfig(nx=10, ny=8),
                regularization=RegularizationConfig(mode='sparse_rectangular', rect_scheme='gradient'),
            ),
            reg_scale=0.1,
            reg_coefficient=1.0,
        )
        lens_light = SersicEllipse(
            R_sersic=0.8,
            n_sersic=2.0,
            e1=0.02,
            e2=-0.01,
            center_x=0.03,
            center_y=-0.02,
            Ie=0.5,
        )
        lens_light.R_sersic.to_static()
        lens_light.n_sersic.to_static()
        lens_light.e1.to_static()
        lens_light.e2.to_static()
        lens_light.center_x.to_static()
        lens_light.center_y.to_static()
        lens_light.Ie.to_static()

        phys_model = PhysicalModel(
            lens_mass=setup['phys_model'].lens_mass,
            source_light=[pix_src],
            lens_light=[lens_light],
        )

        model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=phys_model,
            mask=setup['mask'],
            inversion_backend='matrix',
            include_lens_light=True,
            nonnegative=False,
            lens_light_ridge=1e-6,
        )

        data_vector = model.image_data[~model.mask]
        noise_variance = model.noise_map[~model.mask] ** 2
        reg_scale = model.pix_src_model.reg_scale.value
        reg_coeff = model.pix_src_model.reg_coefficient.value

        inv_matrix = model.simulator.build_inverter(
            data_vector,
            noise_variance,
            reg_scale,
            reg_coeff,
            include_lens_light=True,
            lens_light_ridge=model.lens_light_ridge,
            nonnegative=False,
            inversion_backend='matrix',
        )
        inv_operator = model.simulator.build_inverter(
            data_vector,
            noise_variance,
            reg_scale,
            reg_coeff,
            include_lens_light=True,
            lens_light_ridge=model.lens_light_ridge,
            nonnegative=False,
            inversion_backend='operator',
            evidence_mode='accurate',
            cg_tol=1e-7,
            cg_maxiter=500,
            slq_probes=32,
            slq_steps=60,
        )

        x_matrix = inv_matrix.solve()
        x_operator = inv_operator.solve()

        n_src = 10 * 8
        assert isinstance(inv_matrix, LinearInversion)
        assert isinstance(inv_operator, OperatorInversion)
        assert x_matrix.shape[0] == n_src + 1
        assert x_operator.shape[0] == n_src + 1

        assert_allclose(x_operator, x_matrix, rtol=7e-2, atol=5e-3)

        m_matrix = inv_matrix.model_predict(x_matrix)
        m_operator = inv_operator.model_predict(x_operator)
        assert_allclose(m_operator, m_matrix, rtol=2e-2, atol=2e-3)

        log_ev_matrix = float(np.asarray(inv_matrix.log_evidence()))
        log_ev_operator = float(np.asarray(inv_operator.log_evidence()))
        diff = abs(log_ev_matrix - log_ev_operator)
        assert diff < 2.0 or diff / max(abs(log_ev_matrix), 1.0) < 6e-3
