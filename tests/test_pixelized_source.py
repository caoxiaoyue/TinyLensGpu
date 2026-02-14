"""
Comprehensive unit tests for Pixelized Source Model.

This module tests the pixelized source reconstruction pipeline including:
- Source mesh sampling
- Regularization matrices (all kernel types)
- Lens mapping matrix construction
- PSF matrix construction
- Linear inversion (LinearInversion class)
- PixelizedSourceModel
- PixelizedImageProbModel (end-to-end integration)
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from TinyLensGpu.utils.mesh import (
    sample_points_weighted,
    apply_gaussian_blur,
)
from TinyLensGpu.utils.lensing import (
    exp_cov_matrix_from,
    gauss_cov_matrix_from,
    matern32_cov_matrix_from,
    matern52_cov_matrix_from,
    regularization_matrix_gp_from,
    regularization_sparse_knn_from,
    regularization_sparse_rectangular_from,
    sparse_regularization_dense_from,
    lens_mapping_matrix_from,
    build_psf_matrix_dense,
    apply_psf_to_mapping_matrix,
)
from TinyLensGpu.utils.inversion import (
    LinearInversion,
)
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.pixelized_source import (
    PixelizedSourceModel,
)
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import (
    IrregularGridConfig,
    MappingConfig,
    PixelizedSourceConfig,
    RectangularGridConfig,
    RegularizationConfig,
    SolverConfig,
)
from tests.pixelized_test_factory import build_pixelized_source_model
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
    PixelizedImageProbModel,
)
from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_image():
    """Generate a simple test image with a bright center."""
    np.random.seed(42)
    size = 50
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    # Gaussian blob at center
    image = np.exp(-(X**2 + Y**2) / 0.3) + np.random.randn(size, size) * 0.01
    image = np.clip(image, 0.01, None)
    return image


@pytest.fixture
def simple_mask(simple_image):
    """Generate a circular mask for the image."""
    size = simple_image.shape[0]
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    # Mask: True where data is valid (inside circle)
    mask = R < 0.9
    return mask


@pytest.fixture
def simple_psf():
    """Generate a simple Gaussian PSF kernel."""
    size = 11
    sigma = 1.5
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)
    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf = psf / np.sum(psf)
    return psf


@pytest.fixture
def simple_noise_map(simple_image):
    """Generate a uniform noise map."""
    return np.ones_like(simple_image) * 0.05


@pytest.fixture
def source_points():
    """Generate sample source points for testing."""
    np.random.seed(123)
    n_points = 100
    points = np.random.randn(n_points, 2) * 0.3
    return jnp.array(points, dtype=jnp.float32)


@pytest.fixture
def data_points():
    """Generate sample data points for testing."""
    np.random.seed(456)
    n_points = 200
    points = np.random.randn(n_points, 2) * 0.5
    return jnp.array(points, dtype=jnp.float32)


# =============================================================================
# Test Source Mesh Sampling
# =============================================================================

class TestSourceMeshSampling:
    """Tests for source mesh sampling functions."""
    
    def test_sample_points_weighted_basic(self, simple_image, simple_mask):
        """Test basic sampling returns correct number of points."""
        n_points = 500
        pts, (H, W), _ = sample_points_weighted(
            img=simple_image,
            mask=simple_mask,
            n_points=n_points,
            alpha=1.5,
            seed=42
        )
        
        assert pts.shape == (n_points, 2), f"Expected {n_points} points, got {pts.shape[0]}"
        assert H == simple_image.shape[0], "Height mismatch"
        assert W == simple_image.shape[1], "Width mismatch"
    
    def test_sample_points_weighted_reproducible(self, simple_image, simple_mask):
        """Test that sampling is reproducible with same seed."""
        pts1, _, _ = sample_points_weighted(
            img=simple_image, mask=simple_mask, n_points=100, seed=42
        )
        pts2, _, _ = sample_points_weighted(
            img=simple_image, mask=simple_mask, n_points=100, seed=42
        )
        
        assert_allclose(pts1, pts2, err_msg="Sampling should be reproducible with same seed")
    
    def test_sample_points_weighted_different_seeds(self, simple_image, simple_mask):
        """Test that different seeds produce different samples."""
        pts1, _, _ = sample_points_weighted(
            img=simple_image, mask=simple_mask, n_points=100, seed=42
        )
        pts2, _, _ = sample_points_weighted(
            img=simple_image, mask=simple_mask, n_points=100, seed=123
        )
        
        assert not np.allclose(pts1, pts2), "Different seeds should produce different samples"
    
    def test_sample_points_sobol_method(self, simple_image, simple_mask):
        """Test Sobol quasi-Monte Carlo sampling method."""
        n_points = 256  # Power of 2 for Sobol
        pts, (H, W), _ = sample_points_weighted(
            img=simple_image,
            mask=simple_mask,
            n_points=n_points,
            method='sobol',
            seed=42
        )
        
        assert pts.shape == (n_points, 2), f"Expected {n_points} points, got {pts.shape[0]}"
    
    def test_sample_points_with_blur(self, simple_image, simple_mask):
        """Test sampling with Gaussian blur applied."""
        pts, _, _ = sample_points_weighted(
            img=simple_image,
            mask=simple_mask,
            n_points=200,
            blur_sigma_px=2.0,
            seed=42
        )
        
        assert pts.shape[0] == 200, "Should return requested number of points"
    
    def test_sample_points_alpha_effect(self, simple_image, simple_mask):
        """Test that alpha parameter affects point distribution."""
        # High alpha: more concentration in bright areas
        pts_high, _, _ = sample_points_weighted(
            img=simple_image, mask=simple_mask, n_points=500,
            alpha=3.0, seed=42
        )
        # Low alpha: more uniform distribution
        pts_low, _, _ = sample_points_weighted(
            img=simple_image, mask=simple_mask, n_points=500,
            alpha=0.5, seed=42
        )
        
        # Check that high alpha has more points near center (where image is brightest)
        center = np.array([simple_image.shape[1]/2, simple_image.shape[0]/2])
        dist_high = np.mean(np.linalg.norm(pts_high - center, axis=1))
        dist_low = np.mean(np.linalg.norm(pts_low - center, axis=1))
        
        assert dist_high < dist_low, "Higher alpha should concentrate points in bright areas"
    
    def test_sample_points_mask_validation(self, simple_image):
        """Test that invalid mask raises appropriate errors."""
        with pytest.raises(TypeError):
            sample_points_weighted(
                img=simple_image,
                mask="invalid",  # Not a numpy array
                n_points=100
            )
        
        with pytest.raises(TypeError):
            sample_points_weighted(
                img=simple_image,
                mask=np.ones_like(simple_image),  # Not boolean
                n_points=100
            )
    
    def test_sample_points_invalid_method(self, simple_image, simple_mask):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown sampling method"):
            sample_points_weighted(
                img=simple_image,
                mask=simple_mask,
                n_points=100,
                method='invalid_method'
            )
    
    def test_apply_gaussian_blur(self, simple_image):
        """Test Gaussian blur application."""
        blurred = apply_gaussian_blur(simple_image, sigma=2.0)
        
        assert blurred.shape == simple_image.shape, "Blur should preserve shape"
        # Blur should reduce contrast
        assert np.std(blurred) <= np.std(simple_image), "Blur should reduce variance"
    
    def test_apply_gaussian_blur_zero_sigma(self, simple_image):
        """Test that zero sigma returns original image."""
        blurred = apply_gaussian_blur(simple_image, sigma=0)
        assert_allclose(blurred, simple_image, err_msg="Zero sigma should return original")


# =============================================================================
# Test Regularization Matrices
# =============================================================================

class TestRegularizationMatrices:
    """Tests for regularization matrix construction."""
    
    def test_exp_cov_matrix_shape(self, source_points):
        """Test exponential covariance matrix has correct shape."""
        n_points = source_points.shape[0]
        cov = exp_cov_matrix_from(scale_coefficient=0.1, pixel_points=source_points)
        
        assert cov.shape == (n_points, n_points), "Covariance matrix shape mismatch"
    
    def test_exp_cov_matrix_symmetry(self, source_points):
        """Test exponential covariance matrix is symmetric."""
        cov = exp_cov_matrix_from(scale_coefficient=0.1, pixel_points=source_points)
        
        assert_allclose(cov, cov.T, rtol=1e-5, err_msg="Covariance matrix should be symmetric")
    
    def test_exp_cov_matrix_positive_definite(self, source_points):
        """Test exponential covariance matrix is positive definite."""
        cov = exp_cov_matrix_from(scale_coefficient=0.1, pixel_points=source_points)
        eigenvalues = jnp.linalg.eigvalsh(cov)
        
        assert jnp.all(eigenvalues > 0), "Covariance matrix should be positive definite"
    
    def test_gauss_cov_matrix_shape(self, source_points):
        """Test Gaussian covariance matrix has correct shape."""
        n_points = source_points.shape[0]
        cov = gauss_cov_matrix_from(scale_coefficient=0.1, pixel_points=source_points)
        
        assert cov.shape == (n_points, n_points), "Covariance matrix shape mismatch"
    
    def test_gauss_cov_matrix_symmetry(self, source_points):
        """Test Gaussian covariance matrix is symmetric."""
        cov = gauss_cov_matrix_from(scale_coefficient=0.1, pixel_points=source_points)
        
        assert_allclose(cov, cov.T, rtol=1e-5, err_msg="Covariance matrix should be symmetric")
    
    def test_matern32_cov_matrix_shape(self, source_points):
        """Test Matern-3/2 covariance matrix has correct shape."""
        n_points = source_points.shape[0]
        cov = matern32_cov_matrix_from(scale_coefficient=0.1, pixel_points=source_points)
        
        assert cov.shape == (n_points, n_points), "Covariance matrix shape mismatch"
    
    def test_matern32_cov_matrix_positive_definite(self, source_points):
        """Test Matern-3/2 covariance matrix is positive definite."""
        cov = matern32_cov_matrix_from(scale_coefficient=0.1, pixel_points=source_points)
        eigenvalues = jnp.linalg.eigvalsh(cov)
        
        assert jnp.all(eigenvalues > 0), "Covariance matrix should be positive definite"
    
    def test_matern52_cov_matrix_shape(self, source_points):
        """Test Matern-5/2 covariance matrix has correct shape."""
        n_points = source_points.shape[0]
        cov = matern52_cov_matrix_from(scale_coefficient=0.1, pixel_points=source_points)
        
        assert cov.shape == (n_points, n_points), "Covariance matrix shape mismatch"
    
    def test_matern52_cov_matrix_positive_definite(self, source_points):
        """Test Matern-5/2 covariance matrix is positive definite."""
        cov = matern52_cov_matrix_from(scale_coefficient=0.1, pixel_points=source_points)
        eigenvalues = jnp.linalg.eigvalsh(cov)
        
        assert jnp.all(eigenvalues > 0), "Covariance matrix should be positive definite"
    
    def test_regularization_matrix_gp_all_types(self, source_points):
        """Test regularization matrix construction for all kernel types."""
        for reg_type in ['exp', 'gauss', 'matern32', 'matern52']:
            reg_matrix = regularization_matrix_gp_from(
                scale=0.1,
                coefficient=1.0,
                points=source_points,
                reg_type=reg_type
            )
            
            n_points = source_points.shape[0]
            assert reg_matrix.shape == (n_points, n_points), f"Shape mismatch for {reg_type}"
            assert not jnp.any(jnp.isnan(reg_matrix)), f"NaN in {reg_type} regularization matrix"
    
    def test_regularization_matrix_invalid_type(self, source_points):
        """Test that invalid regularization type raises error."""
        with pytest.raises(ValueError, match="Unknown reg_type"):
            regularization_matrix_gp_from(
                scale=0.1,
                coefficient=1.0,
                points=source_points,
                reg_type='invalid'
            )
    
    def test_regularization_matrix_coefficient_scaling(self, source_points):
        """Test that coefficient properly scales the regularization matrix."""
        reg1 = regularization_matrix_gp_from(
            scale=0.1, coefficient=1.0, points=source_points, reg_type='exp'
        )
        reg2 = regularization_matrix_gp_from(
            scale=0.1, coefficient=2.0, points=source_points, reg_type='exp'
        )
        
        assert_allclose(reg2, 2.0 * reg1, rtol=1e-5,
                       err_msg="Coefficient should linearly scale regularization matrix")
    
    def test_cov_diagonal_is_one(self, source_points):
        """Test that covariance matrix diagonal is approximately 1 (plus jitter)."""
        cov = exp_cov_matrix_from(scale_coefficient=0.1, pixel_points=source_points)
        diagonal = jnp.diag(cov)

        # Diagonal should be 1 + relative jitter
        # With new jitter calculation: jitter = 1e-6 * trace(C) / n
        # For exponential kernel, trace(C) ≈ n (since diagonal ≈ 1)
        # So jitter ≈ 1e-6
        expected = 1.0  # Base value
        assert_allclose(diagonal - 1e-6, expected * jnp.ones_like(diagonal), atol=1e-4,
                       err_msg="Diagonal should be ~1 (self-covariance)")


    def test_sparse_knn_no_self_neighbor_offdiag_under_duplicates(self):
        """Sparse KNN regularization should not include self-edges in off-diagonal terms."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)
        rows, cols, values, n_source = regularization_sparse_knn_from(
            scale=0.1,
            coefficient=1.0,
            points=jnp.array(points),
            reg_type='exp',
            k_neighbors=2,
        )

        rows = np.asarray(rows)
        cols = np.asarray(cols)
        n_diag = int(n_source)

        diag_mask = rows == cols
        # Exactly n_source diagonal terms should come from the explicit diagonal block.
        # Any additional diagonal entry would indicate a leaked self-neighbor edge.
        assert int(diag_mask.sum()) == n_diag

        dense = sparse_regularization_dense_from(rows, cols, values, n_source)
        dense_np = np.asarray(dense)
        assert_allclose(dense_np, dense_np.T, rtol=1e-6, atol=1e-6)

    def test_sparse_knn_is_differentiable_wrt_points(self, source_points):
        """Sparse KNN regularization path should support autodiff through points."""

        def loss_fn(points):
            rows, cols, values, n_source = regularization_sparse_knn_from(
                scale=0.1,
                coefficient=1.2,
                points=points,
                reg_type='exp',
                k_neighbors=8,
            )
            dense = sparse_regularization_dense_from(rows, cols, values, n_source)
            return jnp.sum(dense * dense)

        grad = jax.grad(loss_fn)(source_points)
        assert grad.shape == source_points.shape
        assert jnp.all(jnp.isfinite(grad))

    @pytest.mark.parametrize('scheme', ['zero', 'gradient', 'curvature'])
    def test_rectangular_sparse_regularization_symmetric(self, scheme):
        rows, cols, values, n_source = regularization_sparse_rectangular_from(
            coefficient=1.2,
            nx=12,
            ny=9,
            reg_scheme=scheme,
        )
        dense = sparse_regularization_dense_from(rows, cols, values, n_source)

        assert dense.shape == (108, 108)
        assert not jnp.any(jnp.isnan(dense))
        assert_allclose(dense, dense.T, rtol=1e-5, atol=1e-5)
        assert jnp.all(jnp.diag(dense) > 0.0)

    def test_rectangular_sparse_regularization_invalid_scheme(self):
        with pytest.raises(ValueError, match="Unknown reg_scheme"):
            regularization_sparse_rectangular_from(
                coefficient=1.0,
                nx=8,
                ny=8,
                reg_scheme='invalid',
            )


# =============================================================================
# Test Lens Mapping Matrix
# =============================================================================

class TestLensMappingMatrix:
    """Tests for lens mapping matrix construction."""
    
    def test_lens_mapping_matrix_shape(self, source_points, data_points):
        """Test lens mapping matrix has correct shape."""
        n_source = source_points.shape[0]
        n_data = data_points.shape[0]
        
        map_mat = lens_mapping_matrix_from(
            source_mesh_beta=source_points,
            data_mesh_beta=data_points,
            k_neighbors=5
        )
        
        assert map_mat.shape == (n_data, n_source), "Mapping matrix shape mismatch"
    
    def test_lens_mapping_matrix_row_sums(self, source_points, data_points):
        """Test that mapping matrix rows sum to approximately 1."""
        map_mat = lens_mapping_matrix_from(
            source_mesh_beta=source_points,
            data_mesh_beta=data_points,
            k_neighbors=5
        )
        
        row_sums = jnp.sum(map_mat, axis=1)
        assert_allclose(row_sums, jnp.ones_like(row_sums), rtol=0.1,
                       err_msg="Mapping matrix rows should sum to ~1")
    
    def test_lens_mapping_matrix_non_negative(self, source_points, data_points):
        """Test that mapping matrix entries are non-negative."""
        map_mat = lens_mapping_matrix_from(
            source_mesh_beta=source_points,
            data_mesh_beta=data_points,
            k_neighbors=5
        )
        
        assert jnp.all(map_mat >= 0), "Mapping matrix should have non-negative entries"
    
    def test_lens_mapping_matrix_sparsity(self, source_points, data_points):
        """Test that mapping matrix has expected sparsity pattern."""
        k = 5
        n_source = source_points.shape[0]
        n_data = data_points.shape[0]
        
        map_mat = lens_mapping_matrix_from(
            source_mesh_beta=source_points,
            data_mesh_beta=data_points,
            k_neighbors=k
        )
        
        # Each row should have exactly k non-zero entries
        nnz_per_row = jnp.sum(map_mat > 0, axis=1)
        assert jnp.all(nnz_per_row == k), f"Each row should have {k} non-zero entries"
    
    def test_lens_mapping_matrix_different_kernels(self, source_points, data_points):
        """Test mapping matrix with different kernel types."""
        for kernel in ['wendland_c2', 'wendland_c4', 'wendland_c6']:
            map_mat = lens_mapping_matrix_from(
                source_mesh_beta=source_points,
                data_mesh_beta=data_points,
                k_neighbors=5,
                kernel=kernel
            )
            
            assert not jnp.any(jnp.isnan(map_mat)), f"NaN in {kernel} mapping matrix"
            assert jnp.all(map_mat >= 0), f"Negative values in {kernel} mapping matrix"


# =============================================================================
# Test PSF Matrix
# =============================================================================

class TestPSFMatrix:
    """Tests for PSF matrix construction."""
    
    def test_psf_matrix_dense_shape(self, simple_mask, simple_psf):
        """Test dense PSF matrix has correct shape."""
        # Convert mask (True=valid) to expected format (True=masked out)
        mask_out = ~simple_mask
        n_valid = np.sum(simple_mask)
        
        psf_mat = build_psf_matrix_dense(mask_out, simple_psf)
        
        assert psf_mat.shape == (n_valid, n_valid), "PSF matrix shape mismatch"
    
    def test_psf_matrix_row_sums(self, simple_mask, simple_psf):
        """Test that PSF matrix rows sum to approximately 1 (for interior pixels)."""
        mask_out = ~simple_mask
        psf_mat = build_psf_matrix_dense(mask_out, simple_psf)
        
        row_sums = jnp.sum(psf_mat, axis=1)
        # Interior rows should sum to ~1 (edge effects may cause slight deviation)
        mean_row_sum = jnp.mean(row_sums)
        assert 0.7 < mean_row_sum < 1.1, f"Mean row sum {mean_row_sum} is unexpected"
    
    def test_psf_matrix_non_negative(self, simple_mask, simple_psf):
        """Test that PSF matrix entries are non-negative."""
        mask_out = ~simple_mask
        psf_mat = build_psf_matrix_dense(mask_out, simple_psf)
        
        assert jnp.all(psf_mat >= 0), "PSF matrix should have non-negative entries"
    
    def test_psf_matrix_identity_for_delta_psf(self, simple_mask):
        """Test that delta PSF gives identity-like matrix."""
        mask_out = ~simple_mask
        # Delta PSF (single pixel)
        delta_psf = np.zeros((11, 11))
        delta_psf[5, 5] = 1.0

        psf_mat = build_psf_matrix_dense(mask_out, delta_psf)

        # Should be approximately identity
        n = psf_mat.shape[0]
        identity = jnp.eye(n)
        assert_allclose(psf_mat, identity, rtol=1e-5,
                       err_msg="Delta PSF should give identity matrix")

    def test_psf_matrix_matches_reference_small(self):
        """Test that PSF matrix matches reference implementation for small case."""
        from TinyLensGpu.utils.lensing.psf import build_psf_matrix_dense

        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0] = True
        mask[5, 5] = True

        psf = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.1, 0.6, 0.1],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float32,
        )

        # Reference implementation
        inv_mask = np.array(~mask)
        psf_kernel = np.array(psf, dtype=np.float32)
        h_indices, w_indices = np.where(inv_mask)
        n_pixels = len(h_indices)
        psf_h, psf_w = psf_kernel.shape
        ch, cw = psf_h // 2, psf_w // 2

        ref = np.zeros((n_pixels, n_pixels), dtype=np.float32)
        for i in range(n_pixels):
            hi, wi = h_indices[i], w_indices[i]
            for j in range(n_pixels):
                hj, wj = h_indices[j], w_indices[j]
                dh = hi - hj + ch
                dw = wi - wj + cw
                if 0 <= dh < psf_h and 0 <= dw < psf_w:
                    ref[i, j] = psf_kernel[dh, dw]

        # Test implementation
        out = np.asarray(build_psf_matrix_dense(mask, psf))

        assert_allclose(out, ref, rtol=0.0, atol=0.0,
                       err_msg="PSF matrix should match reference implementation")

    def test_apply_psf_consistency_fft_matrix(self, simple_mask, simple_psf):
        """Test that FFT and Matrix methods for applying PSF produce consistent results."""
        # Setup data
        n_unmasked = np.sum(~simple_mask)
        n_source = 10
        mapping_matrix = jnp.ones((n_unmasked, n_source), dtype=jnp.float32)
        
        # Prepare indices
        y_indices, x_indices = np.where(~simple_mask)
        unmasked_indices = (jnp.array(y_indices), jnp.array(x_indices))
        image_shape = simple_mask.shape
        
        # Apply PSF using both methods
        res_fft = apply_psf_to_mapping_matrix(
            mapping_matrix, jnp.array(simple_psf), image_shape, unmasked_indices, method='fft'
        )
        
        res_matrix = apply_psf_to_mapping_matrix(
            mapping_matrix, jnp.array(simple_psf), image_shape, unmasked_indices, method='matrix'
        )
        
        # Check consistency
        assert_allclose(res_fft, res_matrix, rtol=1e-3, atol=1e-3,
                       err_msg="FFT and Matrix methods should produce consistent results")


# =============================================================================
# Test Linear Inversion
# =============================================================================

class TestLinearInversion:
    """Tests for LinearInversion class."""
    
    @pytest.fixture
    def simple_inversion_setup(self):
        """Create simple test data for inversion."""
        np.random.seed(42)
        n_data = 100
        n_source = 50
        
        # Simple mapping matrix
        F = np.random.randn(n_data, n_source).astype(np.float32)
        F = np.abs(F)
        F = F / F.sum(axis=1, keepdims=True)  # Normalize rows
        
        # True source
        s_true = np.random.randn(n_source).astype(np.float32) ** 2  # Positive
        
        # Data with noise
        d = F @ s_true + np.random.randn(n_data).astype(np.float32) * 0.1
        
        # Noise covariance (diagonal)
        noise_cov = np.ones(n_data, dtype=np.float32) * 0.01
        
        # Regularization (identity scaled)
        H = np.eye(n_source, dtype=np.float32) * 0.1
        
        return d, F, noise_cov, H, s_true
    
    def test_linear_inversion_solve(self, simple_inversion_setup):
        """Test that LinearInversion.solve() returns correct shape."""
        d, F, noise_cov, H, s_true = simple_inversion_setup
        
        inverter = LinearInversion(d=d, F=F, noise_cov=noise_cov, H=H)
        s_recon = inverter.solve()
        
        assert s_recon.shape == s_true.shape, "Reconstructed source shape mismatch"
        assert not jnp.any(jnp.isnan(s_recon)), "Reconstructed source has NaN values"
    
    def test_linear_inversion_invert(self, simple_inversion_setup):
        """Test that LinearInversion.invert() returns source and covariance."""
        d, F, noise_cov, H, _ = simple_inversion_setup
        
        inverter = LinearInversion(d=d, F=F, noise_cov=noise_cov, H=H)
        s_recon, Sigma = inverter.invert()
        
        n_source = F.shape[1]
        assert s_recon.shape == (n_source,), "Source shape mismatch"
        assert Sigma.shape == (n_source, n_source), "Covariance shape mismatch"
        
        # Covariance should be symmetric (relax tolerance for float32 precision)
        assert_allclose(Sigma, Sigma.T, rtol=1e-2, atol=1e-6, err_msg="Covariance should be symmetric")
    
    def test_linear_inversion_log_evidence(self, simple_inversion_setup):
        """Test that log_evidence returns finite value."""
        d, F, noise_cov, H, _ = simple_inversion_setup
        
        inverter = LinearInversion(d=d, F=F, noise_cov=noise_cov, H=H)
        log_ev = inverter.log_evidence()
        
        assert jnp.isfinite(log_ev), "Log evidence should be finite"

    def test_linear_inversion_log_evidence_nonpositive_determinant_returns_ninf(self, simple_inversion_setup):
        d, F, noise_cov, H, _ = simple_inversion_setup

        H_bad = np.array(H)
        H_bad[0, 0] = -abs(H_bad[0, 0])

        inverter = LinearInversion(d=d, F=F, noise_cov=noise_cov, H=H_bad)
        log_ev = inverter.log_evidence()
        assert bool(jnp.isneginf(log_ev))
    
    def test_linear_inversion_full_noise_cov(self, simple_inversion_setup):
        """Test inversion with full noise covariance matrix."""
        d, F, noise_cov, H, _ = simple_inversion_setup
        
        # Convert diagonal to full matrix
        noise_cov_full = np.diag(noise_cov)
        
        inverter = LinearInversion(d=d, F=F, noise_cov=noise_cov_full, H=H)
        s_recon = inverter.solve()
        
        assert not jnp.any(jnp.isnan(s_recon)), "Full noise cov inversion has NaN values"

    def test_linear_inversion_log_evidence_bad_full_noise_cov_returns_ninf(self, simple_inversion_setup):
        d, F, noise_cov, H, _ = simple_inversion_setup

        noise_cov_full = np.diag(noise_cov)
        noise_cov_full[0, 0] = -abs(noise_cov_full[0, 0])

        inverter = LinearInversion(d=d, F=F, noise_cov=noise_cov_full, H=H)
        log_ev = inverter.log_evidence()
        assert bool(jnp.isneginf(log_ev))
    
    def test_linear_inversion_reconstruction_quality(self, simple_inversion_setup):
        """Test that reconstruction is reasonably close to true source."""
        d, F, noise_cov, H, s_true = simple_inversion_setup
        
        inverter = LinearInversion(d=d, F=F, noise_cov=noise_cov, H=H)
        s_recon = inverter.solve()
        
        # Correlation should be positive (reconstruction follows trend)
        correlation = jnp.corrcoef(s_true, s_recon)[0, 1]
        assert correlation > 0.5, f"Reconstruction correlation {correlation} is too low"
    
    def test_linear_inversion_pytree(self, simple_inversion_setup):
        """Test that LinearInversion is properly registered as PyTree."""
        d, F, noise_cov, H, _ = simple_inversion_setup
        
        inverter = LinearInversion(d=d, F=F, noise_cov=noise_cov, H=H)
        
        # Test flatten and unflatten
        children, aux_data = inverter.tree_flatten()
        inverter_restored = LinearInversion.tree_unflatten(aux_data, children)
        
        # Should give same result
        s1 = inverter.solve()
        s2 = inverter_restored.solve()
        
        assert_allclose(s1, s2, rtol=1e-5, err_msg="PyTree round-trip should preserve result")


# =============================================================================
# Test PixelizedSourceModel
# =============================================================================

class TestPixelizedSourceModel:
    """Tests for PixelizedSourceModel class."""
    
    def test_model_default_values(self):
        """Test that default configuration values are set correctly."""
        model = build_pixelized_source_model()
        
        assert model.reg_scale.value == 0.05, "Default reg_scale should be 0.05"
        assert model.reg_coefficient.value == 1.0, "Default reg_coefficient should be 1.0"
        assert model.source_grid_type == 'irregular'
        assert model.regularization.gp_kernel == 'exp', "Default reg_type should be 'exp'"
        assert model.grid.n_source_points == 1500, "Default n_source_points should be 1500"
        assert model.grid.mesh_alpha == 0.0, "Default mesh_alpha should be 0.0"
        assert model.mapping.k_neighbors == 5, "Default k_neighbors should be 5"
        assert model.regularization.rect_scheme == 'gradient'
    
    def test_model_custom_values(self):
        """Test configuration with custom values."""
        model = PixelizedSourceModel(
            config=PixelizedSourceConfig(
                grid=RectangularGridConfig(nx=40, ny=28, margin_frac=0.2),
                mapping=MappingConfig(k_neighbors=7, interp_kernel='wendland_c2', radius_scale=2.0),
                regularization=RegularizationConfig(
                    mode='sparse_rectangular',
                    gp_kernel='matern32',
                    sparse_k_neighbors=16,
                    rect_scheme='curvature',
                ),
            ),
            reg_scale=0.1,
            reg_coefficient=2.0,
        )
        
        assert model.reg_scale.value == 0.1
        assert model.reg_coefficient.value == 2.0
        assert model.regularization.gp_kernel == 'matern32'
        assert model.grid.nx * model.grid.ny == 40 * 28
        assert model.mapping.k_neighbors == 7
        assert model.mapping.interp_kernel == 'wendland_c2'
        assert model.mapping.radius_scale == 2.0
        assert model.source_grid_type == 'rectangular_bilinear'
        assert model.grid.nx == 40
        assert model.grid.ny == 28
        assert model.grid.margin_frac == 0.2
        assert model.regularization.rect_scheme == 'curvature'

    def test_model_invalid_rectangular_config(self):
        with pytest.raises(ValueError, match="IrregularGridConfig cannot use regularization mode"):
            PixelizedSourceConfig(
                grid=IrregularGridConfig(),
                regularization=RegularizationConfig(mode='sparse_rectangular'),
            )

        with pytest.raises(ValueError, match="Unknown rect_scheme"):
            PixelizedSourceModel(
                config=PixelizedSourceConfig(
                    grid=RectangularGridConfig(nx=8, ny=8),
                    mapping=MappingConfig(),
                    regularization=RegularizationConfig(mode='sparse_rectangular', rect_scheme='unknown_scheme'),
                )
            )

    def test_model_rectangular_sparse_regularization_builder(self):
        model = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=RectangularGridConfig(nx=10, ny=7),
                regularization=RegularizationConfig(mode='sparse_rectangular', rect_scheme='gradient'),
            ),
            reg_coefficient=2.0,
        )
        rows, cols, values, n_source = model.regularization_sparse_rectangular(nx=10, ny=7)
        assert n_source == 70
        dense = sparse_regularization_dense_from(rows, cols, values, n_source)
        assert dense.shape == (70, 70)
        assert jnp.all(jnp.diag(dense) > 0.0)

    def test_model_rectangular_disables_dense_regularization_matrix(self):
        model = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=RectangularGridConfig(nx=8, ny=8),
                regularization=RegularizationConfig(mode='sparse_rectangular'),
            )
        )
        points = jnp.zeros((5, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="not available for source_grid_type='rectangular_bilinear'"):
            _ = model.regularization_matrix(points=points)
    
    def test_model_removed_legacy_flat_api_accessors(self):
        """Legacy flat config accessors are removed in favor of typed config."""
        model = build_pixelized_source_model()
        for attr_name in (
            'reg_type',
            'reg_operator_mode',
            'reg_sparse_k_neighbors',
            'rect_reg_type',
            'k_neighbors',
            'interp_kernel',
            'radius_scale',
            'n_source_points',
            'mesh_alpha',
            'mesh_blur_sigma',
            'mesh_method',
            'mesh_seed',
            'source_grid_nx',
            'source_grid_ny',
            'source_grid_margin_frac',
            'source_grid_bounds',
            'get_config_dict',
        ):
            assert not hasattr(model, attr_name), f"Legacy API attr should be removed: {attr_name}"

    def test_model_repr(self):
        """Test model string representation."""
        model = build_pixelized_source_model(
            config=PixelizedSourceConfig(grid=IrregularGridConfig(n_source_points=1000)),
            reg_scale=0.05,
            reg_coefficient=1.5,
        )
        
        repr_str = repr(model)
        assert 'PixelizedSourceModel' in repr_str
        assert '1000' in repr_str  # n_source_points


# =============================================================================
# Test PixelizedImageProbModel (Integration Tests)
# =============================================================================

class TestPixelizedImageProbModel:
    """Integration tests for PixelizedImageProbModel."""
    
    @pytest.fixture
    def mock_lensing_setup(self, simple_image, simple_noise_map, simple_psf, simple_mask):
        """Create a complete mock lensing setup for integration testing."""
        # Create SIE mass model
        sie = SIE(
            theta_E=1.0, e1=0.05, e2=-0.03,
            center_x=0.0, center_y=0.0
        )
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()

        # Create pixelized source model with fewer points for faster testing
        pix_src_model = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=IrregularGridConfig(n_source_points=200, mesh_seed=42),
            ),
            reg_scale=0.05,
            reg_coefficient=1.0,
        )

        phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src_model])
        
        # Convert mask format: our fixture has True=valid, model expects True=masked
        mask_out = ~simple_mask
        sim_config = SimulatorConfig(
            dpix=0.05,
            npix=simple_image.shape[0],
            psf_kernel=simple_psf,
            mask=mask_out,
        )
        
        return {
            'image': simple_image,
            'noise': simple_noise_map,
            'psf': simple_psf,
            'mask': mask_out,
            'sim_config': sim_config,
            'phys_model': phys_model,
            'pix_src_model': pix_src_model,
            'dpix': 0.05
        }
    
    def test_prob_model_construction(self, mock_lensing_setup):
        """Test that PixelizedImageProbModel can be constructed."""
        setup = mock_lensing_setup
        
        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            sim_config=setup['sim_config'],
            phys_model=setup['phys_model'],
        )
        assert prob_model is not None, "Model should be created"
        assert prob_model.npix == setup['image'].shape[0], "npix should match image size"
    
    def test_prob_model_log_evidence(self, mock_lensing_setup):
        """Test that log_evidence computation returns finite value."""
        setup = mock_lensing_setup
        
        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            sim_config=setup['sim_config'],
            phys_model=setup['phys_model'],
        )
        
        log_ev = prob_model.log_evidence()
        
        assert np.isfinite(log_ev), f"Log evidence should be finite, got {log_ev}"
    
    def test_prob_model_call(self, mock_lensing_setup):
        """Test that __call__ returns log evidence."""
        setup = mock_lensing_setup
        
        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            sim_config=setup['sim_config'],
            phys_model=setup['phys_model'],
        )
        
        log_ev = prob_model()
        
        assert jnp.isfinite(log_ev), "Call should return finite log evidence"

    def test_prob_model_position_likelihood_penalty_inactive_returns_zero(self, mock_lensing_setup):
        setup = mock_lensing_setup

        prob_model = PixelizedImageProbModel(
            image_data=setup["image"],
            noise_map=setup["noise"],
            sim_config=setup["sim_config"],
            phys_model=setup["phys_model"],
            position_likelihood={
                "positions": [(0.0, 0.0), (0.1, 0.1)],
                "threshold_arcsec": 1.0e3,
                "min_log_like": -10.0,
            },
        )

        penalty = float(np.asarray(prob_model._position_likelihood_penalty_jax()))
        assert np.isclose(penalty, 0.0)
    
    def test_prob_model_reconstruct_source(self, mock_lensing_setup):
        """Test source reconstruction via simulator."""
        setup = mock_lensing_setup

        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            sim_config=setup['sim_config'],
            phys_model=setup['phys_model'],
        )

        if isinstance(setup['pix_src_model'].grid, IrregularGridConfig):
            n_source = setup['pix_src_model'].grid.n_source_points
        else:
            n_source = setup['pix_src_model'].grid.nx * setup['pix_src_model'].grid.ny
        npix = setup['image'].shape[0]
        n_unmasked = jnp.sum(~setup['mask'])

        # Prepare arguments for simulator
        data_vector = prob_model.image_data[~prob_model.mask]
        noise_variance = prob_model.noise_map[~prob_model.mask] ** 2
        reg_scale = prob_model.pix_src_model.reg_scale.value
        reg_coefficient = prob_model.pix_src_model.reg_coefficient.value

        # Test default behavior (return_2d=False) - returns 1D vector
        source_intensities, source_mesh_beta, model_image_1d, _ = prob_model.simulator.reconstruct_source(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
            return_2d=False
        )

        assert source_intensities.shape == (n_source,), "Source intensities shape mismatch"
        assert source_mesh_beta.shape == (n_source, 2), "Source mesh beta shape mismatch"
        assert model_image_1d.shape == (n_unmasked,), "Model image 1D shape mismatch"
        assert not jnp.any(jnp.isnan(source_intensities)), "Source has NaN"
        assert not jnp.any(jnp.isnan(model_image_1d)), "Model image 1D has NaN"

        # Test return_2d=True - returns 2D array
        source_intensities, source_mesh_beta, model_image_2d, _ = prob_model.simulator.reconstruct_source(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
            return_2d=True
        )

        assert model_image_2d.shape == (npix, npix), "Model image 2D shape mismatch"
        assert not jnp.any(jnp.isnan(model_image_2d)), "Model image 2D has NaN"

        # Verify that the unmasked pixels match
        assert jnp.allclose(model_image_1d, model_image_2d[~setup['mask']]), "1D and 2D results should match at unmasked pixels"
    
    def test_prob_model_repr(self, mock_lensing_setup):
        """Test model string representation."""
        setup = mock_lensing_setup
        
        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            sim_config=setup['sim_config'],
            phys_model=setup['phys_model'],
        )
        
        repr_str = repr(prob_model)
        assert 'PixelizedImageProbModel' in repr_str
    
    def test_prob_model_no_mask(self, mock_lensing_setup):
        """Test model without mask (uses full image)."""
        setup = mock_lensing_setup
        sim_config_no_mask = SimulatorConfig(
            dpix=setup["dpix"],
            npix=setup["image"].shape[0],
            psf_kernel=setup["psf"],
            mask=None,
        )
        
        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            sim_config=sim_config_no_mask,
            phys_model=setup['phys_model'],
        )
        
        log_ev = prob_model.log_evidence()
        assert np.isfinite(log_ev), "Should work without mask"
    
    def test_prob_model_different_reg_types(self, mock_lensing_setup):
        """Test model with different regularization types."""
        setup = mock_lensing_setup
        
        for reg_type in ['exp', 'gauss', 'matern32', 'matern52']:
            pix_src_model = build_pixelized_source_model(
                config=PixelizedSourceConfig(
                    grid=IrregularGridConfig(n_source_points=100, mesh_seed=42),
                    regularization=RegularizationConfig(gp_kernel=reg_type),
                ),
                reg_scale=0.05,
                reg_coefficient=1.0,
            )

            sie = SIE(
                theta_E=1.0, e1=0.05, e2=-0.03,
                center_x=0.0, center_y=0.0
            )
            sie.theta_E.to_static()
            sie.e1.to_static()
            sie.e2.to_static()
            sie.center_x.to_static()
            sie.center_y.to_static()

            phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src_model])
            
            prob_model = PixelizedImageProbModel(
                image_data=setup['image'],
                noise_map=setup['noise'],
                sim_config=setup['sim_config'],
                phys_model=phys_model,
            )
            
            log_ev = prob_model.log_evidence()
            assert np.isfinite(log_ev), f"Log evidence should be finite for {reg_type}"

    def test_prob_model_rectangular_source_grid_operator_backend(self, mock_lensing_setup):
        setup = mock_lensing_setup

        sie = SIE(
            theta_E=1.0, e1=0.05, e2=-0.03,
            center_x=0.0, center_y=0.0
        )
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()

        pix_src_model = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=RectangularGridConfig(nx=24, ny=24, margin_frac=0.15),
                regularization=RegularizationConfig(mode='sparse_rectangular', rect_scheme='gradient'),
                solver=SolverConfig(
                    inversion_backend='operator',
                    cg_tol=1e-5,
                    cg_maxiter=200,
                    slq_probes=8,
                    slq_steps=20,
                ),
            ),
            reg_scale=0.05,
            reg_coefficient=1.0,
        )
        phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src_model])

        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            sim_config=setup['sim_config'],
            phys_model=phys_model,
        )

        log_ev = prob_model.log_evidence()
        assert np.isfinite(log_ev)

        data_vector = prob_model.image_data[~prob_model.mask]
        noise_variance = prob_model.noise_map[~prob_model.mask] ** 2
        s, beta, model_image, inv = prob_model.simulator.reconstruct_source(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=prob_model.pix_src_model.reg_scale.value,
            reg_coefficient=prob_model.pix_src_model.reg_coefficient.value,
            return_2d=True,
        )

        assert s.shape[0] == 24 * 24
        assert beta.shape[0] == 24 * 24
        assert model_image.shape == setup['image'].shape
        assert not jnp.any(jnp.isnan(model_image))
        assert getattr(inv, 'reg_operator_mode', None) == 'sparse_rectangular'

    def test_prob_model_rectangular_matrix_backend(self, mock_lensing_setup):
        """Rectangular source-grid supports matrix backend solve and evidence."""
        setup = mock_lensing_setup

        sie = SIE(theta_E=1.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()

        pix_src_model = build_pixelized_source_model(
            config=PixelizedSourceConfig(
                grid=RectangularGridConfig(nx=16, ny=16),
                regularization=RegularizationConfig(mode='sparse_rectangular', rect_scheme='zero'),
                solver=SolverConfig(inversion_backend='matrix'),
            ),
        )
        phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src_model])

        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            sim_config=setup['sim_config'],
            phys_model=phys_model,
        )

        log_ev = prob_model.log_evidence()
        assert np.isfinite(log_ev)

        data_vector = prob_model.image_data[~prob_model.mask]
        noise_variance = prob_model.noise_map[~prob_model.mask] ** 2
        s, beta, model_image, inv = prob_model.simulator.reconstruct_source(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=prob_model.pix_src_model.reg_scale.value,
            reg_coefficient=prob_model.pix_src_model.reg_coefficient.value,
            return_2d=True,
        )

        assert s.shape[0] == 16 * 16
        assert beta.shape[0] == 16 * 16
        assert model_image.shape == setup['image'].shape
        assert not jnp.any(jnp.isnan(model_image))
        assert isinstance(inv, LinearInversion)
        assert inv.H.shape == (16 * 16, 16 * 16)

    def test_prob_model_accepts_lensed_source_image(self, mock_lensing_setup):
        setup = mock_lensing_setup
        prob_model = PixelizedImageProbModel(
            image_data=setup["image"],
            noise_map=setup["noise"],
            sim_config=setup["sim_config"],
            phys_model=setup["phys_model"],
            lensed_source_image=setup["image"],
        )
        log_ev = prob_model.log_evidence()
        assert np.isfinite(log_ev)

    def test_prob_model_rejects_bad_lensed_source_image_shape(self, mock_lensing_setup):
        setup = mock_lensing_setup
        bad_shape = np.ones((setup["image"].shape[0] - 1, setup["image"].shape[1]))
        with pytest.raises(ValueError, match="lensed_source_image shape mismatch"):
            PixelizedImageProbModel(
                image_data=setup["image"],
                noise_map=setup["noise"],
                sim_config=setup["sim_config"],
                phys_model=setup["phys_model"],
                lensed_source_image=bad_shape,
            )

    def test_prob_model_rejects_image_shape_mismatch_with_sim_config(self, mock_lensing_setup):
        setup = mock_lensing_setup
        bad_image = setup["image"][:-1, :]
        bad_noise = setup["noise"][:-1, :]
        with pytest.raises(ValueError, match="image_data shape mismatch"):
            PixelizedImageProbModel(
                image_data=bad_image,
                noise_map=bad_noise,
                sim_config=setup["sim_config"],
                phys_model=setup["phys_model"],
            )


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_source_point(self):
        """Test with single source point."""
        points = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
        cov = exp_cov_matrix_from(scale_coefficient=0.1, pixel_points=points)

        assert cov.shape == (1, 1), "Should handle single point"
        assert cov[0, 0] > 0, "Diagonal should be positive"

    def test_very_small_scale(self):
        """Test regularization with very small scale."""
        np.random.seed(42)
        points = jnp.array(np.random.randn(10, 2).astype(np.float32) * 0.1)

    def test_very_small_scale(self):
        """Test regularization with very small scale."""
        np.random.seed(42)
        points = jnp.array(np.random.randn(10, 2).astype(np.float32) * 0.1)
        reg_matrix = regularization_matrix_gp_from(
            scale=1e-4,  # Very small
            coefficient=1.0,
            points=points,
            reg_type='exp'
        )

        assert not jnp.any(jnp.isnan(reg_matrix)), "Should handle small scale"

    def test_very_large_coefficient(self):
        """Test regularization with very large coefficient."""
        np.random.seed(42)
        points = jnp.array(np.random.randn(10, 2).astype(np.float32) * 0.1)
        
        reg_matrix = regularization_matrix_gp_from(
            scale=0.1,
            coefficient=1e6,  # Very large
            points=points,
            reg_type='exp'
        )
        
        assert not jnp.any(jnp.isnan(reg_matrix)), "Should handle large coefficient"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
