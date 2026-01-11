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
    lens_mapping_matrix_from,
    build_psf_matrix_dense,
)
from TinyLensGpu.utils.inversion import (
    LinearInversion,
)
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.pixelized_source import (
    PixelizedSourceModel,
)
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
    PixelizedImageProbModel,
)


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
        
        # Diagonal should be 1 + jitter (1e-6)
        expected = 1.0 + 1e-6
        assert_allclose(diagonal, expected * jnp.ones_like(diagonal), rtol=1e-4,
                       err_msg="Diagonal should be ~1 (self-covariance)")


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
    
    def test_linear_inversion_full_noise_cov(self, simple_inversion_setup):
        """Test inversion with full noise covariance matrix."""
        d, F, noise_cov, H, _ = simple_inversion_setup
        
        # Convert diagonal to full matrix
        noise_cov_full = np.diag(noise_cov)
        
        inverter = LinearInversion(d=d, F=F, noise_cov=noise_cov_full, H=H)
        s_recon = inverter.solve()
        
        assert not jnp.any(jnp.isnan(s_recon)), "Full noise cov inversion has NaN values"
    
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
        model = PixelizedSourceModel()
        
        assert model.reg_scale.value == 0.05, "Default reg_scale should be 0.05"
        assert model.reg_coefficient.value == 1.0, "Default reg_coefficient should be 1.0"
        assert model.reg_type == 'exp', "Default reg_type should be 'exp'"
        assert model.n_source_points == 1500, "Default n_source_points should be 1500"
        assert model.mesh_alpha == 1.5, "Default mesh_alpha should be 1.5"
        assert model.k_neighbors == 5, "Default k_neighbors should be 5"
    
    def test_model_custom_values(self):
        """Test configuration with custom values."""
        model = PixelizedSourceModel(
            reg_scale=0.1,
            reg_coefficient=2.0,
            reg_type='matern32',
            n_source_points=1000,
            mesh_alpha=2.0,
            mesh_method='sobol',
            k_neighbors=7,
            interp_kernel='wendland_c2',
            radius_scale=2.0
        )
        
        assert model.reg_scale.value == 0.1
        assert model.reg_coefficient.value == 2.0
        assert model.reg_type == 'matern32'
        assert model.n_source_points == 1000
        assert model.mesh_alpha == 2.0
        assert model.mesh_method == 'sobol'
        assert model.k_neighbors == 7
        assert model.interp_kernel == 'wendland_c2'
        assert model.radius_scale == 2.0
    
    def test_model_get_config_dict(self):
        """Test get_config_dict method."""
        model = PixelizedSourceModel(reg_scale=0.1, reg_coefficient=2.0)
        config_dict = model.get_config_dict()
        
        assert isinstance(config_dict, dict), "Should return a dictionary"
        assert 'reg_scale' in config_dict
        assert 'reg_coefficient' in config_dict
        assert 'reg_type' in config_dict
        assert abs(config_dict['reg_scale'] - 0.1) < 1e-6
        assert abs(config_dict['reg_coefficient'] - 2.0) < 1e-6

    def test_model_repr(self):
        """Test model string representation."""
        model = PixelizedSourceModel(
            reg_scale=0.05,
            reg_coefficient=1.5,
            n_source_points=1000
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
        
        phys_model = PhysicalModel(lens_mass=[sie])
        
        # Create pixelized source model with fewer points for faster testing
        pix_src_model = PixelizedSourceModel(
            reg_scale=0.05,
            reg_coefficient=1.0,
            n_source_points=200,  # Reduced for test speed
            mesh_seed=42
        )
        
        # Convert mask format: our fixture has True=valid, model expects True=masked
        mask_out = ~simple_mask
        
        return {
            'image': simple_image,
            'noise': simple_noise_map,
            'psf': simple_psf,
            'mask': mask_out,
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
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            pix_src_model=setup['pix_src_model'],
            mask=setup['mask']
        )
        
        assert prob_model is not None, "Model should be created"
        assert prob_model.npix == setup['image'].shape[0], "npix should match image size"
    
    def test_prob_model_log_evidence(self, mock_lensing_setup):
        """Test that log_evidence computation returns finite value."""
        setup = mock_lensing_setup
        
        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            pix_src_model=setup['pix_src_model'],
            mask=setup['mask']
        )
        
        log_ev = prob_model.log_evidence()
        
        assert np.isfinite(log_ev), f"Log evidence should be finite, got {log_ev}"
    
    def test_prob_model_call(self, mock_lensing_setup):
        """Test that __call__ returns log evidence."""
        setup = mock_lensing_setup
        
        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            pix_src_model=setup['pix_src_model'],
            mask=setup['mask']
        )
        
        log_ev = prob_model()
        
        assert jnp.isfinite(log_ev), "Call should return finite log evidence"
    
    def test_prob_model_reconstruct_source(self, mock_lensing_setup):
        """Test source reconstruction."""
        setup = mock_lensing_setup
        
        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            pix_src_model=setup['pix_src_model'],
            mask=setup['mask']
        )
        
        source_intensities, source_mesh_beta, model_image = prob_model.reconstruct_source()
        
        n_source = setup['pix_src_model'].n_source_points
        npix = setup['image'].shape[0]
        
        assert source_intensities.shape == (n_source,), "Source intensities shape mismatch"
        assert source_mesh_beta.shape == (n_source, 2), "Source mesh beta shape mismatch"
        assert model_image.shape == (npix, npix), "Model image shape mismatch"
        
        assert not jnp.any(jnp.isnan(source_intensities)), "Source has NaN"
        assert not jnp.any(jnp.isnan(model_image)), "Model image has NaN"
    
    def test_prob_model_caching(self, mock_lensing_setup):
        """Test that repeated calls use caching correctly."""
        setup = mock_lensing_setup
        
        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            pix_src_model=setup['pix_src_model'],
            mask=setup['mask']
        )
        
        # First call
        log_ev1 = prob_model.log_evidence()
        
        # Second call (should use cache)
        log_ev2 = prob_model.log_evidence()
        
        assert log_ev1 == log_ev2, "Cached result should be identical"
    
    def test_prob_model_repr(self, mock_lensing_setup):
        """Test model string representation."""
        setup = mock_lensing_setup
        
        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            pix_src_model=setup['pix_src_model'],
            mask=setup['mask']
        )
        
        repr_str = repr(prob_model)
        assert 'PixelizedImageProbModel' in repr_str
    
    def test_prob_model_no_mask(self, mock_lensing_setup):
        """Test model without mask (uses full image)."""
        setup = mock_lensing_setup
        
        prob_model = PixelizedImageProbModel(
            image_data=setup['image'],
            noise_map=setup['noise'],
            psf_kernel=setup['psf'],
            dpix=setup['dpix'],
            phys_model=setup['phys_model'],
            pix_src_model=setup['pix_src_model'],
            mask=None  # No mask
        )
        
        log_ev = prob_model.log_evidence()
        assert np.isfinite(log_ev), "Should work without mask"
    
    def test_prob_model_different_reg_types(self, mock_lensing_setup):
        """Test model with different regularization types."""
        setup = mock_lensing_setup
        
        for reg_type in ['exp', 'gauss', 'matern32', 'matern52']:
            pix_src_model = PixelizedSourceModel(
                reg_scale=0.05,
                reg_coefficient=1.0,
                n_source_points=100,  # Small for speed
                reg_type=reg_type,
                mesh_seed=42
            )
            
            prob_model = PixelizedImageProbModel(
                image_data=setup['image'],
                noise_map=setup['noise'],
                psf_kernel=setup['psf'],
                dpix=setup['dpix'],
                phys_model=setup['phys_model'],
                pix_src_model=pix_src_model,
                mask=setup['mask']
            )
            
            log_ev = prob_model.log_evidence()
            assert np.isfinite(log_ev), f"Log evidence should be finite for {reg_type}"


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
