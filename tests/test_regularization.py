"""
TDD specifications for dense pixelized-source regularization builders.

These tests define the expected matrix contracts for traditional finite-
difference regularization and Gaussian-process kernel regularization.
"""

# pyright: reportMissingImports=false

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
    PixelizedImageProbModel,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light.pixelized_source import (
    PixelizedSourceModel,
)
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.inversion.regularization import (
    DenseRegularizationBuilder,
    source_template_scale_map,
)


@pytest.fixture
def small_source_grid_shape():
    """Return a small source grid dimension for regularization tests."""
    return 5


def assert_valid_regularization_matrix(matrix, n_pixels):
    """Assert common dense regularization matrix validity constraints."""
    assert matrix.shape == (n_pixels, n_pixels)
    assert jnp.all(jnp.isfinite(matrix))
    assert jnp.allclose(matrix, matrix.T, atol=1e-6)

    eigenvalues = jnp.linalg.eigvalsh(matrix)
    assert jnp.min(eigenvalues) >= -1e-5


def pairwise_source_distances(n, xmin, xmax, ymin, ymax):
    """Return pairwise Euclidean distances between source-grid pixels."""
    x_axis = jnp.linspace(xmin, xmax, n)
    y_axis = jnp.linspace(ymin, ymax, n)
    source_x_mesh, source_y_mesh = jnp.meshgrid(x_axis, y_axis)
    coordinates = jnp.stack(
        [source_x_mesh.reshape(-1), source_y_mesh.reshape(-1)],
        axis=1,
    )
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return jnp.sqrt(jnp.sum(delta**2, axis=-1))


@pytest.mark.unit
class TestDenseRegularizationBuilder:
    """Test dense regularization matrix builder behavior."""

    @pytest.mark.parametrize(
        "regularization_type",
        ["zero-order", "first-order", "second-order", "exponential", "gaussian",
         "matern32", "matern52", "matern72"],
    )
    def test_regularization_matrices_are_valid(self, small_source_grid_shape, regularization_type):
        """Test shape, symmetry, finite values, and PSD stability for all types."""
        n = small_source_grid_shape
        builder = DenseRegularizationBuilder(n, regularization_type)

        result, _ = builder.matrix(-1.0, 1.0, -1.0, 1.0, kernel_scale=0.5)
        matrix = result

        assert_valid_regularization_matrix(matrix, n * n)

    def test_zero_regularization_is_identity(self, small_source_grid_shape):
        """Test that zero-order regularization defaults to an identity penalty."""
        n = small_source_grid_shape
        builder = DenseRegularizationBuilder(n, "zero-order")

        matrix, _ = builder.matrix(-1.0, 1.0, -1.0, 1.0)

        assert jnp.allclose(matrix, jnp.eye(n * n))


@pytest.mark.unit
class TestTraditionalRegularizationScaling:
    """Test physical-grid scaling for finite-difference regularization."""

    def test_first_order_regularization_scales_with_inverse_spacing_squared(self, small_source_grid_shape):
        """Test first-order matrix scaling when physical grid spacing changes."""
        n = small_source_grid_shape
        builder = DenseRegularizationBuilder(n, "first-order")

        matrix_half_size_1, _ = builder.matrix(-1.0, 1.0, -1.0, 1.0)
        matrix_half_size_2, _ = builder.matrix(-2.0, 2.0, -2.0, 2.0)

        # first-order: H = (D/dx).T @ (D/dx), so doubling dx divides H by 4.
        assert jnp.allclose(matrix_half_size_2, matrix_half_size_1 / 4.0, rtol=1e-5, atol=1e-6)

    def test_second_order_regularization_scales_with_inverse_spacing_fourth(self, small_source_grid_shape):
        """Test second-order matrix scaling when physical grid spacing changes."""
        n = small_source_grid_shape
        builder = DenseRegularizationBuilder(n, "second-order")

        matrix_half_size_1, _ = builder.matrix(-1.0, 1.0, -1.0, 1.0)
        matrix_half_size_2, _ = builder.matrix(-2.0, 2.0, -2.0, 2.0)

        # second-order: H = (L/dx^2).T @ (L/dx^2), so doubling dx divides H by 16.
        assert jnp.allclose(matrix_half_size_2, matrix_half_size_1 / 16.0, rtol=1e-5, atol=1e-6)


@pytest.mark.unit

class TestGaussianProcessRegularization:
    """Test distance-kernel construction for GP-style dense regularization."""

    def test_exponential_regularization_uses_distance_kernel(self, small_source_grid_shape):
        """Test exponential kernel returns precision (inverse covariance) matrix."""
        n = small_source_grid_shape
        bbox = (-1.0, 1.0, -1.0, 1.0)
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(n, "exponential", jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(n, *bbox)
        covariance = jnp.exp(-distances / kernel_scale) + jitter * jnp.eye(n * n)
        expected = jnp.linalg.inv(covariance)

        assert jnp.allclose(matrix, expected, rtol=1e-4, atol=1e-4)
        _, expected_logdet = jnp.linalg.slogdet(covariance)
        assert jnp.allclose(logdet_cov, expected_logdet, atol=1e-4)

    def test_gaussian_regularization_uses_distance_kernel(self, small_source_grid_shape):
        """Test Gaussian kernel returns precision (inverse covariance) matrix."""
        n = small_source_grid_shape
        bbox = (-1.0, 1.0, -1.0, 1.0)
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(n, "gaussian", jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(n, *bbox)
        covariance = jnp.exp(-0.5 * (distances / kernel_scale) ** 2) + jitter * jnp.eye(n * n)
        expected = jnp.linalg.inv(covariance)

        assert jnp.allclose(matrix, expected, rtol=1e-3, atol=1e-3)
        _, expected_logdet = jnp.linalg.slogdet(covariance)
        assert jnp.allclose(logdet_cov, expected_logdet, atol=1e-3)

    @pytest.mark.parametrize("reg_type,nu", [("matern32", 1.5), ("matern52", 2.5), ("matern72", 3.5)])
    def test_matern_regularization_uses_distance_kernel(self, small_source_grid_shape, reg_type, nu):
        n = small_source_grid_shape
        bbox = (-1.0, 1.0, -1.0, 1.0)
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(n, reg_type, jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(n, *bbox)
        r = distances / kernel_scale
        if nu == 1.5:
            sqrt3_r = jnp.sqrt(3.0) * r
            covariance = (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)
        elif nu == 2.5:
            sqrt5_r = jnp.sqrt(5.0) * r
            covariance = (1.0 + sqrt5_r + 5.0 * r ** 2 / 3.0) * jnp.exp(-sqrt5_r)
        else:
            sqrt7_r = jnp.sqrt(7.0) * r
            covariance = (1.0 + sqrt7_r + 14.0 * r ** 2 / 5.0 + 7.0 * jnp.sqrt(7.0) * r ** 3 / 15.0) * jnp.exp(-sqrt7_r)
        stabilized = covariance + jitter * jnp.eye(n * n)
        expected = jnp.linalg.inv(stabilized)

        assert jnp.allclose(matrix, expected, rtol=1e-4, atol=1e-4)
        _, expected_logdet = jnp.linalg.slogdet(stabilized)
        assert jnp.allclose(logdet_cov, expected_logdet, atol=1e-4)

@pytest.mark.unit
class TestSlogdetMatchesAnalyticalForSquareGrid:
    """Regression: slogdet on square-grid H must equal the old analytical formula.

    Before the rectangular bbox refactoring, logdet(H) was computed via
    ``logdet(H_unit) + scaling * log(half_size)`` which is exact for square
    grids because H(h) = H_unit / h^{2k}.  The new code uses slogdet
    unconditionally.  This test cross-validates both paths on square grids.
    """

    def test_slogdet_first_order_matches_analytical(self, small_source_grid_shape):
        """slogdet(H) == logdet(H_unit) + n_s * (-2) * log(h) for first-order."""
        n = small_source_grid_shape
        n_s = n * n
        builder = DenseRegularizationBuilder(n, "first-order")

        # --- analytical reference ---
        H_unit, _ = builder.matrix(-1.0, 1.0, -1.0, 1.0)  # h=1, scale=1
        sign_u, logdet_u = jnp.linalg.slogdet(H_unit)
        logdet_u = jnp.where(sign_u > 0.0, logdet_u, -jnp.inf)

        for h in (0.5, 2.0, 4.0):
            H, _ = builder.matrix(-h, h, -h, h)
            sign_h, logdet_h = jnp.linalg.slogdet(H)
            logdet_h = jnp.where(sign_h > 0.0, logdet_h, -jnp.inf)

            expected = logdet_u + n_s * (-2) * jnp.log(h)
            assert jnp.allclose(logdet_h, expected, rtol=1e-5, atol=1e-5), (
                f"first-order logdet mismatch at h={h}: slogdet={logdet_h}, "
                f"analytical={expected}"
            )

    def test_slogdet_second_order_matches_analytical(self, small_source_grid_shape):
        """slogdet(H) == logdet(H_unit) + n_s * (-4) * log(h) for second-order."""
        n = small_source_grid_shape
        n_s = n * n
        builder = DenseRegularizationBuilder(n, "second-order")

        # --- analytical reference ---
        H_unit, _ = builder.matrix(-1.0, 1.0, -1.0, 1.0)  # h=1
        sign_u, logdet_u = jnp.linalg.slogdet(H_unit)
        logdet_u = jnp.where(sign_u > 0.0, logdet_u, -jnp.inf)

        for h in (0.5, 2.0, 4.0):
            H, _ = builder.matrix(-h, h, -h, h)
            sign_h, logdet_h = jnp.linalg.slogdet(H)
            logdet_h = jnp.where(sign_h > 0.0, logdet_h, -jnp.inf)

            expected = logdet_u + n_s * (-4) * jnp.log(h)
            assert jnp.allclose(logdet_h, expected, rtol=1e-5, atol=1e-5), (
                f"second-order logdet mismatch at h={h}: slogdet={logdet_h}, "
                f"analytical={expected}"
            )

    def test_slogdet_zero_order_is_zero(self, small_source_grid_shape):
        """slogdet(identity) == 0 regardless of bbox."""
        n = small_source_grid_shape
        builder = DenseRegularizationBuilder(n, "zero-order")

        for bbox in [(-1.0, 1.0, -1.0, 1.0), (-2.0, 2.0, -0.5, 0.5)]:
            H, _ = builder.matrix(*bbox)
            sign_h, logdet_h = jnp.linalg.slogdet(H)
            logdet_h = jnp.where(sign_h > 0.0, logdet_h, -jnp.inf)
            assert jnp.allclose(logdet_h, 0.0, atol=1e-6), (
                f"zero-order logdet should be 0, got {logdet_h} for bbox={bbox}"
            )


@pytest.mark.unit
class TestGpKernelLogdetNonSquareBbox:
    """GP kernel logdet_cov validation for non-square bboxes on square grids."""

    def test_exponential_logdet_non_square_bbox(self, small_source_grid_shape):
        """Exponential kernel logdet matches slogdet on independently built covariance."""
        n = small_source_grid_shape
        bbox = (-0.5, 1.5, -2.0, 2.0)  # dx≠dy, offset from origin
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(n, "exponential", jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(n, *bbox)
        covariance = jnp.exp(-distances / kernel_scale) + jitter * jnp.eye(n * n)
        _, expected_logdet = jnp.linalg.slogdet(covariance)

        assert jnp.allclose(logdet_cov, expected_logdet, atol=1e-4), (
            f"Exponential logdet_cov mismatch for non-square bbox"
        )

    def test_gaussian_logdet_non_square_bbox(self, small_source_grid_shape):
        """Gaussian kernel logdet matches slogdet on independently built covariance."""
        n = small_source_grid_shape
        bbox = (-0.5, 1.5, -2.0, 2.0)
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(n, "gaussian", jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(n, *bbox)
        covariance = jnp.exp(-0.5 * (distances / kernel_scale) ** 2) + jitter * jnp.eye(n * n)
        _, expected_logdet = jnp.linalg.slogdet(covariance)

        assert jnp.allclose(logdet_cov, expected_logdet, atol=1e-3), (
            f"Gaussian logdet_cov mismatch for non-square bbox"
        )

    def test_matern32_logdet_non_square_bbox(self, small_source_grid_shape):
        """Matern-3/2 kernel logdet matches slogdet on independently built covariance."""
        n = small_source_grid_shape
        bbox = (-0.5, 1.5, -2.0, 2.0)
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(n, "matern32", jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(n, *bbox)
        r = distances / kernel_scale
        sqrt3_r = jnp.sqrt(3.0) * r
        covariance = (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r) + jitter * jnp.eye(n * n)
        _, expected_logdet = jnp.linalg.slogdet(covariance)

        assert jnp.allclose(logdet_cov, expected_logdet, atol=1e-4), (
            f"Matern32 logdet_cov mismatch for non-square bbox"
        )


@pytest.mark.unit
class TestDenseRegularizationBuilderValidation:
    """Test configuration validation for dense regularization builders."""

    def test_invalid_regularization_type_raises_value_error(self, small_source_grid_shape):
        """Test that unsupported regularization types fail with ValueError."""
        n = small_source_grid_shape

        with pytest.raises(ValueError):
            DenseRegularizationBuilder(n, "unsupported")


@pytest.mark.unit
class TestAdaptiveRegUtilities:
    """Test adaptive-regularisation edge-weight and scale-application utilities."""

    # --- integration: scale applied via edge weights preserves PSD ---

    @pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
    def test_scale_preserves_psd(self, reg_type):
        """Regularisation matrix with non-uniform scale remains PSD."""
        n = 8
        builder = DenseRegularizationBuilder(n, reg_type)
        bbox = (-1.0, 1.0, -1.0, 1.0)

        # Non-uniform scale: alternating bright/dark rows
        scale = jnp.ones(n * n, dtype=jnp.float32)
        scale = scale.at[n//2 * n:].set(0.3)  # bottom half "bright"

        matrix, _ = builder.matrix(*bbox, scale=scale)
        eigenvalues = jnp.linalg.eigvalsh(matrix)
        assert jnp.min(eigenvalues) >= -1e-5, (
            f"{reg_type} with non-uniform scale not PSD"
        )

    def test_galan_scale_edge_weight_is_geometric_mean(self):
        """A bright/dark adjacent edge has weight exp(rho / 2)."""
        n = 2
        rho = 2.0
        builder = DenseRegularizationBuilder(n, "first-order")
        scale_2d = jnp.asarray([
            [1.0, jnp.exp(rho)],
            [1.0, 1.0],
        ])

        w_x, _ = builder._edge_weights_first_order(scale_2d)

        assert np.isclose(float(w_x[0, 0]), float(jnp.exp(rho / 2.0)))


# ------------------------------------------------------------------
# Fixtures and helpers for adaptive scale-map computation tests
# ------------------------------------------------------------------

def _delta_psf():
    return jnp.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


def _static_sie():
    sie = SIE(theta_E=0.12, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for p in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
        p.to_static()
    return sie


def _adaptive_source(n=5, *, rho=1.0,
                     reg_type="first-order"):
    log_lambda = ParamU(
        "log_lambda_reg", 0.0, prior_type="uniform",
        prior_settings=[jnp.log(1e-3), jnp.log(1e3)],
    )
    log_lambda.to_dynamic()
    return PixelizedSourceModel(n=n, log_lambda_reg=log_lambda,
        regularization_type=reg_type,
        adaptive_reg_rho=rho,
    )


def _dense_prob(n=5, *, rho=1.0,
                image_data=None, noise_map=None, source_seed_mask=None,
                reg_type="first-order"):
    source = _adaptive_source(
        n, rho=rho, reg_type=reg_type,
    )
    phys = PhysicalModel(
        lens_mass=[_static_sie()], source_light=[source], lens_light=[],
    )
    data = image_data if image_data is not None else jnp.ones((10, 10)) * 0.5
    noise = noise_map if noise_map is not None else jnp.ones((10, 10)) * 0.1
    return PixelizedImageProbModel(
        image_data=data, noise_map=noise, psf_kernel=_delta_psf(),
        dpix=0.08, phys_model=phys, source_seed_mask=source_seed_mask,
    )


def _operator_prob(n=5, *, rho=1.0,
                   image_data=None, noise_map=None, source_seed_mask=None,
                   reg_type="first-order", fixed_source_bbox=None,
                   fixed_reg_scale=None):
    source = _adaptive_source(
        n, rho=rho, reg_type=reg_type,
    )
    phys = PhysicalModel(
        lens_mass=[_static_sie()], source_light=[source], lens_light=[],
    )
    data = image_data if image_data is not None else jnp.ones((10, 10)) * 0.5
    noise = noise_map if noise_map is not None else jnp.ones((10, 10)) * 0.1
    return PixelizedImageProbModelOperator(
        image_data=data, noise_map=noise, psf_kernel=_delta_psf(),
        dpix=0.08, phys_model=phys, source_seed_mask=source_seed_mask,
        fixed_source_bbox=fixed_source_bbox,
        fixed_reg_scale=fixed_reg_scale,
    )


@pytest.mark.unit
class TestSourceTemplateScaleMap:
    """Test S0-derived adaptive regularization scale maps."""

    def test_faint_pixels_receive_stronger_scale(self):
        source = jnp.array([
            [0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        scale = source_template_scale_map(source, 3, rho=2.0)
        assert scale is not None
        center = 1 * 3 + 1
        fainter = 2 * 3 + 1
        dark = 0
        assert float(scale[center]) < float(scale[fainter]) < float(scale[dark])
        assert np.isclose(float(scale[center]), 1.0)
        assert np.isclose(float(scale[dark]), float(jnp.exp(2.0)))

    def test_negative_source_pixels_are_clipped(self):
        source = jnp.array([-5.0, 0.0, 2.0, 0.0])
        scale = source_template_scale_map(source, 2, rho=1.0)
        assert scale is not None
        assert jnp.all(jnp.isfinite(scale))
        assert jnp.all(scale > 0.0)
        assert np.isclose(float(scale[0]), float(jnp.exp(1.0)))
        assert np.isclose(float(scale[2]), 1.0)

    def test_all_dark_template_is_finite_and_uniform(self):
        scale = source_template_scale_map(
            jnp.zeros((2, 2)), 2, rho=1.0,
        )
        assert scale is not None
        assert jnp.all(jnp.isfinite(scale))
        np.testing.assert_allclose(
            np.asarray(scale), np.ones(4) * np.exp(1.0), atol=1e-6,
        )

    def test_rho_zero_uses_uniform_fast_path(self):
        scale = source_template_scale_map(
            jnp.ones((2, 2)), 2, rho=0.0,
        )
        assert scale is None

    def test_traced_rho_changes_scale(self):
        source = jnp.array([0.0, 1.0, 3.0, 0.0])

        def build(rho):
            return source_template_scale_map(source, 2, rho=rho)

        scale_low = jax.jit(build)(jnp.asarray(0.5))
        scale_high = jax.jit(build)(jnp.asarray(3.0))

        assert scale_low is not None
        assert scale_high is not None
        assert scale_low.shape == (4,)
        assert jnp.all(jnp.isfinite(scale_low))
        assert jnp.all(jnp.isfinite(scale_high))
        assert float(scale_high[0]) > float(scale_low[0])
        assert np.isclose(float(scale_high[2]), 1.0)

    def test_traced_zero_rho_returns_uniform_scale(self):
        source = jnp.array([0.0, 1.0, 3.0, 0.0])

        @jax.jit
        def build(rho):
            return source_template_scale_map(
                source, 2, rho=rho,
            )

        scale = build(jnp.asarray(0.0))
        np.testing.assert_allclose(np.asarray(scale), np.ones(4), atol=1e-6)

    def test_accepts_flat_or_2d_source_templates(self):
        source_2d = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        scale_2d = source_template_scale_map(source_2d, 2, rho=1.0)
        scale_1d = source_template_scale_map(
            source_2d.ravel(), 2, rho=1.0,
        )
        np.testing.assert_allclose(np.asarray(scale_2d), np.asarray(scale_1d))

    def test_rejects_wrong_template_shape(self):
        with pytest.raises(ValueError, match="source_pixels must have shape"):
            source_template_scale_map(jnp.ones((3, 2)), 3, rho=1.0)

    def test_rejects_invalid_static_rho(self):
        with pytest.raises(ValueError, match="rho must be"):
            source_template_scale_map(jnp.ones((2, 2)), 2, rho=-1.0)

    def test_percentile_reference_clips_outliers(self):
        source = jnp.concatenate([jnp.ones(399), jnp.asarray([1000.0])])
        scale = source_template_scale_map(source, 20, rho=2.0)
        assert scale is not None
        assert np.isclose(float(scale[-1]), 1.0)


@pytest.mark.unit
class TestRetiredAdaptiveRegPath:
    """Old seed-ray adaptive regularization APIs are no longer active."""

    @pytest.mark.parametrize(
        "kwarg",
        [
            {"adaptive_reg_mode": "brightness_only"},
            {"adaptive_reg_smooth_sigma": 1.0},
            {"adaptive_reg_freeze": True},
        ],
    )
    def test_source_model_rejects_retired_kwargs(self, kwarg):
        with pytest.raises(TypeError):
            PixelizedSourceModel(n=5, log_lambda_reg=0.0,
                regularization_type="first-order",
                **kwarg,
            )

    def test_dense_backend_rejects_adaptive_reg(self):
        with pytest.raises(ValueError, match="no longer supports"):
            _dense_prob(n=5, rho=1.0)

    def test_dense_backend_has_no_freeze_api(self):
        model = _dense_prob(n=5, rho=0.0)
        assert not hasattr(model, "freeze_scale")
        assert not hasattr(model, "unfreeze_scale")


@pytest.mark.integration
class TestAdaptiveRegIntegration:
    """End-to-end evidence finiteness with adaptive regularization enabled."""

    @staticmethod
    def _mock_image():
        """Build a mock lensed image from a compact source for finite evidence."""
        sie = SIE(theta_E=0.12, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        for p in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            p.to_static()
        log_lambda = ParamU(
            "log_lambda_reg", 0.0, prior_type="uniform",
            prior_settings=[jnp.log(1e-3), jnp.log(1e3)],
        )
        log_lambda.to_dynamic()
        source = PixelizedSourceModel(n=5, log_lambda_reg=log_lambda)
        phys = PhysicalModel(
            lens_mass=[sie], source_light=[source], lens_light=[],
        )
        config = SimulatorConfig(
            dpix=0.08, npix=10,
            psf_kernel=_delta_psf(),
        )
        sim = PixelizedLensSimulator(phys, config)
        true_src = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
        mock = sim.simulate(true_src, psf_kernel=_delta_psf())
        noise = jnp.ones((10, 10)) * 0.05
        return mock, noise

    def test_dense_uniform_finite_evidence(self):
        """Dense backend remains finite for uniform pixelized regularization."""
        mock, noise = self._mock_image()
        model = _dense_prob(
            n=5, rho=0.0,
            image_data=mock, noise_map=noise,
        )
        log_ev = model()
        assert jnp.shape(log_ev) == ()
        assert jnp.isfinite(log_ev)

    def test_operator_s0_scale_finite_evidence(self):
        """Operator backend: fixed S0-derived scale yields finite evidence."""
        mock, noise = self._mock_image()
        s0 = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
        fixed_scale = source_template_scale_map(s0, 5, rho=1.0)
        model = _operator_prob(
            n=5, rho=1.0,
            image_data=mock, noise_map=noise,
            fixed_source_bbox=(-0.3, 0.3, -0.3, 0.3),
            fixed_reg_scale=fixed_scale,
        )
        log_ev = model()
        assert jnp.shape(log_ev) == ()
        assert jnp.isfinite(log_ev)

    def test_operator_s0_scale_reuses_evidence_inputs(self):
        """Repeated operator calls reuse the same fixed S0 scale and bbox."""
        mock, noise = self._mock_image()
        s0 = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
        fixed_scale = source_template_scale_map(s0, 5, rho=1.0)
        model = _operator_prob(
            n=5, rho=1.0,
            image_data=mock, noise_map=noise,
            fixed_source_bbox=(-0.3, 0.3, -0.3, 0.3),
            fixed_reg_scale=fixed_scale,
        )
        log_ev_1 = float(model())
        log_ev_2 = float(model())
        assert np.isfinite(log_ev_1)
        assert abs(log_ev_1 - log_ev_2) < 1e-2 * abs(log_ev_1)


if __name__ == "__main__":
    pytest.main()
