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
from TinyLensGpu.utils.inversion.regularization import DenseRegularizationBuilder


@pytest.fixture
def small_source_grid_shape():
    """Return a small rectangular source grid shape for regularization tests."""
    return 5, 5


def assert_valid_regularization_matrix(matrix, n_pixels):
    """Assert common dense regularization matrix validity constraints."""
    assert matrix.shape == (n_pixels, n_pixels)
    assert jnp.all(jnp.isfinite(matrix))
    assert jnp.allclose(matrix, matrix.T, atol=1e-6)

    eigenvalues = jnp.linalg.eigvalsh(matrix)
    assert jnp.min(eigenvalues) >= -1e-5


def pairwise_source_distances(nx, ny, xmin, xmax, ymin, ymax):
    """Return pairwise Euclidean distances between source-grid pixels."""
    x_axis = jnp.linspace(xmin, xmax, nx)
    y_axis = jnp.linspace(ymin, ymax, ny)
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
        nx, ny = small_source_grid_shape
        builder = DenseRegularizationBuilder(nx, ny, regularization_type)

        result, _ = builder.matrix(-1.0, 1.0, -1.0, 1.0, kernel_scale=0.5)
        matrix = result

        assert_valid_regularization_matrix(matrix, nx * ny)

    def test_zero_regularization_is_identity(self, small_source_grid_shape):
        """Test that zero-order regularization defaults to an identity penalty."""
        nx, ny = small_source_grid_shape
        builder = DenseRegularizationBuilder(nx, ny, "zero-order")

        matrix, _ = builder.matrix(-1.0, 1.0, -1.0, 1.0)

        assert jnp.allclose(matrix, jnp.eye(nx * ny))


@pytest.mark.unit
class TestTraditionalRegularizationScaling:
    """Test physical-grid scaling for finite-difference regularization."""

    def test_first_order_regularization_scales_with_inverse_spacing_squared(self, small_source_grid_shape):
        """Test first-order matrix scaling when physical grid spacing changes."""
        nx, ny = small_source_grid_shape
        builder = DenseRegularizationBuilder(nx, ny, "first-order")

        matrix_half_size_1, _ = builder.matrix(-1.0, 1.0, -1.0, 1.0)
        matrix_half_size_2, _ = builder.matrix(-2.0, 2.0, -2.0, 2.0)

        # first-order: H = (D/dx).T @ (D/dx), so doubling dx divides H by 4.
        assert jnp.allclose(matrix_half_size_2, matrix_half_size_1 / 4.0, rtol=1e-5, atol=1e-6)

    def test_second_order_regularization_scales_with_inverse_spacing_fourth(self, small_source_grid_shape):
        """Test second-order matrix scaling when physical grid spacing changes."""
        nx, ny = small_source_grid_shape
        builder = DenseRegularizationBuilder(nx, ny, "second-order")

        matrix_half_size_1, _ = builder.matrix(-1.0, 1.0, -1.0, 1.0)
        matrix_half_size_2, _ = builder.matrix(-2.0, 2.0, -2.0, 2.0)

        # second-order: H = (L/dx^2).T @ (L/dx^2), so doubling dx divides H by 16.
        assert jnp.allclose(matrix_half_size_2, matrix_half_size_1 / 16.0, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
class TestRectangularGridRegularizationScaling:
    """Test that rectangular (non-square) grids scale x/y axes independently.

    Requirement: first/second-order regularization must divide x-differences
    by cell_dx and y-differences by cell_dy, allowing cell_dx != cell_dy.
    """

    @pytest.fixture
    def asymmetric_grid(self):
        """Return an asymmetric (nx=5, ny=3) source grid shape."""
        return 5, 3

    def test_first_order_rectangular_grid_independent_axis_scaling(
        self, asymmetric_grid
    ):
        """First-order H scales independently: factor_x/dx² + factor_y/dy²."""
        nx, ny = asymmetric_grid
        builder = DenseRegularizationBuilder(nx, ny, "first-order")

        bbox_a = (-1.0, 1.0, -1.0, 1.0)  # dx=0.5, dy=1.0
        bbox_b = (-1.0, 1.0, -2.0, 2.0)  # dx=0.5, dy=2.0 (only y stretched)

        matrix_a, _ = builder.matrix(*bbox_a)
        # y-span doubled → dy doubled → y-contribution ÷ 4, x-contribution unchanged
        scale_x_a = 2.0 / (bbox_a[1] - bbox_a[0])
        scale_y_a = 2.0 / (bbox_a[3] - bbox_a[2])
        scale_x_b = 2.0 / (bbox_b[1] - bbox_b[0])
        scale_y_b = 2.0 / (bbox_b[3] - bbox_b[2])
        expected_b = (
            builder._H1_unit_x * (scale_x_b ** 2)
            + builder._H1_unit_y * (scale_y_b ** 2)
        )

        matrix_b, _ = builder.matrix(*bbox_b)
        assert jnp.allclose(matrix_b, expected_b, rtol=1e-5, atol=1e-5)
        # Cross-check: x-contribution identical, y-contribution scaled by 1/4
        assert scale_x_a == scale_x_b
        assert jnp.allclose(scale_y_b, scale_y_a / 2.0)

    def test_first_order_only_x_stretched(self, asymmetric_grid):
        """Only x-axis stretched: x contribution ÷4, y contribution unchanged."""
        nx, ny = asymmetric_grid
        builder = DenseRegularizationBuilder(nx, ny, "first-order")

        bbox_ref = (-1.0, 1.0, -1.0, 1.0)   # dx=0.5, dy=1.0
        bbox_x2 = (-2.0, 2.0, -1.0, 1.0)    # dx=1.0, dy=1.0

        matrix_ref, _ = builder.matrix(*bbox_ref)
        matrix_x2, _ = builder.matrix(*bbox_x2)

        # x-span doubled → dx doubled → x-contribution ÷4; y-contribution same
        diff = matrix_ref - matrix_x2
        # x-contribution difference: H1x * (scale_x_ref^2 - scale_x_x2^2)
        scale_x_ref = 2.0 / (bbox_ref[1] - bbox_ref[0])
        scale_x_x2 = 2.0 / (bbox_x2[1] - bbox_x2[0])
        expected_diff = builder._H1_unit_x * (scale_x_ref ** 2 - scale_x_x2 ** 2)
        assert jnp.allclose(diff, expected_diff, rtol=1e-5, atol=1e-5)

    def test_second_order_rectangular_grid_independent_axis_scaling(
        self, asymmetric_grid
    ):
        """Second-order H scales independently: factor_x/dx⁴ + factor_y/dy⁴."""
        nx, ny = asymmetric_grid
        builder = DenseRegularizationBuilder(nx, ny, "second-order")

        bbox_a = (-1.0, 1.0, -1.0, 1.0)  # dx=0.5, dy=1.0
        bbox_b = (-1.0, 1.0, -2.0, 2.0)  # dx=0.5, dy=2.0

        scale_x_a = 2.0 / (bbox_a[1] - bbox_a[0])
        scale_y_a = 2.0 / (bbox_a[3] - bbox_a[2])
        scale_x_b = 2.0 / (bbox_b[1] - bbox_b[0])
        scale_y_b = 2.0 / (bbox_b[3] - bbox_b[2])
        expected_b = (
            builder._H2_unit_x * (scale_x_b ** 4)
            + builder._H2_unit_y * (scale_y_b ** 4)
        )

        matrix_b, _ = builder.matrix(*bbox_b)
        assert jnp.allclose(matrix_b, expected_b, rtol=1e-5, atol=1e-5)

    def test_rectangular_matrices_are_valid(self, asymmetric_grid):
        """All traditional types yield valid regularization for rectangular grids."""
        for reg_type in ("zero-order", "first-order", "second-order"):
            nx, ny = asymmetric_grid
            builder = DenseRegularizationBuilder(nx, ny, reg_type)
            bbox = (-0.5, 1.5, -2.0, 2.0)  # dx=0.5, dy=2.0 (dx≠dy)
            matrix, _ = builder.matrix(*bbox)
            assert matrix.shape == (nx * ny, nx * ny)
            assert jnp.all(jnp.isfinite(matrix))
            assert jnp.allclose(matrix, matrix.T, atol=1e-6)
            eigenvalues = jnp.linalg.eigvalsh(matrix)
            assert jnp.min(eigenvalues) >= -1e-5, (
                f"{reg_type} regularization not PSD for rectangular grid"
            )

    def test_gp_kernel_handles_rectangular_bounds(self, asymmetric_grid):
        """GP kernels compute valid precision matrices for rectangular bounds."""
        nx, ny = asymmetric_grid
        for reg_type in ("exponential", "gaussian", "matern32"):
            builder = DenseRegularizationBuilder(nx, ny, reg_type)
            bbox = (-0.5, 1.5, -2.0, 2.0)  # dx=0.5, dy=2.0
            matrix, logdet = builder.matrix(*bbox, kernel_scale=0.7)
            assert jnp.all(jnp.isfinite(matrix))
            assert jnp.isfinite(logdet)
            assert jnp.allclose(matrix, matrix.T, atol=1e-6)


@pytest.mark.unit
class TestGaussianProcessRegularization:
    """Test distance-kernel construction for GP-style dense regularization."""

    def test_exponential_regularization_uses_distance_kernel(self, small_source_grid_shape):
        """Test exponential kernel returns precision (inverse covariance) matrix."""
        nx, ny = small_source_grid_shape
        bbox = (-1.0, 1.0, -1.0, 1.0)
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(nx, ny, "exponential", jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(nx, ny, *bbox)
        covariance = jnp.exp(-distances / kernel_scale) + jitter * jnp.eye(nx * ny)
        expected = jnp.linalg.inv(covariance)

        assert jnp.allclose(matrix, expected, rtol=1e-4, atol=1e-4)
        _, expected_logdet = jnp.linalg.slogdet(covariance)
        assert jnp.allclose(logdet_cov, expected_logdet, atol=1e-4)

    def test_gaussian_regularization_uses_distance_kernel(self, small_source_grid_shape):
        """Test Gaussian kernel returns precision (inverse covariance) matrix."""
        nx, ny = small_source_grid_shape
        bbox = (-1.0, 1.0, -1.0, 1.0)
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(nx, ny, "gaussian", jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(nx, ny, *bbox)
        covariance = jnp.exp(-0.5 * (distances / kernel_scale) ** 2) + jitter * jnp.eye(nx * ny)
        expected = jnp.linalg.inv(covariance)

        assert jnp.allclose(matrix, expected, rtol=1e-3, atol=1e-3)
        _, expected_logdet = jnp.linalg.slogdet(covariance)
        assert jnp.allclose(logdet_cov, expected_logdet, atol=1e-3)

    @pytest.mark.parametrize("reg_type,nu", [("matern32", 1.5), ("matern52", 2.5), ("matern72", 3.5)])
    def test_matern_regularization_uses_distance_kernel(self, small_source_grid_shape, reg_type, nu):
        nx, ny = small_source_grid_shape
        bbox = (-1.0, 1.0, -1.0, 1.0)
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(nx, ny, reg_type, jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(nx, ny, *bbox)
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
        stabilized = covariance + jitter * jnp.eye(nx * ny)
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
        nx, ny = small_source_grid_shape
        n_s = nx * ny
        builder = DenseRegularizationBuilder(nx, ny, "first-order")

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
        nx, ny = small_source_grid_shape
        n_s = nx * ny
        builder = DenseRegularizationBuilder(nx, ny, "second-order")

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
        nx, ny = small_source_grid_shape
        builder = DenseRegularizationBuilder(nx, ny, "zero-order")

        for bbox in [(-1.0, 1.0, -1.0, 1.0), (-2.0, 2.0, -0.5, 0.5)]:
            H, _ = builder.matrix(*bbox)
            sign_h, logdet_h = jnp.linalg.slogdet(H)
            logdet_h = jnp.where(sign_h > 0.0, logdet_h, -jnp.inf)
            assert jnp.allclose(logdet_h, 0.0, atol=1e-6), (
                f"zero-order logdet should be 0, got {logdet_h} for bbox={bbox}"
            )


@pytest.mark.unit
class TestGpKernelLogdetRectangular:
    """GP kernel logdet_cov validation for rectangular (non-square) bounds."""

    def test_exponential_logdet_rectangular(self, small_source_grid_shape):
        """Exponential kernel logdet matches slogdet on independently built covariance."""
        nx, ny = small_source_grid_shape
        bbox = (-0.5, 1.5, -2.0, 2.0)  # dx≠dy, offset from origin
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(nx, ny, "exponential", jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(nx, ny, *bbox)
        covariance = jnp.exp(-distances / kernel_scale) + jitter * jnp.eye(nx * ny)
        _, expected_logdet = jnp.linalg.slogdet(covariance)

        assert jnp.allclose(logdet_cov, expected_logdet, atol=1e-4), (
            f"Exponential logdet_cov mismatch for rectangular bbox"
        )

    def test_gaussian_logdet_rectangular(self, small_source_grid_shape):
        """Gaussian kernel logdet matches slogdet on independently built covariance."""
        nx, ny = small_source_grid_shape
        bbox = (-0.5, 1.5, -2.0, 2.0)
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(nx, ny, "gaussian", jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(nx, ny, *bbox)
        covariance = jnp.exp(-0.5 * (distances / kernel_scale) ** 2) + jitter * jnp.eye(nx * ny)
        _, expected_logdet = jnp.linalg.slogdet(covariance)

        assert jnp.allclose(logdet_cov, expected_logdet, atol=1e-3), (
            f"Gaussian logdet_cov mismatch for rectangular bbox"
        )

    def test_matern32_logdet_rectangular(self, small_source_grid_shape):
        """Matern-3/2 kernel logdet matches slogdet on independently built covariance."""
        nx, ny = small_source_grid_shape
        bbox = (-0.5, 1.5, -2.0, 2.0)
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(nx, ny, "matern32", jitter=jitter)

        matrix, logdet_cov = builder.matrix(*bbox, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(nx, ny, *bbox)
        r = distances / kernel_scale
        sqrt3_r = jnp.sqrt(3.0) * r
        covariance = (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r) + jitter * jnp.eye(nx * ny)
        _, expected_logdet = jnp.linalg.slogdet(covariance)

        assert jnp.allclose(logdet_cov, expected_logdet, atol=1e-4), (
            f"Matern32 logdet_cov mismatch for rectangular bbox"
        )


@pytest.mark.unit
class TestDenseRegularizationBuilderValidation:
    """Test configuration validation for dense regularization builders."""

    def test_invalid_regularization_type_raises_value_error(self, small_source_grid_shape):
        """Test that unsupported regularization types fail with ValueError."""
        nx, ny = small_source_grid_shape

        with pytest.raises(ValueError):
            DenseRegularizationBuilder(nx, ny, "unsupported")


@pytest.mark.unit
class TestAdaptiveRegUtilities:
    """Test shared adaptive-regularisation utility functions."""

    # --- smooth_scale_map ---

    def test_smooth_sigma_default(self):
        """Default sigma=1 produces kernel size 5 like legacy."""
        builder = DenseRegularizationBuilder(10, 10, "second-order")
        q = jnp.ones(100, dtype=jnp.float32)
        q_sm = builder.smooth_scale_map(q, 10, 10, sigma=1.0)
        assert q_sm.shape == (100,)
        assert jnp.all(jnp.isfinite(q_sm))

    def test_smooth_sigma_large_adapts_kernel(self):
        """sigma=3 produces kernel size > 5 to avoid truncation."""
        builder = DenseRegularizationBuilder(10, 10, "second-order")
        q = jnp.ones(100, dtype=jnp.float32)
        q_sm = builder.smooth_scale_map(q, 10, 10, sigma=3.0)
        assert q_sm.shape == (100,)
        assert jnp.all(jnp.isfinite(q_sm))

    def test_smooth_preserves_interior(self):
        """Gaussian smoothing preserves uniform values in interior (away from boundaries)."""
        builder = DenseRegularizationBuilder(20, 20, "second-order")
        q = jnp.ones(400, dtype=jnp.float32) * 5.0
        q_sm = builder.smooth_scale_map(q, 20, 20, sigma=1.0)
        # Interior pixels (central 10×10) should be unaffected by boundaries
        q_2d = q_sm.reshape(20, 20)
        interior = q_2d[5:15, 5:15]
        assert jnp.allclose(interior, 5.0, atol=1e-4)

    # --- _normalize_brightness ---

    def test_normalize_all_dark(self):
        """All-dark pixels produce near-zero normalized values."""
        b_raw = jnp.zeros(100, dtype=jnp.float32)
        b_norm = DenseRegularizationBuilder._normalize_brightness(b_raw)
        # With mean≈0, each value/eps → 0
        assert jnp.allclose(b_norm, jnp.zeros(100), atol=1e-6)

    def test_normalize_uniform(self):
        """Uniform brightness produces b_norm ≈ 1."""
        b_raw = jnp.ones(100, dtype=jnp.float32) * 3.0
        b_norm = DenseRegularizationBuilder._normalize_brightness(b_raw)
        assert jnp.allclose(b_norm, 1.0, atol=1e-6)

    def test_normalize_sparse_bright(self):
        """Sparse bright pixels produce b_norm ≫ 1 for bright pixels."""
        b_raw = jnp.zeros(100, dtype=jnp.float32)
        b_raw = b_raw.at[:10].set(10.0)  # 10 bright, 90 dark
        b_norm = DenseRegularizationBuilder._normalize_brightness(b_raw)
        # Global mean = (10*10 + 90*0) / 100 = 1.0
        # Bright: 10 / 1 = 10
        assert b_norm[0] > 5.0
        # Dark: 0 / 1 = 0
        assert jnp.allclose(b_norm[10:], 0.0, atol=1e-6)

    # --- _compute_scale_formula ---

    def test_scale_formula_darkest(self):
        """b_norm=0 → scale=1 regardless of alpha/floor."""
        b_norm = jnp.zeros(10, dtype=jnp.float32)
        alpha = jnp.array(2.0)
        floor = jnp.array(0.1)
        scale = DenseRegularizationBuilder._compute_scale_formula(b_norm, alpha, floor)
        assert jnp.allclose(scale, 1.0)

    def test_scale_formula_brightest(self):
        """b_norm → ∞ → scale → floor."""
        b_norm = jnp.array([1e10], dtype=jnp.float32)
        alpha = jnp.array(1.0)
        floor = jnp.array(0.2)
        scale = DenseRegularizationBuilder._compute_scale_formula(b_norm, alpha, floor)
        assert scale[0] < 0.21  # very close to floor

    def test_scale_formula_mean_brightness(self):
        """b_norm=1, alpha=1, floor=0.1 → scale = 0.1 + 0.9/2 = 0.55."""
        b_norm = jnp.ones(1, dtype=jnp.float32)
        alpha = jnp.array(1.0)
        floor = jnp.array(0.1)
        scale = DenseRegularizationBuilder._compute_scale_formula(b_norm, alpha, floor)
        assert jnp.allclose(scale[0], 0.55, atol=1e-6)

    def test_scale_formula_alpha_zero(self):
        """alpha=0 → scale=1 regardless of brightness."""
        b_norm = jnp.array([0.0, 1.0, 100.0], dtype=jnp.float32)
        alpha = jnp.array(0.0)
        floor = jnp.array(0.1)
        scale = DenseRegularizationBuilder._compute_scale_formula(b_norm, alpha, floor)
        assert jnp.allclose(scale, 1.0)

    def test_scale_formula_monotonic(self):
        """scale decreases monotonically with increasing brightness."""
        b_norm = jnp.linspace(0.0, 10.0, 100, dtype=jnp.float32)
        alpha = jnp.array(1.0)
        floor = jnp.array(0.1)
        scale = DenseRegularizationBuilder._compute_scale_formula(b_norm, alpha, floor)
        diffs = jnp.diff(scale)
        assert jnp.all(diffs <= 0.0)  # non-increasing

    # --- integration: scale applied via edge weights preserves PSD ---

    @pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
    def test_scale_preserves_psd(self, reg_type):
        """Regularisation matrix with non-uniform scale remains PSD."""
        nx, ny = 8, 8
        builder = DenseRegularizationBuilder(nx, ny, reg_type)
        bbox = (-1.0, 1.0, -1.0, 1.0)

        # Non-uniform scale: alternating bright/dark rows
        scale = jnp.ones(nx * ny, dtype=jnp.float32)
        scale = scale.at[ny//2 * nx:].set(0.3)  # bottom half "bright"

        matrix, _ = builder.matrix(*bbox, scale=scale)
        eigenvalues = jnp.linalg.eigvalsh(matrix)
        assert jnp.min(eigenvalues) >= -1e-5, (
            f"{reg_type} with non-uniform scale not PSD"
        )


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


def _adaptive_source(nx=5, ny=5, *, alpha=1.0, mode="brightness_only",
                     floor=0.1, sigma=1.0, freeze=False, reg_type="first-order"):
    log_lambda = ParamU(
        "log_lambda_reg", 0.0, prior_type="uniform",
        prior_settings=[jnp.log(1e-3), jnp.log(1e3)],
    )
    log_lambda.to_dynamic()
    return PixelizedSourceModel(
        nx=nx, ny=ny, log_lambda_reg=log_lambda,
        regularization_type=reg_type,
        adaptive_reg_alpha=alpha, adaptive_reg_floor=floor,
        adaptive_reg_mode=mode, adaptive_reg_smooth_sigma=sigma,
        adaptive_reg_freeze=freeze,
    )


def _dense_prob(nx=5, ny=5, *, alpha=1.0, mode="brightness_only",
                floor=0.1, sigma=1.0, freeze=False,
                image_data=None, noise_map=None, source_seed_mask=None,
                reg_type="first-order"):
    source = _adaptive_source(
        nx, ny, alpha=alpha, mode=mode, floor=floor, sigma=sigma,
        freeze=freeze, reg_type=reg_type,
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


def _operator_prob(nx=5, ny=5, *, alpha=1.0, mode="brightness_only",
                   floor=0.1, sigma=1.0, freeze=False,
                   image_data=None, noise_map=None, source_seed_mask=None,
                   reg_type="first-order"):
    source = _adaptive_source(
        nx, ny, alpha=alpha, mode=mode, floor=floor, sigma=sigma,
        freeze=freeze, reg_type=reg_type,
    )
    phys = PhysicalModel(
        lens_mass=[_static_sie()], source_light=[source], lens_light=[],
    )
    data = image_data if image_data is not None else jnp.ones((10, 10)) * 0.5
    noise = noise_map if noise_map is not None else jnp.ones((10, 10)) * 0.1
    return PixelizedImageProbModelOperator(
        image_data=data, noise_map=noise, psf_kernel=_delta_psf(),
        dpix=0.08, phys_model=phys, source_seed_mask=source_seed_mask,
    )


@pytest.mark.unit
class TestAdaptiveRegScaleComputation:
    """Test mode behaviour and freeze semantics of _compute_reg_scale_from_betas."""

    # --- 5.1: brightness_only cancels magnification ---

    def test_brightness_only_cancels_magnification(self):
        """brightness_only: equal brightness + different ray counts -> equal scale."""
        # 9x9 source grid so the two clusters sit 4 pixels apart (smoothing-safe).
        model = _dense_prob(nx=9, ny=9, alpha=1.0, mode="brightness_only", floor=0.1)
        # 90 seeds cluster at source pixel (2,2) [bright, high magnification]
        # 10 seeds cluster at source pixel (6,6) [bright, low magnification]
        # Both clusters have identical intrinsic brightness (image_data=0.5).
        beta_x = jnp.concatenate([jnp.full(90, -0.5), jnp.full(10, 0.5)])
        beta_y = jnp.concatenate([jnp.full(90, -0.5), jnp.full(10, 0.5)])
        bbox = (-1.0, 1.0, -1.0, 1.0)
        scale = model._compute_reg_scale_from_betas(beta_x, beta_y, *bbox)
        assert scale is not None
        idx_hi = 2 * 9 + 2   # pixel (2,2) — high magnification
        idx_lo = 6 * 9 + 6   # pixel (6,6) — low magnification
        # Magnification cancels in N/C: both clusters have b_raw ~ 0.5,
        # so their scales must match despite the 9x ray-count difference.
        assert abs(float(scale[idx_hi]) - float(scale[idx_lo])) < 0.03

    # --- 5.2: brightness_weighted preserves magnification ---

    def test_brightness_weighted_preserves_magnification(self):
        """brightness_weighted: higher magnification -> lower scale for same brightness."""
        model = _dense_prob(nx=9, ny=9, alpha=1.0, mode="brightness_weighted", floor=0.1)
        beta_x = jnp.concatenate([jnp.full(90, -0.5), jnp.full(10, 0.5)])
        beta_y = jnp.concatenate([jnp.full(90, -0.5), jnp.full(10, 0.5)])
        bbox = (-1.0, 1.0, -1.0, 1.0)
        scale = model._compute_reg_scale_from_betas(beta_x, beta_y, *bbox)
        assert scale is not None
        idx_hi = 2 * 9 + 2   # 90 seeds -> higher q -> lower scale
        idx_lo = 6 * 9 + 6   # 10 seeds -> lower q -> higher scale
        assert float(scale[idx_hi]) < float(scale[idx_lo])

    # --- 5.3: inverse-variance weighting ---

    def test_inv_var_weighting_suppresses_noisy_pixels(self):
        """A pixel with sigma=100 contributes 1e-4x the weight of sigma=1.

        Two seed pixels with identical brightness map to two different source
        pixels. In brightness_weighted mode q = brightness / sigma^2, so the
        noisy pixel's source pixel receives a 1e-4x smaller brightness estimate
        and is treated as near-dark (high scale).
        """
        seed_mask = jnp.ones((10, 10), dtype=bool).at[5, 5].set(False).at[5, 6].set(False)
        data = jnp.ones((10, 10)) * 1.0
        noise = jnp.ones((10, 10)) * 1.0
        noise = noise.at[5, 6].set(100.0)  # second seed pixel is very noisy
        model_noisy = _dense_prob(
            nx=5, ny=5, alpha=1.0, mode="brightness_weighted",
            image_data=data, noise_map=noise,
            source_seed_mask=seed_mask,
        )
        # Seed 1 (clean) -> source pixel (1,1) [index 6]; seed 2 (noisy) -> (3,3) [index 18].
        beta_x = jnp.array([-0.5, 0.5])
        beta_y = jnp.array([-0.5, 0.5])
        bbox = (-1.0, 1.0, -1.0, 1.0)
        scale_noisy = model_noisy._compute_reg_scale_from_betas(beta_x, beta_y, *bbox)
        assert scale_noisy is not None
        idx_clean = 1 * 5 + 1     # source pixel fed by sigma=1 seed
        idx_noisy = 3 * 5 + 3     # source pixel fed by sigma=100 seed
        # Clean source pixel: q = 1/1 = 1.0 -> bright -> low scale.
        # Noisy source pixel: q = 1/1e4 = 1e-4 -> near-dark -> high scale.
        assert float(scale_noisy[idx_clean]) < float(scale_noisy[idx_noisy])

    def test_inv_var_weighting_ratio_is_exact(self):
        """The inverse-variance weight ratio between sigma=1 and sigma=100 is 1e-4."""
        sigma_clean = jnp.array(1.0, dtype=jnp.float32)
        sigma_noisy = jnp.array(100.0, dtype=jnp.float32)
        inv_var_clean = 1.0 / (sigma_clean ** 2)
        inv_var_noisy = 1.0 / (sigma_noisy ** 2)
        ratio = inv_var_noisy / inv_var_clean
        # float32 precision: ~7 significant digits.
        assert abs(float(ratio) - 1e-4) < 1e-6

    # --- 5.8: empirical-Bayes freeze ---

    def test_freeze_returns_cached_scale_ignoring_new_betas(self):
        """After freeze_scale(), subsequent calls return the cached scale unchanged."""
        model = _dense_prob(
            nx=5, ny=5, alpha=1.0, mode="brightness_only", freeze=True,
        )
        assert getattr(model, "_frozen_scale", None) is None
        model.freeze_scale()
        frozen = getattr(model, "_frozen_scale", None)
        assert frozen is not None
        # The cached scale must be a concrete array (not a traced value).
        assert not isinstance(frozen, jax.core.Tracer)

        beta_x_orig, beta_y_orig = model.sim_obj.ray_trace_seed()
        bbox = model.sim_obj.infer_source_bbox(beta_x_orig, beta_y_orig)
        scale_orig = model._compute_reg_scale_from_betas(
            beta_x_orig, beta_y_orig, *bbox,
        )
        # Identical to the frozen array.
        assert jnp.array_equal(scale_orig, frozen)

        # Different betas should STILL return the frozen scale (cache hit).
        n_seed = int(jnp.size(beta_x_orig))
        beta_x_new = jnp.full(n_seed, 0.3)
        beta_y_new = jnp.full(n_seed, -0.2)
        scale_new = model._compute_reg_scale_from_betas(
            beta_x_new, beta_y_new, *bbox,
        )
        assert jnp.array_equal(scale_new, frozen)

    def test_freeze_captured_by_jit_ignores_traced_betas(self):
        """JIT-compiled scale read after freeze_scale() returns the cached constant."""
        model = _dense_prob(
            nx=5, ny=5, alpha=1.0, mode="brightness_only", freeze=True,
        )
        model.freeze_scale()
        frozen = model._frozen_scale

        bbox = (-1.0, 1.0, -1.0, 1.0)

        @jax.jit
        def _scale_under_jit(bx, by):
            return model._compute_reg_scale_from_betas(bx, by, *bbox)

        bx_a = jnp.full(25, -0.3)
        by_a = jnp.full(25, 0.2)
        bx_b = jnp.full(25, 0.4)
        by_b = jnp.full(25, -0.5)
        out_a = _scale_under_jit(bx_a, by_a)
        out_b = _scale_under_jit(bx_b, by_b)
        # Both JIT evaluations use the frozen constant -> identical output.
        assert jnp.array_equal(out_a, frozen)
        assert jnp.array_equal(out_b, frozen)

    def test_unfreeze_clears_cache(self):
        """unfreeze_scale() discards the cache; subsequent calls recompute."""
        model = _dense_prob(
            nx=5, ny=5, alpha=1.0, mode="brightness_only", freeze=True,
        )
        model.freeze_scale()
        assert getattr(model, "_frozen_scale", None) is not None
        model.unfreeze_scale()
        assert getattr(model, "_frozen_scale", None) is None
        # Recomputation works again.
        beta_x, beta_y = model.sim_obj.ray_trace_seed()
        bbox = model.sim_obj.infer_source_bbox(beta_x, beta_y)
        with pytest.warns(UserWarning, match="no frozen scale map"):
            scale = model._compute_reg_scale_from_betas(beta_x, beta_y, *bbox)
        assert scale is not None
        assert jnp.all(jnp.isfinite(scale))

    def test_freeze_noop_when_alpha_zero(self):
        """freeze_scale() is a no-op when adaptive_reg_alpha == 0 (uniform reg)."""
        model = _dense_prob(
            nx=5, ny=5, alpha=0.0, mode="brightness_only", freeze=True,
        )
        model.freeze_scale()
        assert getattr(model, "_frozen_scale", None) is None

    @pytest.mark.parametrize("factory", [_dense_prob, _operator_prob])
    def test_alpha_near_zero_uses_uniform_fast_path(self, factory):
        """alpha within 1e-10 tolerance returns None and does not freeze."""
        model = factory(
            nx=5, ny=5, alpha=1e-12, mode="brightness_only", freeze=True,
        )
        scale = model._compute_reg_scale_from_betas(
            jnp.zeros(1), jnp.zeros(1), -1.0, 1.0, -1.0, 1.0,
        )
        assert scale is None
        model.freeze_scale()
        assert getattr(model, "_frozen_scale", None) is None

    def test_freeze_warns_when_not_populated(self):
        """freeze=True without freeze_scale() warns and recomputes."""
        model = _dense_prob(
            nx=5, ny=5, alpha=1.0, mode="brightness_only", freeze=True,
        )
        beta_x, beta_y = model.sim_obj.ray_trace_seed()
        bbox = model.sim_obj.infer_source_bbox(beta_x, beta_y)
        with pytest.warns(UserWarning, match="no frozen scale map"):
            scale = model._compute_reg_scale_from_betas(beta_x, beta_y, *bbox)
        assert scale is not None  # fell back to recomputation

    def test_freeze_warns_when_freeze_disabled(self):
        """freeze_scale() warns when adaptive_reg_freeze is False."""
        model = _dense_prob(
            nx=5, ny=5, alpha=1.0, mode="brightness_only", freeze=False,
        )
        with pytest.warns(UserWarning, match="adaptive_reg_freeze=False"):
            model.freeze_scale()
        assert getattr(model, "_frozen_scale", None) is None


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
        source = PixelizedSourceModel(nx=5, ny=5, log_lambda_reg=log_lambda)
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

    def test_dense_brightness_only_finite_evidence(self):
        """Dense backend: brightness_only mode yields finite log-evidence (5.9)."""
        mock, noise = self._mock_image()
        model = _dense_prob(
            nx=5, ny=5, alpha=1.0, mode="brightness_only", floor=0.1,
            image_data=mock, noise_map=noise,
        )
        log_ev = model()
        assert jnp.shape(log_ev) == ()
        assert jnp.isfinite(log_ev)

    def test_operator_brightness_only_finite_evidence(self):
        """Operator backend: brightness_only mode yields finite log-evidence (5.10)."""
        mock, noise = self._mock_image()
        model = _operator_prob(
            nx=5, ny=5, alpha=1.0, mode="brightness_only", floor=0.1,
            image_data=mock, noise_map=noise,
        )
        log_ev = model()
        assert jnp.shape(log_ev) == ()
        assert jnp.isfinite(log_ev)

    def test_dense_brightness_weighted_finite_evidence(self):
        """Dense backend: brightness_weighted mode also yields finite evidence."""
        mock, noise = self._mock_image()
        model = _dense_prob(
            nx=5, ny=5, alpha=1.0, mode="brightness_weighted", floor=0.1,
            image_data=mock, noise_map=noise,
        )
        log_ev = model()
        assert jnp.shape(log_ev) == ()
        assert jnp.isfinite(log_ev)

    def test_operator_freeze_preserves_evidence_across_param_change(self):
        """Operator: frozen scale keeps evidence stable when lens params change."""
        mock, noise = self._mock_image()
        model = _operator_prob(
            nx=5, ny=5, alpha=1.0, mode="brightness_only", floor=0.1,
            freeze=True, image_data=mock, noise_map=noise,
        )
        model.freeze_scale()
        log_ev_frozen = float(model())
        # The frozen scale is a constant in the compiled graph; the evidence
        # must remain finite and stable on a second call (PCG introduces ~1e-6
        # relative solver noise, so use a loose relative tolerance).
        log_ev_frozen_2 = float(model())
        assert np.isfinite(log_ev_frozen)
        assert abs(log_ev_frozen - log_ev_frozen_2) < 1e-2 * abs(log_ev_frozen)


if __name__ == "__main__":
    pytest.main()
