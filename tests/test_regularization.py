"""
TDD specifications for dense pixelized-source regularization builders.

These tests define the expected matrix contracts for traditional finite-
difference regularization and Gaussian-process kernel regularization.
"""

# pyright: reportMissingImports=false

import pytest
import jax.numpy as jnp

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


if __name__ == "__main__":
    pytest.main()
