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


def pairwise_source_distances(nx, ny, half_size):
    """Return pairwise Euclidean distances between source-grid pixels."""
    x_axis = jnp.linspace(-half_size, half_size, nx)
    y_axis = jnp.linspace(-half_size, half_size, ny)
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

        matrix = builder.matrix(1.0, kernel_scale=0.5)

        assert_valid_regularization_matrix(matrix, nx * ny)

    def test_zero_regularization_is_identity(self, small_source_grid_shape):
        """Test that zero-order regularization defaults to an identity penalty."""
        nx, ny = small_source_grid_shape
        builder = DenseRegularizationBuilder(nx, ny, "zero-order")

        matrix = builder.matrix(1.0)

        assert jnp.allclose(matrix, jnp.eye(nx * ny))


@pytest.mark.unit
class TestTraditionalRegularizationScaling:
    """Test physical-grid scaling for finite-difference regularization."""

    def test_first_order_regularization_scales_with_inverse_spacing_squared(self, small_source_grid_shape):
        """Test first-order matrix scaling when physical grid spacing changes."""
        nx, ny = small_source_grid_shape
        builder = DenseRegularizationBuilder(nx, ny, "first-order")

        matrix_half_size_1 = builder.matrix(1.0)
        matrix_half_size_2 = builder.matrix(2.0)

        # first-order: H = (D/dx).T @ (D/dx), so doubling dx divides H by 4.
        assert jnp.allclose(matrix_half_size_2, matrix_half_size_1 / 4.0, rtol=1e-5, atol=1e-6)

    def test_second_order_regularization_scales_with_inverse_spacing_fourth(self, small_source_grid_shape):
        """Test second-order matrix scaling when physical grid spacing changes."""
        nx, ny = small_source_grid_shape
        builder = DenseRegularizationBuilder(nx, ny, "second-order")

        matrix_half_size_1 = builder.matrix(1.0)
        matrix_half_size_2 = builder.matrix(2.0)

        # second-order: H = (L/dx^2).T @ (L/dx^2), so doubling dx divides H by 16.
        assert jnp.allclose(matrix_half_size_2, matrix_half_size_1 / 16.0, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
class TestGaussianProcessRegularization:
    """Test distance-kernel construction for GP-style dense regularization."""

    def test_exponential_regularization_uses_distance_kernel(self, small_source_grid_shape):
        """Test exponential kernel returns precision (inverse covariance) matrix."""
        nx, ny = small_source_grid_shape
        half_size = 1.0
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(nx, ny, "exponential", jitter=jitter)

        matrix = builder.matrix(half_size, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(nx, ny, half_size)
        covariance = jnp.exp(-distances / kernel_scale) + jitter * jnp.eye(nx * ny)
        expected = jnp.linalg.inv(covariance)

        assert jnp.allclose(matrix, expected, rtol=1e-5, atol=1e-6)

    def test_gaussian_regularization_uses_distance_kernel(self, small_source_grid_shape):
        """Test Gaussian kernel returns precision (inverse covariance) matrix."""
        nx, ny = small_source_grid_shape
        half_size = 1.0
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(nx, ny, "gaussian", jitter=jitter)

        matrix = builder.matrix(half_size, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(nx, ny, half_size)
        covariance = jnp.exp(-0.5 * (distances / kernel_scale) ** 2) + jitter * jnp.eye(nx * ny)
        expected = jnp.linalg.inv(covariance)

        assert jnp.allclose(matrix, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("reg_type,nu", [("matern32", 1.5), ("matern52", 2.5), ("matern72", 3.5)])
    def test_matern_regularization_uses_distance_kernel(self, small_source_grid_shape, reg_type, nu):
        nx, ny = small_source_grid_shape
        half_size = 1.0
        kernel_scale = 0.7
        jitter = 1e-6
        builder = DenseRegularizationBuilder(nx, ny, reg_type, jitter=jitter)

        matrix = builder.matrix(half_size, kernel_scale=kernel_scale)
        distances = pairwise_source_distances(nx, ny, half_size)
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
        expected = jnp.linalg.inv(covariance + jitter * jnp.eye(nx * ny))

        assert jnp.allclose(matrix, expected, rtol=1e-5, atol=1e-6)

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
