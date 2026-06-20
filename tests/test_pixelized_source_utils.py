"""
TDD specifications for pixelized source grid and lens mapping utilities.

These tests describe the expected public behavior for rectangular source-grid
helpers used by pixelized reconstructions. The implementation is intentionally
not provided here.
"""

# pyright: reportMissingImports=false

import pytest
import jax.numpy as jnp

from TinyLensGpu.utils.lensing.mapping import (
    build_lens_mapping_matrix,
    build_source_grid,
    infer_source_bbox,
)


@pytest.fixture
def small_source_grid():
    """Return a small 5x5 source grid for mapping-matrix tests."""
    nx, ny = 5, 5
    source_x_axis, source_y_axis, source_x_mesh, source_y_mesh = build_source_grid(
        nx,
        ny,
        -1.0,
        1.0,
        -1.0,
        1.0,
    )
    return source_x_axis, source_y_axis, source_x_mesh, source_y_mesh


@pytest.mark.unit
class TestBuildSourceGrid:
    """Test rectangular pixelized source-grid construction."""

    def test_three_by_three_grid_axes_and_mesh_shape(self):
        """Test that a 3x3 unit-half-size grid spans [-1, 0, 1]."""
        source_x_axis, source_y_axis, source_x_mesh, source_y_mesh = build_source_grid(
            3,
            3,
            -1.0,
            1.0,
            -1.0,
            1.0,
        )

        expected_axis = jnp.asarray([-1.0, 0.0, 1.0])

        assert jnp.allclose(source_x_axis, expected_axis)
        assert jnp.allclose(source_y_axis, expected_axis)
        assert source_x_mesh.shape == (3, 3)
        assert source_y_mesh.shape == (3, 3)
        assert jnp.allclose(source_x_mesh[0], expected_axis)
        assert jnp.allclose(source_y_mesh[:, 0], expected_axis)


@pytest.mark.unit
class TestBuildLensMappingMatrix:
    """Test dense image-to-source mapping matrix construction."""

    def test_exact_grid_point_rays_map_to_single_source_pixel(self, small_source_grid):
        """Test that rays at source pixels produce one-hot mapping rows."""
        source_x_axis, source_y_axis, _, _ = small_source_grid
        beta_x = jnp.asarray([-1.0, 0.0, 1.0])
        beta_y = jnp.asarray([-1.0, 0.0, 1.0])

        mapping_matrix = build_lens_mapping_matrix(
            beta_x,
            beta_y,
            source_x_axis,
            source_y_axis,
        )

        expected_columns = jnp.asarray([0, 12, 24])
        expected = jnp.zeros((3, 25))
        expected = expected.at[jnp.arange(3), expected_columns].set(1.0)

        assert mapping_matrix.shape == (3, 25)
        assert jnp.allclose(mapping_matrix, expected)
        assert jnp.allclose(jnp.sum(mapping_matrix, axis=1), 1.0)

    def test_out_of_bounds_rays_produce_zero_rows(self, small_source_grid):
        """Test that rays outside the source grid do not map to any pixel."""
        source_x_axis, source_y_axis, _, _ = small_source_grid
        beta_x = jnp.asarray([-1.5, 0.0, 1.5])
        beta_y = jnp.asarray([0.0, 1.5, 0.0])

        mapping_matrix = build_lens_mapping_matrix(
            beta_x,
            beta_y,
            source_x_axis,
            source_y_axis,
        )

        assert mapping_matrix.shape == (3, 25)
        assert jnp.allclose(mapping_matrix, jnp.zeros((3, 25)))

    def test_mixed_in_bounds_and_out_of_bounds_rows(self, small_source_grid):
        """Test that only valid rays contribute interpolation weights."""
        source_x_axis, source_y_axis, _, _ = small_source_grid
        beta_x = jnp.asarray([0.0, 2.0])
        beta_y = jnp.asarray([0.0, 0.0])

        mapping_matrix = build_lens_mapping_matrix(
            beta_x,
            beta_y,
            source_x_axis,
            source_y_axis,
        )

        assert mapping_matrix.shape == (2, 25)
        assert jnp.allclose(mapping_matrix[0, 12], 1.0)
        assert jnp.allclose(jnp.sum(mapping_matrix[0]), 1.0)
        assert jnp.allclose(mapping_matrix[1], jnp.zeros(25))

    def test_arbitrary_in_bounds_rays_have_row_weights_summing_to_one(self, small_source_grid):
        """Test bilinear interpolation weights sum to 1 for arbitrary in-bounds beta points."""
        source_x_axis, source_y_axis, _, _ = small_source_grid
        # Arbitrary in-bounds points (not on grid nodes)
        beta_x = jnp.asarray([-0.7, 0.3, 0.8, -0.2])
        beta_y = jnp.asarray([0.4, -0.6, 0.1, 0.9])

        mapping_matrix = build_lens_mapping_matrix(
            beta_x,
            beta_y,
            source_x_axis,
            source_y_axis,
        )

        row_sums = jnp.sum(mapping_matrix, axis=1)
        assert mapping_matrix.shape == (4, 25)
        assert jnp.allclose(row_sums, jnp.ones(4), atol=1e-6)

    def test_near_boundary_rays_weights_sum_to_one(self, small_source_grid):
        """Test bilinear interpolation near cell boundaries still sums to 1."""
        source_x_axis, source_y_axis, _, _ = small_source_grid
        # Points very close to cell boundaries (fx, fy near 0 or 1)
        beta_x = jnp.asarray([-1.0 + 1e-7, 1.0 - 1e-7, 0.0 + 1e-7])
        beta_y = jnp.asarray([-1.0 + 1e-7, 1.0 - 1e-7, 0.0 - 1e-7])

        mapping_matrix = build_lens_mapping_matrix(
            beta_x,
            beta_y,
            source_x_axis,
            source_y_axis,
        )

        row_sums = jnp.sum(mapping_matrix, axis=1)
        assert jnp.allclose(row_sums, jnp.ones(3), atol=1e-6)


@pytest.mark.unit
class TestInferSourceBbox:
    """Test standalone infer_source_bbox function."""

    def test_standalone_symmetric_betas(self):
        """Test infer_source_bbox as standalone with symmetric betas."""
        beta_x = jnp.asarray([-1.0, 1.0])
        beta_y = jnp.asarray([-0.5, 0.5])

        xmin, xmax, ymin, ymax = infer_source_bbox(
            beta_x, beta_y, padding=0.05, outlier_frac=0.0
        )

        assert jnp.allclose(xmin, -1.0 - 0.05 * 2.0)
        assert jnp.allclose(xmax, 1.0 + 0.05 * 2.0)
        assert jnp.allclose(ymin, -0.5 - 0.05 * 1.0)
        assert jnp.allclose(ymax, 0.5 + 0.05 * 1.0)

    def test_standalone_asymmetric_span(self):
        """Test infer_source_bbox where x and y spans differ."""
        beta_x = jnp.asarray([0.0, 3.0])
        beta_y = jnp.asarray([-0.2, 0.2])

        xmin, xmax, ymin, ymax = infer_source_bbox(
            beta_x, beta_y, padding=0.05, outlier_frac=0.0
        )

        assert jnp.allclose(xmin, 0.0 - 0.05 * 3.0)
        assert jnp.allclose(xmax, 3.0 + 0.05 * 3.0)
        assert jnp.allclose(ymin, -0.2 - 0.05 * 0.4)
        assert jnp.allclose(ymax, 0.2 + 0.05 * 0.4)

    def test_single_point_floor(self):
        """Test that a single beta point produces non-degenerate bbox."""
        beta = jnp.asarray([5.0])
        xmin, xmax, ymin, ymax = infer_source_bbox(
            beta, beta, padding=0.0, outlier_frac=0.0
        )
        assert xmax > xmin, "bbox should have positive span even for single point"
        assert ymax > ymin
        # With zero span, floor sets span = 1e-6 around the point
        assert jnp.allclose(xmin, 5.0 - 0.5e-6)
        assert jnp.allclose(xmax, 5.0 + 0.5e-6)

    def test_default_outlier_frac_ignores_extreme_points(self):
        """Test that default outlier_frac=0.01 trims 1% from each tail."""
        # 100 points linearly spaced in [0, 99]; 1% and 99% quantiles are 0.99 and 98.01
        beta_x = jnp.arange(100.0)
        beta_y = jnp.arange(100.0)

        xmin, xmax, ymin, ymax = infer_source_bbox(
            beta_x, beta_y, padding=0.0, outlier_frac=0.01
        )

        assert jnp.allclose(xmin, 0.99)
        assert jnp.allclose(xmax, 98.01)
        assert jnp.allclose(ymin, 0.99)
        assert jnp.allclose(ymax, 98.01)

    def test_outlier_frac_with_extreme_outliers(self):
        """Test that extreme outliers do not expand the bbox when trimmed."""
        # Bulk of points in [0, 10]; outliers at -1000 and +1000
        beta_x = jnp.concatenate(
            [jnp.linspace(0.0, 10.0, 100), jnp.asarray([-1000.0, 1000.0])]
        )
        beta_y = beta_x

        xmin, xmax, ymin, ymax = infer_source_bbox(
            beta_x, beta_y, padding=0.0, outlier_frac=0.01
        )

        # The outliers should be trimmed away; bbox should reflect the bulk.
        assert xmin > 0.0
        assert xmax < 10.0
        assert ymin > 0.0
        assert ymax < 10.0

    @pytest.mark.parametrize("bad_frac", [-0.1, 0.5, 1.0])
    def test_outlier_frac_out_of_range_raises(self, bad_frac):
        """Test that outlier_frac outside [0, 0.5) raises ValueError."""
        beta = jnp.asarray([1.0, 2.0])
        with pytest.raises(ValueError, match="outlier_frac"):
            infer_source_bbox(beta, beta, outlier_frac=bad_frac)


@pytest.mark.unit
class TestBuildSourceGridOffset:
    """Test build_source_grid with non-symmetric, offset bounds."""

    def test_offset_bounds_grid_axes(self):
        """Test grid axes span exactly [xmin, xmax] and [ymin, ymax]."""
        x_axis, y_axis, x_mesh, y_mesh = build_source_grid(3, 4, 0.5, 1.5, -2.0, -0.5)

        assert jnp.allclose(x_axis, jnp.asarray([0.5, 1.0, 1.5]))
        assert jnp.allclose(y_axis, jnp.asarray([-2.0, -1.5, -1.0, -0.5]))
        assert x_mesh.shape == (4, 3)
        assert y_mesh.shape == (4, 3)

    def test_offset_bounds_meshgrid_consistency(self):
        """Test that meshgrid coordinates are consistent with axes."""
        x_axis, y_axis, x_mesh, y_mesh = build_source_grid(3, 3, 0.5, 1.5, -2.0, 0.0)

        # All rows of x_mesh should equal x_axis
        assert jnp.allclose(x_mesh[0], x_axis)
        assert jnp.allclose(x_mesh[1], x_axis)
        # All columns of y_mesh should equal y_axis
        assert jnp.allclose(y_mesh[:, 0], y_axis)
        assert jnp.allclose(y_mesh[:, 1], y_axis)


if __name__ == "__main__":
    pytest.main()
