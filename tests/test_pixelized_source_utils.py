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
)


@pytest.fixture
def small_source_grid():
    """Return a small 5x5 source grid for mapping-matrix tests."""
    nx, ny = 5, 5
    half_size = 1.0
    source_x_axis, source_y_axis, source_x_mesh, source_y_mesh = build_source_grid(
        nx,
        ny,
        half_size,
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


if __name__ == "__main__":
    pytest.main()
