"""Mapping strategies for pixelized-source reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import MappingConfig
from TinyLensGpu.utils.interpolation.kernels import get_interpolation_weights
from TinyLensGpu.utils.lensing import (
    dense_mapping_from_weights_indices,
    lens_mapping_operator_bilinear_rectangular_from,
)

from .artifacts import GridArtifacts, MappingArtifacts, OperatorCacheKey


class BaseMappingStrategy:
    """
    Abstract base class for source-to-image mapping strategies.

    Defines the interface for constructing the linear mapping operator :math:`F`
    where :math:`d = F s`. The operator maps the source surface brightness vector
    to the image plane flux.

    This strategy pattern allows for different interpolation schemes (e.g.,
    adaptive k-NN, rectangular bilinear) to be swapped interchangeably.
    """

    def build_dense(self, grid: GridArtifacts) -> jnp.ndarray:
        """
        Build the full dense mapping matrix.

        Parameters
        ----------
        grid : GridArtifacts
            The grid geometry artifacts containing source and data plane coordinates.

        Returns
        -------
        jnp.ndarray
            The dense mapping matrix of shape ``(n_data, n_source)``.
        """
        raise NotImplementedError

    def build_operator(self, grid: GridArtifacts) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Build the sparse mapping operator components (weights and indices).

        This is used for matrix-free operations to save memory. The result represents
        a sparse matrix where each row (image pixel) has a fixed number of non-zero
        entries (interpolation weights) corresponding to source nodes.

        Parameters
        ----------
        grid : GridArtifacts
            The grid geometry artifacts containing source and data plane coordinates.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            - **weights**: Interpolation weights of shape ``(n_data, k_nn)``.
            - **indices**: Source node indices of shape ``(n_data, k_nn)``.
        """
        raise NotImplementedError

    def operator_cache_key(self, grid: GridArtifacts) -> OperatorCacheKey:
        """
        Generate a cache key for the mapping operator.

        The key uniquely identifies the geometric configuration. If the key's signature
        matches a cached version, the expensive operator construction can be skipped.

        Parameters
        ----------
        grid : GridArtifacts
            The grid geometry artifacts.

        Returns
        -------
        OperatorCacheKey
            A hashable key representing the current mapping configuration.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class KnnKernelMappingStrategy(BaseMappingStrategy):
    """
    Mapping strategy using k-Nearest Neighbors (k-NN) interpolation.

    Suitable for irregular (adaptive) grids. It computes the mapping weights based on
    the distance between ray-traced image pixels and their nearest source nodes,
    using a specified kernel function (e.g., Gaussian, linear).
    """

    config: MappingConfig

    def build_dense(self, grid: GridArtifacts) -> jnp.ndarray:
        """
        Build dense matrix via sparse operator expansion.
        See :meth:`BaseMappingStrategy.build_dense`.
        """
        weights, indices = self.build_operator(grid)
        n_source = grid.source_mesh_beta.shape[0]
        return dense_mapping_from_weights_indices(weights, indices, int(n_source))

    def build_operator(self, grid: GridArtifacts) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Build k-NN sparse operator.
        See :meth:`BaseMappingStrategy.build_operator`.
        """
        weights, indices, _ = get_interpolation_weights(
            points=grid.source_mesh_beta,
            query_points=grid.data_mesh_beta,
            k_neighbors=int(self.config.k_neighbors),
            kernel=self.config.interp_kernel,
            radius_scale=float(self.config.radius_scale),
        )
        return weights, indices

    def operator_cache_key(self, grid: GridArtifacts) -> OperatorCacheKey:
        """
        Generate cache key based on data/source point statistics and config.
        See :meth:`BaseMappingStrategy.operator_cache_key`.
        """
        source_np = np.asarray(grid.source_mesh_beta, dtype=np.float32)
        data_np = np.asarray(grid.data_mesh_beta, dtype=np.float32)
        return OperatorCacheKey(
            signature=(
                "irregular_knn",
                tuple(source_np.shape),
                tuple(data_np.shape),
                float(source_np.sum()),
                float((source_np * source_np).sum()),
                float(data_np.sum()),
                float((data_np * data_np).sum()),
                int(self.config.k_neighbors),
                str(self.config.interp_kernel),
                float(self.config.radius_scale),
            )
        )


@dataclass(frozen=True)
class RectBilinearMappingStrategy(BaseMappingStrategy):
    """
    Mapping strategy using bilinear interpolation on a rectangular grid.

    Efficient for structured grids. It computes weights for the 4 surrounding
    grid points for each ray-traced image pixel.
    """

    def _grid_meta(self, grid: GridArtifacts) -> Tuple[int, int, Tuple[float, float, float, float]]:
        """Extract rectangular grid metadata (nx, ny, bounds)."""
        if grid.source_grid_shape is None or grid.source_grid_bounds is None:
            raise RuntimeError("Rectangular grid metadata missing for bilinear mapping.")
        ny, nx = grid.source_grid_shape
        return int(nx), int(ny), tuple(float(v) for v in grid.source_grid_bounds)

    def build_dense(self, grid: GridArtifacts) -> jnp.ndarray:
        """
        Build dense matrix via sparse operator expansion.
        See :meth:`BaseMappingStrategy.build_dense`.
        """
        nx, ny, _ = self._grid_meta(grid)
        weights, indices = self.build_operator(grid)
        return dense_mapping_from_weights_indices(weights, indices, int(nx) * int(ny))

    def build_operator(self, grid: GridArtifacts) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Build bilinear sparse operator (4 neighbors).
        See :meth:`BaseMappingStrategy.build_operator`.
        """
        nx, ny, bounds = self._grid_meta(grid)
        x_min, x_max, y_min, y_max = bounds
        weights, indices, _ = lens_mapping_operator_bilinear_rectangular_from(
            data_mesh_beta=grid.data_mesh_beta,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            nx=nx,
            ny=ny,
        )
        return weights, indices

    def operator_cache_key(self, grid: GridArtifacts) -> OperatorCacheKey:
        """
        Generate cache key based on grid geometry and data point statistics.
        See :meth:`BaseMappingStrategy.operator_cache_key`.
        """
        data_np = np.asarray(grid.data_mesh_beta, dtype=np.float32)
        return OperatorCacheKey(
            signature=(
                "rectangular_bilinear",
                tuple(data_np.shape),
                float(data_np.sum()),
                tuple(grid.source_grid_shape) if grid.source_grid_shape is not None else None,
                tuple(grid.source_grid_bounds) if grid.source_grid_bounds is not None else None,
            )
        )


def build_mapping_artifacts(
    strategy: BaseMappingStrategy,
    grid: GridArtifacts,
    *,
    need_dense: bool,
    need_operator: bool,
) -> MappingArtifacts:
    """
    Construct the mapping artifacts using the provided strategy.

    Helper function to orchestrate the creation of dense matrices or sparse operators
    based on the requirements.

    Parameters
    ----------
    strategy : BaseMappingStrategy
        The mapping strategy to use (e.g., KNN or Bilinear).
    grid : GridArtifacts
        The grid geometry data.
    need_dense : bool
        Whether to build and return the dense mapping matrix.
    need_operator : bool
        Whether to build and return the sparse operator (weights/indices).

    Returns
    -------
    MappingArtifacts
        Container holding the constructed mapping components.
    """
    dense_matrix: Optional[jnp.ndarray] = None
    operator_weights: Optional[jnp.ndarray] = None
    operator_indices: Optional[jnp.ndarray] = None

    if need_dense:
        dense_matrix = strategy.build_dense(grid)

    if need_operator:
        operator_weights, operator_indices = strategy.build_operator(grid)

    return MappingArtifacts(
        dense_matrix=dense_matrix,
        operator_weights=operator_weights,
        operator_indices=operator_indices,
    )
