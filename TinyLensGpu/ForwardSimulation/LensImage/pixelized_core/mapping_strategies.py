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
    Represent the `BaseMappingStrategy` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    def build_dense(self, grid: GridArtifacts) -> jnp.ndarray:
        """
        Compute build dense.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        Raises
        ------
        NotImplementedError
            Raised when input validation fails or required runtime state is missing.
        
        """
        raise NotImplementedError

    def build_operator(self, grid: GridArtifacts) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute build operator.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        Raises
        ------
        NotImplementedError
            Raised when input validation fails or required runtime state is missing.
        
        """
        raise NotImplementedError

    def operator_cache_key(self, grid: GridArtifacts) -> OperatorCacheKey:
        """
        Compute operator cache key.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        Raises
        ------
        NotImplementedError
            Raised when input validation fails or required runtime state is missing.
        
        """
        raise NotImplementedError


@dataclass(frozen=True)
class KnnKernelMappingStrategy(BaseMappingStrategy):
    """
    Represent the `KnnKernelMappingStrategy` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    config: MappingConfig

    def build_dense(self, grid: GridArtifacts) -> jnp.ndarray:
        """
        Compute build dense.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        weights, indices = self.build_operator(grid)
        n_source = grid.source_mesh_beta.shape[0]
        return dense_mapping_from_weights_indices(weights, indices, int(n_source))

    def build_operator(self, grid: GridArtifacts) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute build operator.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
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
        Compute operator cache key.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
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
    Represent the `RectBilinearMappingStrategy` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    def _grid_meta(self, grid: GridArtifacts) -> Tuple[int, int, Tuple[float, float, float, float]]:
        """
        Internal helper to grid meta.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        Raises
        ------
        RuntimeError
            Raised when input validation fails or required runtime state is missing.
        
        """
        if grid.source_grid_shape is None or grid.source_grid_bounds is None:
            raise RuntimeError("Rectangular grid metadata missing for bilinear mapping.")
        ny, nx = grid.source_grid_shape
        return int(nx), int(ny), tuple(float(v) for v in grid.source_grid_bounds)

    def build_dense(self, grid: GridArtifacts) -> jnp.ndarray:
        """
        Compute build dense.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        nx, ny, _ = self._grid_meta(grid)
        weights, indices = self.build_operator(grid)
        return dense_mapping_from_weights_indices(weights, indices, int(nx) * int(ny))

    def build_operator(self, grid: GridArtifacts) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute build operator.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
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
        Compute operator cache key.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
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
    Compute build mapping artifacts.
    
    Parameters
    ----------
    strategy : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    grid : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    need_dense : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    need_operator : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
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

