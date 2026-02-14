"""Regularization strategies for pixelized-source inversion."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import RegularizationConfig
from TinyLensGpu.utils.lensing import (
    regularization_matrix_gp_from,
    regularization_sparse_knn_from,
    regularization_sparse_rectangular_from,
)

from .artifacts import GridArtifacts, RegularizationArtifacts


class BaseRegularizationStrategy:
    """
    Represent the `BaseRegularizationStrategy` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    mode: str

    def build(
        self,
        *,
        grid: GridArtifacts,
        reg_scale: float,
        reg_coefficient: float,
    ) -> RegularizationArtifacts:
        """
        Compute build.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        reg_scale : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        reg_coefficient : Any
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
class DenseGpRegularizationStrategy(BaseRegularizationStrategy):
    """
    Represent the `DenseGpRegularizationStrategy` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    config: RegularizationConfig
    mode: str = "dense_gp"

    def build(
        self,
        *,
        grid: GridArtifacts,
        reg_scale: float,
        reg_coefficient: float,
    ) -> RegularizationArtifacts:
        """
        Compute build.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        reg_scale : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        reg_coefficient : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        Raises
        ------
        ValueError
            Raised when input validation fails or required runtime state is missing.
        
        """
        kernel = self.config.gp_kernel
        if kernel is None:
            raise ValueError("dense_gp regularization requires an irregular_gp_* scheme.")
        dense = regularization_matrix_gp_from(
            scale=float(reg_scale),
            coefficient=float(reg_coefficient),
            points=grid.source_mesh_beta,
            reg_type=kernel,
        )
        return RegularizationArtifacts(
            mode=self.mode,
            dense_matrix=dense,
            sparse_rows=None,
            sparse_cols=None,
            sparse_values=None,
            sparse_n_source=None,
        )


@dataclass(frozen=True)
class SparseKnnRegularizationStrategy(BaseRegularizationStrategy):
    """
    Represent the `SparseKnnRegularizationStrategy` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    config: RegularizationConfig
    mode: str = "sparse_knn"

    def build(
        self,
        *,
        grid: GridArtifacts,
        reg_scale: float,
        reg_coefficient: float,
    ) -> RegularizationArtifacts:
        """
        Compute build.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        reg_scale : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        reg_coefficient : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        Raises
        ------
        ValueError
            Raised when input validation fails or required runtime state is missing.
        
        """
        kernel = self.config.gp_kernel
        if kernel is None:
            raise ValueError("sparse_knn regularization requires an irregular_knn_* scheme.")
        rows, cols, values, n_source = regularization_sparse_knn_from(
            scale=float(reg_scale),
            coefficient=float(reg_coefficient),
            points=grid.source_mesh_beta,
            reg_type=kernel,
            k_neighbors=int(self.config.sparse_k_neighbors),
        )
        return RegularizationArtifacts(
            mode=self.mode,
            dense_matrix=None,
            sparse_rows=rows,
            sparse_cols=cols,
            sparse_values=values,
            sparse_n_source=int(n_source),
        )


@dataclass(frozen=True)
class SparseRectangularRegularizationStrategy(BaseRegularizationStrategy):
    """
    Represent the `SparseRectangularRegularizationStrategy` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    config: RegularizationConfig
    mode: str = "sparse_rectangular"

    def build(
        self,
        *,
        grid: GridArtifacts,
        reg_scale: float,
        reg_coefficient: float,
    ) -> RegularizationArtifacts:
        """
        Compute build.
        
        Parameters
        ----------
        grid : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        reg_scale : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        reg_coefficient : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        Raises
        ------
        ValueError
            Raised when input validation fails or required runtime state is missing.
        RuntimeError
            Raised when input validation fails or required runtime state is missing.
        
        """
        _ = reg_scale
        rect_scheme = self.config.rect_scheme
        if rect_scheme is None:
            raise ValueError(
                "sparse_rectangular regularization requires a rectangular scheme "
                "('rectangular_zero', 'rectangular_first', or 'rectangular_second')."
            )
        if grid.source_grid_shape is None:
            raise RuntimeError("Rectangular regularization requires source_grid_shape.")
        ny, nx = grid.source_grid_shape
        rows, cols, values, n_source = regularization_sparse_rectangular_from(
            coefficient=float(reg_coefficient),
            nx=int(nx),
            ny=int(ny),
            reg_scheme=rect_scheme,
        )
        return RegularizationArtifacts(
            mode=self.mode,
            dense_matrix=None,
            sparse_rows=rows,
            sparse_cols=cols,
            sparse_values=values,
            sparse_n_source=int(n_source),
        )


def select_regularization_strategy(
    regularization_config: RegularizationConfig,
    *,
    resolved_mode: str,
) -> BaseRegularizationStrategy:
    """
    Compute select regularization strategy.
    
    Parameters
    ----------
    regularization_config : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    resolved_mode : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    Raises
    ------
    ValueError
        Raised when input validation fails or required runtime state is missing.
    
    """
    mode = str(resolved_mode).strip().lower()
    if mode == "dense_gp":
        return DenseGpRegularizationStrategy(config=regularization_config)
    if mode == "sparse_knn":
        return SparseKnnRegularizationStrategy(config=regularization_config)
    if mode == "sparse_rectangular":
        return SparseRectangularRegularizationStrategy(config=regularization_config)
    raise ValueError(f"Unknown regularization mode: '{mode}'.")
