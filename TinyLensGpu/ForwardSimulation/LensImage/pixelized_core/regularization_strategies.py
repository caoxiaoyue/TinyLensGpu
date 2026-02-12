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
    """Base interface for source regularization builders."""

    mode: str

    def build(
        self,
        *,
        grid: GridArtifacts,
        reg_scale: float,
        reg_coefficient: float,
    ) -> RegularizationArtifacts:
        raise NotImplementedError


@dataclass(frozen=True)
class DenseGpRegularizationStrategy(BaseRegularizationStrategy):
    """Dense GP inverse-covariance regularization."""

    config: RegularizationConfig
    mode: str = "dense_gp"

    def build(
        self,
        *,
        grid: GridArtifacts,
        reg_scale: float,
        reg_coefficient: float,
    ) -> RegularizationArtifacts:
        dense = regularization_matrix_gp_from(
            scale=float(reg_scale),
            coefficient=float(reg_coefficient),
            points=grid.source_mesh_beta,
            reg_type=self.config.gp_kernel,
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
    """Sparse KNN graph Laplacian regularization."""

    config: RegularizationConfig
    mode: str = "sparse_knn"

    def build(
        self,
        *,
        grid: GridArtifacts,
        reg_scale: float,
        reg_coefficient: float,
    ) -> RegularizationArtifacts:
        rows, cols, values, n_source = regularization_sparse_knn_from(
            scale=float(reg_scale),
            coefficient=float(reg_coefficient),
            points=grid.source_mesh_beta,
            reg_type=self.config.gp_kernel,
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
    """Sparse finite-difference regularization on rectangular grids."""

    config: RegularizationConfig
    mode: str = "sparse_rectangular"

    def build(
        self,
        *,
        grid: GridArtifacts,
        reg_scale: float,
        reg_coefficient: float,
    ) -> RegularizationArtifacts:
        _ = reg_scale
        if grid.source_grid_shape is None:
            raise RuntimeError("Rectangular regularization requires source_grid_shape.")
        ny, nx = grid.source_grid_shape
        rows, cols, values, n_source = regularization_sparse_rectangular_from(
            coefficient=float(reg_coefficient),
            nx=int(nx),
            ny=int(ny),
            reg_scheme=self.config.rect_scheme,
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
    """Select concrete regularization strategy from config."""
    mode = str(resolved_mode).strip().lower()
    if mode == "dense_gp":
        return DenseGpRegularizationStrategy(config=regularization_config)
    if mode == "sparse_knn":
        return SparseKnnRegularizationStrategy(config=regularization_config)
    if mode == "sparse_rectangular":
        return SparseRectangularRegularizationStrategy(config=regularization_config)
    raise ValueError(f"Unknown regularization mode: '{mode}'.")
