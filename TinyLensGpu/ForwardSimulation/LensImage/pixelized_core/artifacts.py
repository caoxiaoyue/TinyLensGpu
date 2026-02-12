"""Shared artifact containers for pixelized-source pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp


@dataclass(frozen=True)
class GridArtifacts:
    """Artifacts produced by source-grid generation."""

    source_mesh: jnp.ndarray
    source_mesh_beta: jnp.ndarray
    data_mesh_beta: jnp.ndarray
    source_grid_shape: Optional[Tuple[int, int]]
    source_grid_bounds: Optional[Tuple[float, float, float, float]]


@dataclass(frozen=True)
class MappingArtifacts:
    """Artifacts for source-to-image mapping operators."""

    dense_matrix: Optional[jnp.ndarray]
    operator_weights: Optional[jnp.ndarray]
    operator_indices: Optional[jnp.ndarray]


@dataclass(frozen=True)
class RegularizationArtifacts:
    """Artifacts for source regularization."""

    mode: str
    dense_matrix: Optional[jnp.ndarray]
    sparse_rows: Optional[jnp.ndarray]
    sparse_cols: Optional[jnp.ndarray]
    sparse_values: Optional[jnp.ndarray]
    sparse_n_source: Optional[int]

    @property
    def is_sparse(self) -> bool:
        return self.mode in {"sparse_knn", "sparse_rectangular"}

    @property
    def n_source(self) -> int:
        if self.is_sparse:
            if self.sparse_n_source is None:
                raise RuntimeError("Sparse regularization artifacts missing n_source.")
            return int(self.sparse_n_source)
        if self.dense_matrix is None:
            raise RuntimeError("Dense regularization artifacts missing matrix.")
        return int(self.dense_matrix.shape[0])


@dataclass(frozen=True)
class OperatorCacheKey:
    """Cache key for mapping operators."""

    signature: Tuple

