"""Shared artifact containers for pixelized-source pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp


@dataclass(frozen=True)
class GridArtifacts:
    """
    Container for grid geometry and ray-tracing results used in pixelized source reconstruction.

    This class holds the coordinate meshes for both the source plane grid (where the source
    light is defined) and the data plane (where the image is observed), mapped back to the
    source plane.

    Attributes
    ----------
    source_mesh : jnp.ndarray
        Coordinates of the points defining the source grid structure.
        - For **irregular grids** (e.g., adaptive), these are typically points in the
          **image plane** (e.g., k-means centers) that generate the grid.
        - For **rectangular grids**, these are the pixel centers in the **source plane**.
        Shape: ``(n_source, 2)``.
    source_mesh_beta : jnp.ndarray
        The coordinates of ``source_mesh`` mapped to the **source plane**.
        - For irregular grids, this is ``ray_trace(source_mesh)``.
        - For rectangular grids, this is identical to ``source_mesh``.
        These points serve as the nodes/centers for the source surface brightness model.
        Shape: ``(n_source, 2)``.
    data_mesh_beta : jnp.ndarray
        The ray-traced coordinates of all unmasked **image pixels** in the source plane.
        These are the locations where the source model is evaluated to produce the model image.
        Shape: ``(n_data, 2)``.
    source_grid_shape : Optional[Tuple[int, int]]
        The shape ``(ny, nx)`` of the source grid if it is structured (rectangular).
        ``None`` for unstructured (irregular) grids.
    source_grid_bounds : Optional[Tuple[float, float, float, float]]
        The physical bounds ``(x_min, x_max, y_min, y_max)`` of the source grid.
    """

    source_mesh: jnp.ndarray
    source_mesh_beta: jnp.ndarray
    data_mesh_beta: jnp.ndarray
    source_grid_shape: Optional[Tuple[int, int]]
    source_grid_bounds: Optional[Tuple[float, float, float, float]]


@dataclass(frozen=True)
class MappingArtifacts:
    """
    Container for the linear mapping operator (Lens Matrix).

    The mapping operator :math:`F` relates the source surface brightness vector :math:`s`
    to the image flux vector :math:`d` via :math:`d = F s`. This class supports both
    dense matrix and sparse matrix-free representations.

    Attributes
    ----------
    dense_matrix : Optional[jnp.ndarray]
        The explicit dense mapping matrix :math:`F`.
        Shape: ``(n_data, n_source)``.
        Used when ``inversion_backend='matrix'``.
    operator_weights : Optional[jnp.ndarray]
        Weights for the sparse interpolation operator. Each row corresponds to an image
        pixel and contains the weights of the ``k`` nearest source nodes.
        Shape: ``(n_data, k_nn)``.
        Used when ``inversion_backend='operator'``.
    operator_indices : Optional[jnp.ndarray]
        Indices of the source nodes corresponding to ``operator_weights``.
        Shape: ``(n_data, k_nn)``.
        Used when ``inversion_backend='operator'``.
    """

    dense_matrix: Optional[jnp.ndarray]
    operator_weights: Optional[jnp.ndarray]
    operator_indices: Optional[jnp.ndarray]


@dataclass(frozen=True)
class RegularizationArtifacts:
    """
    Container for regularization matrices or operators.

    This class holds the regularization matrix :math:`H` (or its decomposition) used in
    the penalty term :math:`R(s) = \lambda s^T H s` or :math:`||Rs||^2`.

    Attributes
    ----------
    mode : str
        The regularization mode identifier (e.g., 'dense_gp', 'sparse_knn', 'sparse_rectangular').
    dense_matrix : Optional[jnp.ndarray]
        The dense regularization matrix.
        Shape: ``(n_source, n_source)``.
    sparse_rows : Optional[jnp.ndarray]
        Row indices for the sparse regularization matrix (COO format).
    sparse_cols : Optional[jnp.ndarray]
        Column indices for the sparse regularization matrix (COO format).
    sparse_values : Optional[jnp.ndarray]
        Values for the sparse regularization matrix (COO format).
    sparse_n_source : Optional[int]
        The dimension of the source vector, required to reconstruct the sparse matrix shape.
    """

    mode: str
    dense_matrix: Optional[jnp.ndarray]
    sparse_rows: Optional[jnp.ndarray]
    sparse_cols: Optional[jnp.ndarray]
    sparse_values: Optional[jnp.ndarray]
    sparse_n_source: Optional[int]

    @property
    def is_sparse(self) -> bool:
        """Check if the regularization artifacts use a sparse representation."""
        return self.mode in {"sparse_knn", "sparse_rectangular"}

    @property
    def n_source(self) -> int:
        """
        Get the number of source pixels (size of the source vector).

        Returns
        -------
        int
            Number of source pixels.

        Raises
        ------
        RuntimeError
            If the necessary shape information is missing from the artifacts.
        """
        if self.is_sparse:
            if self.sparse_n_source is None:
                raise RuntimeError("Sparse regularization artifacts missing n_source.")
            return int(self.sparse_n_source)
        if self.dense_matrix is None:
            raise RuntimeError("Dense regularization artifacts missing matrix.")
        return int(self.dense_matrix.shape[0])


@dataclass(frozen=True)
class OperatorCacheKey:
    """
    Cache key for mapping operators.

    Used to determine if a computationally expensive mapping operator (or its components)
    can be reused based on the geometric configuration of the grid and lens model.

    Attributes
    ----------
    signature : Tuple
        A hashable tuple uniquely identifying the grid geometry and ray-tracing state.
        If the signature matches, the cached operator is considered valid.
    """

    signature: Tuple
