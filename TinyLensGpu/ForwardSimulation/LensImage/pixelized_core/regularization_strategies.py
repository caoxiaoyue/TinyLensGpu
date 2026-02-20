"""Regularization strategies for pixelized-source inversion."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import RegularizationConfig
from TinyLensGpu.utils.lensing import (
    regularization_matrix_gp_from,
    regularization_sparse_rectangular_from,
)

from .artifacts import GridArtifacts, RegularizationArtifacts


class BaseRegularizationStrategy:
    """
    Abstract interface for source regularization construction.

    Implementations produce the regularization operator used by inversion, either
    as a dense matrix or as sparse COO components depending on the configured
    backend and grid type.
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
        Build regularization artifacts for current grid geometry.

        Parameters
        ----------
        grid : GridArtifacts
            Grid geometry and source-node coordinates.
        reg_scale : float
            Correlation length or smoothing scale. Interpretation depends on the
            specific regularization implementation.
        reg_coefficient : float
            Global regularization strength multiplying the penalty term.

        Returns
        -------
        RegularizationArtifacts
            Dense or sparse representation of the regularization operator.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class DenseGpRegularizationStrategy(BaseRegularizationStrategy):
    """
    Dense Gaussian-process-style regularization on source nodes.

    The resulting matrix is typically used with matrix-based inversion backends
    where explicit ``(n_source, n_source)`` storage is acceptable.
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
        Build dense GP regularization matrix.

        Returns
        -------
        RegularizationArtifacts
            Artifacts with ``dense_matrix`` populated and sparse fields set to
            ``None``.

        Raises
        ------
        ValueError
            If GP kernel metadata is missing from the configuration.
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
class SparseRectangularRegularizationStrategy(BaseRegularizationStrategy):
    """
    Sparse finite-difference regularization for rectangular source grids.

    The operator encodes zero/first/second-order smoothness on a structured
    lattice and is emitted in COO form.
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
        Build sparse rectangular-grid regularization operator.

        Returns
        -------
        RegularizationArtifacts
            Artifacts with sparse COO arrays for rectangular finite-difference
            regularization.

        Raises
        ------
        ValueError
            If regularization scheme metadata is not set.
        RuntimeError
            If rectangular grid shape metadata is unavailable.
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
    Select a regularization strategy from resolved mode.

    Parameters
    ----------
    regularization_config : RegularizationConfig
        User configuration object containing kernel/sparsity settings.
    resolved_mode : str
        Canonical regularization mode string.

    Returns
    -------
    BaseRegularizationStrategy
        Instantiated strategy matching ``resolved_mode``.

    Raises
    ------
    ValueError
        If ``resolved_mode`` is not recognized.
    """
    mode = str(resolved_mode).strip().lower()
    if mode == "dense_gp":
        return DenseGpRegularizationStrategy(config=regularization_config)
    if mode == "sparse_rectangular":
        return SparseRectangularRegularizationStrategy(config=regularization_config)
    raise ValueError(f"Unknown regularization mode: '{mode}'.")
