"""Pixelized source model definitions and configuration wrapper."""

from __future__ import annotations

from typing import Optional

import caskade as ck
import jax.numpy as jnp

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import (
    IrregularGridConfig,
    MappingConfig,
    PixelizedSourceConfig,
    RectangularGridConfig,
    RegularizationConfig,
    SolverConfig,
)
from TinyLensGpu.utils.lensing import (
    regularization_matrix_gp_from,
    regularization_sparse_rectangular_from,
)


class PixelizedSourceModel(ck.Module):
    """
    Pixelized source configuration and regularization wrapper.

    This module stores grid/mapping/solver configuration and provides helper
    builders for dense and sparse regularization operators used by pixelized
    source inversion.

    Parameters
    ----------
    config : PixelizedSourceConfig, optional
        Source-grid, mapping, regularization and solver settings.
    reg_scale : float | ParamU, optional
        Regularization correlation/length scale.
    reg_coefficient : float | ParamU, optional
        Overall regularization strength.
    """

    def __init__(
        self,
        config: Optional[PixelizedSourceConfig] = None,
        reg_scale: float | ParamU = 0.05,
        reg_coefficient: float | ParamU = 1.0,
    ) -> None:
        """
        Initialize a `PixelizedSourceModel`.

        This component manages the configuration and regularization parameters for a pixelized source.
        It supports both rectangular grids (with finite difference regularization) and irregular grids
        (with Gaussian Process regularization).

        Parameters
        ----------
        config : Optional[PixelizedSourceConfig]
            The configuration object defining the grid type, mapping strategy, and regularization mode.
            Defaults to a default `PixelizedSourceConfig` if None.
        reg_scale : float | ParamU, optional
            The correlation length scale (for GP) or characteristic scale of the regularization.
            Can be a fixed float or a `ParamU` parameter. Defaults to 0.05.
        reg_coefficient : float | ParamU, optional
            The overall strength of the regularization (log-amplitude).
            Can be a fixed float or a `ParamU` parameter. Defaults to 1.0.
        """
        super().__init__()
        self.reg_scale = reg_scale if isinstance(reg_scale, ParamU) else ParamU("reg_scale", reg_scale)
        self.reg_coefficient = (
            reg_coefficient if isinstance(reg_coefficient, ParamU) else ParamU("reg_coefficient", reg_coefficient)
        )
        object.__setattr__(self, "config", config if config is not None else PixelizedSourceConfig())

    @property
    def is_rectangular_grid(self) -> bool:
        """
        Check if the source grid is rectangular.

        Returns
        -------
        bool
            True if `source_grid_type` is 'rectangular_bilinear', False otherwise.
        """
        return self.config.is_rectangular

    @property
    def source_grid_type(self) -> str:
        """
        Get the type of source grid.

        Returns
        -------
        str
            The grid type identifier (e.g., 'rectangular_bilinear', 'irregular_delaunay').
        """
        return self.config.source_grid_type

    @property
    def grid(self):
        """
        Get the grid configuration object.

        Returns
        -------
        Union[IrregularGridConfig, RectangularGridConfig]
            The configuration for the source grid geometry.
        """
        return self.config.grid

    @property
    def mapping(self):
        """
        Get the mapping configuration object.

        Returns
        -------
        MappingConfig
            The configuration for the lens mapping (ray-tracing) strategy.
        """
        return self.config.mapping

    @property
    def regularization(self):
        """
        Get the regularization configuration object.

        Returns
        -------
        RegularizationConfig
            The configuration for the regularization scheme (e.g., GP kernel, finite difference order).
        """
        return self.config.regularization

    @property
    def solver(self):
        """
        Get the solver configuration object.

        Returns
        -------
        SolverConfig
            The configuration for the linear solver (e.g., matrix inversion, CG, NNLS).
        """
        return self.config.solver

    def regularization_sparse_rectangular(
        self,
        nx: int,
        ny: int,
        reg_coefficient: Optional[float] = None,
        rect_reg_type: Optional[str] = None,
    ):
        """
        Compute the sparse regularization matrix components for a rectangular grid.

        Parameters
        ----------
        nx : int
            Number of pixels in x-direction.
        ny : int
            Number of pixels in y-direction.
        reg_coefficient : Optional[float], optional
            Regularization strength. If None, uses `self.reg_coefficient.value`.
        rect_reg_type : Optional[str], optional
            Type of finite difference scheme (e.g., 'rectangular_zero', 'rectangular_first', 'rectangular_second').
            If None, uses the value from `self.config`.

        Returns
        -------
        rows, cols, values : Tuple[Array, Array, Array]
            The sparse matrix indices and values representing the regularization operator R.
            These can be used to construct a sparse matrix or perform sparse matrix-vector products.

        Raises
        ------
        ValueError
            If the regularization scheme is not specified or invalid.
        """
        coefficient = reg_coefficient if reg_coefficient is not None else self.reg_coefficient.value
        if rect_reg_type is not None:
            scheme = rect_reg_type
        else:
            rect_scheme = self.config.regularization.rect_scheme
            if rect_scheme is None:
                raise ValueError(
                    "regularization_sparse_rectangular() requires a rectangular regularization scheme "
                    "('rectangular_zero', 'rectangular_first', or 'rectangular_second')."
                )
            scheme = rect_scheme
        return regularization_sparse_rectangular_from(
            coefficient=coefficient,
            nx=int(nx),
            ny=int(ny),
            reg_scheme=scheme,
        )

    @ck.forward
    def regularization_matrix(
        self,
        points: jnp.ndarray,
        reg_scale: Optional[float] = None,
        reg_coefficient: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Compute the dense regularization matrix for irregular grids.

        Supports only 'dense_gp' (Gaussian Process) mode.

        Parameters
        ----------
        points : jnp.ndarray
            Source plane coordinates of the grid points. Shape (N_points, 2).
        reg_scale : Optional[float], optional
            Regularization scale length. If None, uses `self.reg_scale.value`.
        reg_coefficient : Optional[float], optional
            Regularization strength. If None, uses `self.reg_coefficient.value`.

        Returns
        -------
        H_reg : jnp.ndarray
            The dense regularization matrix (H_reg = R^T R or similar, depending on definition).
            Shape (N_points, N_points).

        Raises
        ------
        ValueError
            If called for a rectangular grid (which should use `regularization_sparse_rectangular`),
            or if the configured mode/kernel is invalid.
        """
        if self.is_rectangular_grid:
            raise ValueError(
                "regularization_matrix() is not available for source_grid_type='rectangular_bilinear'. "
                "Use regularization_sparse_rectangular(); rectangular matrix-mode densification "
                "is handled internally by PixelizedLensSimulator.build_inverter()."
            )

        scale = reg_scale if reg_scale is not None else self.reg_scale.value
        coefficient = reg_coefficient if reg_coefficient is not None else self.reg_coefficient.value
        kernel_type = self.config.regularization.gp_kernel
        operator_mode = self.config.regularization.mode

        if operator_mode != "dense_gp":
            raise ValueError(
                f"Unknown reg_operator_mode: '{operator_mode}'. "
                "Must be one of {'dense_gp'}."
            )
        if kernel_type is None:
            raise ValueError("dense_gp regularization requires an irregular_gp_* scheme.")

        return regularization_matrix_gp_from(
            scale=scale,
            coefficient=coefficient,
            points=points,
            reg_type=kernel_type,
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the `PixelizedSourceModel`.

        Returns
        -------
        str
            A string summarizing the regularization parameters and grid type.
        """
        if isinstance(self.config.grid, IrregularGridConfig):
            n_source_points = int(self.config.grid.n_source_points)
        elif isinstance(self.config.grid, RectangularGridConfig):
            n_source_points = int(self.config.grid.nx * self.config.grid.ny)
        else:
            n_source_points = 0

        return (
            "PixelizedSourceModel("
            f"reg_scale={float(self.reg_scale.value):.3f}, "
            f"reg_coefficient={float(self.reg_coefficient.value):.2f}, "
            f"n_source_points={n_source_points}, "
            f"source_grid_type='{self.config.source_grid_type}')"
        )


__all__ = [
    "PixelizedSourceModel",
    "PixelizedSourceConfig",
    "IrregularGridConfig",
    "RectangularGridConfig",
    "MappingConfig",
    "RegularizationConfig",
    "SolverConfig",
]
