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
    regularization_sparse_knn_from,
    regularization_sparse_rectangular_from,
    sparse_regularization_dense_from,
)


class PixelizedSourceModel(ck.Module):
    """Pixelized source model driven by typed configuration objects."""

    def __init__(
        self,
        config: Optional[PixelizedSourceConfig] = None,
        reg_scale: float | ParamU = 0.05,
        reg_coefficient: float | ParamU = 1.0,
    ) -> None:
        super().__init__()
        self.reg_scale = reg_scale if isinstance(reg_scale, ParamU) else ParamU("reg_scale", reg_scale)
        self.reg_coefficient = (
            reg_coefficient if isinstance(reg_coefficient, ParamU) else ParamU("reg_coefficient", reg_coefficient)
        )
        object.__setattr__(self, "config", config if config is not None else PixelizedSourceConfig())

    @property
    def is_rectangular_grid(self) -> bool:
        return self.config.is_rectangular

    @property
    def source_grid_type(self) -> str:
        return self.config.source_grid_type

    @property
    def grid(self):
        return self.config.grid

    @property
    def mapping(self):
        return self.config.mapping

    @property
    def regularization(self):
        return self.config.regularization

    @property
    def solver(self):
        return self.config.solver

    def regularization_sparse_rectangular(
        self,
        nx: int,
        ny: int,
        reg_coefficient: Optional[float] = None,
        rect_reg_type: Optional[str] = None,
    ):
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
        sparse_k = int(self.config.regularization.sparse_k_neighbors)

        if operator_mode == "sparse_knn":
            if kernel_type is None:
                raise ValueError("sparse_knn regularization requires an irregular_knn_* scheme.")
            rows, cols, values, n_source = regularization_sparse_knn_from(
                scale=scale,
                coefficient=coefficient,
                points=points,
                reg_type=kernel_type,
                k_neighbors=sparse_k,
            )
            return sparse_regularization_dense_from(rows, cols, values, n_source)

        if operator_mode != "dense_gp":
            raise ValueError(
                f"Unknown reg_operator_mode: '{operator_mode}'. "
                "Must be one of {'dense_gp', 'sparse_knn'}."
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
