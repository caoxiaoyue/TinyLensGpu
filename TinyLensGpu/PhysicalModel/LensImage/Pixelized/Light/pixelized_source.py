"""Pixelized source-plane light model."""

# pyright: reportMissingImports=false

from __future__ import annotations

import caskade as ck
import jax.numpy as jnp
from jax import Array

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.utils.pixelized_source_utils import (
    build_lens_mapping_matrix,
    build_source_grid,
)


class PixelizedSourceModel(ck.Module):
    """Pixelized source light profile with external pixel amplitudes.

    Parameters
    ----------
    nx, ny : int
        Number of source pixels along x and y.
    lambda_reg : float or ParamU or None, optional
        Regularization strength. Wrapped as a ``ParamU`` with a log-uniform
        prior over ``[1e-4, 1e4]`` when provided as a scalar.
    regularization_type : str, optional
        Regularization family. Supported values are ``"zero-order"``,
        ``"first-order"``, ``"second-order"``, ``"exponential"``, and
        ``"gaussian"``.
    kernel_type : str, optional
        Kernel family for GP-style regularization. Kept as a static attribute.
    kernel_scale : float or ParamU or None, optional
        Kernel scale for GP-style regularization only.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        lambda_reg: float | ParamU | None = None,
        regularization_type: str = "second-order",
        kernel_type: str = "gaussian",
        kernel_scale: float | ParamU | None = None,
    ) -> None:
        super().__init__()

        if regularization_type not in {"zero-order", "first-order", "second-order", "exponential", "gaussian"}:
            raise ValueError(f"Unsupported regularization_type: {regularization_type}")

        object.__setattr__(self, "nx", int(nx))
        object.__setattr__(self, "ny", int(ny))
        object.__setattr__(self, "regularization_type", regularization_type)
        object.__setattr__(self, "kernel_type", kernel_type)
        object.__setattr__(self, "is_pixelized_source", True)

        if isinstance(lambda_reg, ParamU):
            self.lambda_reg = lambda_reg
        else:
            self.lambda_reg = ParamU(
                "lambda_reg",
                lambda_reg,
                prior_type="log_uniform",
                prior_settings=[1e-4, 1e4],
                limits=[1e-4, 1e4],
            )

        is_gp_regularization = regularization_type in {"exponential", "gaussian"}
        if is_gp_regularization:
            if isinstance(kernel_scale, ParamU):
                self.kernel_scale = kernel_scale
            else:
                self.kernel_scale = ParamU(
                    "kernel_scale",
                    kernel_scale,
                    prior_type="log_uniform",
                    prior_settings=[1e-4, 1e4],
                    limits=[1e-4, 1e4],
                )
        elif kernel_scale is not None:
            raise ValueError("kernel_scale is only valid for GP regularization types")
        else:
            self.kernel_scale = None

    @ck.forward
    def light(
        self,
        x: Array,
        y: Array,
        source_values: Array,
        source_half_size: Array | float,
    ) -> Array:
        """Interpolate pixelized source brightness onto image-plane coordinates."""
        half_size = jnp.asarray(source_half_size)
        source_x_axis, source_y_axis, _, _ = build_source_grid(self.nx, self.ny, half_size)
        mapping_matrix = build_lens_mapping_matrix(x, y, source_x_axis, source_y_axis)
        brightness = mapping_matrix @ jnp.ravel(jnp.asarray(source_values))
        return brightness.reshape(jnp.shape(x))
