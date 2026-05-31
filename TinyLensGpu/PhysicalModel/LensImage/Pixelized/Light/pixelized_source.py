"""Pixelized source-plane light model."""

# pyright: reportMissingImports=false

from __future__ import annotations

import caskade as ck
import jax.numpy as jnp
from jax import Array

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.utils.inversion.regularization import (
    GP_REGULARIZATION_TYPES,
    VALID_REGULARIZATION_TYPES,
)
from TinyLensGpu.utils.lensing.mapping import (
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
        ``"first-order"``, ``"second-order"``, ``"exponential"``,
        ``"gaussian"``, ``"matern32"``, ``"matern52"``, and ``"matern72"``.
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
        kernel_scale: float | ParamU | None = None,
    ) -> None:
        super().__init__()

        if regularization_type not in VALID_REGULARIZATION_TYPES:
            raise ValueError(f"Unsupported regularization_type: {regularization_type}")

        object.__setattr__(self, "nx", int(nx))
        object.__setattr__(self, "ny", int(ny))
        object.__setattr__(self, "regularization_type", regularization_type)
        object.__setattr__(self, "is_pixelized_source", True)

        self.lambda_reg = (
            lambda_reg if isinstance(lambda_reg, ParamU)
            else ParamU("lambda_reg", lambda_reg, prior_type="log_uniform",
                        prior_settings=[1e-4, 1e4], limits=[1e-4, 1e4])
        )

        if regularization_type in GP_REGULARIZATION_TYPES:
            self.kernel_scale = (
                kernel_scale if isinstance(kernel_scale, ParamU)
                else ParamU("kernel_scale", kernel_scale, prior_type="log_uniform",
                            prior_settings=[1e-4, 1e4], limits=[1e-4, 1e4])
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
        source_bbox: tuple,
    ) -> Array:
        """Interpolate pixelized source brightness onto image-plane coordinates.

        Parameters
        ----------
        x, y : Array
            Image-plane coordinates.
        source_values : Array
            Source pixel intensities, shape (Nx*Ny,).
        source_bbox : tuple
            (xmin, xmax, ymin, ymax) source-plane bounding box.
        """
        xmin, xmax, ymin, ymax = source_bbox
        source_x_axis, source_y_axis, _, _ = build_source_grid(
            self.nx, self.ny, xmin, xmax, ymin, ymax
        )
        mapping_matrix = build_lens_mapping_matrix(x, y, source_x_axis, source_y_axis)
        brightness = mapping_matrix @ jnp.ravel(jnp.asarray(source_values))
        return brightness.reshape(jnp.shape(x))
