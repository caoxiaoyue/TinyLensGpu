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
    n : int
        Number of source pixels per side. The source grid is ``n x n``.
    log_lambda_reg : float or ParamU or None, optional
        Natural-log of the regularization strength. Stored in log-space so
        that the optimiser works with O(1) values even when the physical
        λ is very small (e.g. 1e-6).  Wrapped as a ``ParamU`` with a
        uniform prior over ``[log(1e-4), log(1e4)]`` when provided as a
        scalar.
    regularization_type : str, optional
        Regularization family. Supported values are ``"zero-order"``,
        ``"first-order"``, ``"second-order"``, ``"exponential"``,
        ``"gaussian"``, ``"matern32"``, ``"matern52"``, and ``"matern72"``.
    kernel_scale : float or ParamU or None, optional
        Kernel scale for GP-style regularization only.
    adaptive_reg_rho : float or ParamU, optional
        Galan-style adaptive regularization strength. ``0.0`` (default)
        disables adaptation and recovers uniform regularization. Larger
        values strengthen regularization in faint S0 source-template regions.
    """

    def __init__(
        self,
        n: int,
        *,
        log_lambda_reg: float | ParamU | None = None,
        regularization_type: str = "second-order",
        kernel_scale: float | ParamU | None = None,
        adaptive_reg_rho: float | ParamU = 0.0,
        adaptive_reg_alpha: float | ParamU | None = None,
        adaptive_reg_floor: float | ParamU | None = None,
    ) -> None:
        super().__init__()

        if adaptive_reg_alpha is not None or adaptive_reg_floor is not None:
            raise ValueError(
                "adaptive_reg_alpha and adaptive_reg_floor are retired. "
                "Use adaptive_reg_rho for Galan-style source-template "
                "adaptive regularization."
            )

        if regularization_type not in VALID_REGULARIZATION_TYPES:
            raise ValueError(f"Unsupported regularization_type: {regularization_type}")

        n_int = int(n)
        if n_int < 2:
            raise ValueError(
                f"PixelizedSourceModel requires n >= 2; got n={n_int}."
            )

        self._validate_adaptive_reg_rho(adaptive_reg_rho)

        object.__setattr__(self, "n", n_int)
        object.__setattr__(self, "regularization_type", regularization_type)
        object.__setattr__(self, "is_pixelized_source", True)

        self.log_lambda_reg = (
            log_lambda_reg if isinstance(log_lambda_reg, ParamU)
            else ParamU("log_lambda_reg", log_lambda_reg,
                        prior_type="uniform",
                        prior_settings=[jnp.log(1e-4), jnp.log(1e4)],
                        limits=[jnp.log(1e-4), jnp.log(1e4)])
        )
        self.adaptive_reg_rho = (
            adaptive_reg_rho if isinstance(adaptive_reg_rho, ParamU)
            else float(adaptive_reg_rho)
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

    @staticmethod
    def _validate_adaptive_reg_rho(adaptive_reg_rho: float | ParamU) -> None:
        rho_value = (
            adaptive_reg_rho.value
            if isinstance(adaptive_reg_rho, ParamU)
            else adaptive_reg_rho
        )
        if not 0.0 <= float(rho_value):
            raise ValueError(f"adaptive_reg_rho must be >= 0, got {adaptive_reg_rho}")
        if not isinstance(adaptive_reg_rho, ParamU):
            return

        def _lower_bound(values):
            if values is None:
                return None
            return float(values[0])

        limits_lower = _lower_bound(adaptive_reg_rho.limits)
        if limits_lower is not None and limits_lower < 0.0:
            raise ValueError(
                "adaptive_reg_rho limits must have a non-negative lower bound"
            )

        prior_lower = _lower_bound(adaptive_reg_rho.prior_settings)
        if (
            adaptive_reg_rho.prior_type in ("uniform", "log_uniform")
            and prior_lower is not None
            and prior_lower < 0.0
        ):
            raise ValueError(
                "adaptive_reg_rho prior_settings must have a non-negative lower bound"
            )

        if (
            bool(getattr(adaptive_reg_rho, "dynamic", False))
            and adaptive_reg_rho.prior_type in ("gaussian", "truncated_gaussian")
            and limits_lower is None
        ):
            raise ValueError(
                "dynamic adaptive_reg_rho gaussian priors require non-negative limits"
            )

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
            Source pixel intensities, shape (n * n,).
        source_bbox : tuple
            Square (xmin, xmax, ymin, ymax) source-plane bounding box.
        """
        xmin, xmax, ymin, ymax = source_bbox
        source_x_axis, source_y_axis, _, _ = build_source_grid(
            self.n, xmin, xmax, ymin, ymax
        )
        mapping_matrix = build_lens_mapping_matrix(x, y, source_x_axis, source_y_axis)
        brightness = mapping_matrix @ jnp.ravel(jnp.asarray(source_values))
        return brightness.reshape(jnp.shape(x))
