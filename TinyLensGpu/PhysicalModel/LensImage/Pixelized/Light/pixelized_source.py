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
    adaptive_reg_alpha : float, optional
        Adaptive regularization strength.  ``0.0`` (default) disables
        adaptation and recovers uniform regularisation.  Larger values
        (e.g. ``1.0``) make the per-pixel regularisation scale more
        sensitive to the source brightness estimate.
    adaptive_reg_floor : float, optional
        Minimum per-pixel regularisation scale, relative to the global
        ``lambda_reg``.  Default ``0.1``; must be in ``(0, 1]``.
    adaptive_reg_mode : str, optional
        Brightness estimation mode.  ``"brightness_only"`` (default) uses
        inverse-variance-weighted normalized convolution (``N/C``) to
        cancel magnification dependence.  ``"brightness_weighted"``
        retains the legacy brightness×ray-count product, upgraded with
        inverse-variance weighting.
    adaptive_reg_smooth_sigma : float, optional
        Gaussian kernel sigma (in source pixels) for brightness-map
        smoothing.  Default ``1.0``.  Kernel size auto-adapts as
        ``max(5, 2·ceil(3·sigma)+1)``.
    adaptive_reg_freeze : bool, optional
        When ``True``, evidence models will reuse an explicitly frozen
        brightness scale map.  Call ``prob_model.freeze_scale()`` eagerly
        before JIT tracing / sampling to create the frozen map.  If no
        frozen map exists, evidence evaluation warns and recomputes the
        scale per call.  Default ``False``.
    """

    _ALLOWED_ADAPTIVE_MODES: frozenset[str] = frozenset({"brightness_only", "brightness_weighted"})

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        log_lambda_reg: float | ParamU | None = None,
        regularization_type: str = "second-order",
        kernel_scale: float | ParamU | None = None,
        adaptive_reg_alpha: float = 0.0,
        adaptive_reg_floor: float = 0.1,
        adaptive_reg_mode: str = "brightness_only",
        adaptive_reg_smooth_sigma: float = 1.0,
        adaptive_reg_freeze: bool = False,
    ) -> None:
        super().__init__()

        if regularization_type not in VALID_REGULARIZATION_TYPES:
            raise ValueError(f"Unsupported regularization_type: {regularization_type}")

        if not 0.0 <= float(adaptive_reg_alpha):
            raise ValueError(f"adaptive_reg_alpha must be >= 0, got {adaptive_reg_alpha}")
        if not 0.0 < float(adaptive_reg_floor) <= 1.0:
            raise ValueError(f"adaptive_reg_floor must be in (0, 1], got {adaptive_reg_floor}")
        if adaptive_reg_mode not in self._ALLOWED_ADAPTIVE_MODES:
            raise ValueError(
                f"adaptive_reg_mode must be one of {sorted(self._ALLOWED_ADAPTIVE_MODES)}, "
                f"got {adaptive_reg_mode!r}"
            )
        if not float(adaptive_reg_smooth_sigma) > 0.0:
            raise ValueError(
                f"adaptive_reg_smooth_sigma must be > 0, got {adaptive_reg_smooth_sigma}"
            )

        object.__setattr__(self, "nx", int(nx))
        object.__setattr__(self, "ny", int(ny))
        object.__setattr__(self, "regularization_type", regularization_type)
        object.__setattr__(self, "adaptive_reg_alpha", float(adaptive_reg_alpha))
        object.__setattr__(self, "adaptive_reg_floor", float(adaptive_reg_floor))
        object.__setattr__(self, "adaptive_reg_mode", str(adaptive_reg_mode))
        object.__setattr__(self, "adaptive_reg_smooth_sigma", float(adaptive_reg_smooth_sigma))
        object.__setattr__(self, "adaptive_reg_freeze", bool(adaptive_reg_freeze))
        object.__setattr__(self, "is_pixelized_source", True)

        self.log_lambda_reg = (
            log_lambda_reg if isinstance(log_lambda_reg, ParamU)
            else ParamU("log_lambda_reg", log_lambda_reg,
                        prior_type="uniform",
                        prior_settings=[jnp.log(1e-4), jnp.log(1e4)],
                        limits=[jnp.log(1e-4), jnp.log(1e4)])
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
