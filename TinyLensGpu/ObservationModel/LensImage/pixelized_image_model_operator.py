"""Operator-based Bayesian evidence model for pixelized source inversions.

This module provides :class:`PixelizedImageProbModelOperator`, which uses
matrix-free operators and preconditioned conjugate gradient (PCG) to solve
the source inversion, avoiding explicit construction of the (Nd × Ns)
design matrix.

Phase-1 limitation: NNLS solver is not supported; lens-light joint
inversion is not yet supported.
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import Dict, Optional, Union

import caskade as ck
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit

logger = logging.getLogger(__name__)

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized_operator import (
    PixelizedLensOperator,
    LensOperatorData,
)
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.cg_solver import pcg_solve, PCGInfo
from TinyLensGpu.utils.inversion.regularization import (
    DenseRegularizationBuilder,
)
from TinyLensGpu.utils.lensing.mapping import (
    lens_mapping_operator_bilinear_rectangular_from,
)


class PixelizedImageProbModelOperator(ck.Module):
    """Operator-based evidence model for one pixelized source.

    Same public API as :class:`PixelizedImageProbModel` but uses matrix-free
    operators and PCG instead of explicit Cholesky decomposition.

    Parameters
    ----------
    image_data : array_like
        Observed 2D image data.
    noise_map : array_like
        Positive per-pixel noise standard deviations.
    psf_kernel : array_like
        PSF kernel used by FFT convolution.
    dpix : float
        Native image pixel scale.
    phys_model : PhysicalModel
        Physical model with exactly one pixelized source.
    mask : array_like, optional
        Boolean mask where ``True`` pixels are excluded.
    source_seed_mask : array_like, optional
        Boolean mask for source bounding-box inference.
    nsub : int, optional
        Subsampling factor (default: 1).
    position_likelihood : dict, optional
        Position likelihood constraint configuration.
    """

    def __init__(
        self,
        image_data: Union[np.ndarray, Array],
        noise_map: Union[np.ndarray, Array],
        psf_kernel: Union[np.ndarray, Array],
        dpix: float,
        phys_model: PhysicalModel,
        mask: Union[np.ndarray, Array, None] = None,
        source_seed_mask: Union[np.ndarray, Array, None] = None,
        nsub: int = 1,
        position_likelihood: Optional[Dict] = None,
        block_size: int = 10,
    ) -> None:
        super().__init__("pixelized_image_prob_model_operator")

        self.image_data = jnp.asarray(image_data)
        self.noise_map = jnp.asarray(noise_map)
        if jnp.any(self.noise_map <= 0.0):
            raise ValueError("noise_map must contain only positive values")

        self.phys_model = phys_model
        sim_config = SimulatorConfig(
            dpix=dpix,
            npix=int(self.image_data.shape[0]),
            psf_kernel=psf_kernel,
            nsub=nsub,
            mask=mask,
            source_seed_mask=source_seed_mask,
        )
        self.sim_obj = PixelizedLensOperator(self.phys_model, sim_config)
        self.unmask = ~jnp.asarray(sim_config.mask, dtype=bool)
        self.data_1d = self.image_data[self.unmask]
        self.noise_1d = self.noise_map[self.unmask]
        self.logdet_C = jnp.sum(jnp.log(self.noise_1d**2))

        source_nx = int(self.source_model.nx)
        source_ny = int(self.source_model.ny)
        self.reg_type = self.source_model.regularization_type
        self.reg_builder = DenseRegularizationBuilder(
            source_nx, source_ny, self.reg_type,
        )

        # Block-diagonal preconditioner settings
        self.block_size = int(block_size)

        # PCG settings
        self.pcg_max_iter = 200
        self.pcg_rtol = 1e-6

        self._init_position_likelihood(position_likelihood)

    def _init_position_likelihood(self, config: Optional[Dict]) -> None:
        self._pos_px = None
        self._pos_py = None
        self._pos_thr = jnp.array(0.0, dtype=jnp.float32)
        self._pos_minl = jnp.array(0.0, dtype=jnp.float32)
        self._has_pos_penalty = False

        if config is not None:
            positions = config.get("positions", [])
            if positions is not None and len(positions) >= 2:
                self._pos_px = jnp.array(
                    [p[0] for p in positions], dtype=jnp.float32
                )
                self._pos_py = jnp.array(
                    [p[1] for p in positions], dtype=jnp.float32
                )
                self._pos_thr = jnp.array(
                    float(
                        config.get(
                            "threshold_arcsec",
                            config.get("position_threshold", 0.0),
                        )
                    ),
                    dtype=jnp.float32,
                )
                self._pos_minl = jnp.array(
                    float(
                        config.get(
                            "min_log_like",
                            config.get("min_position_likelihood", 0.0),
                        )
                    ),
                    dtype=jnp.float32,
                )
                self._has_pos_penalty = True

    def get_dynamic_params(self):
        """Return dynamic parameters exposed by the physical model."""
        return self.phys_model.dynamic_params

    def get_values(self, mode="flat"):
        """Return current dynamic-parameter values."""
        if mode == "flat":
            return jnp.asarray(
                [jnp.asarray(param.value) for param in self.get_dynamic_params()]
            )
        return super().get_values(mode)

    @property
    def source_model(self):
        """Return the single pixelized source configuration."""
        return self.phys_model.source_light[0]

    # ------------------------------------------------------------------
    # Source-plane bbox
    # ------------------------------------------------------------------

    def _get_bbox(self):
        """Infer source-plane bounding box from seed-region ray-tracing.

        Returns ``(xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub,
        beta_x_seed, beta_y_seed)`` so that callers can reuse the seed
        betas for adaptive regularisation without a second deflection call.
        """
        beta_x_sub, beta_y_sub, beta_x_seed, beta_y_seed = \
            self.sim_obj._get_beta_sub_and_seed()
        xmin, xmax, ymin, ymax = self.sim_obj._infer_and_fix_bbox(
            beta_x_seed, beta_y_seed
        )
        return xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub, beta_x_seed, beta_y_seed

    # ------------------------------------------------------------------
    # Regularization matrix
    # ------------------------------------------------------------------

    def _regularization_data(
        self, xmin, xmax, ymin, ymax, scale: Array | None = None,
    ) -> tuple:
        """Return the compact :class:`RegData` tuple for the matrix-free PCG path.

        Only finite-difference regularization types are supported by the
        operator backend.  GP types should use the dense backend instead.

        If *scale* is ``None``, uniform edge weights of 1 are used
        (fast-path in the operator matvec); ``None`` is passed through
        rather than materialising an array of ones, avoiding O(Ns)
        wasted work.

        Returns
        -------
        RegData
            Per-pixel adaptive ``scale`` array and physical spacing factors for
            the edge-weighted regularisation term.
        """
        # Pass None through — all operator/matvec paths now handle scale=None
        # with uniform-weight fast-paths, avoiding O(Ns) wasted work.
        return self.reg_builder.make_reg_data(xmin, xmax, ymin, ymax, scale=scale)

    # ------------------------------------------------------------------
    # Adaptive regularization scale map
    # ------------------------------------------------------------------

    def _compute_reg_scale_from_betas(
        self,
        beta_x_seed: Array,
        beta_y_seed: Array,
        xmin, xmax, ymin, ymax,
    ) -> Array | None:
        """Compute per-source-pixel regularization scale factors.

        Supports two brightness-estimation modes (configured via
        ``source_model.adaptive_reg_mode``):

        * ``"brightness_only"`` (default) — inverse-variance-weighted
          normalized convolution ``N/C``; magnification cancels in the
          ratio, yielding a pure brightness proxy.
        * ``"brightness_weighted"`` — inverse-variance-weighted
          brightness×ray-count product; magnification dependence is
          preserved for comparison or legacy use.

        Both modes share the same downstream pipeline: configurable
        Gaussian smoothing via :meth:`DenseRegularizationBuilder.smooth_scale_map`,
        global-mean normalization, and a continuously-differentiable
        scale formula.

        Returns ``None`` when ``adaptive_reg_alpha == 0`` (fast path).

        When ``adaptive_reg_freeze`` is ``True``, the caller MUST first
        populate ``self._frozen_scale`` via :meth:`freeze_scale` (eagerly,
        before JIT tracing).  At trace time this method picks the cached
        branch and the JIT compiler captures the frozen array as a
        constant.  If freeze is requested but no scale has been stored,
        a warning is emitted and the scale is recomputed on every call.
        """
        alpha_val = self.source_model.adaptive_reg_alpha
        # Note: alpha_val is a plain Python float from the model config.
        # If it ever becomes a traced (caskade.Param) value, this check must
        # switch to jnp.isclose / jax.lax.cond for JIT compatibility.
        if abs(alpha_val) < 1e-10:
            return None

        # --- freeze: trace-time check for a cached scale map ---
        # If freeze_scale() was called eagerly before JIT tracing, the
        # cached concrete array is captured by the compiler as a constant
        # and the (traced) betas below become dead args.
        if self.source_model.adaptive_reg_freeze:
            frozen = getattr(self, '_frozen_scale', None)
            if frozen is not None:
                return frozen
            warnings.warn(
                "adaptive_reg_freeze=True but no frozen scale map has been "
                "stored; call .freeze_scale() before JIT tracing (e.g. before "
                "make_likelihood / sampling). Falling back to per-call "
                "recomputation.",
                stacklevel=2,
            )

        return self._compute_scale_core(beta_x_seed, beta_y_seed, xmin, xmax, ymin, ymax)

    def _compute_scale_core(
        self,
        beta_x_seed: Array,
        beta_y_seed: Array,
        xmin, xmax, ymin, ymax,
    ) -> Array:
        """Core brightness → scale computation (no freeze / alpha-0 fast path).

        Shared by :meth:`_compute_reg_scale_from_betas` (JIT path) and
        :meth:`freeze_scale` (eager path) so the freeze-eager call does
        not re-enter the freeze cache check.
        """
        nx = self.sim_obj.source_nx
        ny = self.sim_obj.source_ny
        n_source = nx * ny
        floor = jnp.asarray(self.source_model.adaptive_reg_floor, dtype=jnp.float32)
        alpha = jnp.asarray(self.source_model.adaptive_reg_alpha, dtype=jnp.float32)
        mode = self.source_model.adaptive_reg_mode
        sigma = float(self.source_model.adaptive_reg_smooth_sigma)

        # 1. Brightness and inverse-variance at seed mask pixels
        seed_flat = self.sim_obj.seed_flat_indices
        brightness = jnp.maximum(
            jnp.ravel(self.image_data)[seed_flat], 0.0,
        )
        noise_at_seed = jnp.ravel(self.noise_map)[seed_flat]
        inv_var = 1.0 / (noise_at_seed ** 2)

        # 2. Bilinear weights mapping seed betas → source grid
        data_mesh = jnp.stack(
            [jnp.ravel(beta_x_seed), jnp.ravel(beta_y_seed)], axis=1,
        )
        weights, indices, valid = lens_mapping_operator_bilinear_rectangular_from(
            data_mesh, xmin, xmax, ymin, ymax, nx, ny,
        )
        valid_f = valid[:, None].astype(weights.dtype)

        if mode == "brightness_only":
            # Inverse-variance-weighted normalized convolution: N / C
            w_num = weights * (brightness[:, None] * inv_var[:, None] * valid_f)
            w_den = weights * (inv_var[:, None] * valid_f)

            N = jax.ops.segment_sum(
                w_num.ravel(), indices.ravel(), num_segments=n_source,
            )
            C = jax.ops.segment_sum(
                w_den.ravel(), indices.ravel(), num_segments=n_source,
            )

            N_sm = self.reg_builder.smooth_scale_map(N, nx, ny, sigma=sigma)
            C_sm = self.reg_builder.smooth_scale_map(C, nx, ny, sigma=sigma)
            b_raw = N_sm / (C_sm + 1e-10)
        else:
            # brightness_weighted: inv-var-weighted brightness × ray-count
            w_bright = weights * (brightness[:, None] * inv_var[:, None] * valid_f)
            q = jax.ops.segment_sum(
                w_bright.ravel(), indices.ravel(), num_segments=n_source,
            )
            b_raw = self.reg_builder.smooth_scale_map(q, nx, ny, sigma=sigma)

        # 3. Shared downstream: normalize → scale formula
        b_norm = DenseRegularizationBuilder._normalize_brightness(b_raw)
        scale = DenseRegularizationBuilder._compute_scale_formula(
            b_norm, alpha, floor,
        )

        return scale

    # ------------------------------------------------------------------
    # Empirical-Bayes freeze API
    # ------------------------------------------------------------------

    def freeze_scale(self) -> None:
        """Eagerly compute and cache the adaptive scale map.

        Implements the empirical-Bayes freeze: the scale map is evaluated
        once at the current lens-parameter values and reused for all
        subsequent evidence evaluations.  This prevents the adaptive
        prior from drifting during lens-parameter sampling.

        MUST be called before the likelihood is JIT-traced (i.e. before
        :func:`make_likelihood` / sampler startup) so that the JIT
        compiler captures the frozen array as a closure constant.  Calling
        it after tracing has no effect on the compiled graph.

        No-op when ``adaptive_reg_alpha == 0`` (uniform regularization)
        or ``adaptive_reg_freeze == False``.
        """
        if abs(self.source_model.adaptive_reg_alpha) < 1e-10:
            return
        if not self.source_model.adaptive_reg_freeze:
            warnings.warn(
                "freeze_scale() called but adaptive_reg_freeze=False; "
                "the cached scale will not be used.",
                stacklevel=2,
            )
            return
        (xmin, xmax, ymin, ymax, _beta_x_sub, _beta_y_sub,
         beta_x_seed, beta_y_seed) = self._get_bbox()
        # Bypass the freeze cache check in _compute_reg_scale_from_betas by
        # calling the core directly — otherwise the first eager evaluation
        # would re-enter the freeze branch and spuriously warn.
        scale = self._compute_scale_core(
            beta_x_seed, beta_y_seed, xmin, xmax, ymin, ymax,
        )
        object.__setattr__(self, '_frozen_scale', scale)

    def unfreeze_scale(self) -> None:
        """Discard the cached adaptive scale map.

        Subsequent evidence evaluations recompute the scale on every call.
        Safe to call when no scale is cached.
        """
        if hasattr(self, '_frozen_scale'):
            object.__delattr__(self, '_frozen_scale')

    # ------------------------------------------------------------------
    # Source solve via PCG
    # ------------------------------------------------------------------

    def _solve_source(
        self,
        xmin, xmax, ymin, ymax,
        lambda_reg: Array,
        reg_data: tuple,
        preconditioner,
        op_data=None,  # precomputed LensOperatorData
    ) -> tuple[Array, PCGInfo]:
        """Solve for MAP source pixels using PCG.

        ``preconditioner`` may be a dense Cholesky factor (legacy) or a
        block-diagonal ``(block_chols, block_masks)`` tuple.

        Returns ``(source_pixels, pcg_info)``.
        """
        A_data, _A_jit_prebound = self.sim_obj.build_A_matvec(
            self.noise_1d,
            xmin, xmax, ymin, ymax,
            lambda_reg,
            reg_data,
            op_data=op_data,
        )
        b = self.sim_obj.build_rhs(
            self.data_1d, self.noise_1d,
            xmin, xmax, ymin, ymax,
            op_data=op_data,
        )

        source_pixels, pcg_info = pcg_solve(
            A_data,
            b,
            preconditioner,
            _A_jit_prebound,
            max_iter=self.pcg_max_iter,
            rtol=self.pcg_rtol,
        )
        return source_pixels, pcg_info

    # ------------------------------------------------------------------
    # Log evidence
    # ------------------------------------------------------------------

    def _log_evidence(self) -> Array:
        """Evaluate log evidence using PCG and block-diagonal preconditioner.

        .. warning::
            This uses ``logdet(P)`` as a deterministic approximation to
            ``logdet(A)``, where ``P`` is the block-diagonal preconditioner,
            and a block-diagonal approximation (with the same partition) for
            ``logdet(R)``.  For blurred or asymmetric PSFs, ``nsub > 1``, or
            masked pixels, the evidence will deviate from the exact dense-backend
            value.  Use the dense backend when exact evidence parity is required.
        """
        log_lambda_reg = jnp.asarray(self.source_model.log_lambda_reg.value)
        lambda_reg = jnp.exp(log_lambda_reg)
        (xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub,
         beta_x_seed, beta_y_seed) = self._get_bbox()

        # Compute adaptive regularization scale map (zero extra tracing)
        scale = self._compute_reg_scale_from_betas(
            beta_x_seed, beta_y_seed, xmin, xmax, ymin, ymax,
        )

        # reg_data: compact edge-weighted Laplacian descriptor for matrix-free PCG.
        reg_data = self._regularization_data(xmin, xmax, ymin, ymax, scale=scale)

        # ---- Precompute lens-operator data ONCE ----
        # Reuse sub-grid betas from _get_bbox to avoid a second deflection call.
        op_data = self.sim_obj.precompute_operator_data(
            xmin, xmax, ymin, ymax,
            _betas_sub=(beta_x_sub, beta_y_sub),
        )

        # Build block-diagonal preconditioner
        block_chols, block_masks = self.sim_obj.build_block_diag_preconditioner(
            self.noise_1d,
            xmin, xmax, ymin, ymax,
            lambda_reg,
            self.reg_builder,
            block_size=self.block_size,
            scale=scale,
        )
        preconditioner = (block_chols, block_masks)

        # Solve source via PCG (uses compact reg_data, matrix-free).
        source_pixels, pcg_info = self._solve_source(
            xmin, xmax, ymin, ymax,
            lambda_reg,
            reg_data,
            preconditioner,
            op_data=op_data,
        )

        # Penalize non-converged PCG solves.
        pcg_penalty = jnp.where(
            pcg_info.converged, 0.0, -1.0e10
        )

        # Forward model for χ² and regularization penalty (reuses op_data)
        model_1d = self.sim_obj.forward_model(
            source_pixels, xmin, xmax, ymin, ymax,
            op_data=op_data,
        )

        n_source = self.sim_obj.n_source_pixels
        n_data = int(self.data_1d.size)

        resid = self.data_1d - model_1d
        e_d = 0.5 * jnp.sum((resid / self.noise_1d) ** 2)

        # Regularisation energy — matrix-free via edge-weighted Laplacian matvec.
        e_s = 0.5 * jnp.dot(
            source_pixels,
            self.reg_builder.matvec_free(
                source_pixels, xmin, xmax, ymin, ymax, scale=scale,
            ),
        )

        # logdet(P) from block-diagonal Cholesky factors
        logdet_P = PixelizedLensOperator.logdet_block_diag(block_chols)

        # logdet regularisation matrix — deterministic block-diagonal approximation
        logdet_h = self.reg_builder.logdet_free(
            xmin, xmax, ymin, ymax, scale=scale, block_size=self.block_size,
        )

        log_evidence = (
            -e_d
            - lambda_reg * e_s
            - 0.5 * logdet_P
            + 0.5 * n_source * log_lambda_reg
            + 0.5 * logdet_h
            - 0.5 * n_data * jnp.log(2.0 * jnp.pi)
            - 0.5 * self.logdet_C
            + pcg_penalty
        )

        return jnp.where(jnp.isfinite(log_evidence), log_evidence, -1.0e10)

    # ------------------------------------------------------------------
    # Forward model (public API)
    # ------------------------------------------------------------------

    @ck.forward
    def forward_model(
        self, *, return_source: bool = False, return_components: bool = False
    ):
        """Solve linear params and return the reconstructed model image."""
        lambda_reg = jnp.exp(jnp.asarray(self.source_model.log_lambda_reg.value))
        (xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub,
         beta_x_seed, beta_y_seed) = self._get_bbox()

        scale = self._compute_reg_scale_from_betas(
            beta_x_seed, beta_y_seed, xmin, xmax, ymin, ymax,
        )

        reg_data = self._regularization_data(xmin, xmax, ymin, ymax, scale=scale)

        # Precompute operator data once (reuse betas from _get_bbox).
        op_data = self.sim_obj.precompute_operator_data(
            xmin, xmax, ymin, ymax,
            _betas_sub=(beta_x_sub, beta_y_sub),
        )

        # Build block-diagonal preconditioner and solve
        block_chols, block_masks = self.sim_obj.build_block_diag_preconditioner(
            self.noise_1d,
            xmin, xmax, ymin, ymax,
            lambda_reg,
            self.reg_builder,
            block_size=self.block_size,
            scale=scale,
        )
        preconditioner = (block_chols, block_masks)

        source_pixels, pcg_info = self._solve_source(
            xmin, xmax, ymin, ymax,
            lambda_reg,
            reg_data,
            preconditioner,
            op_data=op_data,
        )

        model_1d = self.sim_obj.forward_model(
            source_pixels, xmin, xmax, ymin, ymax,
            op_data=op_data,
        )
        # Zero out the model image AND the source pixels when PCG fails to
        # converge — prevents silently returning partially-converged garbage.
        # Gating both keeps the (image, source) pair internally consistent for
        # callers that destructure ``forward_model(return_source=True)``.
        converged = pcg_info.converged
        model_1d = jnp.where(converged, model_1d, jnp.zeros_like(model_1d))
        source_pixels = jnp.where(converged, source_pixels, jnp.zeros_like(source_pixels))

        H, W = self.sim_obj.image_shape
        model_image = jnp.zeros(H * W, dtype=model_1d.dtype)
        model_image = model_image.at[self.sim_obj.flat_indices].set(model_1d)
        model_image = model_image.reshape(H, W)

        if return_source:
            return model_image, source_pixels
        return model_image

    # ------------------------------------------------------------------
    # Evidence callable
    # ------------------------------------------------------------------

    @ck.forward
    def __call__(self):
        """Return a finite scalar log evidence approximation."""
        log_ev = self._log_evidence()
        if self._has_pos_penalty:
            log_ev = log_ev + self._position_likelihood_penalty_jax()
        return log_ev

    def _position_likelihood_penalty_jax(self) -> Array:
        r"""Penalize image positions that don't map to the same source position."""
        beta_x, beta_y = self.phys_model.deflection(
            self._pos_px, self._pos_py
        )

        dx = beta_x[:, None] - beta_x[None, :]
        dy = beta_y[:, None] - beta_y[None, :]
        dist = jnp.sqrt(dx * dx + dy * dy)
        max_sep = jnp.max(dist)

        exceed = jnp.maximum(0.0, max_sep - self._pos_thr)
        ratio = jnp.where(self._pos_thr > 0.0, exceed / self._pos_thr, 0.0)
        pen_continuous = self._pos_minl * (1.0 - jnp.exp(-ratio))

        return jnp.clip(pen_continuous, min=self._pos_minl, max=0.0)

    def likelihood(self, debug: bool = True) -> float:
        """Return the current log evidence as a Python float."""
        return float(np.asarray(self.__call__()))


__all__ = ["PixelizedImageProbModelOperator"]
