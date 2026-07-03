"""Operator-based Bayesian evidence model for pixelized source inversions.

This module provides :class:`PixelizedImageProbModelOperator`, which uses
matrix-free operators and PCG or FISTA to solve the source inversion,
avoiding explicit construction of the (Nd × Ns) design matrix.

Phase-1 limitation: lens-light joint inversion is not yet supported.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import caskade as ck
import jax.numpy as jnp
import numpy as np
from jax import Array

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized_operator import (
    PixelizedLensOperator,
    LensOperatorData,
)
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.cg_solver import pcg_solve, PCGInfo
from TinyLensGpu.utils.fista_solver import fista_nnls_solve, FISTAInfo
from TinyLensGpu.utils.inversion.regularization import (
    DenseRegularizationBuilder,
    source_template_scale_map,
)
from ._position_likelihood import resolve_position_likelihood_attrs, compute_position_penalty_jax


class PixelizedImageProbModelOperator(ck.Module):
    """Operator-based evidence model for one pixelized source.

    Same public API as :class:`PixelizedImageProbModel` but uses matrix-free
    operators and iterative source solvers instead of explicit Cholesky
    decomposition. ``solver_type="pcg"`` preserves the unconstrained source
    solve. ``solver_type="fista"`` enforces hard source non-negativity with a
    projected FISTA solve while retaining the existing operator logdet
    approximation.

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
    solver_type : {"pcg", "fista"}, optional
        Source solver. ``"pcg"`` is unconstrained. ``"fista"`` solves the
        matrix-free non-negative quadratic with source pixels constrained to
        be non-negative.
    fixed_source_bbox : tuple, optional
        Fixed square ``(xmin, xmax, ymin, ymax)`` source-plane bbox for
        S0-based adaptive regularization.
    fixed_reg_scale : array_like, optional
        Flat fixed adaptive regularization scale map with shape ``(n * n,)``.
    fixed_reg_template : array_like, optional
        Flat or 2D S0 source template used to generate the adaptive scale map
        from current ``adaptive_reg_rho`` values.
    source_bbox_padding : float, optional
        Fractional padding passed to source-plane bbox inference when
        ``fixed_source_bbox`` is not supplied.
    source_bbox_outlier_frac : float, optional
        Fraction of ray-traced source-plane points trimmed from each tail
        during bbox inference when ``fixed_source_bbox`` is not supplied.
    fista_max_iter : int, optional
        Fixed iteration budget for ``solver_type="fista"``. The default is
        intentionally conservative because evidence evaluation penalizes
        non-converged source solves.
    fista_rtol : float, optional
        Relative tolerance for the projected-gradient convergence metric.
    fista_power_iter : int, optional
        Number of power iterations used for FISTA step-size estimation.
    fista_step_safety : float, optional
        Safety factor applied to the estimated Lipschitz constant.
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
        solver_type: str = "pcg",
        fixed_source_bbox: tuple[float, float, float, float] | None = None,
        fixed_reg_scale: Union[np.ndarray, Array, None] = None,
        fixed_reg_template: Union[np.ndarray, Array, None] = None,
        source_bbox_padding: float = 0.0,
        source_bbox_outlier_frac: float = 0.01,
        fista_max_iter: int = 1000,
        fista_rtol: float = 1e-5,
        fista_power_iter: int = 10,
        fista_step_safety: float = 1.2,
    ) -> None:
        super().__init__("pixelized_image_prob_model_operator")
        if solver_type not in ("pcg", "fista"):
            raise ValueError(
                "solver_type must be 'pcg' or 'fista', "
                f"got {solver_type!r}"
            )
        self.solver_type = solver_type

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
            source_bbox_padding=source_bbox_padding,
            source_bbox_outlier_frac=source_bbox_outlier_frac,
        )
        self.sim_obj = PixelizedLensOperator(self.phys_model, sim_config)
        self.unmask = ~jnp.asarray(sim_config.mask, dtype=bool)
        self.data_1d = self.image_data[self.unmask]
        self.noise_1d = self.noise_map[self.unmask]
        self.logdet_C = jnp.sum(jnp.log(self.noise_1d**2))

        source_n = int(self.source_model.n)
        self.reg_type = self.source_model.regularization_type
        self.reg_builder = DenseRegularizationBuilder(
            source_n, self.reg_type,
        )
        self._fixed_source_bbox = self._validate_fixed_source_bbox(
            fixed_source_bbox
        )
        self._fixed_reg_scale = self._validate_fixed_reg_scale(
            fixed_reg_scale, source_n * source_n
        )
        self._fixed_reg_template = self._validate_fixed_reg_template(
            fixed_reg_template, source_n
        )
        if self._adaptive_reg_enabled():
            if self._fixed_source_bbox is None:
                raise ValueError(
                    "adaptive regularization in the operator backend requires "
                    "fixed_source_bbox from an S0 source template."
                )
            if self._fixed_reg_scale is None and self._fixed_reg_template is None:
                raise ValueError(
                    "adaptive regularization in the operator backend requires "
                    "fixed_reg_scale or fixed_reg_template derived from an S0 "
                    "source template."
                )

        # Block-diagonal preconditioner settings
        self.block_size = int(block_size)

        # PCG settings
        self.pcg_max_iter = 200
        self.pcg_rtol = 1e-6
        self.fista_max_iter = int(fista_max_iter)
        self.fista_rtol = float(fista_rtol)
        self.fista_power_iter = int(fista_power_iter)
        self.fista_step_safety = float(fista_step_safety)

        self._pos_px, self._pos_py, self._pos_thr, self._pos_minl, self._has_pos_penalty = \
            resolve_position_likelihood_attrs(position_likelihood)

    @staticmethod
    def _validate_fixed_source_bbox(
        fixed_source_bbox: tuple[float, float, float, float] | None,
    ) -> tuple[Array, Array, Array, Array] | None:
        if fixed_source_bbox is None:
            return None
        if len(fixed_source_bbox) != 4:
            raise ValueError(
                "fixed_source_bbox must be a 4-tuple "
                "(xmin, xmax, ymin, ymax)."
            )
        bbox = tuple(jnp.asarray(v, dtype=jnp.float32) for v in fixed_source_bbox)
        bbox_np = np.asarray([float(np.asarray(v)) for v in bbox], dtype=np.float64)
        if not np.all(np.isfinite(bbox_np)):
            raise ValueError("fixed_source_bbox values must be finite.")
        xmin, xmax, ymin, ymax = bbox_np
        if not (xmin < xmax and ymin < ymax):
            raise ValueError(
                "fixed_source_bbox must satisfy xmin < xmax and ymin < ymax."
            )
        if not np.isclose(xmax - xmin, ymax - ymin, rtol=1.0e-6, atol=1.0e-7):
            raise ValueError(
                "fixed_source_bbox must be square for pixelized source grids "
                "(xmax - xmin must equal ymax - ymin)."
            )
        return bbox

    @staticmethod
    def _validate_fixed_reg_scale(
        fixed_reg_scale: Union[np.ndarray, Array, None],
        expected_size: int,
    ) -> Array | None:
        if fixed_reg_scale is None:
            return None
        scale = jnp.asarray(fixed_reg_scale, dtype=jnp.float32)
        if scale.shape != (int(expected_size),):
            raise ValueError(
                "fixed_reg_scale must have shape "
                f"({int(expected_size)},), got {scale.shape}."
            )
        valid = bool(np.asarray(jnp.all(jnp.isfinite(scale) & (scale > 0.0))))
        if not valid:
            raise ValueError("fixed_reg_scale values must be finite and positive.")
        return scale

    @staticmethod
    def _validate_fixed_reg_template(
        fixed_reg_template: Union[np.ndarray, Array, None],
        n: int,
    ) -> Array | None:
        if fixed_reg_template is None:
            return None
        template = jnp.asarray(fixed_reg_template, dtype=jnp.float32)
        if template.shape == (int(n), int(n)):
            template = template.reshape(int(n) * int(n))
        elif template.shape != (int(n) * int(n),):
            raise ValueError(
                "fixed_reg_template must have shape "
                f"({int(n) * int(n)},) or ({int(n)}, {int(n)}), "
                f"got {template.shape}."
            )
        valid = bool(np.asarray(jnp.all(jnp.isfinite(template))))
        if not valid:
            raise ValueError("fixed_reg_template values must be finite.")
        return template

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

    @staticmethod
    def _param_value(value):
        return value.value if hasattr(value, "value") else value

    def _adaptive_reg_enabled(self) -> bool:
        rho = self.source_model.adaptive_reg_rho
        if bool(getattr(rho, "dynamic", False)):
            return True
        rho_value = self._param_value(rho)
        try:
            return abs(float(rho_value)) >= 1.0e-10
        except TypeError:
            return True

    # ------------------------------------------------------------------
    # Source-plane bbox
    # ------------------------------------------------------------------

    def _get_bbox(self):
        """Infer square source-plane bounding box from seed-region ray-tracing.

        Returns ``(xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub,
        beta_x_seed, beta_y_seed)`` so that callers can reuse the seed
        betas for adaptive regularisation without a second deflection call.
        """
        beta_x_sub, beta_y_sub, beta_x_seed, beta_y_seed = (
            self.sim_obj._get_beta_sub_and_seed()
        )
        if self._fixed_source_bbox is None:
            xmin, xmax, ymin, ymax = self.sim_obj._infer_and_fix_bbox(
                beta_x_seed, beta_y_seed
            )
        else:
            xmin, xmax, ymin, ymax = self._fixed_source_bbox
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

    def _get_reg_scale(self) -> Array | None:
        """Return the S0-derived adaptive scale map.

        The operator backend no longer constructs adaptive scale maps from
        image-plane seed rays.  Adaptive runs must provide either a fixed
        scale map or an S0 source template before JIT tracing; uniform runs
        keep the ``None`` fast path.
        """
        if not self._adaptive_reg_enabled():
            return None
        if self._fixed_reg_template is not None:
            source = self.source_model
            return source_template_scale_map(
                self._fixed_reg_template,
                int(source.n),
                rho=self._param_value(source.adaptive_reg_rho),
            )
        if self._fixed_reg_scale is None:
            raise ValueError(
                "fixed_reg_scale or fixed_reg_template is required when "
                "adaptive_reg_rho > 0 in PixelizedImageProbModelOperator."
            )
        return self._fixed_reg_scale

    # ------------------------------------------------------------------
    # Source solve via configured iterative solver
    # ------------------------------------------------------------------

    def _solve_source(
        self,
        xmin, xmax, ymin, ymax,
        lambda_reg: Array,
        reg_data: tuple,
        preconditioner,
        op_data=None,  # precomputed LensOperatorData
    ) -> tuple[Array, PCGInfo | FISTAInfo]:
        """Solve for MAP source pixels using the configured source solver.

        ``preconditioner`` may be a dense Cholesky factor (legacy) or a
        block-diagonal ``(block_chols, block_masks)`` tuple. It is used by PCG
        and still built for FISTA because the evidence approximation uses the
        same block-diagonal logdet term.

        Returns ``(source_pixels, solver_info)``. For ``solver_type="fista"``,
        ``source_pixels`` are projected to be non-negative.
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

        if self.solver_type == "fista":
            source_pixels, solver_info = fista_nnls_solve(
                A_data,
                b,
                _A_jit_prebound,
                max_iter=self.fista_max_iter,
                rtol=self.fista_rtol,
                power_iter=self.fista_power_iter,
                step_safety=self.fista_step_safety,
            )
        else:
            source_pixels, solver_info = pcg_solve(
                A_data,
                b,
                preconditioner,
                _A_jit_prebound,
                max_iter=self.pcg_max_iter,
                rtol=self.pcg_rtol,
            )
        return source_pixels, solver_info

    # ------------------------------------------------------------------
    # Log evidence
    # ------------------------------------------------------------------

    def _log_evidence(self) -> Array:
        """Evaluate log evidence using the configured matrix-free solver.

        .. warning::
            This uses ``logdet(P)`` as a deterministic approximation to
            ``logdet(A)``, where ``P`` is the block-diagonal preconditioner,
            and a block-diagonal approximation (with the same partition) for
            ``logdet(R)``.  For blurred or asymmetric PSFs, ``nsub > 1``, or
            masked pixels, the evidence will deviate from the exact dense-backend
            value.  For ``solver_type="fista"``, the source MAP is constrained
            but the logdet terms remain this same unconstrained-style operator
            approximation. Use the dense backend when exact evidence parity is
            required.
        """
        log_lambda_reg = jnp.asarray(self.source_model.log_lambda_reg.value)
        lambda_reg = jnp.exp(log_lambda_reg)
        (xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub,
         beta_x_seed, beta_y_seed) = self._get_bbox()

        # Compute adaptive regularization scale map (zero extra tracing)
        scale = self._get_reg_scale()

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

        # Solve source via configured matrix-free solver.
        source_pixels, solver_info = self._solve_source(
            xmin, xmax, ymin, ymax,
            lambda_reg,
            reg_data,
            preconditioner,
            op_data=op_data,
        )

        # Penalize non-converged source solves.
        solver_penalty = jnp.where(
            solver_info.converged, 0.0, -1.0e10
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
            + solver_penalty
        )

        return jnp.where(jnp.isfinite(log_evidence), log_evidence, -1.0e10)

    # ------------------------------------------------------------------
    # Forward model (public API)
    # ------------------------------------------------------------------

    @ck.forward
    def forward_model(
        self, *, return_source: bool = False, return_components: bool = False
    ):
        """Solve source pixels and return the reconstructed model image.

        With ``solver_type="fista"``, returned source pixels are constrained
        to be non-negative.
        """
        lambda_reg = jnp.exp(jnp.asarray(self.source_model.log_lambda_reg.value))
        (xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub,
         beta_x_seed, beta_y_seed) = self._get_bbox()

        scale = self._get_reg_scale()

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

        source_pixels, solver_info = self._solve_source(
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
        # Zero out the model image AND the source pixels when solving fails to
        # converge — prevents silently returning partially-converged garbage.
        # Gating both keeps the (image, source) pair internally consistent for
        # callers that destructure ``forward_model(return_source=True)``.
        converged = solver_info.converged
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
        """Compute position-likelihood penalty via shared utility (see ``_position_likelihood``)."""
        return compute_position_penalty_jax(
            self.phys_model, self._pos_px, self._pos_py, self._pos_thr, self._pos_minl,
        )

    def likelihood(self, debug: bool = True) -> float:
        """Return the current log evidence as a Python float."""
        return float(np.asarray(self.__call__()))


__all__ = ["PixelizedImageProbModelOperator"]
