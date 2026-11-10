"""Operator-based Bayesian evidence model for pixelized source inversions.

This module provides :class:`PixelizedImageProbModelOperator`, which uses
matrix-free operators and PCG or PNPG to solve the source inversion,
avoiding explicit construction of the (Nd × Ns) design matrix.

"""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral
from typing import Dict, NamedTuple, Optional, Union

import caskade as ck
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax import Array

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized_operator import (
    PixelizedLensOperator,
    SourceCurvatureData,
)
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.cg_solver import (
    pcg_solve,
    pcg_solve_with_implicit_grad,
    PCGInfo,
    BlockSchurPreconditioner,
    solve_source_blocks,
)
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import (
    EPSILON_REG,
    validate_lens_light_regularization,
)
from TinyLensGpu.utils.pnpg_solver import pnpg_nnls_solve_with_implicit_grad, PNPGInfo
from TinyLensGpu.utils.inversion.regularization import (
    FiniteDifferenceRegularization,
    RegData,
)
from .inversion_result import InversionDiagnostics, InversionResult
from ._inversion_diagnostics import reconstruction_diagnostics
from ._overlap_logdet import OverlapLogdet
from ._pixelized_regularization import (
    is_fixed as _is_fixed,
    adaptive_reg_scale,
    prepare_adapt_image,
    project_adapt_image,
)
from ._position_likelihood import (
    resolve_position_likelihood_attrs,
    compute_position_penalty_jax,
)


class _PreparedData(NamedTuple):
    image_adjoint: Array
    noise_weights: Array
    projected_brightness: Array | None
    regularization: tuple | None = None
    mapping: tuple | None = None


class _EvaluationDetails(NamedTuple):
    block_chols: Array | list[Array]
    reg_blocks: Array | list[Array]
    lambda_reg: Array
    model_source: Array
    model_lens: Array | None
    scale: Array | None
    evidence_valid: Array | None
    schur_solved: Array | None


@jax.custom_vjp
def _penalize_invalid_evidence(score: Array, valid: Array, penalty: Array) -> Array:
    """Return a finite sampling score and reject gradients of invalid scores."""
    return jnp.where(valid, score, penalty)


def _penalize_invalid_evidence_fwd(score, valid, penalty):
    return jnp.where(valid, score, penalty), valid


def _penalize_invalid_evidence_bwd(valid, cotangent):
    return jnp.where(valid, cotangent, jnp.full_like(cotangent, jnp.nan)), None, None


_penalize_invalid_evidence.defvjp(
    _penalize_invalid_evidence_fwd, _penalize_invalid_evidence_bwd,
)


class PixelizedImageProbModelOperator(ck.Module):
    """Operator evidence model for a pixelized source and optional lens light.

    Same public API as :class:`PixelizedImageProbModel` but uses matrix-free
    operators and iterative source solvers instead of explicit Cholesky
    decomposition. ``solver_type="pcg"`` is unconstrained. The ``"pnpg"``
    solver enforces non-negativity on source pixels and lens-light
    intensities while retaining the unconstrained-curvature operator logdet
    approximation. PNPG uses diagonal metric equilibration and is preferred for
    ill-conditioned joint source/MGE systems.

    Fixed inputs are prepared at construction. Configure parameter modes first;
    rebuild this model and its compiled likelihood after changing fixed physical
    parameters, parameter modes, observation data, or source-grid settings.
    Dynamic parameter values and solver controls are not cached. ``evaluate``
    returns evidence and reconstruction from the same solve.

    With ``fixed_source_bbox``, PCG and PNPG support first-order reverse-mode
    derivatives of reconstructions and evidence via implicit differentiation.
    PNPG requires a locally stable active set. Failed solves or invalid
    active sets yield NaN gradients. A failed primary solve returns zero
    reconstruction arrays, and an evidence failure retains any valid primary
    reconstruction. Both failures yield a finite sampling penalty.

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
    solver_type : {"pcg", "pnpg"}, optional
        Source solver. ``"pcg"`` is unconstrained. ``"pnpg"`` solves the
        matrix-free non-negative quadratic with all joint linear parameters
        constrained to be non-negative.
    evidence_logdet : {"block", "overlap"}, optional
        The default "overlap" uses exact principal-window determinant
        ratios and adds the lens-light Schur term for joint fits. Explicit
        "block" uses the historical preconditioner-based calculation.
    evidence_block_size : int, optional
        Source tile width for Overlap (default: 10), independent of
        the solver preconditioner block size.
    fixed_source_bbox : tuple, optional
        Fixed square ``(xmin, xmax, ymin, ymax)`` source-plane bbox. It keeps
        the source grid invariant under mass proposals and is required for
        adaptive regularization.
    adapt_image : array_like, optional
        Fixed, PSF-convolved reference source arc on the native image grid.
        The active subpixel rays project its brightness onto the fixed source
        nodes at each mass proposal.
        Required together with the fixed source bbox for adaptive rho,
        including dynamic rho initialized at zero.
    source_bbox_padding : float, optional
        Fractional padding passed to source-plane bbox inference when
        ``fixed_source_bbox`` is not supplied (default: 0.05 — 5% per side).
    source_bbox_outlier_frac : float, optional
        Fraction of ray-traced source-plane points trimmed from each tail
        during bbox inference when ``fixed_source_bbox`` is not supplied
        (default: 0.0 — use absolute min/max).
    pnpg_max_iter : int, optional
        Fixed projected-Nesterov iteration budget.
    pnpg_rtol : float, optional
        Componentwise-scaled KKT tolerance used to gate the likelihood.
    pnpg_power_iter : int, optional
        Power iterations used to estimate the equilibrated Lipschitz constant.
    pnpg_comp_rel_tol : float, optional
        Relative strict-complementarity margin for implicit derivatives
        (default: 1e-4). This does not change forward convergence. Redundant
        bright bases may need a smaller margin in double precision; verify
        KKT accuracy and finite-difference derivatives before using it.
    initial_source : array_like, optional
        Finite nodal intensities for the initial solver iterate. Requires
        ``fixed_source_bbox`` so the node coordinates stay fixed. Joint fits
        initialize the lens-light intensities at zero.
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
        source_bbox_padding: float = 0.05,
        source_bbox_outlier_frac: float = 0.0,
        pnpg_max_iter: int = 1000,
        pnpg_rtol: float = 2e-2,
        pnpg_power_iter: int = 20,
        lens_light_regularization: float = EPSILON_REG,
        pnpg_comp_rel_tol: float = 1e-4,
        *,
        initial_source: Union[np.ndarray, Array, None] = None,
        adapt_image: Union[np.ndarray, Array, None] = None,
        evidence_logdet: str = "overlap",
        evidence_block_size: int = 10,
    ) -> None:
        super().__init__("pixelized_image_prob_model_operator")
        if solver_type not in ("pcg", "pnpg"):
            raise ValueError(
                "solver_type must be 'pcg' or 'pnpg', " f"got {solver_type!r}"
            )
        if evidence_logdet not in ("block", "overlap"):
            raise ValueError("evidence_logdet must be 'block' or 'overlap'.")
        if (
            isinstance(evidence_block_size, bool)
            or not isinstance(evidence_block_size, Integral)
            or evidence_block_size < 1
        ):
            raise ValueError("evidence_block_size must be a positive integer.")
        self.evidence_logdet = evidence_logdet
        self.evidence_block_size = int(evidence_block_size)
        self.solver_type = solver_type
        self.lens_light_regularization = validate_lens_light_regularization(
            lens_light_regularization
        )

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
        self.n_lens_light = self.sim_obj.n_lens_light
        self.has_lens_light = self.sim_obj.has_lens_light
        self.unmask = ~jnp.asarray(sim_config.mask, dtype=bool)
        self.data_1d = self.image_data[self.unmask]
        self.noise_1d = self.noise_map[self.unmask]
        self.logdet_C = jnp.sum(jnp.log(self.noise_1d**2))

        source_n = int(self.source_model.n)
        self.reg_type = self.source_model.regularization_type
        self.reg_builder = FiniteDifferenceRegularization(
            source_n,
            self.reg_type,
        )
        self._fixed_source_bbox = self._validate_fixed_source_bbox(fixed_source_bbox)
        self._initial_linear = None
        if initial_source is not None:
            if self._fixed_source_bbox is None:
                raise ValueError("initial_source requires a fixed_source_bbox.")
            initial = jnp.asarray(initial_source)
            if initial.shape == (source_n, source_n):
                initial = initial.ravel()
            if initial.shape != (source_n**2,) or not jnp.all(jnp.isfinite(initial)):
                raise ValueError("initial_source must contain n² finite node intensities.")
            self._initial_linear = jnp.pad(initial, (0, self.n_lens_light))
        self._adapt_brightness = prepare_adapt_image(
            adapt_image, self.image_data, self.sim_obj,
            self.source_model.adaptive_reg_rho, self._fixed_source_bbox,
        )

        # Block-diagonal preconditioner settings
        self.block_size = int(block_size)

        # PCG settings
        self.pcg_max_iter = 400
        self.pcg_rtol = 1e-5
        self.pnpg_max_iter = int(pnpg_max_iter)
        self.pnpg_rtol = float(pnpg_rtol)
        self.pnpg_power_iter = int(pnpg_power_iter)
        if not np.isfinite(pnpg_comp_rel_tol) or pnpg_comp_rel_tol < 0.0:
            raise ValueError("pnpg_comp_rel_tol must be finite and non-negative")
        self.pnpg_comp_rel_tol = float(pnpg_comp_rel_tol)

        (
            self._pos_px,
            self._pos_py,
            self._pos_thr,
            self._pos_minl,
            self._has_pos_penalty,
        ) = resolve_position_likelihood_attrs(position_likelihood)

        with ck.ActiveContext(self):
            image_adjoint, noise_weights = self.sim_obj.prepare_observation(
                self.data_1d,
                self.noise_1d,
            )
            regularization = mapping = projected_brightness = None
            bbox = self._fixed_source_bbox
            if bbox is not None:
                mass_fixed = all(_is_fixed(mass) for mass in self.phys_model.lens_mass)
                if mass_fixed:
                    op_data = self.sim_obj.precompute_operator_data(*bbox)
                    rhs = self.sim_obj.build_rhs(
                        self.data_1d,
                        self.noise_1d,
                        *bbox,
                        op_data=op_data,
                        image_adjoint=image_adjoint,
                    )
                    data_blocks = self.sim_obj.build_data_blocks(
                        self.noise_1d,
                        *bbox,
                        self.block_size,
                        noise_weights,
                    )
                    mapping = (op_data, rhs, data_blocks)
                    if self._adapt_brightness is not None:
                        projected_brightness = self._project_reference(op_data)
                rho = self.source_model.adaptive_reg_rho
                if _is_fixed(rho) and (self._adapt_brightness is None or mass_fixed):
                    scale = adaptive_reg_scale(projected_brightness, rho)
                    reg_data, blocks = self._build_regularization(bbox, scale)
                    regularization = (
                        reg_data,
                        blocks,
                        self._regularization_logdet(bbox, scale, blocks)
                        if evidence_logdet == "block" else None,
                    )
            object.__setattr__(
                self, "_prepared",
                _PreparedData(image_adjoint, noise_weights, projected_brightness, regularization, mapping),
            )

        self._overlap_logdet = None
        if evidence_logdet == "overlap":
            self._overlap_logdet = OverlapLogdet(
                self.sim_obj, self.reg_builder, self.noise_1d,
                self.evidence_block_size,
                fixed_geometry=(bbox, mapping[0]) if mapping is not None else None,
            )

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
        bbox = tuple(jnp.asarray(v, dtype=jnp.result_type(v, jnp.float32)) for v in fixed_source_bbox)
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
        """Infer square source-plane bounding box from seed-region ray-tracing.

        Returns ``(xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub,
        beta_x_seed, beta_y_seed)`` with the subpixel rays needed to build
        the interpolation stencil.
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

    def infer_source_bbox(self) -> tuple[float, float, float, float]:
        """Return the source bbox implied by the current reference model.

        This public helper is intended for constructing a second probability
        model with ``fixed_source_bbox=...``.  Fixing the source coordinate
        system is important during mass-model sampling: otherwise every mass
        proposal changes both the lens mapping and the pixelized source basis.
        """
        bbox = self._fixed_source_bbox
        if bbox is None:
            bbox = self._get_bbox()[:4]
        return tuple(float(np.asarray(value)) for value in bbox)

    def _mapping_terms(self):
        """Return proposal geometry and any invariant data normal terms."""
        bbox = self._fixed_source_bbox
        if self._prepared.mapping is not None:
            return bbox, *self._prepared.mapping
        if bbox is None:
            *bounds, bx, by, _, _ = self._get_bbox()
            bbox = tuple(bounds)
            op_data = self.sim_obj.precompute_operator_data(*bbox, _betas_sub=(bx, by))
        else:
            op_data = self.sim_obj.precompute_operator_data(*bbox)
        return bbox, op_data, None, None

    # ------------------------------------------------------------------
    # Regularization matrix
    # ------------------------------------------------------------------

    def _regularization_data(
        self, xmin, xmax, ymin, ymax, scale: Array | None = None,
    ) -> tuple:
        """Return the compact :class:`RegData` tuple for the matrix-free PCG path.

        Only finite-difference regularization types are supported by the
        operator backend.

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

    def _project_reference(self, op_data) -> Array:
        return project_adapt_image(
            self._adapt_brightness, op_data.weights, op_data.indices,
            op_data.n_source, op_data.nsub,
            interpolation=getattr(self.source_model, "interpolation", "bilinear"),
        )

    def _get_reg_scale(self, op_data=None) -> Array | None:
        """Compute the scale from fixed brightness or this proposal's stencil."""
        projected_brightness = self._prepared.projected_brightness
        if self._adapt_brightness is not None and projected_brightness is None:
            if op_data is None:
                _, op_data, _, _ = self._mapping_terms()
            projected_brightness = self._project_reference(op_data)
        return adaptive_reg_scale(projected_brightness, self.source_model.adaptive_reg_rho)

    # ------------------------------------------------------------------
    # Source solve via configured iterative solver
    # ------------------------------------------------------------------

    @staticmethod
    def _equilibrated_cholesky(
        matrix: Array,
    ) -> tuple[Array, Array]:
        """Return a finite Cholesky factor and its stabilized SPD matrix.

        Highly redundant MGE columns can make a mathematically positive lens
        Gram block appear slightly indefinite in float32.  The physical joint
        operator is left untouched; only its small dense preconditioner block
        is diagonally equilibrated and floored at the precision-resolvable
        relative eigenvalue scale.
        """
        matrix = 0.5 * (matrix + matrix.T)
        dtype = matrix.dtype
        eps = jnp.finfo(dtype).eps
        diagonal = jnp.maximum(jnp.diag(matrix), eps)
        diagonal_scale = jnp.sqrt(diagonal)
        equilibrated = matrix / (
            diagonal_scale[:, None] * diagonal_scale[None, :]
        )
        equilibrated = 0.5 * (equilibrated + equilibrated.T)
        minimum_eigenvalue = jnp.min(jnp.linalg.eigvalsh(equilibrated))
        eigenvalue_floor = jnp.asarray(10.0 * eps, dtype=dtype)
        jitter = jnp.maximum(
            eigenvalue_floor,
            eigenvalue_floor - minimum_eigenvalue,
        )
        stabilized_equilibrated = equilibrated + jitter * jnp.eye(
            matrix.shape[0], dtype=dtype,
        )
        equilibrated_chol = jnp.linalg.cholesky(stabilized_equilibrated)
        chol = diagonal_scale[:, None] * equilibrated_chol
        stabilized = matrix + jitter * jnp.diag(diagonal)
        return chol, stabilized

    def _build_block_schur_preconditioner(
        self, source_chols: Array, source_masks: Array,
        source_data: SourceCurvatureData, lens_matrix: Array,
    ) -> tuple[BlockSchurPreconditioner, Array]:
        """Build the preconditioner and return the physical cross curvature."""
        noise_var = source_data.noise_variance

        def cross_column(_, lens_column):
            column = self.sim_obj.apply_source_data_adjoint(
                lens_column / noise_var, source_data,
            )
            return None, column

        _, cross_columns = jax.lax.scan(
            cross_column, None, lens_matrix.T,
        )
        cross = cross_columns.T
        lens_block = lens_matrix.T @ (lens_matrix / noise_var[:, None])
        lens_block = lens_block + self.lens_light_regularization * jnp.eye(
            self.n_lens_light, dtype=lens_matrix.dtype,
        )
        inverse_cross = jax.vmap(
            lambda column: solve_source_blocks(
                source_chols, source_masks, column,
            ),
            in_axes=1, out_axes=1,
        )(cross)
        cross_curvature = cross.T @ inverse_cross

        # A block-diagonal approximation to the source curvature can be too
        # weak for the exact source/lens cross block, making the approximate
        # Schur complement indefinite even though the true joint curvature is
        # SPD.  Spectrally damp only the preconditioner cross block when this
        # happens; the joint operator and physical regularization are unchanged.
        lens_chol, stabilized_lens_block = self._equilibrated_cholesky(
            lens_block
        )
        whitened = jax.scipy.linalg.solve_triangular(
            lens_chol, cross_curvature, lower=True,
        )
        whitened = jax.scipy.linalg.solve_triangular(
            lens_chol, whitened.T, lower=True,
        ).T
        max_coupling = jnp.max(jnp.linalg.eigvalsh(
            0.5 * (whitened + whitened.T)
        ))
        margin = jnp.asarray(1.0e-3, dtype=lens_matrix.dtype)
        cross_scale = jnp.minimum(
            1.0,
            jnp.sqrt((1.0 - margin) / jnp.maximum(max_coupling, margin)),
        )
        schur = stabilized_lens_block - (cross_scale ** 2) * cross_curvature
        schur = 0.5 * (schur + schur.T)
        schur_chol, _ = self._equilibrated_cholesky(
            schur
        )
        return BlockSchurPreconditioner(
            source_chols, source_masks, cross_scale * cross, schur_chol,
        ), cross

    @staticmethod
    def _logdet_block_schur(preconditioner: BlockSchurPreconditioner) -> Array:
        return (
            PixelizedLensOperator.logdet_block_diag(preconditioner.source_chols)
            + 2.0 * jnp.sum(jnp.log(jnp.diag(preconditioner.schur_chol)))
        )

    def _logdet_lens_schur(
        self, source_operator, source_preconditioner, cross, lens_matrix, op_data, bbox,
    ) -> tuple[Array, Array]:
        """Compute the joint determinant correction from the physical cross block."""
        def solve_column(rhs):
            return pcg_solve_with_implicit_grad(
                source_operator, rhs, source_preconditioner,
                max_iter=2 * self.pcg_max_iter, rtol=min(self.pcg_rtol, 1e-6),
            )

        response, info = jax.vmap(solve_column, in_axes=1, out_axes=(1, 0))(cross)
        source_image = jax.vmap(
            lambda pixels: self.sim_obj.forward_model(pixels, *bbox, op_data=op_data),
            in_axes=1, out_axes=1,
        )(response)
        data = source_operator.data
        residual = (lens_matrix - source_image) / self.noise_1d[:, None]
        source_factor = jnp.sqrt(data.lambda_reg) * self.reg_builder.apply_factor_free(
            response, data.regularization,
        )
        lens_strength = jnp.asarray(self.lens_light_regularization, dtype=residual.dtype)
        lens_factor = jnp.sqrt(lens_strength) * jnp.eye(self.n_lens_light, dtype=residual.dtype)

        # With A X = B, this augmented residual has Gram D - B.T @ X.
        # QR avoids explicitly subtracting nearly equal curvature blocks.
        factor = jnp.concatenate((residual, source_factor, lens_factor), axis=0)
        _, triangular = jnp.linalg.qr(factor, mode="reduced")
        logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(triangular))))
        solved = jnp.all(info.converged & ~info.failed) & jnp.all(jnp.isfinite(response))
        return logdet, solved

    def _solve_source(
        self,
        xmin,
        xmax,
        ymin,
        ymax,
        lambda_reg: Array,
        reg_data: RegData,
        preconditioner,
        op_data=None,  # precomputed LensOperatorData
        source_rhs=None,
        lens_matrix=None,
    ) -> tuple[Array, PCGInfo | PNPGInfo]:
        """Solve for joint MAP linear parameters with the configured solver.

        ``preconditioner`` may be a dense Cholesky factor (legacy) or a
        block-diagonal ``(block_chols, block_masks)`` tuple. It is used by PCG
        and still built for constrained solvers because PNPG uses its
        diagonal for equilibration. Explicit Block evidence reuses its
        determinant.

        Returns ``(linear_params, solver_info)``. For constrained solvers, all
        returned linear parameters are projected to be non-negative.
        """
        if self.has_lens_light:
            operator, b, _ = self.sim_obj.build_joint_system(
                self.noise_1d,
                self.data_1d,
                xmin,
                xmax,
                ymin,
                ymax,
                lambda_reg,
                reg_data,
                self.lens_light_regularization,
                op_data=op_data,
                image_adjoint=self._prepared.image_adjoint,
                source_rhs=source_rhs,
                lens_matrix=lens_matrix,
            )
        else:
            operator = self.sim_obj.build_source_curvature_operator(
                self.noise_1d,
                xmin,
                xmax,
                ymin,
                ymax,
                lambda_reg,
                reg_data,
                op_data=op_data,
            )
            b = source_rhs
            if b is None:
                b = self.sim_obj.build_rhs(
                    self.data_1d,
                    self.noise_1d,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    op_data=op_data,
                    image_adjoint=self._prepared.image_adjoint,
                )

        if self.solver_type == "pnpg":
            warm_start = self._initial_linear
            if warm_start is None:
                warm_start, warm_info = pcg_solve(
                    operator,
                    b,
                    preconditioner,
                    max_iter=self.pcg_max_iter,
                    rtol=self.pcg_rtol,
                )
                warm_start = jnp.where(
                    warm_info.converged & jnp.all(jnp.isfinite(warm_start)),
                    warm_start,
                    jnp.zeros_like(warm_start),
                )
            source_pixels, solver_info = pnpg_nnls_solve_with_implicit_grad(
                operator,
                b,
                preconditioner,
                x0=warm_start,
                max_iter=self.pnpg_max_iter,
                rtol=self.pnpg_rtol,
                power_iter=self.pnpg_power_iter,
                comp_rel_tol=self.pnpg_comp_rel_tol,
                backward_pcg_max_iter=2 * self.pcg_max_iter,
                backward_pcg_rtol=min(self.pcg_rtol, 1e-7),
            )
        else:
            source_pixels, solver_info = pcg_solve_with_implicit_grad(
                operator,
                b,
                preconditioner,
                x0=self._initial_linear,
                max_iter=self.pcg_max_iter,
                rtol=self.pcg_rtol,
                backward_max_iter=2 * self.pcg_max_iter,
                backward_rtol=min(self.pcg_rtol, 1e-7),
            )
        return source_pixels, solver_info

    # ------------------------------------------------------------------
    # Log evidence
    # ------------------------------------------------------------------

    def _build_regularization(self, bbox, scale):
        return (
            self._regularization_data(*bbox, scale=scale),
            self.reg_builder.block_matrices(
                *bbox, scale=scale, block_size=self.block_size
            ),
        )

    def _regularization_logdet(self, bbox, scale, blocks):
        if self.reg_type == "zero-order":
            return self.reg_builder.logdet_free(
                *bbox, scale=scale, block_size=self.block_size
            )
        return self.reg_builder.logdet_blocks(blocks)

    def _evaluate(
        self, *, compute_evidence: bool, include_position: bool = False,
        return_details: bool = False,
    ) -> InversionResult | tuple[InversionResult, _EvaluationDetails]:
        """Prepare, solve and reconstruct once; optionally compute evidence."""
        log_lambda = jnp.asarray(self.source_model.log_lambda_reg.value)
        lambda_reg = jnp.exp(log_lambda)
        bbox, op_data, source_rhs, data_blocks = self._mapping_terms()

        logdet_h = None
        if self._prepared.regularization is not None:
            reg_data, reg_blocks, logdet_h = self._prepared.regularization
            scale = reg_data.scale
        else:
            scale = self._get_reg_scale(op_data)
            reg_data, reg_blocks = self._build_regularization(bbox, scale)
        block_chols, block_masks = self.sim_obj.build_block_diag_preconditioner(
            self.noise_1d,
            *bbox,
            lambda_reg,
            self.reg_builder,
            block_size=self.block_size,
            scale=scale,
            noise_weights=self._prepared.noise_weights,
            data_blocks=data_blocks,
            reg_blocks=reg_blocks,
        )
        lens_matrix = None
        if self.has_lens_light:
            lens_matrix = self.sim_obj.build_lens_light_matrix()
            operator = self.sim_obj.build_source_curvature_operator(
                self.noise_1d,
                *bbox,
                lambda_reg,
                reg_data,
                op_data=op_data,
            )
            preconditioner, cross = self._build_block_schur_preconditioner(
                block_chols,
                block_masks,
                operator.data,
                lens_matrix,
            )
        else:
            preconditioner = (block_chols, block_masks)
        linear_params, solver_info = self._solve_source(
            *bbox,
            lambda_reg,
            reg_data,
            preconditioner,
            op_data=op_data,
            source_rhs=source_rhs,
            lens_matrix=lens_matrix,
        )
        n_source = self.sim_obj.n_source_pixels
        source = linear_params[:n_source]
        lens = linear_params[n_source:] if self.has_lens_light else None
        model_source = self.sim_obj.forward_model(source, *bbox, op_data=op_data)
        model_lens = lens_matrix @ lens if lens is not None else None
        model = model_source + model_lens if model_lens is not None else model_source

        log_evidence = evidence_valid = schur_solved = None
        if compute_evidence:
            e_d = 0.5 * jnp.sum(((self.data_1d - model) / self.noise_1d) ** 2)
            e_s = 0.5 * jnp.dot(
                source, self.reg_builder.matvec_free(source, *bbox, scale=scale)
            )
            log_evidence = -e_d - lambda_reg * e_s
            if self._overlap_logdet is None:
                logdet_p = (
                    self._logdet_block_schur(preconditioner)
                    if self.has_lens_light
                    else self.sim_obj.logdet_block_diag(block_chols)
                )
                if logdet_h is None:
                    logdet_h = self._regularization_logdet(bbox, scale, reg_blocks)
                log_evidence = (
                    log_evidence - 0.5 * logdet_p
                    + 0.5 * n_source * log_lambda + 0.5 * logdet_h
                )
            else:
                ratio = self._overlap_logdet(op_data, bbox, reg_data, lambda_reg)
                if lens is not None:
                    schur_logdet, schur_solved = self._logdet_lens_schur(
                        operator, (block_chols, block_masks), cross, lens_matrix, op_data, bbox,
                    )
                    ratio = ratio + schur_logdet
                log_evidence = log_evidence - 0.5 * ratio
            log_evidence = (
                log_evidence
                - 0.5 * self.data_1d.size * jnp.log(2.0 * jnp.pi)
                - 0.5 * self.logdet_C
            )
            if lens is not None:
                log_evidence = (
                    log_evidence
                    - 0.5 * self.lens_light_regularization * jnp.sum(lens**2)
                    + 0.5 * self.n_lens_light * jnp.log(self.lens_light_regularization)
                )
            evidence_valid = solver_info.converged & jnp.isfinite(log_evidence)
            if schur_solved is not None:
                evidence_valid = evidence_valid & schur_solved

            penalty = jnp.full_like(log_evidence, -1.0e10)
            if include_position and self._has_pos_penalty:
                position_penalty = self._position_likelihood_penalty_jax()
                log_evidence = log_evidence + position_penalty
                evidence_valid = evidence_valid & jnp.isfinite(log_evidence)
                # Historically the primary failure score also included the
                # position penalty whenever their sum was finite.
                primary_penalty = penalty + position_penalty
                penalty = jnp.where(
                    ~solver_info.converged & jnp.isfinite(primary_penalty),
                    primary_penalty, penalty,
                )
            log_evidence = _penalize_invalid_evidence(log_evidence, evidence_valid, penalty)

        # Preserve forward_model's failure contract for every reconstruction output.
        source = jnp.where(solver_info.converged, source, jnp.zeros_like(source))
        model = jnp.where(solver_info.converged, model, jnp.zeros_like(model))
        if lens is not None:
            lens = jnp.where(solver_info.converged, lens, jnp.zeros_like(lens))
        image = jnp.zeros(self.image_data.size, dtype=model.dtype)
        image = (
            image.at[self.sim_obj.flat_indices]
            .set(model)
            .reshape(self.image_data.shape)
        )
        result = InversionResult(log_evidence, image, source, lens, bbox, solver_info)
        if not return_details:
            return result
        return result, _EvaluationDetails(
            block_chols, reg_blocks, lambda_reg, model_source, model_lens,
            scale, evidence_valid, schur_solved,
        )

    def _log_evidence(self) -> Array:
        """Evidence with the selected curvature determinant approximation.

        Constrained solvers retain the unconstrained-style determinant terms;
        this is not the exact evidence of a truncated non-negative prior.
        """
        return self._evaluate(compute_evidence=True).log_evidence

    @ck.forward
    def evaluate(self) -> InversionResult:
        """Return evidence, model image, source and solver status from one solve.

        Includes the configured position penalty. Failed solves return the same
        evidence penalty and zero reconstruction arrays as the legacy methods.
        This method is JIT/vmap compatible and keeps no last-result cache.
        """
        return self._evaluate(compute_evidence=True, include_position=True)

    @ck.forward
    def evaluate_with_diagnostics(
        self,
        *,
        include_effective_dof: bool = False,
        include_components: bool = False,
    ) -> tuple[InversionResult, InversionDiagnostics]:
        """Return the inversion result plus host-side diagnostics from one solve.

        Shares ``evaluate``'s evidence and position-penalty convention and its
        solver-status/zero-reconstruction failure contract. Source-only
        ``include_effective_dof`` fills the block-preconditioner
        effective-parameter statistics; joint requests report
        ``operator_joint_unsupported`` without marking the reconstruction
        invalid. ``include_components`` returns the source and lens-light
        images from the same solve. Unrequested fields stay ``None``.
        """
        result, details = self._evaluate(
            compute_evidence=True, include_position=True, return_details=True,
        )
        diagnostics = reconstruction_diagnostics(self, result, details.scale)
        evidence_valid = bool(details.evidence_valid)
        evidence_reason = None
        if not evidence_valid:
            if not diagnostics.valid:
                evidence_reason = "Evidence requires a valid primary reconstruction."
            elif details.schur_solved is not None and not bool(details.schur_solved):
                evidence_reason = "Overlap auxiliary source-curvature solve failed."
            else:
                evidence_reason = "Evidence calculation returned a nonfinite value."
        diagnostics = replace(
            diagnostics,
            evidence_logdet=self.evidence_logdet,
            evidence_valid=evidence_valid,
            evidence_invalid_reason=evidence_reason,
        )
        if include_effective_dof:
            if self.has_lens_light:
                diagnostics = replace(
                    diagnostics,
                    effective_dof_method="operator_joint_unsupported",
                    effective_dof_reason=(
                        "Effective-parameter diagnostics are unsupported for "
                        "joint operator inversion."
                    ),
                )
            elif diagnostics.valid:
                trace_inv_pr = sum(
                    jnp.trace(jsl.cho_solve((chol, True), rblock))
                    for chol, rblock in zip(details.block_chols, details.reg_blocks)
                )
                effective = float(
                    self.sim_obj.n_source_pixels - details.lambda_reg * trace_inv_pr
                )
                dof = int(self.data_1d.size) - effective
                diagnostics = replace(
                    diagnostics,
                    effective_parameters=effective,
                    residual_dof=dof,
                    reduced_chi2=diagnostics.chi2 / dof if dof > 0 else 0.0,
                    effective_dof_method="operator_source_block_preconditioner",
                )
            else:
                diagnostics = replace(
                    diagnostics, effective_dof_reason=diagnostics.invalid_reason
                )
        if include_components:
            model_source, model_lens = details.model_source, details.model_lens
            zeros_active = jnp.zeros_like(model_source)
            source_component = jnp.where(
                result.solver_info.converged, model_source, zeros_active
            )
            lens_component = (
                jnp.where(result.solver_info.converged, model_lens, zeros_active)
                if model_lens is not None
                else zeros_active
            )

            def _component_image(values):
                full = jnp.zeros(self.image_data.size, dtype=values.dtype)
                return (
                    full.at[self.sim_obj.flat_indices]
                    .set(values)
                    .reshape(self.image_data.shape)
                )

            diagnostics = replace(
                diagnostics,
                source_image=_component_image(source_component),
                lens_light_image=_component_image(lens_component),
            )
        return result, diagnostics

    @ck.forward
    def forward_model(self, *, return_source=False, return_components=False):
        """Reconstruct without computing evidence; preserve legacy return formats."""
        result = self._evaluate(compute_evidence=False)
        if return_components and self.has_lens_light:
            return result.model_image, result.source_pixels, result.lens_amplitudes
        if return_source:
            return result.model_image, result.source_pixels
        return result.model_image

    def reconstruct_component_images(
        self, source_pixels: Array, lens_light_intensities: Array | None,
    ) -> tuple[Array, Array, tuple[Array, Array, Array, Array]]:
        """Reconstruct masked 2D source/lens images at the current parameters.

        The reconstruction uses the parameter values current at call time —
        fill dynamic parameters first (``with ck.ActiveContext(model):
        model.fill_params(theta)``) to match a chosen proposal; no last-result
        state is kept. Masked pixels (``mask=True``) are zero. With no lens light,
        pass ``None`` or an empty vector and the lens image is zero.
        """
        if self.has_lens_light and lens_light_intensities is None:
            raise ValueError("lens_light_intensities is required when lens_light is present")
        if not self.has_lens_light:
            lens_light_intensities = jnp.empty((0,), dtype=jnp.asarray(source_pixels).dtype)
        bbox, op_data, _, _ = self._mapping_terms()
        source_1d, lens_1d = self.sim_obj.reconstruct_components(
            source_pixels, lens_light_intensities, *bbox, op_data=op_data,
        )
        H, W = self.sim_obj.image_shape
        source_image = jnp.zeros(H * W, dtype=source_1d.dtype)
        source_image = source_image.at[self.sim_obj.flat_indices].set(source_1d)
        lens_image = jnp.zeros(H * W, dtype=lens_1d.dtype)
        lens_image = lens_image.at[self.sim_obj.flat_indices].set(lens_1d)
        return (
            source_image.reshape(H, W), lens_image.reshape(H, W),
            bbox,
        )

    # ------------------------------------------------------------------
    # Evidence callable
    # ------------------------------------------------------------------

    @ck.forward
    def __call__(self):
        """Return a finite scalar log evidence approximation."""
        return self.evaluate().log_evidence

    def _position_likelihood_penalty_jax(self) -> Array:
        """Compute position-likelihood penalty via shared utility (see ``_position_likelihood``)."""
        return compute_position_penalty_jax(
            self.phys_model, self._pos_px, self._pos_py, self._pos_thr, self._pos_minl,
        )

    def likelihood(self, debug: bool = True) -> float:
        """Return the current log evidence as a Python float."""
        return float(np.asarray(self.__call__()))


__all__ = ["PixelizedImageProbModelOperator", "InversionResult"]
