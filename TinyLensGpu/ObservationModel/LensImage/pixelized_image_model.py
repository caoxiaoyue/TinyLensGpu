"""Bayesian evidence model for pixelized source inversions."""

# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Union

import caskade as ck
import functools
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax import Array, jit

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import (
    PixelizedLensSimulator,
    EPSILON_REG,
    validate_lens_light_regularization,
)
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.inversion.regularization import (
    DenseRegularizationBuilder,
    GP_REGULARIZATION_TYPES,
)
from TinyLensGpu.utils.linear_solver import fnnls_jax
from ._position_likelihood import resolve_position_likelihood_attrs, compute_position_penalty_jax


class _DensePrepared(NamedTuple):
    """Invariant terms for one dense pixelized evidence model.

    ``regularization`` is ``(matrix, log_determinant)`` when the fixed source
    grid makes the regularizer invariant. The determinant is ``logdet(R)`` for
    finite differences and ``log|K|`` for a fixed Gaussian-process kernel.
    ``design`` is ``(M, M^T C^{-1} M, M^T C^{-1} d)`` when the lens mapping and
    lens-light bases are fixed too. Either slot is absent when that tier does
    not apply.
    """

    regularization: tuple | None = None
    design: tuple | None = None


def _is_fixed(value) -> bool:
    """Use the operator fixedness rule, including transitive dependencies.

    A parameter is fixed when it is not dynamic. A pointer counts as fixed
    only when it has children, and those children must be fixed as well.
    """
    if not isinstance(value, (ck.Module, ck.Param)):
        return True
    params = (param for param in value.topological_ordering() if isinstance(param, ck.Param))
    return all(not param.dynamic and (not param.pointer or bool(param.children)) for param in params)


def _modules_are_fixed(modules) -> bool:
    """An empty component list is fixed; otherwise every component must be."""
    return all(_is_fixed(module) for module in modules)


class PixelizedImageProbModel(ck.Module):
    """Evidence probability model for one pixelized source.

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
        Physical model accepted by :class:`PixelizedLensSimulator`.
    mask : array_like, optional
        Boolean mask where ``True`` pixels are excluded.
    nsub : int, optional
        Subsampling factor for image-plane ray-tracing (default: 1).
    position_likelihood : dict, optional
        Position likelihood constraint configuration. If provided, the
        log-evidence is penalized when the observed image-plane positions
        do not map back to a common source-plane position. Expected keys:
        ``positions`` (list of [x, y] pairs), ``threshold_arcsec``,
        and ``min_log_like``.
    solver_type : str, optional
        Linear solver for MAP source/lens-light amplitudes. ``"cholesky"``
        (default) uses unconstrained normal equations; ``"nnls"`` enforces
        non-negativity via FNNLS.
    fixed_source_bbox : tuple, optional
        Fixed square ``(xmin, xmax, ymin, ymax)`` source-plane bbox. When
        omitted, each evaluation infers a bbox from seed rays. When supplied,
        finite-difference regularization (and a Gaussian-process kernel whose
        scale is fixed) is stored at construction. If every lens-mass component
        and every lens-light basis is fixed as well, the joint design matrix
        and its data-only normal equations are stored too. Regularization
        strength stays outside that cache. Rebuild this model and its compiled
        likelihood after changing a fixed value, a parameter mode, the
        observations, or the bbox.
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
        solver_type: str = "cholesky",
        lens_light_regularization: float = EPSILON_REG,
        fixed_source_bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        super().__init__("pixelized_image_prob_model")
        if solver_type not in ("nnls", "cholesky"):
            raise ValueError(f"solver_type must be 'nnls' or 'cholesky', got {solver_type}")
        self.solver_type = solver_type
        self.eps_reg = validate_lens_light_regularization(
            lens_light_regularization
        )
        self._fixed_source_bbox = self._validate_fixed_source_bbox(fixed_source_bbox)
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
        self.sim_obj = PixelizedLensSimulator(self.phys_model, sim_config)
        self.unmask = ~jnp.asarray(sim_config.mask, dtype=bool)
        self.data_1d = self.image_data[self.unmask]
        self.noise_1d = self.noise_map[self.unmask]
        self.logdet_C = jnp.sum(jnp.log(self.noise_1d**2))
        source_n = int(self.source_model.n)
        rho = self.source_model.adaptive_reg_rho
        rho_value = rho.value if hasattr(rho, "value") else rho
        rho_dynamic = bool(getattr(rho, "dynamic", False))
        if rho_dynamic or abs(float(rho_value)) >= 1.0e-10:
            raise ValueError(
                "PixelizedImageProbModel no longer supports mass-dependent "
                "seed-ray adaptive regularization. Use "
                "PixelizedImageProbModelOperator with fixed_source_bbox and "
                "fixed_reg_scale or fixed_reg_template derived from an S0 "
                "source template."
            )
        self.reg_type = self.source_model.regularization_type
        self.reg_builder = DenseRegularizationBuilder(
            source_n,
            self.reg_type,
        )

        # Lens-light configuration
        self.n_lens_light = self.sim_obj.n_lens_light
        self.has_lens_light = self.sim_obj.has_lens_light

        self._pos_px, self._pos_py, self._pos_thr, self._pos_minl, self._has_pos_penalty = \
            resolve_position_likelihood_attrs(position_likelihood)
        self._store_fixed_grid_cache()

    @staticmethod
    def _validate_fixed_source_bbox(
        fixed_source_bbox: tuple[float, float, float, float] | None,
    ) -> tuple[Array, Array, Array, Array] | None:
        """Accept a square source bbox or reject it before any solve."""
        if fixed_source_bbox is None:
            return None
        if len(fixed_source_bbox) != 4:
            raise ValueError(
                "fixed_source_bbox must be a 4-tuple "
                "(xmin, xmax, ymin, ymax)."
            )
        xmin_b, xmax_b, ymin_b, ymax_b = (
            jnp.asarray(v, dtype=jnp.float32) for v in fixed_source_bbox
        )
        bbox = (xmin_b, xmax_b, ymin_b, ymax_b)
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
        """Return current dynamic-parameter values (``"flat"`` mode returns a JAX array)."""
        if mode == "flat":
            return jnp.asarray([jnp.asarray(param.value) for param in self.get_dynamic_params()])
        return super().get_values(mode)

    @property
    def source_model(self):
        """Return the single pixelized source configuration."""
        return self.phys_model.source_light[0]

    def infer_source_bbox(self) -> tuple[float, float, float, float]:
        """Return the square source bbox a second model can freeze.

        A model constructed with ``fixed_source_bbox`` reports that grid.
        Otherwise the bbox is inferred from the current seed rays, using the
        simulator's padding and outlier fraction.
        """
        if self._fixed_source_bbox is not None:
            values = self._fixed_source_bbox
        else:
            beta_x_seed, beta_y_seed = self.sim_obj.ray_trace_seed()
            values = self.sim_obj.infer_source_bbox(beta_x_seed, beta_y_seed)
        xmin, xmax, ymin, ymax = values
        return (
            float(np.asarray(xmin)),
            float(np.asarray(xmax)),
            float(np.asarray(ymin)),
            float(np.asarray(ymax)),
        )

    def _inversion_source_bbox(self) -> tuple:
        """Return the source bbox for this evaluation.

        A fixed bbox is used as stored. An inferred bbox is detached when the
        simulator is configured to keep seed-ray bounds out of the gradient.
        """
        if self._fixed_source_bbox is not None:
            return self._fixed_source_bbox
        beta_x_seed, beta_y_seed = self.sim_obj.ray_trace_seed()
        source_bbox = self.sim_obj.infer_source_bbox(beta_x_seed, beta_y_seed)
        if self.sim_obj.detach_bbox:
            source_bbox = tuple(jax.lax.stop_gradient(b) for b in source_bbox)
        return source_bbox

    def _regularization_geometry_is_fixed(self) -> bool:
        """Finite differences are invariant on a fixed grid; a GP kernel is not."""
        if self._fixed_source_bbox is None:
            return False
        if self.reg_type not in GP_REGULARIZATION_TYPES:
            return True
        return _is_fixed(self.source_model.kernel_scale)

    def _lens_mapping_is_fixed(self) -> bool:
        """The joint design matrix is invariant only when mass and lens light are."""
        if self._fixed_source_bbox is None:
            return False
        return _modules_are_fixed(self.phys_model.lens_mass) and _modules_are_fixed(
            self.phys_model.lens_light
        )

    def _store_fixed_grid_cache(self) -> None:
        """Fill invariant regularization and, when possible, the design triple.

        The cache is not invalidated later. A changed fixed value, parameter
        mode, observation, or bbox needs a new evidence model.
        """
        regularization = None
        design = None
        if self._fixed_source_bbox is not None:
            with ck.ActiveContext(self):
                if self._regularization_geometry_is_fixed():
                    regularization = self._build_cached_regularization(self._fixed_source_bbox)
                if self._lens_mapping_is_fixed():
                    design_matrix, _ = self.sim_obj.design_matrix(
                        source_bbox=self._fixed_source_bbox
                    )
                    design = self._normal_terms(design_matrix)
        object.__setattr__(
            self,
            "_prepared",
            _DensePrepared(regularization=regularization, design=design),
        )

    def _bbox_matches_fixed(self, source_bbox) -> bool:
        """Return whether ``source_bbox`` is the stored fixed grid."""
        fixed = self._fixed_source_bbox
        if fixed is None or source_bbox is None:
            return False
        if source_bbox is fixed:
            return True
        try:
            offered = np.array(
                [np.float32(np.asarray(value)) for value in source_bbox],
                dtype=np.float32,
            )
            stored = np.array(
                [np.float32(np.asarray(value)) for value in fixed],
                dtype=np.float32,
            )
        except (TypeError, ValueError):
            return False
        return bool(np.array_equal(offered, stored))

    def _matched_regularization(self, source_bbox):
        """Return the stored regularizer when ``source_bbox`` is the fixed grid."""
        if self._prepared.regularization is None or not self._bbox_matches_fixed(source_bbox):
            return None
        return self._prepared.regularization

    def _build_cached_regularization(self, source_bbox) -> tuple[Array, Array]:
        """Build the matrix and the log determinant the evidence reuses."""
        matrix, covariance_logdet = self._regularization_from_builder(source_bbox)
        if self.reg_type in GP_REGULARIZATION_TYPES:
            return matrix, covariance_logdet
        sign, logdet = jnp.linalg.slogdet(matrix)
        return matrix, jnp.where(sign > 0.0, logdet, -jnp.inf)

    def _normal_terms(self, design_matrix: Array) -> tuple[Array, Array, Array]:
        """Return the design matrix and the data-only normal-equation terms."""
        weighted = design_matrix / self.noise_1d[:, None]
        gram = weighted.T @ weighted
        rhs = weighted.T @ (self.data_1d / self.noise_1d)
        return design_matrix, gram, rhs

    def _design_terms(self, source_bbox) -> tuple[Array, Array, Array]:
        """Return ``M``, the data Gram matrix, and the data right-hand side."""
        if self._prepared.design is not None and self._bbox_matches_fixed(source_bbox):
            return self._prepared.design
        design_matrix, _ = self.sim_obj.design_matrix(source_bbox=source_bbox)
        return self._normal_terms(design_matrix)

    def _solve_source(
        self,
        design_matrix: Array,
        reg_matrix: Array,
        lambda_reg: Array,
        *,
        gram: Array | None = None,
        rhs: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """Solve for MAP source and lens-light pixels.

        For source-only models (Nl=0), solves the standard source inversion.
        For joint models (Nl>0), solves the joint system with block-diagonal
        regularization: block_diag(lambda_reg * R_src, eps * I).

        When ``solver_type='nnls'``, enforces non-negativity on all linear
        parameters (source pixels and lens light amplitudes) using FNNLS.

        Returns (linear_params, chol_factor, curvature).
        linear_params contains [s | A] for joint models, or just s otherwise.
        curvature = M^T C^{-1} M + R_tilde; chol_factor is its Cholesky.
        """
        n_source = self.sim_obj.n_source_pixels
        n_total = design_matrix.shape[1]  # Ns + Nl

        if gram is None or rhs is None:
            _, gram, rhs = self._normal_terms(design_matrix)
        # Add the current regularization to a copy. The stored Gram matrix
        # remains the data term, and the Cholesky factor is not reused.
        curvature = gram.at[:n_source, :n_source].add(lambda_reg * reg_matrix)

        # Add lens-light Tikhonov regularization (eps * I) if present
        if self.has_lens_light:
            lens_light_block = jnp.eye(n_total - n_source, dtype=curvature.dtype) * self.eps_reg
            curvature = curvature.at[n_source:, n_source:].add(lens_light_block)

        chol = jnp.linalg.cholesky(curvature)

        if self.solver_type == "nnls":
            # Solve min ||d||^2_A - 2 b^T d  s.t. d >= 0
            # via augmented NNLS: Z_aug = L^T, x_aug = L^{-1} b
            # where curvature = L @ L^T (Cholesky), so Z_aug^T @ Z_aug = curvature
            # and Z_aug^T @ x_aug = b.
            Z_aug = chol.T  # L^T
            x_aug = jsl.solve_triangular(chol, rhs, lower=True)
            linear_params, _ = fnnls_jax(Z_aug, x_aug)
        else:
            linear_params = jsl.cho_solve((chol, True), rhs)

        return linear_params, chol, curvature

    def _log_evidence(self) -> Array:
        """Evaluate log evidence for current parameter values."""
        log_lambda_reg = jnp.asarray(self.source_model.log_lambda_reg.value)
        lambda_reg = jnp.exp(log_lambda_reg)

        source_bbox = self._inversion_source_bbox()
        design_matrix, gram, rhs = self._design_terms(source_bbox)
        reg_matrix, logdet_cov = self._regularization_matrix(source_bbox)
        logdet_h = self._evidence_logdet(source_bbox, reg_matrix, logdet_cov)
        linear_params, chol, curvature = self._solve_source(
            design_matrix, reg_matrix, lambda_reg, gram=gram, rhs=rhs,
        )

        n_source = self.sim_obj.n_source_pixels
        n_data = self.data_1d.size

        # Split parameters if joint model
        if self.has_lens_light:
            source_pixels = linear_params[:n_source]
            lens_amplitudes = linear_params[n_source:]
        else:
            source_pixels = linear_params
            lens_amplitudes = None

        resid = self.data_1d - design_matrix @ linear_params
        e_d = 0.5 * jnp.sum((resid / self.noise_1d) ** 2)
        e_s = 0.5 * jnp.dot(source_pixels, reg_matrix @ source_pixels)

        # logdet curvature from Cholesky diagonal. This factor includes the
        # current regularization strength and is not cached.
        logdet_a = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

        log_evidence = (
            -e_d
            - lambda_reg * e_s
            - 0.5 * logdet_a
            + 0.5 * n_source * log_lambda_reg
            + 0.5 * logdet_h
            - 0.5 * n_data * jnp.log(2.0 * jnp.pi)
            - 0.5 * self.logdet_C
        )

        # Add lens-light regularization contribution if present
        if self.has_lens_light and lens_amplitudes is not None:
            n_lens = self.n_lens_light
            # Prior quadratic penalty: -0.5 * eps * ||A||^2
            e_lens = 0.5 * self.eps_reg * jnp.sum(lens_amplitudes ** 2)
            # Prior logdet: +0.5 * log|eps * I| = +0.5 * Nl * log(eps)
            log_evidence = log_evidence - e_lens + 0.5 * n_lens * jnp.log(self.eps_reg)

        return jnp.where(jnp.isfinite(log_evidence), log_evidence, -1.0e10)

    def _regularization_from_builder(
        self, source_bbox: tuple, scale: Array | None = None,
    ) -> tuple[Array, Array | None]:
        """Build one regularization matrix without reading or writing the cache."""
        xmin, xmax, ymin, ymax = source_bbox
        if self.reg_type in GP_REGULARIZATION_TYPES:
            kernel_scale = jnp.asarray(self.source_model.kernel_scale.value)
            precision, logdet_cov = self.reg_builder.matrix(
                xmin, xmax, ymin, ymax, kernel_scale=kernel_scale, scale=scale,
            )
            return jnp.asarray(precision, dtype=self.image_data.dtype), logdet_cov
        reg_matrix_raw, _ = self.reg_builder.matrix(xmin, xmax, ymin, ymax, scale=scale)
        return jnp.asarray(reg_matrix_raw, dtype=self.image_data.dtype), None

    def _evidence_logdet(self, source_bbox, reg_matrix: Array, covariance_logdet: Array | None) -> Array:
        """Return the regularization log determinant added to the evidence.

        A cache hit reads the stored determinant and does not call ``slogdet``.
        Finite differences use ``logdet(R)``; a Gaussian process uses ``-log|K|``.
        """
        cached = self._matched_regularization(source_bbox)
        if cached is not None:
            stored_logdet = cached[1]
            if self.reg_type in GP_REGULARIZATION_TYPES:
                return -stored_logdet
            return stored_logdet
        if self.reg_type in GP_REGULARIZATION_TYPES:
            return -covariance_logdet
        sign, logdet = jnp.linalg.slogdet(reg_matrix)
        return jnp.where(sign > 0.0, logdet, -jnp.inf)

    def _regularization_matrix(self, source_bbox: tuple, scale: Array | None = None) -> tuple[Array, Array | None]:
        """Return (reg_matrix, logdet_covariance) for the configured regularization.

        For GP types, ``logdet_covariance`` is ``log|K|``. For finite-difference
        types it is absent. A bbox other than the stored fixed grid is built
        immediately and does not replace the cache.
        """
        if scale is None:
            cached = self._matched_regularization(source_bbox)
            if cached is not None:
                matrix, stored_logdet = cached
                if self.reg_type in GP_REGULARIZATION_TYPES:
                    return matrix, stored_logdet
                return matrix, None
        return self._regularization_from_builder(source_bbox, scale=scale)

    @ck.forward
    def forward_model(self, *, return_source: bool = False, return_components: bool = False):
        """Solve linear params and return the reconstructed model image.

        Parameters
        ----------
        return_source : bool, optional
            If ``True``, return ``(model_image, source_pixels)``.
        return_components : bool, optional
            If ``True`` and lens_light is present, return
            ``(model_image, source_pixels, lens_amplitudes)``.

        Returns
        -------
        Array or tuple
            Model image, optionally with source pixels and lens light.
        """
        source_bbox = self._inversion_source_bbox()
        design_matrix, gram, rhs = self._design_terms(source_bbox)
        reg_matrix, _ = self._regularization_matrix(source_bbox)
        lambda_reg = jnp.exp(jnp.asarray(self.source_model.log_lambda_reg.value))
        linear_params, _, _ = self._solve_source(
            design_matrix, reg_matrix, lambda_reg, gram=gram, rhs=rhs,
        )

        n_source = self.sim_obj.n_source_pixels
        source_pixels = linear_params[:n_source]

        if self.has_lens_light:
            lens_amplitudes = linear_params[n_source:]
            L_cols = design_matrix[:, n_source:]
            source_1d = design_matrix[:, :n_source] @ source_pixels
            lens_1d = L_cols @ lens_amplitudes
            model_1d = source_1d + lens_1d
        else:
            model_1d = design_matrix @ source_pixels
            lens_amplitudes = None

        H, W = self.sim_obj.image_shape
        model_image = jnp.zeros(H * W, dtype=model_1d.dtype)
        model_image = model_image.at[self.sim_obj.flat_indices].set(model_1d)
        model_image = model_image.reshape(H, W)

        if return_components and self.has_lens_light:
            return model_image, source_pixels, lens_amplitudes
        if return_source:
            return model_image, source_pixels
        return model_image

    @ck.forward
    def __call__(self):
        """Return a finite scalar log evidence approximation."""
        log_ev = self._log_evidence()
        if self._has_pos_penalty:
            log_ev = log_ev + self._position_likelihood_penalty_jax()
        return jnp.where(jnp.isfinite(log_ev), log_ev, -1.0e10)

    @functools.partial(jit, static_argnums=(0,))
    def _position_likelihood_penalty_jax(self) -> Array:
        """Compute position-likelihood penalty via shared utility (see ``_position_likelihood``)."""
        return compute_position_penalty_jax(
            self.phys_model, self._pos_px, self._pos_py, self._pos_thr, self._pos_minl,
        )

    def likelihood(self, debug: bool = True) -> float:
        """Return the current log evidence as a Python float."""
        return float(np.asarray(self.__call__()))


__all__ = ["PixelizedImageProbModel"]
