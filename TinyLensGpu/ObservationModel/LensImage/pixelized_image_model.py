"""Bayesian evidence model for pixelized source inversions."""

# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Dict, Optional, Union

import caskade as ck
import functools
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.scipy.signal as jsp_signal
import numpy as np
from jax import Array, jit

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator, EPSILON_REG
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.inversion.regularization import (
    DenseRegularizationBuilder,
    GP_REGULARIZATION_TYPES,
)
from TinyLensGpu.utils.lensing.mapping import (
    build_source_grid,
    lens_mapping_operator_bilinear_rectangular_from,
)
from TinyLensGpu.utils.linear_solver import fnnls_jax


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
    ) -> None:
        super().__init__("pixelized_image_prob_model")
        if solver_type not in ("nnls", "cholesky"):
            raise ValueError(f"solver_type must be 'nnls' or 'cholesky', got {solver_type}")
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
        )
        self.sim_obj = PixelizedLensSimulator(self.phys_model, sim_config)
        self.unmask = ~jnp.asarray(sim_config.mask, dtype=bool)
        self.data_1d = self.image_data[self.unmask]
        self.noise_1d = self.noise_map[self.unmask]
        self.logdet_C = jnp.sum(jnp.log(self.noise_1d**2))
        source_nx = int(self.source_model.nx)
        source_ny = int(self.source_model.ny)
        self.reg_type = self.source_model.regularization_type
        self.reg_builder = DenseRegularizationBuilder(
            source_nx,
            source_ny,
            self.reg_type,
        )

        # Lens-light configuration
        self.n_lens_light = self.sim_obj.n_lens_light
        self.has_lens_light = self.sim_obj.has_lens_light
        self.eps_reg = EPSILON_REG

        self._init_position_likelihood(position_likelihood)

    def _init_position_likelihood(self, config: Optional[Dict]) -> None:
        self._pos_px = None
        self._pos_py = None
        self._pos_thr = jnp.array(0.0, dtype=jnp.float32)
        self._pos_minl = jnp.array(0.0, dtype=jnp.float32)
        self._has_pos_penalty = False

        if config is not None:
            positions = config.get('positions', [])
            if positions is not None and len(positions) >= 2:
                self._pos_px = jnp.array([p[0] for p in positions], dtype=jnp.float32)
                self._pos_py = jnp.array([p[1] for p in positions], dtype=jnp.float32)
                self._pos_thr = jnp.array(
                    float(config.get('threshold_arcsec', config.get('position_threshold', 0.0))),
                    dtype=jnp.float32,
                )
                self._pos_minl = jnp.array(
                    float(config.get('min_log_like', config.get('min_position_likelihood', 0.0))),
                    dtype=jnp.float32,
                )
                self._has_pos_penalty = True

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

    def _solve_source(
        self, design_matrix: Array, reg_matrix: Array, lambda_reg: Array
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

        weighted_design = design_matrix / self.noise_1d[:, None]
        curvature = weighted_design.T @ weighted_design

        # Add source regularization
        curvature = curvature.at[:n_source, :n_source].add(lambda_reg * reg_matrix)

        # Add lens-light Tikhonov regularization (eps * I) if present
        if self.has_lens_light:
            lens_light_block = jnp.eye(n_total - n_source, dtype=curvature.dtype) * self.eps_reg
            curvature = curvature.at[n_source:, n_source:].add(lens_light_block)

        rhs = weighted_design.T @ (self.data_1d / self.noise_1d)
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

        # Trace seed pixels once; reuse for bbox + adaptive scale map
        beta_x_seed, beta_y_seed = self.sim_obj.ray_trace_seed()
        source_bbox = self.sim_obj.infer_source_bbox(beta_x_seed, beta_y_seed)
        if self.sim_obj.detach_bbox:
            source_bbox = tuple(jax.lax.stop_gradient(b) for b in source_bbox)
        xmin, xmax, ymin, ymax = source_bbox
        scale = self._compute_reg_scale_from_betas(
            beta_x_seed, beta_y_seed, xmin, xmax, ymin, ymax,
        )

        design_matrix, _ = self.sim_obj.design_matrix(source_bbox=source_bbox)
        reg_matrix, logdet_cov = self._regularization_matrix(source_bbox, scale=scale)
        linear_params, chol, curvature = self._solve_source(design_matrix, reg_matrix, lambda_reg)

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

        # logdet curvature from Cholesky diagonal
        logdet_a = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

        # logdet reg matrix: slogdet for finite-difference types;
        # for GP types use logdet_cov from Cholesky (more stable).
        if self.reg_type in GP_REGULARIZATION_TYPES:
            assert logdet_cov is not None
            logdet_h = -logdet_cov
        else:
            sign_h, logdet_h = jnp.linalg.slogdet(reg_matrix)
            logdet_h = jnp.where(sign_h > 0.0, logdet_h, -jnp.inf)

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

    def _regularization_matrix(self, source_bbox: tuple, scale: Array | None = None) -> tuple[Array, Array | None]:
        """Return (reg_matrix, logdet_covariance) for the configured regularization.

        For GP types, ``logdet_covariance`` is ``log|K|`` extracted from the Cholesky
        factorization inside ``_gp_matrix`` — no extra ``slogdet`` needed.
        For finite-difference types, ``logdet_covariance`` is ``None`` (the caller
        computes ``logdet`` via ``slogdet`` on the returned matrix).
        """
        xmin, xmax, ymin, ymax = source_bbox
        if self.reg_type in GP_REGULARIZATION_TYPES:
            kernel_scale = jnp.asarray(self.source_model.kernel_scale.value)
            precision, logdet_cov = self.reg_builder.matrix(
                xmin, xmax, ymin, ymax, kernel_scale=kernel_scale, scale=scale,
            )
            return jnp.asarray(precision, dtype=self.image_data.dtype), logdet_cov
        reg_matrix_raw, _ = self.reg_builder.matrix(xmin, xmax, ymin, ymax, scale=scale)
        reg_matrix = jnp.asarray(
            reg_matrix_raw,
            dtype=self.image_data.dtype,
        )
        return reg_matrix, None

    # ------------------------------------------------------------------
    # Adaptive regularization scale map
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_scale_map(q_1d: Array, nx: int, ny: int) -> Array:
        """Gaussian-smooth the scale map on the 2-D source grid.

        Uses a 5x5 separable Gaussian kernel with sigma = 1 source pixel.
        Assumes row-major flat layout ``(x + y * nx)``.
        """
        sigma = 1.0
        ksize = 5
        x_k = jnp.arange(ksize, dtype=jnp.float32) - (ksize - 1) / 2
        kernel_1d = jnp.exp(-0.5 * (x_k / sigma) ** 2)
        kernel_1d = kernel_1d / jnp.sum(kernel_1d)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]

        # Row-major flat -> (ny, nx)
        q_2d = q_1d.reshape(ny, nx)
        q_smooth = jsp_signal.convolve2d(q_2d, kernel_2d, mode='same')
        return q_smooth.ravel()  # back to (Ns,)

    def _compute_reg_scale_from_betas(
        self,
        beta_x_seed: Array,
        beta_y_seed: Array,
        xmin, xmax, ymin, ymax,
    ) -> Array | None:
        """Compute per-source-pixel regularization scale factors.

        Returns ``None`` when ``adaptive_reg_alpha == 0`` (fast path — uniform
        regularisation).  Otherwise returns a ``(Ns,)`` array of scale factors
        in ``[floor, 1.0]``.
        """
        alpha = self.source_model.adaptive_reg_alpha
        if alpha == 0.0:
            return None

        nx = self.sim_obj.source_nx
        ny = self.sim_obj.source_ny
        n_source = nx * ny
        floor = jnp.asarray(self.source_model.adaptive_reg_floor, dtype=jnp.float32)
        alpha = jnp.asarray(alpha, dtype=jnp.float32)

        # 1. Brightness at seed mask active pixels
        brightness = jnp.maximum(
            jnp.ravel(self.image_data)[self.sim_obj.seed_flat_indices], 0.0,
        )

        # 2. Bilinear weights mapping seed betas → source grid
        data_mesh = jnp.stack(
            [jnp.ravel(beta_x_seed), jnp.ravel(beta_y_seed)], axis=1,
        )
        weights, indices, valid = lens_mapping_operator_bilinear_rectangular_from(
            data_mesh, xmin, xmax, ymin, ymax, nx, ny,
        )
        # weights: (N_seed, 4), indices: (N_seed, 4), valid: (N_seed,)

        # 3. Brightness-weighted histogram via segment_sum
        # w_bright shape is (N_seed, 4) since each seed pixel splits its brightness 
        # across 4 adjacent source pixels via bilinear interpolation.
        w_bright = weights * (brightness[:, None] * valid[:, None].astype(weights.dtype))
        
        # q Accumulate all scattered brightness fractions into their corresponding 
        # source pixels to build a rough initial source brightness map.
        q = jax.ops.segment_sum(
            w_bright.ravel(), indices.ravel(),
            num_segments=n_source,
        )  # (Ns,)

        # 4. Normalize by mean positive count
        q_pos = jnp.where(q > 0, q, 0.0)
        q_sum = jnp.sum(q_pos)
        q_count = jnp.maximum(jnp.sum(q > 0), 1.0)
        q_mean = q_sum / q_count
        q_norm = q / jnp.maximum(q_mean, 1e-10)

        # 5. Smooth on source grid
        q_smooth = self._smooth_scale_map(q_norm, nx, ny)

        # 6. Compute scale
        scale = 1.0 / (1.0 + alpha * q_smooth)
        scale = jnp.maximum(scale, floor)
        return scale

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
        beta_x_seed, beta_y_seed = self.sim_obj.ray_trace_seed()
        source_bbox = self.sim_obj.infer_source_bbox(beta_x_seed, beta_y_seed)
        if self.sim_obj.detach_bbox:
            source_bbox = tuple(jax.lax.stop_gradient(b) for b in source_bbox)
        xmin, xmax, ymin, ymax = source_bbox
        scale = self._compute_reg_scale_from_betas(
            beta_x_seed, beta_y_seed, xmin, xmax, ymin, ymax,
        )
        design_matrix, _ = self.sim_obj.design_matrix(source_bbox=source_bbox)
        reg_matrix, _ = self._regularization_matrix(source_bbox, scale=scale)
        lambda_reg = jnp.exp(jnp.asarray(self.source_model.log_lambda_reg.value))
        linear_params, _, _ = self._solve_source(design_matrix, reg_matrix, lambda_reg)

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
        return log_ev

    @functools.partial(jit, static_argnums=(0,))
    def _position_likelihood_penalty_jax(self) -> Array:
        r"""Penalize image positions that don't map to the same source position.

        $Penalty = min\_log\_like \cdot (1 - \exp(-ratio))$
        where $ratio = \max(0, \max(separation) - threshold) / threshold$.
        """
        beta_x, beta_y = self.phys_model.deflection(self._pos_px, self._pos_py)

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


__all__ = ["PixelizedImageProbModel"]
