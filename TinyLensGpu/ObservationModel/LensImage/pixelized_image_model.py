"""Bayesian evidence model for pixelized source inversions."""

# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Dict, Optional, Union

import caskade as ck
import functools
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax import Array, jit

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.inversion.regularization import (
    DenseRegularizationBuilder,
    GP_REGULARIZATION_TYPES,
)


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
    """

    def __init__(
        self,
        image_data: Union[np.ndarray, Array],
        noise_map: Union[np.ndarray, Array],
        psf_kernel: Union[np.ndarray, Array],
        dpix: float,
        phys_model: PhysicalModel,
        mask: Union[np.ndarray, Array, None] = None,
        nsub: int = 1,
        position_likelihood: Optional[Dict] = None,
    ) -> None:
        super().__init__("pixelized_image_prob_model")
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
        # Precompute logdet(H(half_size)) = logdet_H_unit + scaling * log(half_size)
        # for finite-difference regularization where H(h) = h^{-k} * H_unit.
        # logdet H(h) = logdet H_unit + (-k * n_s) * log(h)
        # zero-order: k=0; first-order: k=2 (H scales as h^{-2}); second-order: k=4.
        n_s = source_nx * source_ny
        _exponent = {"zero-order": 0, "first-order": -2, "second-order": -4}.get(self.reg_type, None)
        self._logdet_H_scaling = _exponent * n_s if _exponent is not None else None
        if self._logdet_H_scaling is not None:
            H_unit_raw, _ = self.reg_builder.matrix(1.0)
            H_unit = jnp.asarray(H_unit_raw, dtype=self.image_data.dtype)
            sign_h, logdet_h_unit = jnp.linalg.slogdet(H_unit)
            self._logdet_H_unit = jnp.where(sign_h > 0.0, logdet_h_unit, -jnp.inf)
        else:
            self._logdet_H_unit = None

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
        """Solve normal equations via Cholesky for MAP source pixels.

        Returns (source_pixels, chol_factor, curvature).
        curvature = F^T C^{-1} F + λH; chol_factor is its lower-triangular Cholesky.
        """
        weighted_design = design_matrix / self.noise_1d[:, None]
        curvature = weighted_design.T @ weighted_design + lambda_reg * reg_matrix
        rhs = weighted_design.T @ (self.data_1d / self.noise_1d)
        chol = jnp.linalg.cholesky(curvature)
        source_pixels = jsl.cho_solve((chol, True), rhs)
        return source_pixels, chol, curvature

    def _log_evidence(self) -> Array:
        """Evaluate log evidence for current parameter values."""
        lambda_reg = jnp.asarray(self.source_model.lambda_reg.value)
        design_matrix, source_half_size = self.sim_obj.design_matrix()
        reg_matrix, logdet_cov = self._regularization_matrix(source_half_size)
        source_pixels, chol, curvature = self._solve_source(design_matrix, reg_matrix, lambda_reg)

        resid = self.data_1d - design_matrix @ source_pixels
        e_d = 0.5 * jnp.sum((resid / self.noise_1d) ** 2)
        e_s = 0.5 * jnp.dot(source_pixels, reg_matrix @ source_pixels)

        # logdet curvature from Cholesky diagonal
        logdet_a = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

        # logdet reg matrix: analytical for finite-difference types;
        # for GP types use logdet_cov from _gp_matrix (no extra slogdet needed).
        if self._logdet_H_scaling is not None:
            logdet_h = self._logdet_H_unit + self._logdet_H_scaling * jnp.log(source_half_size)
        else:
            # GP types always return a non-None logdet_cov from _regularization_matrix.
            logdet_h = -logdet_cov

        n_source = self.sim_obj.n_source_pixels
        n_data = self.data_1d.size
        log_evidence = (
            -e_d
            - lambda_reg * e_s
            - 0.5 * logdet_a
            + 0.5 * n_source * jnp.log(lambda_reg)
            + 0.5 * logdet_h
            - 0.5 * n_data * jnp.log(2.0 * jnp.pi)
            - 0.5 * self.logdet_C
        )
        return jnp.where(jnp.isfinite(log_evidence), log_evidence, -1.0e10)

    def _regularization_matrix(self, source_half_size: Array | float) -> tuple[Array, Array | None]:
        """Return (reg_matrix, logdet_covariance) for the configured regularization.

        For GP types, ``logdet_covariance`` is ``log|K|`` extracted from the Cholesky
        factorization inside ``_gp_matrix`` — no extra ``slogdet`` needed.
        For finite-difference types, ``logdet_covariance`` is ``None`` (the caller uses
        the precomputed ``_logdet_H_unit`` / ``_logdet_H_scaling`` path instead).
        """
        if self.reg_type in GP_REGULARIZATION_TYPES:
            kernel_scale = jnp.asarray(self.source_model.kernel_scale.value)
            precision, logdet_cov = self.reg_builder.matrix(
                source_half_size, kernel_scale=kernel_scale
            )
            return jnp.asarray(precision, dtype=self.image_data.dtype), logdet_cov
        reg_matrix_raw, _ = self.reg_builder.matrix(source_half_size)
        reg_matrix = jnp.asarray(
            reg_matrix_raw,
            dtype=self.image_data.dtype,
        )
        return reg_matrix, None

    @ck.forward
    def forward_model(self, *, return_source: bool = False):
        """Solve source pixels and return the reconstructed model image.

        Parameters
        ----------
        return_source : bool, optional
            If ``True``, return ``(model_image, source_pixels)``.

        Returns
        -------
        Array or tuple[Array, Array]
            Model image, optionally with solved source pixels.
        """
        design_matrix, source_half_size = self.sim_obj.design_matrix()
        reg_matrix, _ = self._regularization_matrix(source_half_size)
        lambda_reg = jnp.asarray(self.source_model.lambda_reg.value)
        source_pixels, _, _ = self._solve_source(design_matrix, reg_matrix, lambda_reg)

        model_1d = design_matrix @ source_pixels
        H, W = self.sim_obj.image_shape
        model_image = jnp.zeros(H * W, dtype=model_1d.dtype)
        model_image = model_image.at[self.sim_obj.flat_indices].set(model_1d)
        model_image = model_image.reshape(H, W)

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
