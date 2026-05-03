"""Bayesian evidence model for pixelized source inversions."""

# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Union

import caskade as ck
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax import Array

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.inversion.regularization import DenseRegularizationBuilder


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
        self.kernel_type = self.source_model.kernel_type
        self.reg_builder = DenseRegularizationBuilder(
            source_nx,
            source_ny,
            self.reg_type,
            kernel_type=self.kernel_type,
        )
        # Precompute logdet(H(half_size)) = logdet_H_unit + scaling * log(half_size)
        # for finite-difference regularization where H(h) = h^{-k} * H_unit.
        # logdet H(h) = logdet H_unit + (-k * n_s) * log(h)
        # zero-order: k=0; first-order: k=2 (H scales as h^{-2}); second-order: k=4.
        n_s = source_nx * source_ny
        _exponent = {"zero-order": 0, "first-order": -2, "second-order": -4}.get(self.reg_type, None)
        self._logdet_H_scaling = _exponent * n_s if _exponent is not None else None
        if self._logdet_H_scaling is not None:
            H_unit = jnp.asarray(self.reg_builder.matrix(1.0), dtype=self.image_data.dtype)
            sign_h, logdet_h_unit = jnp.linalg.slogdet(H_unit)
            self._logdet_H_unit = jnp.where(sign_h > 0.0, logdet_h_unit, -jnp.inf)
        else:
            self._logdet_H_unit = None

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
        reg_matrix = self._regularization_matrix(source_half_size)
        source_pixels, chol, curvature = self._solve_source(design_matrix, reg_matrix, lambda_reg)

        resid = self.data_1d - design_matrix @ source_pixels
        e_d = 0.5 * jnp.sum((resid / self.noise_1d) ** 2)
        e_s = 0.5 * jnp.dot(source_pixels, reg_matrix @ source_pixels)

        # logdet curvature from Cholesky diagonal
        logdet_a = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

        # logdet reg matrix: analytical for finite-difference types, numeric otherwise
        if self._logdet_H_scaling is not None:
            logdet_h = self._logdet_H_unit + self._logdet_H_scaling * jnp.log(source_half_size)
        else:
            sign_h, logdet_h_raw = jnp.linalg.slogdet(reg_matrix)
            logdet_h = jnp.where(sign_h > 0.0, logdet_h_raw, -jnp.inf)

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

    def _regularization_matrix(self, source_half_size: Array | float) -> Array:
        """Return the configured dense source regularization matrix."""
        kernel_scale = None
        if self.reg_type in {"gp", "exponential", "gaussian"}:
            kernel_scale = jnp.asarray(self.source_model.kernel_scale.value)
        return jnp.asarray(
            self.reg_builder.matrix(source_half_size, kernel_scale=kernel_scale),
            dtype=self.image_data.dtype,
        )

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
        reg_matrix = self._regularization_matrix(source_half_size)
        lambda_reg = jnp.asarray(self.source_model.lambda_reg.value)
        source_pixels, _, _ = self._solve_source(design_matrix, reg_matrix, lambda_reg)

        model_1d = design_matrix @ source_pixels
        model_image = jnp.zeros(self.sim_obj.image_shape, dtype=model_1d.dtype)
        model_image = model_image.at[self.sim_obj.active_mask].set(model_1d)

        if return_source:
            return model_image, source_pixels
        return model_image

    @ck.forward
    def __call__(self):
        """Return a finite scalar log evidence approximation."""
        return self._log_evidence()

    def likelihood(self, debug: bool = True) -> float:
        """Return the current log evidence as a Python float."""
        return float(np.asarray(self.__call__()))


__all__ = ["PixelizedImageProbModel"]
