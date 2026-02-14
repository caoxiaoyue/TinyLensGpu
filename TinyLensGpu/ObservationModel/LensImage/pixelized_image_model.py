"""Probability model for pixelized source gravitational lensing image fitting."""

from __future__ import annotations

import functools
from typing import Dict, Optional, Union

import caskade as ck
import jax.numpy as jnp
import numpy as np
from jax import Array, jit

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import IrregularGridConfig, RectangularGridConfig
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel


class PixelizedImageProbModel(ck.Module):
    """Bayesian evidence model for pixelized source reconstruction."""

    def __init__(
        self,
        image_data: Union[np.ndarray, Array],
        noise_map: Union[np.ndarray, Array],
        sim_config: SimulatorConfig,
        phys_model: PhysicalModel,
        lensed_source_image: Optional[Union[np.ndarray, Array]] = None,
        position_likelihood: Optional[Dict] = None,
    ) -> None:
        super().__init__("pixelized_image_prob_model")

        self.image_data = jnp.asarray(image_data)
        self.noise_map = jnp.asarray(noise_map)
        self.sim_config = sim_config
        self.phys_model = phys_model
        extracted_pix_src_model = self.phys_model.get_pixelized_source_model()
        object.__setattr__(self, "pix_src_model", extracted_pix_src_model)

        self.npix = int(self.sim_config.npix)
        expected_shape = (self.npix, self.npix)

        if self.image_data.shape != expected_shape:
            raise ValueError(
                f"image_data shape mismatch: expected {expected_shape}, got {self.image_data.shape}."
            )
        if self.noise_map.shape != expected_shape:
            raise ValueError(
                f"noise_map shape mismatch: expected {expected_shape}, got {self.noise_map.shape}."
            )

        self.mask = jnp.asarray(self.sim_config.mask, dtype=bool)
        if self.mask.shape != expected_shape:
            raise ValueError(
                f"sim_config.mask shape mismatch: expected {expected_shape}, got {self.mask.shape}."
            )

        self.lensed_source_image = lensed_source_image
        if self.lensed_source_image is not None and np.asarray(self.lensed_source_image).shape != expected_shape:
            raise ValueError(
                "lensed_source_image shape mismatch: "
                f"expected {expected_shape}, got {np.asarray(self.lensed_source_image).shape}."
            )

        self.unmask = ~self.mask
        self._data_vector = self.image_data[self.unmask]
        self._noise_variance = self.noise_map[self.unmask] ** 2

        self.position_like_config = position_likelihood
        self._init_position_likelihood(self.position_like_config)

        self.simulator = PixelizedLensSimulator(
            phys_model=self.phys_model,
            sim_config=self.sim_config,
            lensed_source_image=(
                None if self.lensed_source_image is None else np.asarray(self.lensed_source_image)
            ),
        )

    def _init_position_likelihood(self, config: Optional[Dict]) -> None:
        self._pos_px = None
        self._pos_py = None
        self._pos_thr = jnp.array(0.0, dtype=jnp.float32)
        self._pos_minl = jnp.array(0.0, dtype=jnp.float32)
        self._has_pos_penalty = False

        if not config:
            return

        positions = config.get("positions", [])
        if positions is None or len(positions) < 2:
            return

        self._pos_px = jnp.array([p[0] for p in positions], dtype=jnp.float32)
        self._pos_py = jnp.array([p[1] for p in positions], dtype=jnp.float32)

        def get_val(keys, default):
            for key in keys:
                if key in config:
                    return config[key]
            return default

        threshold = get_val(["threshold_arcsec", "position_threshold"], 0.0)
        min_log_like = get_val(["min_log_like", "min_position_likelihood"], 0.0)

        self._pos_thr = jnp.array(float(threshold), dtype=jnp.float32)
        self._pos_minl = jnp.array(float(min_log_like), dtype=jnp.float32)
        self._has_pos_penalty = True

    @ck.forward
    def _build_inverter(self):
        model = self.pix_src_model

        inverter = self.simulator.build_inverter(
            data_vector=self._data_vector,
            noise_variance=self._noise_variance,
            reg_scale=model.reg_scale.value,
            reg_coefficient=model.reg_coefficient.value,
        )
        return inverter

    @ck.forward
    def __call__(self):
        inverter = self._build_inverter()
        log_ev = inverter.log_evidence()
        if self._has_pos_penalty:
            log_ev = log_ev + self._position_likelihood_penalty_jax()
        return jnp.where(jnp.isfinite(log_ev), log_ev, -jnp.inf)

    def log_evidence(self) -> float:
        return float(np.asarray(self.__call__()))

    @functools.partial(jit, static_argnums=(0,))
    def _position_likelihood_penalty_jax(self) -> Array:
        beta_x, beta_y = self.phys_model.deflection(self._pos_px, self._pos_py)

        dx = beta_x[:, None] - beta_x[None, :]
        dy = beta_y[:, None] - beta_y[None, :]
        dist = jnp.sqrt(dx * dx + dy * dy)
        max_sep = jnp.max(dist)

        exceed = jnp.maximum(0.0, max_sep - self._pos_thr)
        ratio = jnp.where(self._pos_thr > 0.0, exceed / self._pos_thr, 0.0)
        pen_continuous = self._pos_minl * (1.0 - jnp.exp(-ratio))

        pen_clipped = jnp.clip(pen_continuous, min=self._pos_minl, max=0.0)
        return pen_clipped

    def __repr__(self) -> str:
        if isinstance(self.pix_src_model.grid, IrregularGridConfig):
            n_source_points = int(self.pix_src_model.grid.n_source_points)
        elif isinstance(self.pix_src_model.grid, RectangularGridConfig):
            n_source_points = int(self.pix_src_model.grid.nx * self.pix_src_model.grid.ny)
        else:
            n_source_points = 0

        return (
            "PixelizedImageProbModel("
            f"npix={self.npix}, "
            f"n_source_points={n_source_points}, "
            f"source_grid_type='{self.pix_src_model.source_grid_type}')"
        )
