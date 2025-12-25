"""
Probability model for gravitational lensing image fitting.

This module provides the probability model using PhysicalModel
and LensSimulator for computing image likelihoods.
"""

import functools
import caskade as ck
import jax.numpy as jnp
import jax
from jax import jit, Array
import numpy as np
from typing import Optional, Dict, Tuple, Union

from TinyLensGpu.Simulator.lens_simulator import LensSimulator
from TinyLensGpu.Simulator.config import SimulatorConfig
from TinyLensGpu.Models.composite import PhysicalModel


class ImageProbModel(ck.Module):
    """
    Probability model for gravitational lensing images.

    This class computes the likelihood of model parameters given observed data,
    using a forward model (PhysicalModel + LensSimulator).

    Parameters
    ----------
    image_data : array_like
        Observed image data, shape (npix, npix)
    noise_map : array_like
        Noise map (standard deviations), shape (npix, npix)
    psf_kernel : array_like
        Point spread function kernel
    dpix : float
        Pixel scale in arcsec/pixel
    nsub : int
        Subsampling factor for ray-tracing
    phys_model : PhysicalModel
        Physical model containing mass and light components
    use_linear : bool
        Whether to use linear solver for intensity parameters
    mask : array_like, optional
        Boolean mask array (True = masked out)
    solver_type : str, optional
        Linear solver type: 'nnls' or 'normal' (default: 'nnls')
    position_likelihood : dict, optional
        Position likelihood constraint configuration

    Attributes
    ----------
    image_data : jnp.ndarray
        Observed image
    noise_map : jnp.ndarray
        Noise map
    sim_obj : LensSimulator
        lens simulator instance
    use_linear : bool
        Linear solver flag
    unmask : jnp.ndarray
        Boolean mask (True = unmasked pixels)
    position_like_config : dict
        Position likelihood configuration
    """

    def __init__(
        self,
        image_data: np.ndarray,
        noise_map: np.ndarray,
        psf_kernel: np.ndarray,
        dpix: float,
        nsub: int,
        phys_model: PhysicalModel,
        use_linear: bool,
        mask: Optional[np.ndarray] = None,
        solver_type: str = 'nnls',
        position_likelihood: Optional[Dict] = None,
    ) -> None:
        super().__init__("image_prob_model")

        self.image_data = jnp.array(image_data)
        self.noise_map = jnp.array(noise_map)

        # Keep phys_model in the caskade module tree so @ck.forward can inject theta
        self.phys_model = phys_model

        # Create simulator configuration
        sim_config = SimulatorConfig(
            dpix=dpix,
            npix=image_data.shape[0],
            psf_kernel=psf_kernel,
            nsub=nsub,
            mask=mask,
        )

        # Create simulator
        self.sim_obj = LensSimulator(
            phys_model=self.phys_model,
            sim_config=sim_config,
            solver_type=solver_type,
        )

        self.use_linear = use_linear
        self.unmask = jnp.array(~sim_config.mask)
        self.position_like_config = position_likelihood

        # Precompute static flags for JIT-friendly branching
        self._check_nnls = bool(self.use_linear and self.sim_obj.solver_type == "nnls")

    def get_dynamic_params(self):
        """Get dynamic parameters from the underlying physical model."""
        return self.phys_model.dynamic_params

    def forward_model(self) -> Tuple[Array, Optional[Array]]:
        """
        Run forward model to generate simulated image.

        With caskade, parameters are already set in the PhysicalModel,
        so we don't need to pass them explicitly.

        Returns
        -------
        image_model : jnp.ndarray
            Simulated image, shape (npix, npix)
        intensity_list : jnp.ndarray or None
            Intensity values if linear solver used, else None
        """
        return self.sim_obj.simulate(
            use_linear=self.use_linear,
            return_intensity=True,
            image_map=self.image_data if self.use_linear else None,
            noise_map=self.noise_map if self.use_linear else None,
        )

    @ck.forward
    @functools.partial(jit, static_argnums=(0,))
    def __call__(self, theta: Optional[jnp.ndarray] = None):
        """Vectorization-friendly log-likelihood evaluation.

        This function is designed to be used with `make_likelihood(..., vectorized=True)`
        where `theta` is vmapped over.
        """
        image_model, intensity_list = self.forward_model()

        log_like = self._likelihood_helper(
            image_model=image_model,
            image_data=self.image_data,
            noise_map=self.noise_map,
            unmask=self.unmask,
        )

        # If NNLS is configured and returns any negative intensity, treat as invalid
        if self._check_nnls:
            ok = jnp.all(intensity_list >= 0.0)
            log_like = jnp.where(ok, log_like, -jnp.inf)

        # Add position likelihood penalty (JIT/vmap safe)
        if self.position_like_config is not None:
            log_like = log_like + self._position_likelihood_penalty_jax()

        # Guard against NaN/Inf
        log_like = jnp.where(jnp.isfinite(log_like), log_like, -jnp.inf)
        return log_like

    @functools.partial(jit, static_argnums=(0,))
    def _likelihood_helper(
        self,
        image_model: Array,
        image_data: Array,
        noise_map: Array,
        unmask: Array,
    ) -> float:
        """
        Compute chi-square likelihood.

        Parameters
        ----------
        image_model : jnp.ndarray
            Model image, shape (npix, npix)
        image_data : jnp.ndarray
            Observed image, shape (npix, npix)
        noise_map : jnp.ndarray
            Noise map, shape (npix, npix)
        unmask : jnp.ndarray
            Boolean mask, shape (npix, npix)

        Returns
        -------
        log_like : float
            Log-likelihood value
        """
        chi2_image = (image_model - image_data) ** 2 / noise_map ** 2
        chi2_image = chi2_image * unmask
        return -0.5 * jnp.sum(chi2_image)

    def likelihood(self) -> float:
        """
        Compute log-likelihood of current model parameters.

        With caskade, parameters are already set in the PhysicalModel via
        caskade's parameter management system, so we don't pass them here.

        Parameters
        ----------
        debug : bool, optional
            Whether to check for NaN/Inf values (default: True)

        Returns
        -------
        log_like : float
            Log-likelihood value
        """
        like = float(np.asarray(self.__call__()))
        return like

    def _position_likelihood_penalty(self) -> float:
        """
        Compute position likelihood penalty.

        This enforces that multiple lensed images of the same source
        should map to the same location in the source plane.

        Returns
        -------
        penalty : float
            Penalty value
        """
        cfg = self.position_like_config
        if cfg is None:
            return 0.0

        positions = cfg.get('positions', [])
        if positions is None or len(positions) < 2:
            return 0.0

        threshold = float(cfg.get('threshold_arcsec', cfg.get('position_threshold', 0.0)))
        min_like = float(cfg.get('min_log_like', cfg.get('min_position_likelihood', 0.0)))

        # NOTE: this method is kept for backward compatibility (non-jitted usage)
        return float(np.asarray(self._position_likelihood_penalty_jax()))

    def _position_likelihood_penalty_jax(self) -> Array:
        """JAX-compatible position likelihood penalty (JIT/vmap safe)."""
        cfg = self.position_like_config
        if cfg is None:
            return jnp.array(0.0, dtype=jnp.float32)

        positions = cfg.get('positions', [])
        if positions is None or len(positions) < 2:
            return jnp.array(0.0, dtype=jnp.float32)

        threshold = float(cfg.get('threshold_arcsec', cfg.get('position_threshold', 0.0)))
        min_like = float(cfg.get('min_log_like', cfg.get('min_position_likelihood', 0.0)))

        px = jnp.array([p[0] for p in positions], dtype=jnp.float32)
        py = jnp.array([p[1] for p in positions], dtype=jnp.float32)

        beta_x, beta_y = self.phys_model.deflection(px, py)

        dx = beta_x[:, None] - beta_x[None, :]
        dy = beta_y[:, None] - beta_y[None, :]
        dist = jnp.sqrt(dx * dx + dy * dy)
        max_sep = jnp.max(dist)

        thr = jnp.array(threshold, dtype=jnp.float32)
        minl = jnp.array(min_like, dtype=jnp.float32)

        exceed = jnp.maximum(0.0, max_sep - thr)
        ratio = jnp.where(thr > 0.0, exceed / thr, 0.0)
        pen_continuous = minl * (1.0 - jnp.exp(-ratio))

        pen_clipped = jnp.clip(pen_continuous, a_min=minl, a_max=jnp.array(0.0, dtype=jnp.float32))
        pen = jnp.where(jnp.logical_or(thr <= 0.0, exceed <= 0.0), 0.0, pen_clipped)
        return pen

    def __repr__(self) -> str:
        return (f"ImageProbModel("
                f"npix={self.image_data.shape[0]}, "
                f"use_linear={self.use_linear}, "
                f"solver={self.sim_obj.solver_type})")
