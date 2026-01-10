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
from typing import Optional, Dict, Tuple, Union, Sequence

from TinyLensGpu.ForwardModel.LensImage.lens_forward_model import LensSimulator
from TinyLensGpu.ForwardModel.LensImage.config import SimulatorConfig
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel


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

        # Precompute Gaussian log-likelihood constant term: -0.5 * sum(ln(2*pi*sigma^2))
        # only for unmasked pixels
        log_sigma_sq = 2 * jnp.log(self.noise_map)
        log_2pi = jnp.log(2 * jnp.pi)
        self.log_like_const = -0.5 * jnp.sum((log_sigma_sq + log_2pi) * self.unmask)

        # Initialize position likelihood for JIT
        self._init_position_likelihood(self.position_like_config)

        # Precompute static flags for JIT-friendly branching
        self._check_nnls = bool(self.use_linear and self.sim_obj.solver_type == "nnls")

    def _init_position_likelihood(self, config: Optional[Dict]) -> None:
        """Initialize position likelihood parameters for JIT optimization."""
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
                    dtype=jnp.float32
                )
                self._pos_minl = jnp.array(
                    float(config.get('min_log_like', config.get('min_position_likelihood', 0.0))),
                    dtype=jnp.float32
                )
                self._has_pos_penalty = True

    def get_dynamic_params(self):
        """Get dynamic parameters from the underlying physical model."""
        return self.phys_model.dynamic_params

    @ck.forward
    def forward_model(
        self,
        *,
        use_linear: Optional[bool] = None,
        return_intensity: bool = True,
        ret_each_plane: bool = False,
        image_map: Optional[np.ndarray] = None,
        noise_map: Optional[np.ndarray] = None,
        xgrid_sub: Optional[np.ndarray] = None,
        ygrid_sub: Optional[np.ndarray] = None,
        psf_kernel: Optional[np.ndarray] = None,
    ) -> Union[
        Array,
        Tuple[Array, Optional[Array]],
        Tuple[Array, Array],
        Tuple[Array, Array, Optional[Array]],
    ]:
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
        linear_flag = self.use_linear if use_linear is None else use_linear

        sim_kwargs = dict(
            use_linear=linear_flag,
            return_intensity=return_intensity,
            ret_each_plane=ret_each_plane,
            xgrid_sub=xgrid_sub,
            ygrid_sub=ygrid_sub,
            psf_kernel=psf_kernel,
        )

        if linear_flag:
            sim_kwargs["image_map"] = image_map if image_map is not None else self.image_data
            sim_kwargs["noise_map"] = noise_map if noise_map is not None else self.noise_map

        return self.sim_obj.simulate(**sim_kwargs)  # theta injected by caskade into phys_model

    @ck.forward
    @functools.partial(jit, static_argnums=(0,))
    def __call__(self):
        """Vectorization-friendly log-likelihood evaluation.

        This function is designed to be used with `make_likelihood(..., vectorized=True)`
        where `theta` is vmapped over.
        """
        image_model, intensity_list = self.forward_model()

        chi2_image = (image_model - self.image_data) ** 2 / self.noise_map ** 2
        chi2_image = chi2_image * self.unmask
        log_like = -0.5 * jnp.sum(chi2_image) + self.log_like_const

        # If NNLS is configured and returns any negative intensity, treat as invalid
        if self._check_nnls:
            ok = jnp.all(intensity_list >= 0.0)
            log_like = jnp.where(ok, log_like, -jnp.inf)

        # Add position likelihood penalty (JIT/vmap safe)
        if self._has_pos_penalty:
            log_like = log_like + self._position_likelihood_penalty_jax()

        # Guard against NaN/Inf
        log_like = jnp.where(jnp.isfinite(log_like), log_like, -jnp.inf)
        return log_like

    def likelihood(self, debug: bool = True) -> float:
        """
        Compute log-likelihood of current model parameters.

        With caskade, parameters are already set in the PhysicalModel via
        caskade's parameter management system, so we don't pass them here.

        Parameters
        ----------
        debug : bool, optional
            Whether to check for NaN/Inf values (default: True).
            Note: In the current JAX implementation, NaN/Inf checks are always performed.

        Returns
        -------
        log_like : float
            Log-likelihood value
        """
        like = float(np.asarray(self.__call__()))
        return like


    @functools.partial(jit, static_argnums=(0,))
    def _position_likelihood_penalty_jax(self) -> Array:
        """JAX-compatible position likelihood penalty (JIT/vmap safe)."""
        beta_x, beta_y = self.phys_model.deflection(self._pos_px, self._pos_py)

        dx = beta_x[:, None] - beta_x[None, :]
        dy = beta_y[:, None] - beta_y[None, :]
        dist = jnp.sqrt(dx * dx + dy * dy)
        max_sep = jnp.max(dist)

        exceed = jnp.maximum(0.0, max_sep - self._pos_thr)
        ratio = jnp.where(self._pos_thr > 0.0, exceed / self._pos_thr, 0.0)
        pen_continuous = self._pos_minl * (1.0 - jnp.exp(-ratio))

        pen_clipped = jnp.clip(pen_continuous, a_min=self._pos_minl, a_max=0.0)
        pen = jnp.where(jnp.logical_or(self._pos_thr <= 0.0, exceed <= 0.0), 0.0, pen_clipped)
        return pen

    def __repr__(self) -> str:
        return (f"ImageProbModel("
                f"npix={self.image_data.shape[0]}, "
                f"use_linear={self.use_linear}, "
                f"solver={self.sim_obj.solver_type})")
