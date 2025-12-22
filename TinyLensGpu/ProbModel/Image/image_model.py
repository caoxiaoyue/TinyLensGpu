"""
Probability model for gravitational lensing image fitting.

This module provides the probability model using PhysicalModel
and LensSimulator for computing image likelihoods.
"""

import functools
import jax.numpy as jnp
import jax
import numpy as np
from typing import Optional, Dict
from jax import jit

from TinyLensGpu.Simulator.lens_simulator import LensSimulator
from TinyLensGpu.Simulator.config import SimulatorConfig
from TinyLensGpu.Models.composite import PhysicalModel


class ImageProbModel:
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
    ):
        self.image_data = jnp.array(image_data)
        self.noise_map = jnp.array(noise_map)

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
            phys_model=phys_model,
            sim_config=sim_config,
            solver_type=solver_type,
        )

        self.use_linear = use_linear
        self.unmask = jnp.array(~sim_config.mask)
        self.position_like_config = position_likelihood

    def forward_model(self):
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

    @functools.partial(jit, static_argnums=(0,))
    def _likelihood_helper(
        self,
        image_model: jnp.ndarray,
        image_data: jnp.ndarray,
        noise_map: jnp.ndarray,
        unmask: jnp.ndarray,
    ):
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

    def likelihood(self, debug: bool = True):
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
        # Run forward model
        image_model, intensity_list = self.forward_model()

        # Check NNLS success
        if self.use_linear and self.sim_obj.solver_type == "nnls":
            intensity_list = np.asarray(intensity_list)
            nnls_success = np.all(intensity_list >= 0.0)
            if not nnls_success:
                raise ValueError("NNLS failed to find a solution")

        # Compute chi-square likelihood
        like = self._likelihood_helper(
            image_model,
            self.image_data,
            self.noise_map,
            self.unmask,
        )
        like = float(np.asarray(like))

        # Add position likelihood penalty if configured
        if self.position_like_config is not None:
            penalty = self._position_likelihood_penalty()
            like = like + penalty

        # Debug checks
        if debug:
            if np.isnan(like):
                import warnings
                warnings.warn("NaN detected in likelihood calculation")
                return -np.inf
            if np.isinf(like):
                import warnings
                warnings.warn("Inf detected in likelihood calculation")
                return -np.inf

        return like

    def _position_likelihood_penalty(self):
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

        # Convert positions to JAX arrays
        px = jnp.array([p[0] for p in positions])
        py = jnp.array([p[1] for p in positions])

        # Compute deflection to source plane
        # With caskade, deflection is computed directly from PhysicalModel
        beta_x, beta_y = self.sim_obj.phys_model.deflection(px, py)

        # Compute pairwise distances in source plane
        dx = beta_x[:, None] - beta_x[None, :]
        dy = beta_y[:, None] - beta_y[None, :]
        dist = jnp.sqrt(dx * dx + dy * dy)
        max_sep = jnp.max(dist)

        # Compute penalty
        exceed = jnp.maximum(0.0, max_sep - threshold)
        if threshold <= 0.0 or exceed <= 0.0:
            pen = 0.0
        else:
            ratio = exceed / threshold
            pen_continuous = min_like * (1.0 - float(jnp.exp(-ratio)))
            pen = float(jnp.clip(pen_continuous, min_like, 0.0))

        return pen

    def __repr__(self):
        return (f"ImageProbModel("
                f"npix={self.image_data.shape[0]}, "
                f"use_linear={self.use_linear}, "
                f"solver={self.sim_obj.solver_type})")
