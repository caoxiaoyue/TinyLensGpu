"""
Probability model for pixelized source gravitational lensing image fitting.

This module provides the observation model for pixelized source reconstruction,
computing the Bayesian evidence (log evidence) which is analogous to the log
likelihood in parametric source modeling.
"""

import functools
import caskade as ck
import jax.numpy as jnp
import jax
from jax import jit, Array
import numpy as np
from typing import Optional, Dict, Union

from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Pixelized import PixelizedSourceModel
from TinyLensGpu.utils.inversion import LinearInversion


class PixelizedImageProbModel(ck.Module):
    """
    Probability model for pixelized source gravitational lensing images.
    
    This class computes the Bayesian evidence (log evidence) for pixelized source
    reconstruction. The log evidence is analogous to the log likelihood in parametric
    source modeling and can be used for:
    1. Optimizing hyperparameters (regularization scale/coefficient)
    2. Optimizing mass model parameters
    3. Nested sampling for full Bayesian inference
    
    The key difference from parametric source modeling is that the source intensities
    are marginalized out analytically, resulting in the log evidence rather than a
    simple likelihood.
    
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
    phys_model : PhysicalModel
        Physical model containing mass components (lens_mass)
    mask : array_like, optional
        Boolean mask array (True = masked out)
    position_likelihood : dict, optional
        Position likelihood constraint configuration
    inversion_backend : str, optional
        Inversion backend, "matrix" or "operator".
        Legacy aliases: "exact" -> "matrix", "fast" -> "operator".
    cg_tol : float, optional
        Conjugate gradient tolerance for operator inversion.
    cg_maxiter : int, optional
        Maximum number of conjugate gradient iterations. Default is 40.
    slq_seed : int, optional
        Random seed for SLQ. Default is 0.
    slq_probes : int, optional
        Number of probe vectors for SLQ. Default is 2.
    slq_steps : int, optional
        Number of Lanczos steps for SLQ. Default is 10.
    evidence_mode : str, optional
        Operator evidence mode: "accurate" (default) or "fast".
    operator_cache_policy : str, optional
        Operator mapping cache policy: "safe" (default), "unsafe_static", or "off".
    
    Attributes
    ----------
    image_data : jnp.ndarray
        Observed image
    noise_map : jnp.ndarray
        Noise map
    simulator : PixelizedLensSimulator
        Forward model simulator
    phys_model : PhysicalModel
        Physical model (mass components)
    pix_src_model : PixelizedSourceModel
        Pixelized source model
    
    Examples
    --------
    >>> from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
    >>> from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
    >>> from TinyLensGpu.PhysicalModel.LensImage.Pixelized import PixelizedSourceModel
    >>> 
    >>> # Create mass model
    >>> sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    >>> phys_model = PhysicalModel(lens_mass=[sie])
    >>> 
    >>> # Create pixelized source model
    >>> pix_src = PixelizedSourceModel(reg_scale=0.05, reg_coefficient=1.0)
    >>> 
    >>> # Create probability model
    >>> prob_model = PixelizedImageProbModel(
    ...     image_data=image,
    ...     noise_map=noise,
    ...     psf_kernel=psf,
    ...     dpix=0.05,
    ...     phys_model=PhysicalModel(lens_mass=[sie], source_light=[pix_src]),
    ...     mask=mask
    ... )
    >>> 
    >>> # Compute log evidence
    >>> log_ev = prob_model()
    """
    
    def __init__(
        self,
        image_data: Union[np.ndarray, Array],
        noise_map: Union[np.ndarray, Array],
        psf_kernel: Union[np.ndarray, Array],
        dpix: float,
        phys_model: PhysicalModel,
        mask: Optional[Union[np.ndarray, Array]] = None,
        position_likelihood: Optional[Dict] = None,
        include_lens_light: bool = False,
        nonnegative: bool = False,
        lens_light_ridge: float = 1e-8,
        inversion_backend: str = "matrix",
        cg_tol: float = 1e-4,
        cg_maxiter: int = 120,
        slq_seed: int = 0,
        slq_probes: int = 32,
        slq_steps: int = 60,
        evidence_mode: str = "accurate",
        operator_cache_policy: str = "safe",
        nnls_maxiter: int = 600,
        nnls_tol: float = 1e-6,
        nnls_lipschitz_iters: int = 12,
    ) -> None:
        super().__init__("pixelized_image_prob_model")
        
        self.image_data = jnp.array(image_data)
        self.noise_map = jnp.array(noise_map)
        
        self.phys_model = phys_model
        # Use the centralized extraction method
        extracted_pix_src_model = self.phys_model.get_pixelized_source_model()
        object.__setattr__(self, "pix_src_model", extracted_pix_src_model)
        
        if mask is None:
            mask = np.zeros_like(image_data, dtype=bool)
        self.mask = jnp.array(mask)
        self.unmask = ~self.mask
        self._data_vector = self.image_data[self.unmask]
        self._noise_variance = self.noise_map[self.unmask] ** 2
        
        self.position_like_config = position_likelihood
        
        npix = image_data.shape[0]
        self.npix = npix
        
        self._init_position_likelihood(self.position_like_config)
        
        self.simulator = PixelizedLensSimulator(
            image_data=image_data,
            dpix=dpix,
            phys_model=self.phys_model,
            psf_kernel=psf_kernel,
            mask=mask,
        )

        object.__setattr__(self, "include_lens_light", bool(include_lens_light))
        object.__setattr__(self, "nonnegative", bool(nonnegative))
        object.__setattr__(self, "lens_light_ridge", float(lens_light_ridge))

        backend = str(inversion_backend).strip().lower()
        if backend == "exact":
            backend = "matrix"
        elif backend == "fast":
            backend = "operator"
        object.__setattr__(self, "inversion_backend", backend)
        object.__setattr__(self, "cg_tol", float(cg_tol))
        object.__setattr__(self, "cg_maxiter", int(cg_maxiter))
        object.__setattr__(self, "slq_seed", int(slq_seed))
        object.__setattr__(self, "slq_probes", int(slq_probes))
        object.__setattr__(self, "slq_steps", int(slq_steps))
        object.__setattr__(self, "evidence_mode", str(evidence_mode))
        object.__setattr__(self, "operator_cache_policy", str(operator_cache_policy))
        object.__setattr__(self, "nnls_maxiter", int(nnls_maxiter))
        object.__setattr__(self, "nnls_tol", float(nnls_tol))
        object.__setattr__(self, "nnls_lipschitz_iters", int(nnls_lipschitz_iters))
        
        self._inverter_cache = None
        self._cached_params = None
    
    def _init_position_likelihood(self, config: Optional[Dict]) -> None:
        """Initialize position likelihood parameters for JIT optimization."""
        # Default values
        self._pos_px = None
        self._pos_py = None
        self._pos_thr = jnp.array(0.0, dtype=jnp.float32)
        self._pos_minl = jnp.array(0.0, dtype=jnp.float32)
        self._has_pos_penalty = False
        
        if not config:
            return
            
        positions = config.get('positions', [])
        if positions is None or len(positions) < 2:
            return
            
        self._pos_px = jnp.array([p[0] for p in positions], dtype=jnp.float32)
        self._pos_py = jnp.array([p[1] for p in positions], dtype=jnp.float32)
        
        # Helper to get value with fallback keys
        def get_val(keys, default):
            for k in keys:
                if k in config:
                    return config[k]
            return default
            
        threshold = get_val(['threshold_arcsec', 'position_threshold'], 0.0)
        min_log_like = get_val(['min_log_like', 'min_position_likelihood'], 0.0)
        
        self._pos_thr = jnp.array(float(threshold), dtype=jnp.float32)
        self._pos_minl = jnp.array(float(min_log_like), dtype=jnp.float32)
        self._has_pos_penalty = True
    
    @ck.forward
    def _build_inverter(self):
        """
        Build a new LinearInversion object.

        This method prepares the data and delegates the actual reconstruction
        to the simulator.

        Note: This method needs @ck.forward decorator because it calls simulator methods
        that use self.phys_model.deflection(), which requires caskade parameter injection.

        """
        model = self.pix_src_model

        # Get current parameter values
        reg_scale_val = model.reg_scale.value
        reg_coeff_val = model.reg_coefficient.value

        # Prepare data vectors
        data_vector = self._data_vector
        noise_variance = self._noise_variance

        inverter = self.simulator.build_inverter(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale_val,
            reg_coefficient=reg_coeff_val,
            include_lens_light=self.include_lens_light,
            lens_light_ridge=self.lens_light_ridge,
            nonnegative=self.nonnegative,
            inversion_backend=self.inversion_backend,
            cg_tol=self.cg_tol,
            cg_maxiter=self.cg_maxiter,
            slq_seed=self.slq_seed,
            slq_probes=self.slq_probes,
            slq_steps=self.slq_steps,
            evidence_mode=self.evidence_mode,
            operator_cache_policy=self.operator_cache_policy,
            nnls_maxiter=self.nnls_maxiter,
            nnls_tol=self.nnls_tol,
            nnls_lipschitz_iters=self.nnls_lipschitz_iters,
        )

        return inverter
    
    @ck.forward
    def __call__(self):
        """
        Compute log evidence for pixelized source reconstruction.
        
        This is the main function for computing the Bayesian evidence, which is
        analogous to the log likelihood in parametric source modeling.
        
        Note: This method is NOT JIT-compiled to allow caching of LinearInversion.
        The actual computation inside LinearInversion is JIT-compiled for speed.
        
        Returns
        -------
        log_evidence : float
            Log of the Bayesian evidence
        """
        inverter = self._build_inverter()
        
        log_ev = inverter.log_evidence()
        
        if self._has_pos_penalty:
            log_ev = log_ev + self._position_likelihood_penalty_jax()
        
        log_ev = jnp.where(jnp.isfinite(log_ev), log_ev, -jnp.inf)
        return log_ev
    
    def log_evidence(self) -> float:
        """
        Compute log evidence of current model parameters.
        
        Returns
        -------
        log_evidence : float
            Log evidence value
        """
        log_ev = float(np.asarray(self.__call__()))
        return log_ev
    
    
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
        
        pen_clipped = jnp.clip(pen_continuous, min=self._pos_minl, max=0.0)
        pen = jnp.where(jnp.logical_or(self._pos_thr <= 0.0, exceed <= 0.0), 0.0, pen_clipped)
        return pen
    
    def __repr__(self) -> str:
        return (f"PixelizedImageProbModel("
                f"npix={self.npix}, "
                f"n_source_points={self.pix_src_model.n_source_points})")
