"""
Probability model for pixelized source gravitational lensing image fitting.

This module provides the probability model for pixelized source reconstruction,
computing the Bayesian evidence (log evidence) which is analogous to the log
likelihood in parametric source modeling.
"""

import functools
import caskade as ck
import jax.numpy as jnp
import jax
from jax import jit, Array
import numpy as np
from typing import Optional, Dict, Tuple

from TinyLensGpu.Simulator.config import SimulatorConfig
from TinyLensGpu.Models.composite import PhysicalModel
from TinyLensGpu.Models.pixelized_source import PixelizedSourceModel
from TinyLensGpu.PixelizedSource import (
    LinearInversion,
    regularization_matrix_gp_from,
    lens_mapping_matrix_from,
    build_psf_matrix_dense,
    sample_points_weighted,
)


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
    pix_src_model : PixelizedSourceModel
        Pixelized source model with configuration
    mask : array_like, optional
        Boolean mask array (True = masked out)
    position_likelihood : dict, optional
        Position likelihood constraint configuration
    
    Attributes
    ----------
    image_data : jnp.ndarray
        Observed image
    noise_map : jnp.ndarray
        Noise map
    phys_model : PhysicalModel
        Physical model (mass components)
    pix_src_model : PixelizedSourceModel
        Pixelized source model
    source_mesh : jnp.ndarray
        Source mesh coordinates in image plane
    source_mesh_beta : jnp.ndarray
        Source mesh coordinates in source plane (cached)
    psf_matrix : jnp.ndarray
        PSF convolution matrix (cached)
    
    Examples
    --------
    >>> from TinyLensGpu.Models.mass import SIE
    >>> from TinyLensGpu.Models.composite import PhysicalModel
    >>> from TinyLensGpu.Models.pixelized_source import PixelizedSourceModel, PixelizedSourceConfig
    >>> 
    >>> # Create mass model
    >>> sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    >>> phys_model = PhysicalModel(lens_mass=[sie])
    >>> 
    >>> # Create pixelized source model
    >>> config = PixelizedSourceConfig(reg_scale=0.05, reg_coefficient=1.0)
    >>> pix_src = PixelizedSourceModel(config=config)
    >>> 
    >>> # Create probability model
    >>> prob_model = PixelizedImageProbModel(
    ...     image_data=image,
    ...     noise_map=noise,
    ...     psf_kernel=psf,
    ...     dpix=0.05,
    ...     phys_model=phys_model,
    ...     pix_src_model=pix_src,
    ...     mask=mask
    ... )
    >>> 
    >>> # Compute log evidence
    >>> log_ev = prob_model()
    """
    
    def __init__(
        self,
        image_data: np.ndarray,
        noise_map: np.ndarray,
        psf_kernel: np.ndarray,
        dpix: float,
        phys_model: PhysicalModel,
        pix_src_model: PixelizedSourceModel,
        mask: Optional[np.ndarray] = None,
        position_likelihood: Optional[Dict] = None,
    ) -> None:
        super().__init__("pixelized_image_prob_model")
        
        self.image_data = jnp.array(image_data)
        self.noise_map = jnp.array(noise_map)
        self.psf_kernel = jnp.array(psf_kernel)
        self.dpix = dpix
        
        self.phys_model = phys_model
        self.pix_src_model = pix_src_model
        
        if mask is None:
            mask = np.zeros_like(image_data, dtype=bool)
        self.mask = jnp.array(mask)
        self.unmask = ~self.mask
        
        self.position_like_config = position_likelihood
        
        npix = image_data.shape[0]
        self.npix = npix
        
        self._init_position_likelihood(self.position_like_config)
        
        self._generate_source_mesh()
        
        self.psf_matrix = build_psf_matrix_dense(np.array(self.mask), np.array(psf_kernel))
        
        self._source_mesh_beta_cache = None
        self._lens_map_matrix_cache = None
        self._inverter_cache = None
        self._cached_params = None
    
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
    
    def _generate_source_mesh(self) -> None:
        """Generate source mesh points in image plane based on observed image."""
        config = self.pix_src_model.config
        
        image_np = np.array(self.image_data)
        mask_np = np.array(self.unmask)
        
        source_mesh, (H, W), _ = sample_points_weighted(
            img=image_np,
            mask=mask_np,
            n_points=config.n_source_points,
            alpha=config.mesh_alpha,
            blur_sigma_px=config.mesh_blur_sigma,
            replace=False,
            normalize_xy=False,
            pixel_jitter=False,
            method=config.mesh_method,
            seed=config.mesh_seed,
        )
        
        source_mesh = source_mesh - np.array([(W-1)/2, (H-1)/2])
        source_mesh *= self.dpix
        
        self.source_mesh = jnp.array(source_mesh, dtype=jnp.float32)
    
    @ck.forward
    def _compute_source_mesh_beta(self) -> jnp.ndarray:
        """Compute source mesh coordinates in source plane via ray-tracing."""
        beta_x, beta_y = self.phys_model.deflection(
            self.source_mesh[:, 0],
            self.source_mesh[:, 1]
        )
        return jnp.stack([beta_x, beta_y], axis=1)
    
    @ck.forward
    def _compute_data_mesh_beta(self) -> jnp.ndarray:
        """Compute data mesh coordinates in source plane via ray-tracing."""
        xgrid = jnp.arange(self.npix) - (self.npix - 1) / 2
        ygrid = jnp.arange(self.npix) - (self.npix - 1) / 2
        xgrid_2d, ygrid_2d = jnp.meshgrid(xgrid * self.dpix, ygrid * self.dpix)
        
        xgrid_1d = xgrid_2d[self.unmask]
        ygrid_1d = ygrid_2d[self.unmask]
        
        beta_x, beta_y = self.phys_model.deflection(xgrid_1d, ygrid_1d)
        return jnp.stack([beta_x, beta_y], axis=1)
    
    def _get_or_build_inverter(self):
        """
        Get cached LinearInversion object or build a new one if parameters changed.
        
        This method caches the expensive LinearInversion initialization to avoid
        redundant matrix precomputation when called multiple times with same parameters.
        
        Returns
        -------
        inverter : LinearInversion
            Cached or newly created LinearInversion object
        source_mesh_beta : jnp.ndarray
            Source mesh coordinates in source plane
        blurred_lens_map_matrix : jnp.ndarray
            Blurred lensing mapping matrix
        """
        config = self.pix_src_model.config
        
        # Get current parameter values
        reg_scale_val = config.reg_scale.value
        reg_coeff_val = config.reg_coefficient.value
        
        # Create a simple hash of mass model parameters by extracting all parameter values
        mass_param_values = []
        for mass_comp in self.phys_model.lens_mass:
            # Get all parameter values from the mass component
            for param_name in dir(mass_comp):
                param = getattr(mass_comp, param_name)
                if hasattr(param, 'value'):
                    mass_param_values.append(float(param.value))
        
        current_params = (reg_scale_val, reg_coeff_val, tuple(mass_param_values))
        
        # Check if we can use cached inverter
        if self._inverter_cache is not None and self._cached_params == current_params:
            return self._inverter_cache
        
        # Need to rebuild inverter
        source_mesh_beta = self._compute_source_mesh_beta()
        data_mesh_beta = self._compute_data_mesh_beta()
        
        lens_map_matrix = lens_mapping_matrix_from(
            source_mesh_beta=source_mesh_beta,
            data_mesh_beta=data_mesh_beta,
            k_neighbors=config.k_neighbors,
            kernel=config.interp_kernel,
            radius_scale=config.radius_scale,
        )
        
        blurred_lens_map_matrix = self.psf_matrix @ lens_map_matrix
        
        reg_matrix = regularization_matrix_gp_from(
            scale=reg_scale_val,
            coefficient=reg_coeff_val,
            points=source_mesh_beta,
            reg_type=config.reg_type,
        )
        
        data_vector = self.image_data[self.unmask]
        noise_variance = self.noise_map[self.unmask] ** 2
        
        inverter = LinearInversion(
            d=data_vector,
            F=blurred_lens_map_matrix,
            noise_cov=noise_variance,
            H=reg_matrix,
        )
        
        # Cache the inverter and related data
        self._inverter_cache = (
            inverter,
            source_mesh_beta,
            blurred_lens_map_matrix,
        )
        self._cached_params = current_params
        
        return self._inverter_cache
    
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
        inverter, _, _ = self._get_or_build_inverter()
        
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
    
    @ck.forward
    def reconstruct_source(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Reconstruct the source given current parameters.
        
        Returns
        -------
        source_intensities : jnp.ndarray
            Reconstructed source intensities at source mesh points
        source_mesh_beta : jnp.ndarray
            Source mesh coordinates in source plane
        model_image : jnp.ndarray
            Model image (full 2D array)
        """
        inverter, source_mesh_beta, blurred_lens_map_matrix = self._get_or_build_inverter()
        
        source_intensities, _ = inverter.invert()
        
        model_data = blurred_lens_map_matrix @ source_intensities
        
        model_image = jnp.zeros_like(self.image_data)
        model_image = model_image.at[self.unmask].set(model_data)
        
        return source_intensities, source_mesh_beta, model_image
    
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
        return (f"PixelizedImageProbModel("
                f"npix={self.npix}, "
                f"n_source_points={self.pix_src_model.config.n_source_points})")
