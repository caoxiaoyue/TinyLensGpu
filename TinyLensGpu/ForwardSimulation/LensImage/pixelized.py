"""
Forward model for pixelized source gravitational lensing.

This module provides the forward modeling components for pixelized source
reconstruction, including ray-tracing, mesh generation, and mapping matrix
construction.
"""

from __future__ import annotations

import functools
import jax.numpy as jnp
from jax import jit, Array
import numpy as np
from typing import Optional, Tuple, Union

from .config import make_grid_2d
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Pixelized import PixelizedSourceModel
from TinyLensGpu.utils.lensing import (
    lens_mapping_matrix_from,
    build_psf_matrix_dense,
)
from TinyLensGpu.utils.mesh import sample_points_weighted
from TinyLensGpu.utils.inversion import LinearInversion


def _extract_pixelized_source_model(
    phys_model: PhysicalModel,
) -> PixelizedSourceModel:
    matches = [m for m in getattr(phys_model, "source_light", []) if isinstance(m, PixelizedSourceModel)]
    if len(getattr(phys_model, "source_light", [])) != 1 or len(matches) != 1:
        raise ValueError(
            "PixelizedLensSimulator requires PhysicalModel(source_light=[PixelizedSourceModel])."
        )
    return matches[0]


class PixelizedLensSimulator:
    def __init__(
        self,
        image_data: np.ndarray,
        dpix: float,
        phys_model: PhysicalModel,
        psf_kernel: np.ndarray,
        mask: Optional[np.ndarray] = None,
        lensed_source_image: Optional[np.ndarray] = None,
    ) -> None:
        
        self.dpix = dpix
        self.npix = image_data.shape[0]
        self.phys_model = phys_model
        self.pix_src_model = _extract_pixelized_source_model(phys_model=self.phys_model)
        self.psf_kernel = jnp.array(psf_kernel)
        
        if mask is None:
            mask = np.zeros_like(image_data, dtype=bool)
        self.mask = jnp.array(mask)
        
        if lensed_source_image is None:
            lensed_source_image = np.ones_like(image_data)
        self.lensed_source_image = jnp.array(lensed_source_image)
        
        xgrid_2d, ygrid_2d = make_grid_2d(self.npix, self.dpix)
        self.xgrid_unmask = jnp.array(xgrid_2d[~mask], dtype=jnp.float32)
        self.ygrid_unmask = jnp.array(ygrid_2d[~mask], dtype=jnp.float32)
        
        self._generate_source_mesh(self.lensed_source_image, self.mask)
        self.psf_matrix = build_psf_matrix_dense(np.array(self.mask), np.array(self.psf_kernel))
    
    def _generate_source_mesh(self, lensed_source_image: Array | np.ndarray, mask: Array | np.ndarray) -> None:
        model = self.pix_src_model
        
        source_mesh, (H, W), _ = sample_points_weighted(
            img=np.array(lensed_source_image),
            mask=~np.array(mask),
            n_points=model.n_source_points,
            alpha=model.mesh_alpha,
            blur_sigma_px=model.mesh_blur_sigma,
            replace=False,
            normalize_xy=False,
            pixel_jitter=False,
            method=model.mesh_method,
            seed=model.mesh_seed,
        )
        
        source_mesh = (source_mesh - np.array([(W-1)/2, (H-1)/2])) * self.dpix
        self.source_mesh = jnp.array(source_mesh, dtype=jnp.float32)
    
    @functools.partial(jit, static_argnums=(0,))
    def ray_trace(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        beta_x, beta_y = self.phys_model.deflection(x=x, y=y)
        return jnp.stack([beta_x, beta_y], axis=1)

    @property
    def source_mesh_beta(self) -> jnp.ndarray:
        return self.ray_trace(self.source_mesh[:, 0], self.source_mesh[:, 1])
    
    @property
    def data_mesh_beta(self) -> jnp.ndarray:
        return self.ray_trace(self.xgrid_unmask, self.ygrid_unmask)
    
    def build_lens_mapping_matrix(self) -> jnp.ndarray:
        return lens_mapping_matrix_from(
            source_mesh_beta=self.source_mesh_beta,
            data_mesh_beta=self.data_mesh_beta,
            k_neighbors=self.pix_src_model.k_neighbors,
            kernel=self.pix_src_model.interp_kernel,
            radius_scale=self.pix_src_model.radius_scale,
        )
    
    def build_blurred_lens_mapping_matrix(self) -> jnp.ndarray:
        return self.psf_matrix @ self.build_lens_mapping_matrix()
    
    def build_regularization_matrix(self, reg_scale: float, reg_coefficient: float) -> jnp.ndarray:
        return self.pix_src_model.regularization_matrix(
            points=self.source_mesh_beta,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
        )

    def reconstruct_source(
        self,
        data_vector: Union[jnp.ndarray, np.ndarray],
        noise_variance: Union[jnp.ndarray, np.ndarray],
        reg_scale: float,
        reg_coefficient: float,
        return_2d: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, LinearInversion]:
        """
        Reconstruct the source given observed data and noise.

        This method performs the full source reconstruction pipeline:
        1. Build lensing mapping matrix with PSF convolution
        2. Build regularization matrix
        3. Solve the linear inverse problem for source intensities
        4. Generate model image

        Parameters
        ----------
        data_vector : jnp.ndarray or np.ndarray
            Observed image data vector (unmasked pixels only)
        noise_variance : jnp.ndarray or np.ndarray
            Noise variance for each unmasked pixel
        reg_scale : float
            Regularization scale parameter
        reg_coefficient : float
            Regularization coefficient (lambda)
        return_2d : bool, optional
            If True, returns the model image as a 2D array.
            If False (default), returns the model image as a 1D vector (unmasked pixels only).

        Returns
        -------
        source_intensities : jnp.ndarray
            Reconstructed source intensities at source mesh points
        source_mesh_beta : jnp.ndarray
            Source mesh coordinates in source plane (shape: n_source x 2)
        model_image : jnp.ndarray
            Model image. Shape is (npix, npix) if return_2d=True, 
            else (n_unmasked_pixels,) if return_2d=False.
        inverter : LinearInversion
            Linear inversion solver object (cached for reuse)
        """
        # Convert inputs to JAX arrays
        d = jnp.array(data_vector)
        noise_var = jnp.array(noise_variance)

        # Build matrices
        blurred_lens_map_matrix = self.build_blurred_lens_mapping_matrix()
        reg_matrix = self.pix_src_model.regularization_matrix(
            points=self.source_mesh_beta,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
        )

        # Create linear inversion solver
        inverter = LinearInversion(
            d=d,
            F=blurred_lens_map_matrix,
            noise_cov=noise_var,
            H=reg_matrix,
        )

        # Solve for source intensities
        source_intensities = inverter.solve()

        # Generate model data
        model_data = blurred_lens_map_matrix @ source_intensities

        if return_2d:
            # Place model data into full image
            model_image = jnp.zeros((self.npix, self.npix))
            model_image = model_image.at[~self.mask].set(model_data)
        else:
            model_image = model_data

        return source_intensities, self.source_mesh_beta, model_image, inverter

    def __repr__(self) -> str:
        return (f"PixelizedLensSimulator("
                f"npix={self.npix}, "
                f"n_source_points={self.pix_src_model.n_source_points}, "
                f"n_mass={len(self.phys_model.lens_mass)})")
