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
    apply_psf_to_mapping_matrix,
)
from TinyLensGpu.utils.interpolation.kernels import get_interpolation_weights
from TinyLensGpu.utils.mesh import sample_points_weighted
from TinyLensGpu.utils.inversion import LinearInversion, OperatorInversion, NNLSInversion


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
        # Use the centralized extraction method
        self.pix_src_model = self.phys_model.get_pixelized_source_model()
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
        
        # Precompute unmasked indices for PSF application
        # mask is boolean, True where masked. We want indices of False (unmasked).
        y_indices, x_indices = np.where(~mask)
        self.unmasked_indices = (jnp.array(y_indices), jnp.array(x_indices))
        self.image_shape = (self.npix, self.npix)

        psf_h, psf_w = self.psf_kernel.shape
        self.psf_shape = (int(psf_h), int(psf_w))
        self.fft_shape = (self.npix + int(psf_h) - 1, self.npix + int(psf_w) - 1)
        self.psf_fft = jnp.fft.rfft2(self.psf_kernel, s=self.fft_shape)
        
        self._generate_source_mesh(self.lensed_source_image, self.mask)
    
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

    def build_lens_mapping_operator(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        weights, indices, _ = get_interpolation_weights(
            points=self.source_mesh_beta,
            query_points=self.data_mesh_beta,
            k_neighbors=self.pix_src_model.k_neighbors,
            kernel=self.pix_src_model.interp_kernel,
            radius_scale=self.pix_src_model.radius_scale,
        )
        return weights, indices
    
    def build_blurred_lens_mapping_matrix(self, method: str = 'fft') -> jnp.ndarray:
        return apply_psf_to_mapping_matrix(
            mapping_matrix=self.build_lens_mapping_matrix(),
            psf_kernel=self.psf_kernel,
            image_shape=self.image_shape,
            unmasked_indices=self.unmasked_indices,
            method=method,
            psf_fft=self.psf_fft if method == 'fft' else None,
            psf_shape=self.psf_shape if method == 'fft' else None,
        )

    def build_lens_light_basis_matrix(self, method: str = 'fft') -> jnp.ndarray:
        n_lens_light = len(self.phys_model.lens_light)
        if n_lens_light == 0:
            return jnp.zeros((self.xgrid_unmask.shape[0], 0), dtype=jnp.float32)

        basis = []
        for light_model in self.phys_model.lens_light:
            basis.append(light_model.light(x=self.xgrid_unmask, y=self.ygrid_unmask))
        basis_unblurred = jnp.stack(basis, axis=1).astype(jnp.float32)

        return apply_psf_to_mapping_matrix(
            mapping_matrix=basis_unblurred,
            psf_kernel=self.psf_kernel,
            image_shape=self.image_shape,
            unmasked_indices=self.unmasked_indices,
            method=method,
            psf_fft=self.psf_fft if method == 'fft' else None,
            psf_shape=self.psf_shape if method == 'fft' else None,
        )
    
    def build_regularization_matrix(self, reg_scale: float, reg_coefficient: float) -> jnp.ndarray:
        return self.pix_src_model.regularization_matrix(
            points=self.source_mesh_beta,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
        )

    def build_inverter(
        self,
        data_vector: Union[jnp.ndarray, np.ndarray],
        noise_variance: Union[jnp.ndarray, np.ndarray],
        reg_scale: float,
        reg_coefficient: float,
        *,
        include_lens_light: bool = False,
        lens_light_ridge: float = 1e-8,
        nonnegative: bool = False,
        inversion_backend: str = "exact",
        cg_tol: float = 1e-4,
        cg_maxiter: int = 40,
        slq_seed: int = 0,
        slq_probes: int = 2,
        slq_steps: int = 10,
    ) -> Union[LinearInversion, OperatorInversion]:
        d = jnp.array(data_vector)
        noise_var = jnp.array(noise_variance)

        reg_matrix_src = self.pix_src_model.regularization_matrix(
            points=self.source_mesh_beta,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
        )

        if inversion_backend == "fast":
            if include_lens_light or nonnegative:
                raise ValueError("include_lens_light/nonnegative currently require inversion_backend='exact'.")
            weights, indices = self.build_lens_mapping_operator()
            return OperatorInversion(
                d=d,
                noise_var=noise_var,
                H=reg_matrix_src,
                weights=weights,
                indices=indices,
                psf_fft=self.psf_fft,
                image_shape=self.image_shape,
                psf_shape=self.psf_shape,
                unmasked_indices=self.unmasked_indices,
                cg_tol=cg_tol,
                cg_maxiter=cg_maxiter,
                slq_seed=slq_seed,
                slq_probes=slq_probes,
                slq_steps=slq_steps,
            )

        blurred_lens_map_matrix = self.build_blurred_lens_mapping_matrix()
        if not include_lens_light:
            if nonnegative:
                return NNLSInversion(d=d, F=blurred_lens_map_matrix, noise_cov=noise_var, H=reg_matrix_src)
            return LinearInversion(d=d, F=blurred_lens_map_matrix, noise_cov=noise_var, H=reg_matrix_src)

        lens_basis_matrix = self.build_lens_light_basis_matrix()
        F_total = jnp.concatenate([blurred_lens_map_matrix, lens_basis_matrix], axis=1)

        n_src = int(reg_matrix_src.shape[0])
        n_lens = int(lens_basis_matrix.shape[1])
        ridge = jnp.array(max(float(lens_light_ridge), 0.0), dtype=jnp.float32)
        H_lens = ridge * jnp.eye(n_lens, dtype=jnp.float32)
        H_total = jnp.block(
            [
                [reg_matrix_src, jnp.zeros((n_src, n_lens), dtype=jnp.float32)],
                [jnp.zeros((n_lens, n_src), dtype=jnp.float32), H_lens],
            ]
        )

        if nonnegative:
            return NNLSInversion(d=d, F=F_total, noise_cov=noise_var, H=H_total)
        return LinearInversion(d=d, F=F_total, noise_cov=noise_var, H=H_total)

    def reconstruct_source_and_lens_light(
        self,
        data_vector: Union[jnp.ndarray, np.ndarray],
        noise_variance: Union[jnp.ndarray, np.ndarray],
        reg_scale: float,
        reg_coefficient: float,
        lens_light_ridge: float = 1e-8,
        nonnegative: bool = True,
        return_2d: bool = False,
    ):
        inverter = self.build_inverter(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
            include_lens_light=True,
            lens_light_ridge=lens_light_ridge,
            nonnegative=nonnegative,
            inversion_backend="exact",
        )

        x_total = inverter.solve()
        n_src = int(self.source_mesh_beta.shape[0])
        source_intensities = x_total[:n_src]
        lens_light_intensities = x_total[n_src:]

        model_data = inverter.model_predict(x_total)

        if return_2d:
            model_image = jnp.zeros((self.npix, self.npix))
            model_image = model_image.at[~self.mask].set(model_data)
        else:
            model_image = model_data

        return source_intensities, lens_light_intensities, self.source_mesh_beta, model_image, inverter

    def reconstruct_source(
        self,
        data_vector: Union[jnp.ndarray, np.ndarray],
        noise_variance: Union[jnp.ndarray, np.ndarray],
        reg_scale: float,
        reg_coefficient: float,
        return_2d: bool = False,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Union[LinearInversion, OperatorInversion]]:
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
        **kwargs : dict
            Additional arguments passed to build_inverter (e.g. inversion_backend, cg_tol, etc.)

        Returns
        -------
        source_intensities : jnp.ndarray
            Reconstructed source intensities at source mesh points
        source_mesh_beta : jnp.ndarray
            Source mesh coordinates in source plane (shape: n_source x 2)
        model_image : jnp.ndarray
            Model image. Shape is (npix, npix) if return_2d=True, 
            else (n_unmasked_pixels,) if return_2d=False.
        inverter : Union[LinearInversion, OperatorInversion]
            Linear inversion solver object (cached for reuse)
        """
        inverter = self.build_inverter(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
            **kwargs,
        )

        # Solve for source intensities
        source_intensities = inverter.solve()

        # Generate model data
        model_data = inverter.model_predict(source_intensities)

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
