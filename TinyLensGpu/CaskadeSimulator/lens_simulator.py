"""
Gravitational lens simulator using caskade models.

This module provides the main simulation engine for gravitational lensing,
handling ray-tracing, PSF convolution, and linear parameter solving.
"""

import functools
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
from typing import Optional
import numpy as np

from ..CaskadeModels.composite import PhysicalModel
from .config import SimulatorConfig
from ..CaskadeInference.linear_solver import LinearSolver, prepare_linear_system


def bin_image_general(img, nsub):
    """
    Bin an image by averaging over nsub x nsub blocks.

    Parameters
    ----------
    img : array_like
        Image to bin, shape can be (ny, nx, ...) where ny and nx are divisible by nsub
    nsub : int
        Binning factor

    Returns
    -------
    img_binned : array_like
        Binned image with shape (ny//nsub, nx//nsub, ...)
    """
    if nsub == 1:
        return img

    # Handle different input shapes
    if img.ndim == 2:
        ny, nx = img.shape
        ny_bin, nx_bin = ny // nsub, nx // nsub
        img_reshaped = img.reshape(ny_bin, nsub, nx_bin, nsub)
        img_binned = jnp.mean(img_reshaped, axis=(1, 3))
    else:
        # For 3D and higher dimensional arrays
        ny, nx = img.shape[:2]
        ny_bin, nx_bin = ny // nsub, nx // nsub
        other_dims = img.shape[2:]

        img_reshaped = img.reshape(ny_bin, nsub, nx_bin, nsub, *other_dims)
        img_binned = jnp.mean(img_reshaped, axis=(1, 3))

    return img_binned


class LensSimulator:
    """
    Gravitational lens simulator using caskade models.

    This class performs forward simulation of gravitational lensing,
    including ray-tracing, surface brightness calculation, PSF convolution,
    and optional linear parameter solving for intensity values.

    Parameters
    ----------
    phys_model : PhysicalModel
        Physical model containing mass and light components
    sim_config : SimulatorConfig
        Simulation configuration
    solver_type : str, optional
        Linear solver type: 'nnls' or 'normal' (default: 'nnls')

    Attributes
    ----------
    phys_model : PhysicalModel
        Physical model
    sim_config : SimulatorConfig
        Simulation configuration
    solver_type : str
        Linear solver type
    linear_solver : LinearSolver
        Linear solver instance
    """

    def __init__(
        self,
        phys_model: PhysicalModel,
        sim_config: SimulatorConfig,
        solver_type: str = 'nnls'
    ):
        self.phys_model = phys_model
        self.sim_config = sim_config
        self.solver_type = solver_type

        if self.solver_type not in ['nnls', 'normal']:
            raise ValueError("solver_type must be either 'nnls' or 'normal'")

        self.linear_solver = LinearSolver(solver_type)

        # Pre-convert grids to JAX arrays
        self.xgrid_sub = jnp.array(self.sim_config.xgrid_sub)
        self.ygrid_sub = jnp.array(self.sim_config.ygrid_sub)
        self.psf_kernel = jnp.array(self.sim_config.psf_kernel)

    def simulate(
        self,
        bs: int = 1,
        use_linear: bool = False,
        return_intensity: bool = False,
        image_map: Optional[np.ndarray] = None,
        noise_map: Optional[np.ndarray] = None,
        xgrid_sub: Optional[np.ndarray] = None,
        ygrid_sub: Optional[np.ndarray] = None,
        psf_kernel: Optional[np.ndarray] = None,
    ):
        """
        Simulate gravitational lensing image.

        Parameters
        ----------
        bs : int, optional
            Batch size (default: 1)
        use_linear : bool, optional
            Whether to use linear solver for intensity parameters (default: False)
        return_intensity : bool, optional
            Whether to return intensity values (default: False)
        image_map : array_like, optional
            Observed image for linear solver
        noise_map : array_like, optional
            Noise map for linear solver
        xgrid_sub : array_like, optional
            Override subsampled x-coordinates
        ygrid_sub : array_like, optional
            Override subsampled y-coordinates
        psf_kernel : array_like, optional
            Override PSF kernel

        Returns
        -------
        img : array_like
            Simulated image, shape (npix, npix) or (npix, npix, bs)
        intensity_list : array_like, optional
            Intensity values if return_intensity=True

        Examples
        --------
        >>> # Simple forward simulation
        >>> img = simulator.simulate(bs=1)
        >>>
        >>> # With linear solver
        >>> img, intensities = simulator.simulate(
        ...     bs=100,
        ...     use_linear=True,
        ...     return_intensity=True,
        ...     image_map=observed_data,
        ...     noise_map=noise
        ... )
        """
        # Use default grids if not provided
        if xgrid_sub is None:
            xgrid_sub = self.xgrid_sub
        else:
            xgrid_sub = jnp.array(xgrid_sub)

        if ygrid_sub is None:
            ygrid_sub = self.ygrid_sub
        else:
            ygrid_sub = jnp.array(ygrid_sub)

        if psf_kernel is None:
            psf_kernel = self.psf_kernel
        else:
            psf_kernel = jnp.array(psf_kernel)

        # Replicate for batch
        xgrid_sub = jnp.repeat(xgrid_sub[..., jnp.newaxis], bs, axis=-1)
        ygrid_sub = jnp.repeat(ygrid_sub[..., jnp.newaxis], bs, axis=-1)
        psf_kernel = jnp.repeat(psf_kernel[..., jnp.newaxis], bs, axis=-1)

        # Get component counts
        n_src = len(self.phys_model.source_light)
        n_lens_light = len(self.phys_model.lens_light)
        n_lens_mass = len(self.phys_model.lens_mass)

        # Generate ideal model (before PSF convolution)
        img_lens_sub, img_arc_sub = self._generate_ideal_model(
            xgrid_sub, ygrid_sub, n_src, n_lens_light, n_lens_mass
        )

        # Apply PSF and solve for intensities
        if not use_linear:
            img, X_vec = self._simulate_nonlinear(
                img_lens_sub, img_arc_sub, psf_kernel
            )
        else:
            if image_map is None or noise_map is None:
                raise ValueError("image_map and noise_map required for linear simulation")

            img, X_vec = self._simulate_linear(
                img_lens_sub, img_arc_sub, psf_kernel,
                image_map, noise_map, bs, n_lens_light, n_src
            )

        # Remove batch axis for single image
        if bs == 1:
            img = jnp.squeeze(img, axis=-1)

        if return_intensity:
            return img, X_vec
        else:
            return img

    def _generate_ideal_model(
        self,
        xgrid_sub: jnp.ndarray,
        ygrid_sub: jnp.ndarray,
        n_src: int,
        n_lens_light: int,
        n_lens_mass: int,
    ):
        """
        Generate ideal model image before PSF convolution.

        This method performs ray-tracing and computes surface brightness
        for all light components.

        Parameters
        ----------
        xgrid_sub : array_like
            X-coordinates at subsampled resolution, shape [ny_sub, nx_sub, bs]
        ygrid_sub : array_like
            Y-coordinates at subsampled resolution, shape [ny_sub, nx_sub, bs]
        n_src : int
            Number of source light components
        n_lens_light : int
            Number of lens light components
        n_lens_mass : int
            Number of mass components

        Returns
        -------
        img_lens_sub : array_like
            Lens light images, shape [ny_sub, nx_sub, bs, n_lens_light]
        img_arc_sub : array_like
            Source light images, shape [ny_sub, nx_sub, bs, n_src]
        """
        # Initialize output arrays
        img_sub = jnp.zeros_like(xgrid_sub)  # [ny_sub, nx_sub, bs]
        img_arc_sub = jnp.repeat(img_sub[..., jnp.newaxis], n_src, axis=-1)
        img_lens_sub = jnp.repeat(img_sub[..., jnp.newaxis], n_lens_light, axis=-1)

        # Ray-tracing: compute source plane coordinates
        if n_lens_mass > 0:
            beta_x, beta_y = self.phys_model.deflection(xgrid_sub, ygrid_sub)
        else:
            beta_x, beta_y = xgrid_sub, ygrid_sub

        # Compute source light
        if n_src > 0:
            for i, light_model in enumerate(self.phys_model.source_light):
                img_arc_sub = img_arc_sub.at[..., i].set(
                    light_model.light(beta_x, beta_y)
                )

        # Compute lens light
        if n_lens_light > 0:
            for i, light_model in enumerate(self.phys_model.lens_light):
                img_lens_sub = img_lens_sub.at[..., i].set(
                    light_model.light(xgrid_sub, ygrid_sub)
                )

        return img_lens_sub, img_arc_sub

    @functools.partial(jit, static_argnums=(0,))
    def _simulate_nonlinear(
        self,
        img_lens_sub: jnp.ndarray,
        img_arc_sub: jnp.ndarray,
        psf_kernel: jnp.ndarray,
    ):
        """
        Non-linear simulation (no intensity optimization).

        Simply sums all components, bins, and convolves with PSF.

        Parameters
        ----------
        img_lens_sub : array_like
            Lens light, shape [ny_sub, nx_sub, bs, n_lens]
        img_arc_sub : array_like
            Source light, shape [ny_sub, nx_sub, bs, n_src]
        psf_kernel : array_like
            PSF kernel, shape [ny_psf, nx_psf, bs]

        Returns
        -------
        img : array_like
            Final image, shape [ny, nx, bs]
        X_vec : None
            No intensity values for non-linear case
        """
        # Sum all components
        img_sub = jnp.sum(img_lens_sub, axis=-1) + jnp.sum(img_arc_sub, axis=-1)

        # Bin to full resolution
        img = bin_image_general(img_sub, self.sim_config.nsub)

        # Convolve with PSF
        img = jsp.signal.fftconvolve(img, psf_kernel, mode='same', axes=(0, 1))

        return img, None

    @functools.partial(jit, static_argnums=(0, 6, 7, 8))
    def _simulate_linear(
        self,
        img_lens_sub: jnp.ndarray,
        img_arc_sub: jnp.ndarray,
        psf_kernel: jnp.ndarray,
        image_map: jnp.ndarray,
        noise_map: jnp.ndarray,
        bs: int,
        n_lens_light: int,
        n_src: int,
    ):
        """
        Linear simulation with intensity optimization.

        Solves for optimal intensity values using linear least squares or NNLS.

        Parameters
        ----------
        img_lens_sub : array_like
            Lens light, shape [ny_sub, nx_sub, bs, n_lens]
        img_arc_sub : array_like
            Source light, shape [ny_sub, nx_sub, bs, n_src]
        psf_kernel : array_like
            PSF kernel, shape [ny_psf, nx_psf, bs]
        image_map : array_like
            Observed image, shape [ny, nx]
        noise_map : array_like
            Noise map, shape [ny, nx]
        bs : int
            Batch size (static)
        n_lens_light : int
            Number of lens components (static)
        n_src : int
            Number of source components (static)

        Returns
        -------
        img : array_like
            Final image, shape [ny, nx, bs]
        X_vec : array_like
            Intensity values, shape [n_lens+n_src, bs]
        """
        # Prepare linear system
        A_mat, D_mat = prepare_linear_system(
            img_lens_sub, img_arc_sub, psf_kernel,
            image_map, noise_map,
            self.sim_config.nsub, bs, n_lens_light, n_src,
            bin_image_general, jsp.signal.fftconvolve
        )

        # Solve linear system
        X_vec, _ = self.linear_solver.solve(A_mat, D_mat)

        # Reconstruct image with solved intensities
        # First, bin and convolve each component
        psf_kernel_lens = jnp.repeat(psf_kernel, n_lens_light, axis=-1)
        psf_kernel_src = jnp.repeat(psf_kernel, n_src, axis=-1)

        img_lens_sub_flat = jnp.reshape(img_lens_sub,
                                        (img_lens_sub.shape[0], img_lens_sub.shape[1], -1))
        img_arc_sub_flat = jnp.reshape(img_arc_sub,
                                       (img_arc_sub.shape[0], img_arc_sub.shape[1], -1))

        img_lens = bin_image_general(img_lens_sub_flat, self.sim_config.nsub)
        img_lens = jsp.signal.fftconvolve(img_lens, psf_kernel_lens, mode='same', axes=(0, 1))

        img_arc = bin_image_general(img_arc_sub_flat, self.sim_config.nsub)
        img_arc = jsp.signal.fftconvolve(img_arc, psf_kernel_src, mode='same', axes=(0, 1))

        # Reshape back
        img_lens = jnp.reshape(img_lens, (img_lens.shape[0], img_lens.shape[1], bs, n_lens_light))
        img_arc = jnp.reshape(img_arc, (img_arc.shape[0], img_arc.shape[1], bs, n_src))

        # Concatenate and apply intensities
        img = jnp.concatenate([img_arc, img_lens], axis=-1)  # [ny, nx, bs, n_total]
        img = jnp.reshape(img, (-1, bs, n_src + n_lens_light))  # [ny*nx, bs, n_total]
        img = jnp.transpose(img, (0, 2, 1))  # [ny*nx, n_total, bs]

        # Apply intensities
        img = jnp.einsum('ijl,jl->il', img, X_vec)  # [ny*nx, bs]
        img = jnp.reshape(img, (self.sim_config.npix, self.sim_config.npix, bs))

        # Squeeze if single batch
        if bs == 1:
            X_vec = jnp.squeeze(X_vec, axis=-1)

        return img, X_vec

    def __repr__(self):
        return (f"LensSimulator("
                f"n_mass={len(self.phys_model.lens_mass)}, "
                f"n_src={len(self.phys_model.source_light)}, "
                f"n_lens={len(self.phys_model.lens_light)}, "
                f"solver={self.solver_type})")
