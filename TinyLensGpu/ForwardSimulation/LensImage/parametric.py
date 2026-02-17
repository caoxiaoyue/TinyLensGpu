"""
Gravitational lens simulator  models.

This module provides the main simulation engine for gravitational lensing,
handling ray-tracing, PSF convolution, and linear parameter solving.
"""

import functools
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, Array
from typing import Optional, Tuple, Union
import numpy as np

from ...PhysicalModel.LensImage.composite import PhysicalModel
from .config import SimulatorConfig
from ...utils.linear_solver import LinearSolver, prepare_linear_system


def bin_image_general(img: Array, nsub: int) -> Array:
    """
    Bin an image by averaging blocks of pixels.

    Parameters
    ----------
    img : Array
        Image array to bin.
    nsub : int
        Subsampling factor for binning operation.

    Returns
    -------
    img_binned : Array
        Binned image array.
    """
    if nsub == 1:
        return img

    if img.ndim == 2:
        ny, nx = img.shape
        ny_bin, nx_bin = ny // nsub, nx // nsub
        img_reshaped = img.reshape(ny_bin, nsub, nx_bin, nsub)
        img_binned = jnp.mean(img_reshaped, axis=(1, 3))
    else:
        ny, nx = img.shape[:2]
        ny_bin, nx_bin = ny // nsub, nx // nsub
        other_dims = img.shape[2:]

        img_reshaped = img.reshape(ny_bin, nsub, nx_bin, nsub, *other_dims)
        img_binned = jnp.mean(img_reshaped, axis=(1, 3))

    return img_binned


class LensSimulator:
    """
    Forward simulator for parametric lens/source light models.

    The simulator computes ray tracing, evaluates parametric light profiles,
    applies PSF convolution, and optionally solves linear intensity coefficients
    using NNLS or normal-equation solvers.

    Parameters
    ----------
    phys_model : PhysicalModel
        Physical model containing lens mass, lens light, and source light profiles.
    sim_config : SimulatorConfig
        Configuration for grid, PSF, sub-sampling, and masks.
    solver_type : str, optional
        Linear solver type for intensity parameters ('nnls' or 'normal'), by default 'nnls'.

    """

    def __init__(
        self,
        phys_model: PhysicalModel,
        sim_config: SimulatorConfig,
        solver_type: str = 'nnls'
    ) -> None:
        """
        Initialize the LensSimulator for parametric models.

        Parameters
        ----------
        phys_model : PhysicalModel
            Physical model containing lens mass, lens light, and source light profiles.
        sim_config : SimulatorConfig
            Configuration for grid, PSF, sub-sampling, and masks.
        solver_type : str, optional
            Linear solver type for intensity parameters ('nnls' for non-negative least squares,
            'normal' for standard least squares), by default 'nnls'.
        
        Raises
        ------
        ValueError
            If solver_type is not 'nnls' or 'normal'.
        """
        self.phys_model = phys_model
        self.sim_config = sim_config
        self.solver_type = solver_type

        if self.solver_type not in ['nnls', 'normal']:
            raise ValueError("solver_type must be either 'nnls' or 'normal'")

        self.linear_solver = LinearSolver(solver_type)

        self.xgrid_sub = jnp.array(self.sim_config.xgrid_sub_1d)
        self.ygrid_sub = jnp.array(self.sim_config.ygrid_sub_1d)

        self.psf_kernel = jnp.array(self.sim_config.psf_kernel)

    def _restore_2d_from_1d(self, img_1d: Array) -> Array:
        """
        Restore masked/unmasked flattened arrays to sub-grid image shape.

        Parameters
        ----------
        img_1d : Array
            1D image array to be restored to 2D.

        Returns
        -------
        img_2d : Array
            Restored 2D image array.
        """
        shape = self.sim_config.subgrid_shape
        flat_indices = self.sim_config.flat_indices

        N, M = shape
        n_pixels = N * M

        if flat_indices.shape[0] == n_pixels:
            if img_1d.ndim > 1:
                n_channels = img_1d.shape[-1]
                return img_1d.reshape(N, M, n_channels)
            else:
                return img_1d.reshape(N, M)

        if img_1d.ndim > 1:
            n_channels = img_1d.shape[-1]
            flat_img = jnp.zeros((n_pixels, n_channels), dtype=img_1d.dtype)
            flat_img = flat_img.at[flat_indices].set(img_1d)
            return flat_img.reshape(N, M, n_channels)
        else:
            flat_img = jnp.zeros(n_pixels, dtype=img_1d.dtype)
            flat_img = flat_img.at[flat_indices].set(img_1d)
            return flat_img.reshape(N, M)

    def simulate(
        self,
        use_linear: bool = False,
        return_intensity: bool = False,
        ret_each_plane: bool = False,
        image_map: Optional[Array] = None,
        noise_map: Optional[Array] = None,
        xgrid_sub: Optional[Array] = None,
        ygrid_sub: Optional[Array] = None,
        psf_kernel: Optional[Array] = None,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, Array, Array],
    ]:
        """
        Simulate the lensed image.

        Parameters
        ----------
        use_linear : bool, optional
            If True, solve for linear intensity parameters using the provided data.
            If False, use the intensity parameters currently in the physical model.
            By default False.
        return_intensity : bool, optional
            If True, return the solved intensity vector along with the image.
            Only relevant if use_linear=True (or technically if you want the intensities used).
            By default False.
        ret_each_plane : bool, optional
            If True, return separate images for the source plane and lens plane.
            By default False.
        image_map : Optional[Array], optional
            Observed image data, required if use_linear=True.
        noise_map : Optional[Array], optional
            Noise map (sigma), required if use_linear=True.
        xgrid_sub : Optional[Array], optional
            Custom sub-grid x-coordinates.
        ygrid_sub : Optional[Array], optional
            Custom sub-grid y-coordinates.
        psf_kernel : Optional[Array], optional
            Custom PSF kernel.

        Returns
        -------
        Union[Array, Tuple[Array, Array], Tuple[Array, Array, Array]]
            - If ret_each_plane=False, return_intensity=False:
              Returns `image` (2D array).
            - If ret_each_plane=False, return_intensity=True:
              Returns `(image, intensity_vector)`.
            - If ret_each_plane=True, return_intensity=False:
              Returns `(source_plane_image, lens_plane_image)`.
            - If ret_each_plane=True, return_intensity=True:
              Returns `(source_plane_image, lens_plane_image, intensity_vector)`.
        
        Raises
        ------
        ValueError
            If use_linear is True but image_map or noise_map are missing.
        """
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

        n_src = len(self.phys_model.source_light)
        n_lens_light = len(self.phys_model.lens_light)
        n_lens_mass = len(self.phys_model.lens_mass)

        img_lens_sub, img_arc_sub = self._generate_ideal_model(
            xgrid_sub, ygrid_sub, n_src, n_lens_light, n_lens_mass
        )

        if not use_linear:
            sim_out = self._simulate_nonlinear(
                img_lens_sub, img_arc_sub, psf_kernel, ret_each_plane=ret_each_plane
            )
        else:
            if image_map is None or noise_map is None:
                raise ValueError("image_map and noise_map required for linear simulation")

            sim_out = self._simulate_linear(
                img_lens_sub, img_arc_sub, psf_kernel,
                image_map, noise_map, n_lens_light, n_src,
                ret_each_plane=ret_each_plane
            )

        if ret_each_plane:
            img_arc, img_lens, X_vec = sim_out
            if return_intensity:
                return img_arc, img_lens, X_vec
            return img_arc, img_lens

        img, X_vec = sim_out

        if return_intensity:
            return img, X_vec
        else:
            return img

    def _generate_ideal_model(
        self,
        xgrid_sub: Array,
        ygrid_sub: Array,
        n_src: int,
        n_lens_light: int,
        n_lens_mass: int,
    ) -> Tuple[Array, Array]:
        """
        Generate ideal (unconvolved, noiseless) model images on the sub-grid.

        Parameters
        ----------
        xgrid_sub : Array
            Sub-grid x-coordinates.
        ygrid_sub : Array
            Sub-grid y-coordinates.
        n_src : int
            Number of source light profiles.
        n_lens_light : int
            Number of lens light profiles.
        n_lens_mass : int
            Number of lens mass profiles.

        Returns
        -------
        img_lens_sub : Array
            Lens light image on the sub-grid.
        img_arc_sub : Array
            Lensed source (arc) image on the sub-grid.
        """
        img_sub = jnp.zeros_like(xgrid_sub)
        img_arc_sub = jnp.repeat(img_sub[..., jnp.newaxis], n_src, axis=-1)
        img_lens_sub = jnp.repeat(img_sub[..., jnp.newaxis], n_lens_light, axis=-1)

        if n_lens_mass > 0:
            beta_x, beta_y = self.phys_model.deflection(x=xgrid_sub, y=ygrid_sub)
        else:
            beta_x, beta_y = xgrid_sub, ygrid_sub

        if n_src > 0:
            for i, light_model in enumerate(self.phys_model.source_light):
                img_arc_sub = img_arc_sub.at[..., i].set(
                    light_model.light(x=beta_x, y=beta_y)
                )

        if n_lens_light > 0:
            for i, light_model in enumerate(self.phys_model.lens_light):
                img_lens_sub = img_lens_sub.at[..., i].set(
                    light_model.light(x=xgrid_sub, y=ygrid_sub)
                )

        return img_lens_sub, img_arc_sub

    @functools.partial(jit, static_argnums=(0, 4))
    def _simulate_nonlinear(
        self,
        img_lens_sub: Array,
        img_arc_sub: Array,
        psf_kernel: Array,
        ret_each_plane: bool = False,
    ) -> Union[Tuple[Array, None], Tuple[Array, Array, None]]:
        """
        Simulate image for non-linear optimization (no linear intensity solving).

        Parameters
        ----------
        img_lens_sub : Array
            Lens light image on the sub-grid.
        img_arc_sub : Array
            Lensed source (arc) image on the sub-grid.
        psf_kernel : Array
            PSF kernel for convolution.
        ret_each_plane : bool, optional
            Whether to return separate images for lens and source planes.

        Returns
        -------
        Union[Tuple[Array, None], Tuple[Array, Array, None]]
            - (image, None) if ret_each_plane=False
            - (source_plane_image, lens_plane_image, None) if ret_each_plane=True
        """
        img_lens_sub = self._restore_2d_from_1d(img_lens_sub)
        img_arc_sub = self._restore_2d_from_1d(img_arc_sub)

        if not ret_each_plane:
            img_sub = jnp.sum(img_lens_sub, axis=-1) + jnp.sum(img_arc_sub, axis=-1)
            img = bin_image_general(img_sub, self.sim_config.nsub)
            img = jsp.signal.fftconvolve(img, psf_kernel, mode='same')
            return img, None

        img_lens = bin_image_general(jnp.sum(img_lens_sub, axis=-1), self.sim_config.nsub)
        img_arc = bin_image_general(jnp.sum(img_arc_sub, axis=-1), self.sim_config.nsub)

        img_lens = jsp.signal.fftconvolve(img_lens, psf_kernel, mode='same')
        img_arc = jsp.signal.fftconvolve(img_arc, psf_kernel, mode='same')

        return img_arc, img_lens, None

    @functools.partial(jit, static_argnums=(0, 6, 7, 8))
    def _simulate_linear(
        self,
        img_lens_sub: Array,
        img_arc_sub: Array,
        psf_kernel: Array,
        image_map: Array,
        noise_map: Array,
        n_lens_light: int,
        n_src: int,
        ret_each_plane: bool = False,
    ) -> Union[Tuple[Array, Array], Tuple[Array, Array, Array]]:
        """
        Simulate image with linear intensity parameter solving.

        Parameters
        ----------
        img_lens_sub : Array
            Lens light image on the sub-grid.
        img_arc_sub : Array
            Lensed source (arc) image on the sub-grid.
        psf_kernel : Array
            PSF kernel for convolution.
        image_map : Array
            Observed image data (for linear solving).
        noise_map : Array
            Noise map (sigma) (for linear solving).
        n_lens_light : int
            Number of lens light profiles.
        n_src : int
            Number of source light profiles.
        ret_each_plane : bool, optional
            Whether to return separate images for lens and source planes.

        Returns
        -------
        Union[Tuple[Array, Array], Tuple[Array, Array, Array]]
            - (image, intensity_vector) if ret_each_plane=False
            - (source_plane_image, lens_plane_image, intensity_vector) if ret_each_plane=True
        """
        img_lens_sub = self._restore_2d_from_1d(img_lens_sub)
        img_arc_sub = self._restore_2d_from_1d(img_arc_sub)

        A_mat, D_vec = prepare_linear_system(
            img_lens_sub, img_arc_sub, psf_kernel,
            image_map, noise_map,
            self.sim_config.nsub, n_lens_light, n_src,
            bin_image_general, jsp.signal.fftconvolve
        )

        X_vec, _ = self.linear_solver.solve(A_mat, D_vec)

        img_lens = bin_image_general(img_lens_sub, self.sim_config.nsub)
        img_arc = bin_image_general(img_arc_sub, self.sim_config.nsub)

        # Vectorized convolution using vmap
        def convolve_func(x):
            """
            Convolve one 2D component image with the PSF kernel.

            Parameters
            ----------
            x : Array
                Input 2D image.

            Returns
            -------
            convolved_x : Array
                Convolved 2D image.
            """
            return jsp.signal.fftconvolve(x, psf_kernel, mode='same')

        if n_lens_light > 0:
            img_lens_convolved = jax.vmap(convolve_func, in_axes=-1, out_axes=-1)(img_lens)
        else:
            img_lens_convolved = jnp.zeros((img_lens.shape[0], img_lens.shape[1], 0))

        if n_src > 0:
            img_arc_convolved = jax.vmap(convolve_func, in_axes=-1, out_axes=-1)(img_arc)
        else:
            img_arc_convolved = jnp.zeros((img_arc.shape[0], img_arc.shape[1], 0))

        img_components = jnp.concatenate([img_arc_convolved, img_lens_convolved], axis=-1)
        img = jnp.einsum('ijk,k->ij', img_components, X_vec)

        if ret_each_plane:
            img_arc_sum = (
                jnp.einsum('ijk,k->ij', img_arc_convolved, X_vec[:n_src])
                if n_src > 0 else jnp.zeros(img.shape, dtype=img.dtype)
            )
            img_lens_sum = (
                jnp.einsum('ijk,k->ij', img_lens_convolved, X_vec[n_src:])
                if n_lens_light > 0 else jnp.zeros(img.shape, dtype=img.dtype)
            )
            return img_arc_sum, img_lens_sum, X_vec

        return img, X_vec

    def __repr__(self) -> str:
        """
        Return concise simulator summary for logging/debugging.

        Returns
        -------
        str
            Summary including component counts and selected linear solver.
        """
        return (f"LensSimulator("
                f"n_mass={len(self.phys_model.lens_mass)}, "
                f"n_src={len(self.phys_model.source_light)}, "
                f"n_lens={len(self.phys_model.lens_light)}, "
                f"solver={self.solver_type})")
