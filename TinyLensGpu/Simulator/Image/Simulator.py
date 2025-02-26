import functools
from typing import List, Dict, Tuple, Optional
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import jit, vmap
from abc import ABC
from TinyLensGpu.Simulator.Image import util
import TinyLensGpu.Profile.ProfileBase as ProfileBase
from .fnnls import fnnls_vec

class PhysicalModel:
    def __init__(
            self,
            lens_mass: List[ProfileBase.MassProfile],
            source_light: List[ProfileBase.LightProfile],
            lens_light: List[ProfileBase.LightProfile],
    ):
        self.lens_mass = lens_mass
        self.source_light = source_light
        self.lens_light = lens_light


class SimulatorConfig:
    def __init__(
            self, 
            dpix: float, 
            npix: int, 
            psf_kernel: np.ndarray = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), 
            nsub: Optional[int] = 1,
            mask: Optional[np.ndarray] = None,
    ):
        self.dpix = dpix
        self.npix = npix
        self.psf_kernel = psf_kernel
        self.nsub = nsub
        if mask is None:
            mask = np.zeros((npix, npix))
        self.mask = mask.astype('bool')
        self.xgrid, self.ygrid, self.xgrid_sub, self.ygrid_sub = self.get_coords(self.npix, self.dpix, self.nsub)


    @staticmethod
    def get_coords(npix, dpix, nsub=1):
        """Generate coordinate grids for image simulation.
        
        Args:
            npix: Number of pixels per side
            dpix: Pixel scale
            nsub: Subsampling factor
            
        Returns:
            Tuple of (xgrid, ygrid, xgrid_sub, ygrid_sub) as JAX arrays
        """
        xgrid, ygrid = util.make_grid_2d(npix, dpix, 1)
        xgrid_sub, ygrid_sub = util.make_grid_2d(npix, dpix, nsub)
        return xgrid, ygrid, xgrid_sub, ygrid_sub


@jit
def solve_linear_vec(A, b):
    """Vectorized least squares solver using pseudoinverse.
    
    Args:
        A: Matrix of shape [m, n, bs] where m >= n.
        b: Vector of shape [m, bs].
        
    Returns:
        x: Solution vector of shape [n, bs].
    """
    ATA = jnp.einsum('ijl,jkl->ikl', jnp.transpose(A, (1, 0, 2)), A) #shape: [n, n, bs]
    ATA = jnp.transpose(ATA, (2, 0, 1)) #shape: [bs, n, n]
    ATA_inv = jnp.linalg.pinv(ATA, rcond=1e-6) #shape: [bs, n, n]
    ATA_inv = jnp.transpose(ATA_inv, (1, 2, 0)) #shape: [n, n, bs]
    x = jnp.einsum(
        'ijl, jl->il',
        jnp.einsum('ijl, jkl->ikl', ATA_inv, jnp.transpose(A, (1, 0, 2))), #shape: [n, m, bs]
        b, #shape: [m, bs]
    )
    return x


class LensSimulator(ABC):
    def __init__(
            self,
            phys_model: PhysicalModel,
            sim_config: SimulatorConfig,
            solver_type: str = 'nnls'  # Add solver type parameter
    ):
        self.phys_model = phys_model
        self.sim_config = sim_config
        self.solver_type = solver_type
        if self.solver_type not in ['nnls', 'normal']:
            raise ValueError("solver_type must be either 'nnls' or 'normal'")
        self.xgrid_sub = jnp.array(self.sim_config.xgrid_sub)
        self.ygrid_sub = jnp.array(self.sim_config.ygrid_sub)
        self.psf_kernel = jnp.array(self.sim_config.psf_kernel)


    @functools.partial(jit, static_argnums=(0,))
    def _beta(self, xgrid: jnp.ndarray, ygrid: jnp.ndarray, lens_params: List[Dict]):
        # https://github.com/google/jax/pull/4717
        beta_x, beta_y = xgrid, ygrid
        for lens, p in zip(self.phys_model.lens_mass, lens_params):
            alpha_x, alpha_y = lens.deriv(xgrid, ygrid, **p)
            beta_x, beta_y = beta_x - alpha_x, beta_y - alpha_y
        return beta_x, beta_y

    @functools.partial(jit, static_argnums=(0,))
    def _simulate_nonlinear(
        self,
        img_lens_sub: jnp.ndarray,  #shape: [n_y_pix_sub, n_x_pix_sub, bs, n_lens_light]
        img_arc_sub: jnp.ndarray,  #shape: [n_y_pix_sub, n_x_pix_sub, bs, n_src]
        psf_kernel: jnp.ndarray,  #shape: [ny_psf, nx_psf, bs]
    ):
        """Helper method for non-linear simulation case."""
        img_sub = jnp.sum(img_lens_sub, axis=-1) + jnp.sum(img_arc_sub, axis=-1)  #shape: [n_y_pix_sub, n_x_pix_sub, bs]
        img = util.bin_image_general(img_sub, self.sim_config.nsub)
        img = jsp.signal.fftconvolve(img, psf_kernel, mode='same', axes=(0, 1))  #only convolve along non-batch axis; shape: [n_y_pix, n_x_pix, bs]
        return img, None

    @functools.partial(jit, static_argnums=(0,6,7,8))
    def _simulate_linear(
        self,
        img_lens_sub: jnp.ndarray,  #shape: [n_y_pix_sub, n_x_pix_sub, bs, n_lens_light]
        img_arc_sub: jnp.ndarray,  #shape: [n_y_pix_sub, n_x_pix_sub, bs, n_src]
        psf_kernel: jnp.ndarray,  #shape: [ny_psf, nx_psf, bs]
        image_map: jnp.ndarray,
        noise_map: jnp.ndarray,
        bs: int,  # Now static
        n_lens_light: int,  # Now static
        n_src: int,  # Now static
    ):
        """Helper method for linear simulation case."""
        img_1d = jnp.ravel(image_map)  #shape: [n_y_pix*n_x_pix]
        n_1d = jnp.ravel(noise_map)  #shape: [n_y_pix*n_x_pix]
        snr_1d = img_1d/n_1d

        psf_kernel_lens = jnp.repeat(psf_kernel, n_lens_light, axis=-1)  #shape: [ny_psf, nx_psf, n_lens_light*bs]
        psf_kernel_src = jnp.repeat(psf_kernel, n_src, axis=-1)  #shape: [ny_psf, nx_psf, n_src*bs]
        img_lens_sub = jnp.reshape(img_lens_sub, (img_lens_sub.shape[0], img_lens_sub.shape[1], -1))  #shape: [n_y_pix_sub, n_x_pix_sub, bs, n_lens_light] -> [n_y_pix_sub, n_x_pix_sub, n_lens_light*bs]
        img_arc_sub = jnp.reshape(img_arc_sub, (img_arc_sub.shape[0], img_arc_sub.shape[1], -1))  #shape: [n_y_pix_sub, n_x_pix_sub, bs, n_src] -> [n_y_pix_sub, n_x_pix_sub, n_src*bs]
        img_lens = util.bin_image_general(img_lens_sub, self.sim_config.nsub)  #shape: [n_y_pix, n_x_pix, n_lens_light*bs]
        img_lens = jsp.signal.fftconvolve(img_lens, psf_kernel_lens, mode='same', axes=(0, 1))  #shape: [n_y_pix, n_x_pix, n_lens_light*bs]
        img_arc = util.bin_image_general(img_arc_sub, self.sim_config.nsub)  #shape: [n_y_pix, n_x_pix, n_src*bs]
        img_arc = jsp.signal.fftconvolve(img_arc, psf_kernel_src, mode='same', axes=(0, 1))  #shape: [n_y_pix, n_x_pix, n_src*bs]
        img_lens = jnp.reshape(img_lens, (img_lens.shape[0], img_lens.shape[1], bs, n_lens_light))
        img_arc = jnp.reshape(img_arc, (img_arc.shape[0], img_arc.shape[1], bs, n_src))
        img = jnp.concatenate([img_arc, img_lens], axis=-1)  #shape: [n_y_pix, n_x_pix, bs, n_lens_light+n_src]
        img = jnp.reshape(img, (-1, bs, n_lens_light+n_src))  #shape: [n_y_pix*n_x_pix, bs, n_lens_light+n_src]

        #non-negative least square fitting, |AX-D|^2 subject to X>=0
        img = jnp.transpose(img, (0, 2, 1))  #shape: [n_y_pix*n_x_pix, n_lens_light+n_src, bs]
        D_mat = jnp.repeat(snr_1d[..., jnp.newaxis], bs, axis=-1)  #shape: [n_y_pix*n_x_pix, bs]
        A_mat = img / n_1d[:, jnp.newaxis, jnp.newaxis]  #shape: [n_y_pix*n_x_pix, n_lens_light+n_src, bs]
        #a small regularization term is added to stabilize the solution; see https://arxiv.org/pdf/2403.16253 eq.15
        Reg_mat = jnp.eye(n_lens_light+n_src) * 0.001  #shape: [n_lens_light+n_src, n_lens_light+n_src]
        Reg_mat = jnp.repeat(Reg_mat[..., jnp.newaxis], bs, axis=-1)  #shape: [n_lens_light+n_src, n_lens_light+n_src, bs]
        A_mat = jnp.concatenate([A_mat, Reg_mat], axis=0)  #shape: [n_y_pix*n_x_pix+n_lens_light+n_src, n_lens_light+n_src, bs]
        D_mat = jnp.concatenate([D_mat, jnp.zeros((n_lens_light+n_src, bs))], axis=0)  #shape: [n_y_pix*n_x_pix+n_lens_light+n_src, bs]
        
        # Use the configured solver type
        if self.solver_type == 'nnls':
            X_vec, _ = fnnls_vec(A_mat, D_mat)
        else:  # normal least squares
            # jax.debug.print("Solving normal least squares")
            X_vec = solve_linear_vec(A_mat, D_mat)
            
        img = jnp.einsum('ijl,jl->il', img, X_vec)  #shape: [n_y_pix*n_x_pix, bs]
        img = jnp.reshape(img, (self.sim_config.npix, self.sim_config.npix, bs))  #shape: [n_y_pix, n_x_pix, bs]

        if bs == 1:
            X_vec = jnp.squeeze(X_vec, axis=-1)
            
        return img, X_vec

    @functools.partial(jit, static_argnums=(0,4,5,6))
    def _generate_ideal_model(
        self,
        xgrid_sub: jnp.ndarray,
        ygrid_sub: jnp.ndarray,
        params: List[List[Dict]],
        n_src: int,
        n_lens_light: int,
        n_lens_mass: int,
    ):
        """Helper method to generate ideal model image before PSF convolution and binning.
        
        Args:
            xgrid_sub: Subsampled x coordinates [n_y_pix_sub, n_x_pix_sub, bs]
            ygrid_sub: Subsampled y coordinates [n_y_pix_sub, n_x_pix_sub, bs]
            params: Physical parameters for lens mass, source light and lens light
            n_src: Number of source light profiles
            n_lens_light: Number of lens light profiles 
            n_lens_mass: Number of lens mass profiles
        """
        img_sub = jnp.zeros_like(xgrid_sub) #shape: [n_y_pix_sub, n_x_pix_sub, bs]
        img_arc_sub = jnp.repeat(img_sub[..., jnp.newaxis], n_src, axis=-1) #shape: [n_y_pix_sub, n_x_pix_sub, bs, n_src]
        img_lens_sub = jnp.repeat(img_sub[..., jnp.newaxis], n_lens_light, axis=-1) #shape: [n_y_pix_sub, n_x_pix_sub, bs, n_lens_light]

        if n_lens_mass > 0:
            lens_mass_params = params[0]
            beta_x, beta_y = self._beta(xgrid_sub, ygrid_sub, lens_mass_params)
        else:
            beta_x, beta_y = xgrid_sub, ygrid_sub

        if n_src > 0:
            source_light_params = params[1]
            i = 0
            for lightModel, p in zip(self.phys_model.source_light, source_light_params):
                img_arc_sub = img_arc_sub.at[..., i].set(lightModel.light(beta_x, beta_y, **p))
                i += 1

        if n_lens_light > 0:
            lens_light_params = params[2]
            i = 0
            for lightModel, p in zip(self.phys_model.lens_light, lens_light_params):
                img_lens_sub = img_lens_sub.at[..., i].set(lightModel.light(xgrid_sub, ygrid_sub, **p))
                i += 1
                
        return img_lens_sub, img_arc_sub

    # @functools.partial(jit, static_argnums=(0,2,3,4))
    def simulate(
        self, 
        params: List[List[Dict]], 
        bs=1, 
        use_linear=False,
        return_intensity=False,
        image_map: np.ndarray=None, #Large size array should be dynamically allocated, instead of compiling into the jit function.
        noise_map: np.ndarray=None,
        xgrid_sub: np.ndarray=None,
        ygrid_sub: np.ndarray=None,
        psf_kernel: np.ndarray=None,
    ):
        """Simulates lensing image with physical parameters.
        
        Args:
            params: Physical parameters for lens mass, source light and lens light
            bs: Batch size
            use_linear: Whether to use linear optimization
            return_intensity: Whether to return intensity values
            image_map: Image map for linear optimization
            noise_map: Noise map for linear optimization
            xgrid_sub: Subsampled x coordinates (dynamically allocated)
            ygrid_sub: Subsampled y coordinates (dynamically allocated) 
            psf_kernel: PSF kernel (dynamically allocated)
        """
        #Large size array should be dynamically allocated, instead of compiling into the jit function.
        #otherwise, the compiling time would be longer
        xgrid_sub = jnp.repeat(xgrid_sub[..., jnp.newaxis], bs, axis=-1) #shape: [n_y_pix_sub, n_x_pix_sub, bs]
        ygrid_sub = jnp.repeat(ygrid_sub[..., jnp.newaxis], bs, axis=-1)
        psf_kernel = jnp.repeat(psf_kernel[..., jnp.newaxis], bs, axis=-1)

        n_src = len(self.phys_model.source_light)
        n_lens_light = len(self.phys_model.lens_light)
        n_lens_mass = len(self.phys_model.lens_mass)

        img_lens_sub, img_arc_sub = self._generate_ideal_model(
            xgrid_sub, ygrid_sub, params, n_src, n_lens_light, n_lens_mass
        )

        if not use_linear:
            img, X_vec = self._simulate_nonlinear(img_lens_sub, img_arc_sub, psf_kernel)
        else:
            img, X_vec = self._simulate_linear(
                img_lens_sub, img_arc_sub, psf_kernel, 
                image_map, noise_map, bs, n_lens_light, n_src
            )

        if bs == 1:
            #remove the bs axis for single image
            img = jnp.squeeze(img, axis=-1)

        if return_intensity:
            return img, X_vec
        else:
            return img  