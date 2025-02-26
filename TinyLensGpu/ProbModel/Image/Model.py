from jax import jit
from jax import numpy as jnp
from abc import ABC, abstractmethod
from TinyLensGpu.Simulator.Image.Simulator import SimulatorConfig
from TinyLensGpu.Simulator.Image.Simulator import PhysicalModel
from TinyLensGpu.Simulator.Image import Simulator
import jax
import numpy as np
import functools
from typing import List, Dict, Tuple, Optional

class ImageProbModel(ABC):
    def __init__(
        self,
        image_data: np.ndarray,
        noise_map: np.ndarray,
        psf_kernel: np.ndarray,
        dpix: float,
        nsub: int,
        use_linear: bool,
        phys_model: PhysicalModel,
        mask: Optional[np.ndarray] = None,
        solver_type: str = 'nnls',
    ):
        self.image_data = jnp.array(image_data)
        self.noise_map = jnp.array(noise_map)
        sim_config = SimulatorConfig(
            dpix=dpix,
            npix=image_data.shape[0],
            psf_kernel=psf_kernel,
            nsub=nsub,
            mask=mask,
        )
        self.sim_obj = Simulator.LensSimulator(
            phys_model,
            sim_config,
            solver_type=solver_type,
        )
        self.use_linear = use_linear
        self.unmask = jnp.array(~sim_config.mask)

    @functools.partial(jit, static_argnums=(0,2))
    def forward_model(
        self, 
        param_dict, 
        bs: int = 1, 
        image_map: np.ndarray = None, 
        noise_map: np.ndarray = None, 
        xgrid_sub: np.ndarray = None, 
        ygrid_sub: np.ndarray = None, 
        psf_kernel: np.ndarray = None
    ):
        return self.sim_obj.simulate(
            param_dict, 
            bs=bs, 
            use_linear=self.use_linear, 
            return_intensity=False, 
            image_map=image_map, 
            noise_map=noise_map,
            xgrid_sub=xgrid_sub,
            ygrid_sub=ygrid_sub,
            psf_kernel=psf_kernel,
        )


    @functools.partial(jit, static_argnums=(0,5))
    def _likelihood_helper(self, image_model, image_data, noise_map, unmask, bs=1):
        if bs > 1:
            image_data = jnp.repeat(image_data[..., jnp.newaxis], bs, axis=-1)
            noise_map = jnp.repeat(noise_map[..., jnp.newaxis], bs, axis=-1)
            unmask = jnp.repeat(unmask[..., jnp.newaxis], bs, axis=-1)
        chi2_image = (image_model-image_data)**2/noise_map**2
        chi2_image = chi2_image*unmask
        return -0.5*jnp.sum(chi2_image, axis=(0, 1)) #Sum pixels of non-batch axis if batch dimension exists


    def likelihood(self, param_dict, bs=1, debug=True):
        image_model = self.forward_model(
            param_dict, 
            bs=bs, 
            image_map=self.image_data, 
            noise_map=self.noise_map,
            xgrid_sub=self.sim_obj.xgrid_sub,
            ygrid_sub=self.sim_obj.ygrid_sub,
            psf_kernel=self.sim_obj.psf_kernel,
        )
        like =  self._likelihood_helper(
            image_model, 
            self.image_data, 
            self.noise_map, 
            self.unmask, 
            bs,
        )
        like = np.asarray(like)

        if debug:
            #check if there is nan in like
            if np.isnan(like).any():
                import warnings
                warnings.warn("NaN detected in likelihood calculation")
                # print("Parameters that caused NaN:")
                # id_nan = np.where(np.isnan(like))[0]
                # print(param_dict)
                return -np.inf
            #check if there is inf in like
            if np.isinf(like).any():
                import warnings
                warnings.warn("Inf detected in likelihood calculation")
                # print("Parameters that caused Inf:")
                # id_inf = np.where(np.isinf(like))[0]
                # print(param_dict)
                return -np.inf

        return like
