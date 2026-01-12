"""
Pixelized source model for gravitational lensing.

This module provides the PixelizedSourceModel class that represents a pixelized
source reconstruction approach, where the source is represented by discrete pixels
rather than parametric profiles.
"""

import functools
import caskade as ck
import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Optional, Dict, Any

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.utils.lensing import regularization_matrix_gp_from


class PixelizedSourceModel(ck.Module):
    """
    Pixelized source model for gravitational lensing.
    
    This class represents a source galaxy using a pixelized reconstruction approach,
    where the source is represented by discrete pixels in the source plane rather than
    parametric light profiles. The source reconstruction is regularized using Gaussian
    Process priors.
    
    This model is designed to work alongside mass models in a composite physical model,
    similar to how parametric source light profiles work in TinyLensGpu.
    
    Parameters
    ----------
    reg_scale : float
        Regularization scale parameter (length scale for GP kernel)
    reg_coefficient : float
        Regularization strength coefficient
    reg_type : str
        Type of regularization kernel: 'exp', 'gauss', 'matern32', or 'matern52'
    n_source_points : int
        Number of source mesh points
    mesh_alpha : float
        Density bias exponent for source mesh sampling (>1 favors bright areas)
    mesh_blur_sigma : float
        Gaussian blur sigma for source mesh sampling (in pixels)
    mesh_method : str
        Sampling method: 'random' or 'sobol'
    mesh_seed : int
        Random seed for source mesh generation
    k_neighbors : int
        Number of nearest neighbors for interpolation
    interp_kernel : str
        Interpolation kernel: 'wendland_c2', 'wendland_c4', or 'wendland_c6'
    radius_scale : float
        Scale factor for interpolation kernel support radius
    
    Examples
    --------
    >>> from TinyLensGpu.PhysicalModel import PixelizedSourceModel
    >>> 
    >>> # Create pixelized source model
    >>> pix_src = PixelizedSourceModel(
    ...     reg_scale=0.05,
    ...     reg_coefficient=1.0,
    ...     reg_type='exp',
    ...     n_source_points=1500
    ... )
    
    Notes
    -----
    Unlike parametric source models, this model does not have a `light()` method.
    Instead, it works with the PixelizedImageProbModel which handles the source
    reconstruction and evidence calculation.
    """
    
    def __init__(
        self,
        reg_scale: float = 0.05,
        reg_coefficient: float = 1.0,
        reg_type: str = 'exp',
        n_source_points: int = 1500,
        mesh_alpha: float = 0.0,
        mesh_blur_sigma: float = 0.0,
        mesh_method: str = 'random',
        mesh_seed: int = 42,
        k_neighbors: int = 5,
        interp_kernel: str = 'wendland_c4',
        radius_scale: float = 1.5,
    ) -> None:
        super().__init__()
        
        self.reg_scale = reg_scale if isinstance(reg_scale, ParamU) else ParamU("reg_scale", reg_scale)
        self.reg_coefficient = reg_coefficient if isinstance(reg_coefficient, ParamU) else ParamU("reg_coefficient", reg_coefficient)
        
        object.__setattr__(self, '_reg_type', reg_type)
        object.__setattr__(self, '_n_source_points', n_source_points)
        object.__setattr__(self, '_mesh_alpha', mesh_alpha)
        object.__setattr__(self, '_mesh_blur_sigma', mesh_blur_sigma)
        object.__setattr__(self, '_mesh_method', mesh_method)
        object.__setattr__(self, '_mesh_seed', mesh_seed)
        object.__setattr__(self, '_k_neighbors', k_neighbors)
        object.__setattr__(self, '_interp_kernel', interp_kernel)
        object.__setattr__(self, '_radius_scale', radius_scale)
    
    @property
    def reg_type(self) -> str:
        return self._reg_type
    
    @property
    def n_source_points(self) -> int:
        return self._n_source_points
    
    @property
    def mesh_alpha(self) -> float:
        return self._mesh_alpha
    
    @property
    def mesh_blur_sigma(self) -> float:
        return self._mesh_blur_sigma
    
    @property
    def mesh_method(self) -> str:
        return self._mesh_method
    
    @property
    def mesh_seed(self) -> int:
        return self._mesh_seed
    
    @property
    def k_neighbors(self) -> int:
        return self._k_neighbors
    
    @property
    def interp_kernel(self) -> str:
        return self._interp_kernel
    
    @property
    def radius_scale(self) -> float:
        return self._radius_scale
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as a dictionary."""
        return {
            'reg_scale': float(self.reg_scale.value),
            'reg_coefficient': float(self.reg_coefficient.value),
            'reg_type': self.reg_type,
            'n_source_points': self.n_source_points,
            'mesh_alpha': self.mesh_alpha,
            'mesh_blur_sigma': self.mesh_blur_sigma,
            'mesh_method': self.mesh_method,
            'mesh_seed': self.mesh_seed,
            'k_neighbors': self.k_neighbors,
            'interp_kernel': self.interp_kernel,
            'radius_scale': self.radius_scale,
        }
    

    @ck.forward
    @functools.partial(jit, static_argnums=(0,))
    def regularization_matrix(
        self,
        points: jnp.ndarray,
        reg_scale: Optional[float] = None,
        reg_coefficient: Optional[float] = None,
        reg_type: Optional[str] = None,
    ) -> jnp.ndarray:
        scale = reg_scale if reg_scale is not None else self.reg_scale.value
        coefficient = (
            reg_coefficient if reg_coefficient is not None else self.reg_coefficient.value
        )
        kernel_type = reg_type if reg_type is not None else self.reg_type

        return regularization_matrix_gp_from(
            scale=scale,
            coefficient=coefficient,
            points=points,
            reg_type=kernel_type,
        )

    def __repr__(self) -> str:
        config_dict = self.get_config_dict()
        return (f"PixelizedSourceModel("
                f"reg_scale={config_dict['reg_scale']:.3f}, "
                f"reg_coefficient={config_dict['reg_coefficient']:.2f}, "
                f"n_source_points={config_dict['n_source_points']})")
