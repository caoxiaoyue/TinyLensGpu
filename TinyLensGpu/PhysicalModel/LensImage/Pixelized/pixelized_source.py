"""
Pixelized source model for gravitational lensing.

This module provides the PixelizedSourceModel class that represents a pixelized
source reconstruction approach, where the source is represented by discrete pixels
rather than parametric profiles.
"""

import caskade as ck
import jax.numpy as jnp
from typing import Optional, Dict, Any, Tuple

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.utils.lensing import (
    regularization_matrix_gp_from,
    regularization_sparse_knn_from,
    regularization_sparse_rectangular_from,
    sparse_regularization_dense_from,
)


class PixelizedSourceModel(ck.Module):
    """
    Pixelized source model for gravitational lensing.
    
    This class represents a source galaxy using a pixelized reconstruction approach,
    where the source is represented by discrete pixels in the source plane rather than
    parametric light profiles. The source reconstruction is regularized using Gaussian
    Process priors.
    
    This model is designed to work alongside mass models in a composite physical model,
    similar to how parametric source light profiles work in TinyLensGpu.

    The model supports two source-grid layouts:

    - ``source_grid_type='irregular'``: adaptive source mesh points with GP-based
      dense regularization.
    - ``source_grid_type='rectangular_bilinear'``: regular source-plane grid with
      bilinear lens mapping and sparse stencil regularization (zero/gradient/
      curvature). Sparse rectangular regularization can be consumed directly by
      operator backends or densified downstream by matrix backends.
    
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
        reg_operator_mode: str = 'dense_gp',
        reg_sparse_k_neighbors: int = 16,
        source_grid_type: str = 'irregular',
        source_grid_nx: int = 64,
        source_grid_ny: int = 64,
        source_grid_margin_frac: float = 0.10,
        source_grid_bounds: Optional[Tuple[float, float, float, float]] = None,
        rect_reg_type: str = 'gradient',
    ) -> None:
        super().__init__()
        
        self.reg_scale = reg_scale if isinstance(reg_scale, ParamU) else ParamU("reg_scale", reg_scale)
        self.reg_coefficient = reg_coefficient if isinstance(reg_coefficient, ParamU) else ParamU("reg_coefficient", reg_coefficient)
        
        # Configuration attributes
        object.__setattr__(self, 'reg_type', reg_type)
        object.__setattr__(self, 'n_source_points', n_source_points)
        object.__setattr__(self, 'mesh_alpha', mesh_alpha)
        object.__setattr__(self, 'mesh_blur_sigma', mesh_blur_sigma)
        object.__setattr__(self, 'mesh_method', mesh_method)
        object.__setattr__(self, 'mesh_seed', mesh_seed)
        object.__setattr__(self, 'k_neighbors', k_neighbors)
        object.__setattr__(self, 'interp_kernel', interp_kernel)
        object.__setattr__(self, 'radius_scale', radius_scale)

        grid_type = str(source_grid_type).strip().lower()
        if grid_type not in {'irregular', 'rectangular_bilinear'}:
            raise ValueError(
                f"Unknown source_grid_type: '{source_grid_type}'. Must be one of {'irregular', 'rectangular_bilinear'}."
            )
        object.__setattr__(self, 'source_grid_type', grid_type)
        object.__setattr__(self, 'source_grid_nx', max(1, int(source_grid_nx)))
        object.__setattr__(self, 'source_grid_ny', max(1, int(source_grid_ny)))
        object.__setattr__(self, 'source_grid_margin_frac', float(source_grid_margin_frac))

        if source_grid_bounds is None:
            bounds_value = None
        else:
            if len(source_grid_bounds) != 4:
                raise ValueError(
                    "source_grid_bounds must be a 4-tuple: (x_min, x_max, y_min, y_max)."
                )
            bounds_value = tuple(float(v) for v in source_grid_bounds)
        object.__setattr__(self, 'source_grid_bounds', bounds_value)

        rect_reg = str(rect_reg_type).strip().lower()
        if rect_reg not in {'zero', 'gradient', 'curvature'}:
            raise ValueError(
                f"Unknown rect_reg_type: '{rect_reg_type}'. Must be one of {'zero', 'gradient', 'curvature'}."
            )
        object.__setattr__(self, 'rect_reg_type', rect_reg)

        mode = str(reg_operator_mode).strip().lower()
        if mode not in {'dense_gp', 'sparse_knn'}:
            raise ValueError(
                f"Unknown reg_operator_mode: '{reg_operator_mode}'. Must be one of {'dense_gp', 'sparse_knn'}."
            )
        object.__setattr__(self, 'reg_operator_mode', mode)
        object.__setattr__(self, 'reg_sparse_k_neighbors', max(1, int(reg_sparse_k_neighbors)))
    
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
            'reg_operator_mode': self.reg_operator_mode,
            'reg_sparse_k_neighbors': self.reg_sparse_k_neighbors,
            'source_grid_type': self.source_grid_type,
            'source_grid_nx': self.source_grid_nx,
            'source_grid_ny': self.source_grid_ny,
            'source_grid_margin_frac': self.source_grid_margin_frac,
            'source_grid_bounds': self.source_grid_bounds,
            'rect_reg_type': self.rect_reg_type,
        }

    @property
    def is_rectangular_grid(self) -> bool:
        """Return ``True`` when the model uses rectangular bilinear source-grid mode."""
        return self.source_grid_type == 'rectangular_bilinear'

    def regularization_sparse_rectangular(
        self,
        nx: int,
        ny: int,
        reg_coefficient: Optional[float] = None,
        rect_reg_type: Optional[str] = None,
    ):
        """Build sparse rectangular-grid regularization in COO form.

        Parameters
        ----------
        nx, ny : int
            Source-plane rectangular grid dimensions.
        reg_coefficient : float, optional
            Regularization coefficient override. If omitted, the model's
            ``reg_coefficient`` parameter value is used.
        rect_reg_type : str, optional
            Rectangular regularization scheme override in
            ``{'zero', 'gradient', 'curvature'}``. If omitted, the model's
            configured ``rect_reg_type`` is used.

        Returns
        -------
        tuple
            ``(rows, cols, values, n_source)`` COO entries for the sparse
            regularization operator.
        """
        coefficient = (
            reg_coefficient if reg_coefficient is not None else self.reg_coefficient.value
        )
        scheme = rect_reg_type if rect_reg_type is not None else self.rect_reg_type
        return regularization_sparse_rectangular_from(
            coefficient=coefficient,
            nx=int(nx),
            ny=int(ny),
            reg_scheme=scheme,
        )
    

    @ck.forward
    def regularization_matrix(
        self,
        points: jnp.ndarray,
        reg_scale: Optional[float] = None,
        reg_coefficient: Optional[float] = None,
        reg_type: Optional[str] = None,
        reg_operator_mode: Optional[str] = None,
        reg_sparse_k_neighbors: Optional[int] = None,
    ) -> jnp.ndarray:
        """Construct a dense source regularization matrix for irregular grids.

        This API is intentionally scoped to irregular source meshes where
        regularization depends on arbitrary source-point coordinates. For
        rectangular grids, sparse stencil regularization is the canonical
        representation and should be obtained via
        :meth:`regularization_sparse_rectangular`.

        Notes
        -----
        Matrix backend support for rectangular grids is implemented in the
        simulator inversion layer by densifying sparse rectangular COO operators.
        """
        if self.is_rectangular_grid:
            raise ValueError(
                "regularization_matrix() is not available for source_grid_type='rectangular_bilinear'. "
                "Use regularization_sparse_rectangular(); rectangular matrix-mode densification "
                "is handled internally by PixelizedLensSimulator.build_inverter()."
            )

        scale = reg_scale if reg_scale is not None else self.reg_scale.value
        coefficient = (
            reg_coefficient if reg_coefficient is not None else self.reg_coefficient.value
        )
        kernel_type = reg_type if reg_type is not None else self.reg_type
        operator_mode = reg_operator_mode if reg_operator_mode is not None else self.reg_operator_mode
        sparse_k = (
            int(reg_sparse_k_neighbors)
            if reg_sparse_k_neighbors is not None
            else int(self.reg_sparse_k_neighbors)
        )

        if operator_mode == 'sparse_knn':
            rows, cols, values, n_source = regularization_sparse_knn_from(
                scale=scale,
                coefficient=coefficient,
                points=points,
                reg_type=kernel_type,
                k_neighbors=sparse_k,
            )
            return sparse_regularization_dense_from(rows, cols, values, n_source)
        if operator_mode != 'dense_gp':
            raise ValueError(
                f"Unknown reg_operator_mode: '{operator_mode}'. "
                "Must be one of {'dense_gp', 'sparse_knn'}."
            )

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
                f"n_source_points={config_dict['n_source_points']}, "
                f"source_grid_type='{config_dict['source_grid_type']}')")
