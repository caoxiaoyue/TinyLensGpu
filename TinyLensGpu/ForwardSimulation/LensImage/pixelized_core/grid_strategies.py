"""Grid-generation strategies for pixelized-source reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np

from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import (
    IrregularGridConfig,
    RectangularGridConfig,
)
from TinyLensGpu.utils.mesh import sample_points_weighted

from .artifacts import GridArtifacts


RayTraceFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


class BaseGridStrategy:
    """
    Represent the `BaseGridStrategy` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    def build(
        self,
        *,
        lensed_source_image: np.ndarray,
        mask: np.ndarray,
        dpix: float,
        data_mesh_beta: jnp.ndarray,
        ray_trace: RayTraceFn,
    ) -> GridArtifacts:
        """
        Compute build.
        
        Parameters
        ----------
        lensed_source_image : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        mask : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        dpix : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        data_mesh_beta : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        ray_trace : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        Raises
        ------
        NotImplementedError
            Raised when input validation fails or required runtime state is missing.
        
        """
        raise NotImplementedError


@dataclass(frozen=True)
class IrregularGridStrategy(BaseGridStrategy):
    """
    Represent the `IrregularGridStrategy` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    config: IrregularGridConfig

    def build(
        self,
        *,
        lensed_source_image: np.ndarray,
        mask: np.ndarray,
        dpix: float,
        data_mesh_beta: jnp.ndarray,
        ray_trace: RayTraceFn,
    ) -> GridArtifacts:
        """
        Compute build.
        
        Parameters
        ----------
        lensed_source_image : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        mask : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        dpix : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        data_mesh_beta : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        ray_trace : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        source_mesh_px, (height, width), _ = sample_points_weighted(
            img=np.array(lensed_source_image),
            mask=~np.array(mask),
            n_points=self.config.n_source_points,
            alpha=self.config.mesh_alpha,
            blur_sigma_px=self.config.mesh_blur_sigma,
            replace=False,
            normalize_xy=False,
            pixel_jitter=False,
            method=self.config.mesh_method,
            seed=self.config.mesh_seed,
        )
        source_mesh = (source_mesh_px - np.array([(width - 1) / 2, (height - 1) / 2])) * float(dpix)
        source_mesh = jnp.asarray(source_mesh, dtype=jnp.float32)
        source_mesh_beta = ray_trace(source_mesh[:, 0], source_mesh[:, 1])
        return GridArtifacts(
            source_mesh=source_mesh,
            source_mesh_beta=source_mesh_beta,
            data_mesh_beta=jnp.asarray(data_mesh_beta, dtype=jnp.float32),
            source_grid_shape=None,
            source_grid_bounds=None,
        )


@dataclass(frozen=True)
class RectangularGridStrategy(BaseGridStrategy):
    """
    Represent the `RectangularGridStrategy` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    config: RectangularGridConfig

    def build(
        self,
        *,
        lensed_source_image: np.ndarray,
        mask: np.ndarray,
        dpix: float,
        data_mesh_beta: jnp.ndarray,
        ray_trace: RayTraceFn,
    ) -> GridArtifacts:
        """
        Compute build.
        
        Parameters
        ----------
        lensed_source_image : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        mask : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        dpix : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        data_mesh_beta : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        ray_trace : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        _ = lensed_source_image, mask, dpix, ray_trace

        data_mesh_beta_np = np.asarray(data_mesh_beta, dtype=np.float32)
        if self.config.bounds is None:
            x_min = float(np.min(data_mesh_beta_np[:, 0]))
            x_max = float(np.max(data_mesh_beta_np[:, 0]))
            y_min = float(np.min(data_mesh_beta_np[:, 1]))
            y_max = float(np.max(data_mesh_beta_np[:, 1]))
            margin_frac = float(self.config.margin_frac)

            x_span = max(x_max - x_min, 1e-5)
            y_span = max(y_max - y_min, 1e-5)
            x_margin = margin_frac * x_span
            y_margin = margin_frac * y_span

            x_min -= x_margin
            x_max += x_margin
            y_min -= y_margin
            y_max += y_margin
        else:
            x_min, x_max, y_min, y_max = [float(v) for v in self.config.bounds]

        x_lin = jnp.linspace(x_min, x_max, int(self.config.nx), dtype=jnp.float32)
        y_lin = jnp.linspace(y_min, y_max, int(self.config.ny), dtype=jnp.float32)
        xx, yy = jnp.meshgrid(x_lin, y_lin, indexing="xy")
        source_mesh = jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

        return GridArtifacts(
            source_mesh=source_mesh,
            source_mesh_beta=source_mesh,
            data_mesh_beta=jnp.asarray(data_mesh_beta, dtype=jnp.float32),
            source_grid_shape=(int(self.config.ny), int(self.config.nx)),
            source_grid_bounds=(x_min, x_max, y_min, y_max),
        )

