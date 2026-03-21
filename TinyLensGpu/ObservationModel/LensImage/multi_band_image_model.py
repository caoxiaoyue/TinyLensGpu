# pyright: reportMissingImports=false

"""
Multi-band probability model scaffolding for lensing image fitting.

This module provides a minimal observation-layer wrapper for assembling
per-band image metadata and physical models with constructor-time validation.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import caskade as ck
import jax.numpy as jnp
import numpy as np
from jax import Array

from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel


@dataclass(frozen=True)
class BandImageData:
    """
    Container for one observed imaging band.

    Parameters
    ----------
    name : str
        Band identifier used for deterministic ordering and lookup.
    image_data : Union[np.ndarray, Array]
        Observed image for this band.
    noise_map : Union[np.ndarray, Array]
        Per-pixel noise standard deviation map.
    psf_kernel : Union[np.ndarray, Array]
        Point spread function kernel for this band.
    dpix : float
        Pixel scale in arcsec/pixel.
    nsub : int
        Subsampling factor for ray-tracing.
    mask : Optional[Union[np.ndarray, Array]], optional
        Optional boolean mask where True indicates masked pixels.
    """

    name: str
    image_data: Union[np.ndarray, Array]
    noise_map: Union[np.ndarray, Array]
    psf_kernel: Union[np.ndarray, Array]
    dpix: float
    nsub: int
    mask: Optional[Union[np.ndarray, Array]] = None


class MultiBandImageProbModel(ck.Module):
    """
    Minimal multi-band image probability model scaffold.

    This class currently validates multi-band constructor inputs and stores
    deterministic band ordering. Likelihood evaluation is intentionally deferred.

    Parameters
    ----------
    bands : Sequence[BandImageData]
        Observed data containers for each band.
    phys_models : Sequence[PhysicalModel]
        Physical models aligned one-to-one with ``bands``.
    use_linear : bool
        Whether linear solving is enabled for per-band simulators.
    solver_type : str, optional
        Linear solver type, by default ``'nnls'``.

    Raises
    ------
    ValueError
        If ``bands`` is empty, band/model counts mismatch, or names are duplicated.
    """

    def __init__(
        self,
        bands: Sequence[BandImageData],
        phys_models: Sequence[PhysicalModel],
        *,
        use_linear: bool,
        solver_type: str = "nnls",
    ) -> None:
        super().__init__("multi_band_image_prob_model")

        if len(bands) == 0:
            raise ValueError("bands must be a non-empty sequence")

        if len(bands) != len(phys_models):
            raise ValueError("bands and phys_models must have the same length")

        band_names = [band.name for band in bands]
        if len(set(band_names)) != len(band_names):
            raise ValueError("band names must be unique")

        self.bands = tuple(bands)
        self.phys_models = tuple(phys_models)
        self.use_linear = bool(use_linear)
        self.solver_type = solver_type
        self.band_names = tuple(band_names)

        # Keep jnp imported and ready for subsequent multi-band likelihood work.
        self._num_bands = jnp.array(len(self.bands), dtype=jnp.int32)
