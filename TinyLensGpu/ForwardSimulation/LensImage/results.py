"""Shared result types for lens-image simulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, TypeAlias

from jax import Array

if TYPE_CHECKING:
    from TinyLensGpu.utils.inversion import (
        LinearInversion,
        NNLSInversion,
        OperatorInversion,
        OperatorNNLSInversion,
    )

ImageArray: TypeAlias = Array
LinearParameterArray: TypeAlias = Array
IntensityArray: TypeAlias = Array
MeshCoordinateArray: TypeAlias = Array
InversionResult: TypeAlias = "LinearInversion | NNLSInversion | OperatorInversion | OperatorNNLSInversion"


@dataclass(slots=True)
class SimulationResult:
    """Top-level result returned by simulator ``forward(...)`` methods."""

    model_image: Optional[ImageArray]
    source_image: Optional[ImageArray] = None
    lens_image: Optional[ImageArray] = None
    linear_params: Optional[LinearParameterArray] = None
    source_intensities: Optional[IntensityArray] = None
    lens_light_intensities: Optional[IntensityArray] = None
    source_mesh_beta: Optional[MeshCoordinateArray] = None
    inverter: Optional[InversionResult] = None
