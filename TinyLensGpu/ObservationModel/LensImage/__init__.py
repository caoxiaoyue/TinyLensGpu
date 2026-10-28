"""
Image-based probability models for gravitational lensing.
"""

from .inversion_result import InversionDiagnostics, InversionResult, PositionDiagnostics
from .parametric_image_model import ImageProbModel
from .pixelized_image_model import PixelizedImageProbModel
from .point_source_model import PointSourceProbModel
from .multi_band_image_model import BandImageData, MultiBandImageProbModel
from .potential_correction import SingleStepPotentialCorrection
from .potential_correction_result import (
    PotentialCorrectionResult, PotentialCorrectionStrengths, PotentialSolverInfo,
)

# Alias for backward compatibility
LensLikelihood = ImageProbModel

__all__ = [
    'BandImageData',
    'ImageProbModel',
    'InversionDiagnostics',
    'InversionResult',
    'LensLikelihood',  # Alias
    'MultiBandImageProbModel',
    'PixelizedImageProbModel',
    'PointSourceProbModel',
    'PositionDiagnostics',
    'SingleStepPotentialCorrection',
    'PotentialCorrectionResult',
    'PotentialCorrectionStrengths',
    'PotentialSolverInfo',
]
