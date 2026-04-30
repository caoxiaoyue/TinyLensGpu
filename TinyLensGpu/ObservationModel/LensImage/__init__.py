"""
Image-based probability models for gravitational lensing.
"""

from .parametric_image_model import ImageProbModel
from .point_source_model import PointSourceProbModel
from .multi_band_image_model import BandImageData, MultiBandImageProbModel

# Alias for backward compatibility
LensLikelihood = ImageProbModel

__all__ = [
    'BandImageData',
    'ImageProbModel',
    'LensLikelihood',  # Alias
    'MultiBandImageProbModel',
    'PointSourceProbModel',
]
