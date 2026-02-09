"""
Image-based probability models for gravitational lensing.
"""

from .parametric_image_model import ImageProbModel
from .pixelized_image_model import PixelizedImageProbModel
from .point_source_model import PointSourceProbModel

# Alias for backward compatibility
LensLikelihood = ImageProbModel

__all__ = [
    'ImageProbModel',
    'PixelizedImageProbModel',
    'PointSourceProbModel',
    'LensLikelihood',  # Alias
]
