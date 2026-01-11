"""
Image-based probability models for gravitational lensing.
"""

from .parametric_image_model import ImageProbModel
from .pixelized_image_model import PixelizedImageProbModel

# Alias for backward compatibility
LensLikelihood = ImageProbModel

__all__ = [
    'ImageProbModel',
    'PixelizedImageProbModel',
    'LensLikelihood',  # Alias
]
