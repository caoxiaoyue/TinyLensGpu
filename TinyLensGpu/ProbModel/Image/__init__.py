"""
Image-based probability models for gravitational lensing.
"""

from .image_model import ImageProbModel

# Alias for backward compatibility
LensLikelihood = ImageProbModel

__all__ = [
    'ImageProbModel',
    'LensLikelihood',  # Alias
]
