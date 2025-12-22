"""
Image-based probability models for gravitational lensing.
"""

from .image_model import ImageProbModel
from .vectorized_likelihood import VectorizedLensLikelihood

# Alias for backward compatibility
LensLikelihood = VectorizedLensLikelihood

__all__ = [
    'ImageProbModel',
    'VectorizedLensLikelihood',
    'LensLikelihood',  # Alias
]
