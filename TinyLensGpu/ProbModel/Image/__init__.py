"""
Image-based probability models for gravitational lensing.
"""

from .caskade_model import CaskadeImageProbModel
from .vectorized_likelihood import VectorizedLensLikelihood

# Alias for backward compatibility
LensLikelihood = VectorizedLensLikelihood

__all__ = [
    'CaskadeImageProbModel',
    'VectorizedLensLikelihood',
    'LensLikelihood',  # Alias
]
