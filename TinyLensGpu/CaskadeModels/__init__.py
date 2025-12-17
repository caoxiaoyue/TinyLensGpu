"""
Caskade-based implementations of gravitational lensing models.

This module provides caskade implementations of physical models for
gravitational lensing, including mass profiles and light profiles.
"""

import os
# Force JAX backend for caskade
os.environ['CASKADE_BACKEND'] = 'jax'

from . import mass
from . import light
from .composite import PhysicalModel

__all__ = ['mass', 'light', 'PhysicalModel']
