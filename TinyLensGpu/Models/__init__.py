"""
Implementations of gravitational lensing models.

This module provides physical models for gravitational lensing,
including mass profiles, light profiles, and supporting utilities.

Main Components
---------------
ParamU : Parameter class with prior metadata
PhysicalModel : Composite physical model
SIE, Shear : Mass profile models
SersicEllipse, GaussianEllipse : Light profile models
prior_spec : Prior specification and transformation
likelihood : Likelihood interface utilities
"""

from .composite import PhysicalModel
from .mass import SIE, Shear
from .light import SersicEllipse, GaussianEllipse
from .pixelized_source import PixelizedSourceModel, PixelizedSourceConfig

__all__ = [
    'PhysicalModel',
    'SIE',
    'Shear',
    'SersicEllipse',
    'GaussianEllipse',
    'PixelizedSourceModel',
    'PixelizedSourceConfig',
]
