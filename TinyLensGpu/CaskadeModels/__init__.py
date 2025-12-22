"""
Caskade-based implementations of gravitational lensing models.

This module provides caskade implementations of physical models for
gravitational lensing, including mass profiles, light profiles, and
programmatic model building utilities.

Main Components
---------------
ParamU : Parameter class with prior metadata
PhysicalModel : Composite physical model
SIE, Shear : Mass profile models
SersicEllipse, GaussianEllipse : Light profile models
builder : Programmatic model building functions (no YAML)
prior_spec : Prior specification and transformation
likelihood : Likelihood interface utilities
"""

import os
# Force JAX backend for caskade
os.environ['CASKADE_BACKEND'] = 'jax'

from .composite import PhysicalModel
from .mass import SIE, Shear
from .light import SersicEllipse, GaussianEllipse
from .param_u import ParamU
from . import builder
from . import prior_spec
from . import likelihood

__all__ = [
    'PhysicalModel',
    'SIE',
    'Shear',
    'SersicEllipse',
    'GaussianEllipse',
    'ParamU',
    'builder',
    'prior_spec',
    'likelihood',
]
