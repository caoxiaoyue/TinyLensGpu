"""
Utility functions for TinyLensGpu.

This module provides general-purpose utility functions and tools
that are used across different parts of the package.

Main Classes
------------
LinearSolver : Linear solver for intensity parameters (NNLS or normal equations)
prepare_linear_system : Prepare linear system for solving

Submodules
----------
geometry : Geometric transformation utilities
interpolation : Interpolation kernel utilities
lensing : Lensing operation utilities (mapping, PSF, regularization)
inversion : Linear inversion solver for source reconstruction
mesh : Source mesh sampling utilities
"""

from .linear_solver import LinearSolver, prepare_linear_system
from .misc import auto_mkdir_path, load_lens_data

__all__ = [
    'LinearSolver',
    'prepare_linear_system',
    'auto_mkdir_path',
    'load_lens_data',
    'geometry',
    'interpolation',
    'lensing',
    'inversion',
    'mesh',
]
