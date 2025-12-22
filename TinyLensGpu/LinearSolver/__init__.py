"""
Linear solver utilities for gravitational lensing.

This module provides linear solver utilities for intensity parameter optimization.

Main Classes
------------
LinearSolver : Linear solver for intensity parameters (NNLS or normal equations)
prepare_linear_system : Prepare linear system for solving
"""

from .linear_solver import LinearSolver, prepare_linear_system

__all__ = [
    'LinearSolver',
    'prepare_linear_system',
]
