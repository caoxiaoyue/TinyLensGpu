"""
Gravitational lens simulator.

This module provides simulation capabilities supporting linear
parameter solving and PSF convolution.
"""

from .config import SimulatorConfig, make_grid_2d
from .lens_forward_model import LensSimulator

__all__ = ['SimulatorConfig', 'LensSimulator', 'make_grid_2d']
