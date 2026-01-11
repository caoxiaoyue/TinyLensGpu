"""
Gravitational lens simulator.

This module provides simulation capabilities supporting linear
parameter solving and PSF convolution.
"""

from .config import SimulatorConfig, make_grid_2d
from .parametric import LensSimulator
from .pixelized import PixelizedLensSimulator

__all__ = ['SimulatorConfig', 'LensSimulator', 'PixelizedLensSimulator', 'make_grid_2d']
