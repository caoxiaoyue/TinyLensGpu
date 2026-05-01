"""
Gravitational lens simulator.

This module provides simulation capabilities supporting linear
parameter solving and PSF convolution.
"""

from .config import SimulatorConfig, make_grid_2d
from .parametric import LensSimulator
from .pixelized import PixelizedLensSimulator
from .results import SimulationResult

__all__ = ['SimulatorConfig', 'LensSimulator', 'PixelizedLensSimulator', 'SimulationResult', 'make_grid_2d']
