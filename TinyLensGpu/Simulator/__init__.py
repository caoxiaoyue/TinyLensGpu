"""
Gravitational lens simulator.

This module provides simulation capabilities supporting linear
parameter solving and PSF convolution.
"""

from .config import SimulatorConfig
from .lens_simulator import LensSimulator

__all__ = ['SimulatorConfig', 'LensSimulator']
