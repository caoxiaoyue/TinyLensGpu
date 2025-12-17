"""
Caskade-based gravitational lens simulator.

This module provides simulation capabilities using caskade models,
supporting batch processing, linear parameter solving, and PSF convolution.
"""

from .config import SimulatorConfig
from .lens_simulator import LensSimulator

__all__ = ['SimulatorConfig', 'LensSimulator']
