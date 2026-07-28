"""Tests for critical-line and caustic computation."""

import numpy as np
import pytest

from TinyLensGpu.PhysicalModel import EPL, SIS, Shear
from TinyLensGpu.utils.lensing import find_critical_lines


@pytest.mark.unit
def test_find_critical_lines_separates_subhalo_contours():
    """A subhalo's disconnected critical line should be a separate path."""
    lens_mass = [
        EPL(
            theta_E=1.359571135084649,
            gamma=2.190525144683694,
            e1=0.021365440417320743,
            e2=-5.938880395760709e-05,
            center_x=0.01861582233827087,
            center_y=-0.011558181211533423,
        ),
        Shear(
            gamma1=0.06496314572575379,
            gamma2=-0.05747366682788987,
        ),
        SIS(
            theta_E=0.07230902608033254,
            center_x=-0.7074372737837108,
            center_y=1.0066050253438716,
        ),
    ]

    critical_lines = find_critical_lines(lens_mass, n_grid=64)

    assert len(critical_lines) == 2
    for x_path, y_path in critical_lines:
        point_separations = np.hypot(np.diff(x_path), np.diff(y_path))
        assert np.max(point_separations) < 0.2
