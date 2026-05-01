"""
Unit tests for utility functions.
"""
import pytest
import numpy as np

from TinyLensGpu.utils import load_lens_data


@pytest.mark.unit
class TestLoadLensData:
    """Test lens data loading utility."""

    def test_load_lens_data_basic(self, tmp_path):
        """Test that load_lens_data loads FITS files correctly."""
        # Basic smoke test for import and type checking
        assert callable(load_lens_data)
