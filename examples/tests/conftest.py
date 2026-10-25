"""Paths shared by the example regression tests."""
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def examples_dir():
    """Return the examples directory independently of the working directory."""
    return Path(__file__).resolve().parents[1]
