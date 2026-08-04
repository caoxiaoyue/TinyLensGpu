"""
Tests for the 2×3 parametric diagnostic plotter ``plot_model_results``.

Covers the configurable source-plane grid: default resolution,
explicit overrides, and input validation of ``source_npix`` /
``source_dpix``.
"""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from TinyLensGpu.visualizer import plot_model_results


class _StubConfig:
    """Minimal sim_config: 32×32 image at 0.1 arcsec/pixel, no mask."""

    dpix = 0.1
    npix = 32
    mask = None


class _StubLikelihood:
    """Minimal likelihood interface exercised by ``plot_model_results``."""

    use_linear = True

    def __init__(self, image: np.ndarray) -> None:
        self.image_data = image
        self.noise_map = np.ones_like(image)
        self.sim_obj = SimpleNamespace(
            sim_config=_StubConfig(),
            phys_model=SimpleNamespace(source_light=[]),
        )
        self._theta = None

    def set_values(self, theta) -> None:
        self._theta = list(theta)

    def forward_model(self, **kwargs):
        zeros = np.zeros_like(self.image_data)
        return zeros, zeros, None


@pytest.fixture
def stub_likelihood():
    rng = np.random.default_rng(0)
    return _StubLikelihood(rng.random((32, 32)))


def _source_panel(axes):
    return axes[1, 2]


def test_source_panel_default_grid(stub_likelihood):
    """Default kwargs render a 200×200 source panel spanning ±1 arcsec."""
    fig, axes = plot_model_results(stub_likelihood, [0.0])
    try:
        source = np.asarray(_source_panel(axes).images[0].get_array())
        assert source.shape == (200, 200)
        assert np.allclose(axes[1, 2].get_xlim(), (-1.0, 1.0), atol=1e-9)
        assert np.allclose(axes[1, 2].get_ylim(), (-1.0, 1.0), atol=1e-9)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_source_panel_override_grid(stub_likelihood):
    """Explicit source_npix/source_dpix override the grid and its extent."""
    fig, axes = plot_model_results(
        stub_likelihood, [0.0], source_npix=64, source_dpix=0.05
    )
    try:
        source = np.asarray(_source_panel(axes).images[0].get_array())
        assert source.shape == (64, 64)
        assert np.allclose(axes[1, 2].get_xlim(), (-1.6, 1.6), atol=1e-9)
        assert np.allclose(axes[1, 2].get_ylim(), (-1.6, 1.6), atol=1e-9)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.parametrize(
    "source_npix",
    [0, -5, 1.5, True, "200", None],
)
def test_source_npix_invalid_values_raise(stub_likelihood, source_npix):
    with pytest.raises(ValueError, match="source_npix must be a positive integer"):
        plot_model_results(
            stub_likelihood, [0.0], source_npix=source_npix, source_dpix=0.01
        )


@pytest.mark.parametrize(
    "source_dpix",
    [0, -0.01, float("nan"), float("inf"), float("-inf"), "0.01", None],
)
def test_source_dpix_invalid_values_raise(stub_likelihood, source_dpix):
    with pytest.raises(ValueError, match="source_dpix must be a positive finite number"):
        plot_model_results(
            stub_likelihood, [0.0], source_npix=200, source_dpix=source_dpix
        )
