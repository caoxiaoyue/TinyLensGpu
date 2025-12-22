"""
Pytest configuration and shared fixtures for TinyLensGpu tests.
"""
import pytest
import numpy as np
import jax.numpy as jnp


@pytest.fixture
def sample_image_data():
    """Generate sample image data for testing."""
    np.random.seed(42)
    image_shape = (50, 50)
    image = np.random.randn(*image_shape) * 0.1 + 1.0
    image = np.abs(image)  # Ensure positive values
    return image


@pytest.fixture
def sample_noise_map():
    """Generate sample noise map for testing."""
    np.random.seed(43)
    noise_shape = (50, 50)
    noise = np.abs(np.random.randn(*noise_shape) * 0.1 + 0.1)
    return noise


@pytest.fixture
def sample_psf_kernel():
    """Generate sample PSF kernel for testing."""
    # Create a simple Gaussian PSF
    size = 15
    sigma = 2.0
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)
    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf = psf / np.sum(psf)  # Normalize
    return psf


@pytest.fixture
def coordinate_grids():
    """Generate coordinate grids for testing."""
    npix = 50
    pixel_scale = 0.05
    half_width = npix * pixel_scale / 2.0
    x = jnp.linspace(-half_width, half_width, npix)
    y = jnp.linspace(-half_width, half_width, npix)
    xx, yy = jnp.meshgrid(x, y)
    return xx, yy

