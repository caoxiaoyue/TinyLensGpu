"""
Pytest configuration and shared fixtures for TinyLensGpu tests.
"""
import pytest
import numpy as np
import jax.numpy as jnp
import yaml
import os
from pathlib import Path


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
def sample_config_simple(tmp_path):
    """Create a simple configuration file for testing."""
    config = {
        'dataset': {
            'data_path': 'test_data.fits',
            'noise_path': 'test_noise.fits',
            'psf_path': 'test_psf.fits',
            'pixel_scale': 0.05
        },
        'model_components': {
            'lens_mass_list': [],
            'source_light_list': [],
            'lens_light_list': [
                {
                    'type': 'Sersic',
                    'params': {
                        'R_sersic': {
                            'prior_type': 'uniform',
                            'prior_settings': [0.1, 2.0],
                            'limits': [0.0, 5.0],
                            'fixed': False
                        },
                        'n_sersic': {
                            'prior_type': 'uniform',
                            'prior_settings': [1.0, 4.0],
                            'limits': [0.3, 6.0],
                            'fixed': False
                        },
                        'e1': {
                            'prior_type': 'gaussian',
                            'prior_settings': [0.0, 0.3],
                            'limits': [-1.0, 1.0],
                            'fixed': False
                        },
                        'e2': {
                            'prior_type': 'gaussian',
                            'prior_settings': [0.0, 0.3],
                            'limits': [-1.0, 1.0],
                            'fixed': False
                        },
                        'center_x': {
                            'fixed': True,
                            'fixed_value': 0.0
                        },
                        'center_y': {
                            'fixed': True,
                            'fixed_value': 0.0
                        },
                        'Ie': {
                            'use_linear': True
                        }
                    }
                }
            ]
        },
        'inference': {
            'type': 'sampler',
            'method': 'nautilus',
            'settings': {
                'nlive': 100,
                'batch_size': 100
            }
        },
        'output': {
            'path': 'test_output',
            'figures': {
                'results': 'results.png',
                'corner': 'corner.png'
            },
            'tables': {
                'samples': 'samples.csv',
                'summary': 'summary.csv'
            },
            'datasets': {
                'subplot': 'subplot.png'
            }
        }
    }
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)


@pytest.fixture
def sample_config_full(tmp_path):
    """Create a full configuration file with all components for testing."""
    config = {
        'dataset': {
            'data_path': 'test_data.fits',
            'noise_path': 'test_noise.fits',
            'psf_path': 'test_psf.fits',
            'pixel_scale': 0.05
        },
        'model_components': {
            'lens_mass_list': [
                {
                    'type': 'SIE',
                    'params': {
                        'theta_E': {
                            'prior_type': 'uniform',
                            'prior_settings': [0.5, 2.0],
                            'limits': [0.0, 5.0],
                            'fixed': False
                        },
                        'e1': {
                            'prior_type': 'gaussian',
                            'prior_settings': [0.0, 0.3],
                            'limits': [-1.0, 1.0],
                            'fixed': False
                        },
                        'e2': {
                            'prior_type': 'gaussian',
                            'prior_settings': [0.0, 0.3],
                            'limits': [-1.0, 1.0],
                            'fixed': False
                        },
                        'center_x': {
                            'fixed': True,
                            'fixed_value': 0.0
                        },
                        'center_y': {
                            'fixed': True,
                            'fixed_value': 0.0
                        }
                    }
                },
                {
                    'type': 'SHEAR',
                    'params': {
                        'gamma1': {
                            'prior_type': 'uniform',
                            'prior_settings': [-0.1, 0.1],
                            'limits': [-0.5, 0.5],
                            'fixed': False
                        },
                        'gamma2': {
                            'prior_type': 'uniform',
                            'prior_settings': [-0.1, 0.1],
                            'limits': [-0.5, 0.5],
                            'fixed': False
                        }
                    }
                }
            ],
            'source_light_list': [
                {
                    'type': 'Sersic',
                    'params': {
                        'R_sersic': {
                            'prior_type': 'uniform',
                            'prior_settings': [0.1, 1.0],
                            'limits': [0.0, 3.0],
                            'fixed': False
                        },
                        'n_sersic': {
                            'prior_type': 'uniform',
                            'prior_settings': [0.5, 4.0],
                            'limits': [0.3, 6.0],
                            'fixed': False
                        },
                        'e1': {
                            'prior_type': 'gaussian',
                            'prior_settings': [0.0, 0.3],
                            'limits': [-1.0, 1.0],
                            'fixed': False
                        },
                        'e2': {
                            'prior_type': 'gaussian',
                            'prior_settings': [0.0, 0.3],
                            'limits': [-1.0, 1.0],
                            'fixed': False
                        },
                        'center_x': {
                            'prior_type': 'gaussian',
                            'prior_settings': [0.0, 0.5],
                            'limits': [-2.0, 2.0],
                            'fixed': False
                        },
                        'center_y': {
                            'prior_type': 'gaussian',
                            'prior_settings': [0.0, 0.5],
                            'limits': [-2.0, 2.0],
                            'fixed': False
                        },
                        'Ie': {
                            'use_linear': True
                        }
                    }
                }
            ],
            'lens_light_list': [
                {
                    'type': 'Sersic',
                    'params': {
                        'R_sersic': {
                            'prior_type': 'uniform',
                            'prior_settings': [0.5, 2.0],
                            'limits': [0.0, 5.0],
                            'fixed': False
                        },
                        'n_sersic': {
                            'prior_type': 'uniform',
                            'prior_settings': [2.0, 5.0],
                            'limits': [0.3, 6.0],
                            'fixed': False
                        },
                        'e1': {
                            'prior_type': 'gaussian',
                            'prior_settings': [0.0, 0.3],
                            'limits': [-1.0, 1.0],
                            'fixed': False
                        },
                        'e2': {
                            'prior_type': 'gaussian',
                            'prior_settings': [0.0, 0.3],
                            'limits': [-1.0, 1.0],
                            'fixed': False
                        },
                        'center_x': {
                            'fixed': True,
                            'fixed_value': 0.0
                        },
                        'center_y': {
                            'fixed': True,
                            'fixed_value': 0.0
                        },
                        'Ie': {
                            'use_linear': True
                        }
                    }
                }
            ]
        },
        'inference': {
            'type': 'sampler',
            'method': 'nautilus',
            'settings': {
                'nlive': 100,
                'batch_size': 100
            }
        },
        'output': {
            'path': 'test_output',
            'figures': {
                'results': 'results.png',
                'corner': 'corner.png'
            },
            'tables': {
                'samples': 'samples.csv',
                'summary': 'summary.csv'
            },
            'datasets': {
                'subplot': 'subplot.png'
            }
        }
    }
    
    config_path = tmp_path / "test_config_full.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)


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

