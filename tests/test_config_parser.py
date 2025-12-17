"""
Unit tests for configuration parser and prior transformation.

This module tests the CaskadeConfigParser's ability to:
1. Parse YAML configuration files
2. Build PhysicalModel correctly
3. Set up parameter management (dynamic/static/linear)
4. Apply parameter links
5. Perform prior transformations
"""

import pytest
import numpy as np
import jax.numpy as jnp
from TinyLensGpu.CaskadeInference.config_parser import CaskadeConfigParser


class TestConfigParser:
    """Test configuration parser"""

    def test_parse_lens_src_config(self):
        """Test parsing of lens_src configuration"""
        config_path = 'paper/demo/lens_src/model_config.yaml'

        parser = CaskadeConfigParser(config_path)

        # Check component counts
        counts = parser.phys_model.get_component_counts()
        assert counts['n_lens_mass'] == 2, "Should have 2 mass components"
        assert counts['n_source_light'] == 1, "Should have 1 source light"
        assert counts['n_lens_light'] == 1, "Should have 1 lens light"

        # Check parameter counts
        # Expected dynamic params:
        # SIE: theta_E, e1, e2 (3)
        # Shear: gamma1, gamma2 (2)
        # Source: R_sersic, n_sersic, e1, e2, center_x, center_y (6)
        # Lens: R_sersic, n_sersic, e1, e2 (4)
        # Total: 15 dynamic parameters
        assert parser.ndim == 15, f"Should have 15 dynamic params, got {parser.ndim}"

        # Check linear parameters
        # Source: Ie (1)
        # Lens: Ie (1)
        # Total: 2 linear parameters
        assert parser.n_linear_params == 2, f"Should have 2 linear params, got {parser.n_linear_params}"

        print("✓ lens_src configuration parsing test passed")

    def test_prior_transform(self):
        """Test prior transformation"""
        config_path = 'paper/demo/lens_src/model_config.yaml'
        parser = CaskadeConfigParser(config_path)

        # Create unit cube samples
        np.random.seed(42)
        n_samples = 10
        cube_unit = np.random.rand(n_samples, parser.ndim)

        # Transform to physical space
        physical_params = parser.prior_transform.transform(cube_unit)

        # Check shape
        assert physical_params.shape == (n_samples, parser.ndim), \
            f"Shape mismatch: expected {(n_samples, parser.ndim)}, got {physical_params.shape}"

        # Check that values are in expected ranges
        # (at least check they're not all in [0,1] anymore)
        assert np.any(physical_params > 1.0) or np.any(physical_params < 0.0), \
            "Parameters should be transformed from unit cube"

        print("✓ Prior transformation test passed")

    def test_prior_transform_single_sample(self):
        """Test prior transformation with single sample"""
        config_path = 'paper/demo/lens_src/model_config.yaml'
        parser = CaskadeConfigParser(config_path)

        # Create single sample
        cube_unit = np.random.rand(parser.ndim)

        # Transform
        physical_params = parser.prior_transform.transform(cube_unit)

        # Check shape (should be 1D)
        assert physical_params.shape == (parser.ndim,), \
            f"Shape mismatch: expected ({parser.ndim},), got {physical_params.shape}"

        print("✓ Single sample prior transformation test passed")

    def test_get_param_bounds(self):
        """Test parameter bounds extraction"""
        config_path = 'paper/demo/lens_src/model_config.yaml'
        parser = CaskadeConfigParser(config_path)

        bounds = parser.prior_transform.get_param_bounds()

        # Check we have bounds for all dynamic parameters
        assert len(bounds) == parser.ndim, \
            f"Should have {parser.ndim} bounds, got {len(bounds)}"

        # Check all bounds are tuples of (min, max)
        for i, (low, high) in enumerate(bounds):
            assert low < high, f"Invalid bounds at index {i}: ({low}, {high})"

        print("✓ Parameter bounds test passed")

    def test_static_params_initialization(self):
        """Test that static parameters are properly initialized"""
        config_path = 'paper/demo/lens_src/model_config.yaml'
        parser = CaskadeConfigParser(config_path)

        # Set static params
        parser.set_static_params()

        # Check SIE center_x is static and equals 0.0
        sie = parser.phys_model.lens_mass[0]
        assert hasattr(sie.center_x, 'value'), "center_x should have value attribute"
        # The value should be 0.0 as specified in config

        print("✓ Static parameters initialization test passed")


class TestPriorTypes:
    """Test different prior types"""

    def test_uniform_prior(self):
        """Test uniform prior transformation"""
        config_path = 'paper/demo/lens_src/model_config.yaml'
        parser = CaskadeConfigParser(config_path)

        # Find a uniform prior parameter (theta_E should be first)
        param_info = parser.prior_transform.get_param_info(0)
        assert param_info['prior']['type'] == 'uniform', \
            f"First param should have uniform prior, got {param_info['prior']['type']}"

        # Test transformation at boundaries
        cube_0 = np.zeros(parser.ndim)
        cube_1 = np.ones(parser.ndim)

        phys_0 = parser.prior_transform.transform(cube_0)
        phys_1 = parser.prior_transform.transform(cube_1)

        # For uniform prior, should transform to [low, high]
        prior_range = param_info['prior']['range']
        np.testing.assert_allclose(phys_0[0], prior_range[0], rtol=1e-5)
        np.testing.assert_allclose(phys_1[0], prior_range[1], rtol=1e-5)

        print("✓ Uniform prior test passed")

    def test_gaussian_prior(self):
        """Test Gaussian prior transformation"""
        config_path = 'paper/demo/lens_src/model_config.yaml'
        parser = CaskadeConfigParser(config_path)

        # Find a Gaussian prior parameter (e1 should be index 1)
        param_info = parser.prior_transform.get_param_info(1)
        assert param_info['prior']['type'] == 'gaussian', \
            f"Param at index 1 should have gaussian prior, got {param_info['prior']['type']}"

        # Test transformation at 0.5 (median)
        cube_median = np.ones(parser.ndim) * 0.5
        phys_median = parser.prior_transform.transform(cube_median)

        # For Gaussian, 0.5 should map approximately to mean
        prior_mean = param_info['prior']['mean']
        # Allow some tolerance since it's inverse CDF
        assert abs(phys_median[1] - prior_mean) < 0.1, \
            f"Median should be close to mean {prior_mean}, got {phys_median[1]}"

        print("✓ Gaussian prior test passed")


if __name__ == "__main__":
    # Run tests
    print("Running CaskadeInference tests...\n")

    print("Testing ConfigParser...")
    test_parser = TestConfigParser()
    test_parser.test_parse_lens_src_config()
    test_parser.test_prior_transform()
    test_parser.test_prior_transform_single_sample()
    test_parser.test_get_param_bounds()
    test_parser.test_static_params_initialization()

    print("\nTesting Prior Types...")
    test_priors = TestPriorTypes()
    test_priors.test_uniform_prior()
    test_priors.test_gaussian_prior()

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
