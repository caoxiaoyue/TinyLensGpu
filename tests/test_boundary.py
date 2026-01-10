"""
Boundary and edge case tests for TinyLensGpu.

This module tests edge cases, boundary values, and error handling
to ensure robustness of the codebase.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from TinyLensGpu.Models import SIE, Shear, SersicEllipse, GaussianEllipse, PhysicalModel
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Simulator import LensSimulator, SimulatorConfig


@pytest.mark.unit
class TestParameterBoundaries:
    """Test parameter boundary values and edge cases."""
    
    def test_sie_zero_einstein_radius(self):
        """Test SIE with zero Einstein radius."""
        sie = SIE(theta_E=0.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        x = jnp.array([1.0, 2.0])
        y = jnp.array([1.0, 2.0])
        alpha_x, alpha_y = sie.deriv(x, y)
        
        # Should return zero deflection
        assert jnp.allclose(alpha_x, 0.0)
        assert jnp.allclose(alpha_y, 0.0)
    
    def test_sie_extreme_ellipticity(self):
        """Test SIE with extreme ellipticity values."""
        # Maximum ellipticity (close to 1)
        sie = SIE(theta_E=1.5, e1=0.99, e2=0.0, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        x = jnp.linspace(-2, 2, 5)
        y = jnp.linspace(-2, 2, 5)
        X, Y = jnp.meshgrid(x, y)
        alpha_x, alpha_y = sie.deriv(X, Y)
        
        # Should not produce NaN or Inf
        assert not jnp.any(jnp.isnan(alpha_x))
        assert not jnp.any(jnp.isnan(alpha_y))
        assert not jnp.any(jnp.isinf(alpha_x))
        assert not jnp.any(jnp.isinf(alpha_y))
    
    def test_sie_at_singularity(self):
        """Test SIE at the singularity (center)."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        # Evaluate at center
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        alpha_x, alpha_y = sie.deriv(x, y)
        
        # Should handle singularity gracefully (numerical stability)
        assert not jnp.any(jnp.isnan(alpha_x))
        assert not jnp.any(jnp.isnan(alpha_y))
    
    def test_shear_zero_values(self):
        """Test Shear with zero shear components."""
        shear = Shear(gamma1=0.0, gamma2=0.0)
        shear.gamma1.to_static()
        shear.gamma2.to_static()
        
        x = jnp.array([1.0, 2.0])
        y = jnp.array([1.0, 2.0])
        alpha_x, alpha_y = shear.deriv(x, y)
        
        assert jnp.allclose(alpha_x, 0.0)
        assert jnp.allclose(alpha_y, 0.0)
    
    def test_sersic_extreme_index(self):
        """Test Sersic with extreme index values."""
        # Very low index
        sersic_low = SersicEllipse(R_sersic=1.0, n_sersic=0.5, e1=0.0, e2=0.0,
                                   center_x=0.0, center_y=0.0, Ie=1.0)
        for param in [sersic_low.R_sersic, sersic_low.n_sersic, sersic_low.e1,
                      sersic_low.e2, sersic_low.center_x, sersic_low.center_y, sersic_low.Ie]:
            param.to_static()
        
        x = jnp.array([1.0])
        y = jnp.array([1.0])
        light_low = sersic_low.light(x, y)
        
        assert not jnp.any(jnp.isnan(light_low))
        assert not jnp.any(jnp.isinf(light_low))
        
        # Very high index
        sersic_high = SersicEllipse(R_sersic=1.0, n_sersic=10.0, e1=0.0, e2=0.0,
                                    center_x=0.0, center_y=0.0, Ie=1.0)
        for param in [sersic_high.R_sersic, sersic_high.n_sersic, sersic_high.e1,
                      sersic_high.e2, sersic_high.center_x, sersic_high.center_y, sersic_high.Ie]:
            param.to_static()
        
        light_high = sersic_high.light(x, y)
        
        assert not jnp.any(jnp.isnan(light_high))
        assert not jnp.any(jnp.isinf(light_high))
    
    def test_gaussian_zero_sigma(self):
        """Test Gaussian with very small sigma."""
        gaussian = GaussianEllipse(flux=1.0, sigma=1e-5, e1=0.0, e2=0.0,
                                   center_x=0.0, center_y=0.0)
        for param in [gaussian.flux, gaussian.sigma, gaussian.e1,
                      gaussian.e2, gaussian.center_x, gaussian.center_y]:
            param.to_static()
        
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        light = gaussian.light(x, y)
        
        # Should be very large but not infinite
        assert not jnp.any(jnp.isnan(light))
        assert not jnp.any(jnp.isinf(light))


@pytest.mark.unit
class TestNegativeValues:
    """Test handling of negative and invalid values."""
    
    def test_negative_einstein_radius(self):
        """Test SIE with negative Einstein radius."""
        sie = SIE(theta_E=-1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        x = jnp.array([1.0])
        y = jnp.array([1.0])
        alpha_x, alpha_y = sie.deriv(x, y)
        
        # Should handle negative values (physical interpretation may vary)
        assert not jnp.any(jnp.isnan(alpha_x))
        assert not jnp.any(jnp.isnan(alpha_y))
    
    def test_negative_flux(self):
        """Test Gaussian with negative flux."""
        gaussian = GaussianEllipse(flux=-1.0, sigma=1.0, e1=0.0, e2=0.0,
                                   center_x=0.0, center_y=0.0)
        for param in [gaussian.flux, gaussian.sigma, gaussian.e1,
                      gaussian.e2, gaussian.center_x, gaussian.center_y]:
            param.to_static()
        
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        light = gaussian.light(x, y)
        
        # Should produce negative light (unphysical but mathematically valid)
        assert light < 0


@pytest.mark.unit
class TestLargeArrays:
    """Test with large arrays to check memory and performance."""
    
    def test_sie_large_grid(self):
        """Test SIE with large coordinate grid."""
        sie = SIE(theta_E=1.5, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        # Large grid (100x100)
        x = jnp.linspace(-5, 5, 100)
        y = jnp.linspace(-5, 5, 100)
        X, Y = jnp.meshgrid(x, y)
        
        alpha_x, alpha_y = sie.deriv(X, Y)
        
        assert alpha_x.shape == (100, 100)
        assert alpha_y.shape == (100, 100)
        assert not jnp.any(jnp.isnan(alpha_x))
        assert not jnp.any(jnp.isnan(alpha_y))


@pytest.mark.unit
class TestEmptyAndNoneValues:
    """Test handling of empty arrays and None values."""
    
    def test_empty_coordinate_arrays(self):
        """Test models with empty coordinate arrays."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        x = jnp.array([])
        y = jnp.array([])
        alpha_x, alpha_y = sie.deriv(x, y)
        
        assert alpha_x.shape == (0,)
        assert alpha_y.shape == (0,)
    
    def test_param_u_none_value(self):
        """Test ParamU with None value."""
        param = ParamU("test_param", None)
        assert param.value is None


@pytest.mark.unit
class TestPhysicalModelBoundaries:
    """Test PhysicalModel with edge cases."""
    
    def test_empty_model(self):
        """Test PhysicalModel with no components."""
        model = PhysicalModel(lens_mass=[], source_light=[], lens_light=[])
        
        counts = model.get_component_counts()
        assert counts['n_lens_mass'] == 0
        assert counts['n_source_light'] == 0
        assert counts['n_lens_light'] == 0
    
    def test_single_component_model(self):
        """Test PhysicalModel with single component of each type."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        gaussian = GaussianEllipse(flux=1.0, sigma=1.0, e1=0.0, e2=0.0,
                                   center_x=0.0, center_y=0.0)
        
        model = PhysicalModel(lens_mass=[sie], source_light=[gaussian], lens_light=[])
        
        counts = model.get_component_counts()
        assert counts['n_lens_mass'] == 1
        assert counts['n_source_light'] == 1
        assert counts['n_lens_light'] == 0
    
    def test_many_components_model(self):
        """Test PhysicalModel with many components (MGE-like)."""
        # Create 15 Gaussian components (typical MGE)
        gaussians = [
            GaussianEllipse(flux=1.0, sigma=0.1 * (i + 1), e1=0.0, e2=0.0,
                           center_x=0.0, center_y=0.0)
            for i in range(15)
        ]
        
        model = PhysicalModel(lens_mass=[], source_light=gaussians, lens_light=[])
        
        counts = model.get_component_counts()
        assert counts['n_source_light'] == 15


@pytest.mark.unit
class TestSimulatorBoundaries:
    """Test Simulator with edge cases."""
    
    def test_single_pixel_simulation(self):
        """Test simulation with 1x1 pixel."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        gaussian = GaussianEllipse(flux=1.0, sigma=1.0, e1=0.0, e2=0.0,
                                   center_x=0.0, center_y=0.0)
        
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [gaussian.flux, gaussian.sigma, gaussian.e1, gaussian.e2,
                      gaussian.center_x, gaussian.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie], source_light=[gaussian], lens_light=[])
        
        config = SimulatorConfig(dpix=0.05, npix=1, nsub=1)
        simulator = LensSimulator(model, config)
        
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (1, 1)
        assert not jnp.isnan(img).any()
    
    def test_large_subsampling(self):
        """Test simulation with large subsampling factor."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        gaussian = GaussianEllipse(flux=1.0, sigma=1.0, e1=0.0, e2=0.0,
                                   center_x=0.0, center_y=0.0)
        
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [gaussian.flux, gaussian.sigma, gaussian.e1, gaussian.e2,
                      gaussian.center_x, gaussian.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie], source_light=[gaussian], lens_light=[])
        
        # Large subsampling
        config = SimulatorConfig(dpix=0.05, npix=10, nsub=10)
        simulator = LensSimulator(model, config)
        
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (10, 10)
        assert not jnp.isnan(img).any()


@pytest.mark.unit
class TestNumericalStability:
    """Test numerical stability with extreme values."""
    
    def test_very_large_coordinates(self):
        """Test with very large coordinate values."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        x = jnp.array([1000.0, 2000.0])
        y = jnp.array([1000.0, 2000.0])
        alpha_x, alpha_y = sie.deriv(x, y)
        
        # Should handle large coordinates
        assert not jnp.any(jnp.isnan(alpha_x))
        assert not jnp.any(jnp.isnan(alpha_y))
    
    def test_very_small_coordinates(self):
        """Test with very small coordinate values."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        x = jnp.array([1e-10, 2e-10])
        y = jnp.array([1e-10, 2e-10])
        alpha_x, alpha_y = sie.deriv(x, y)
        
        # Should handle small coordinates
        assert not jnp.any(jnp.isnan(alpha_x))
        assert not jnp.any(jnp.isnan(alpha_y))
    
    def test_mixed_scale_values(self):
        """Test with mixed scale values (large and small)."""
        x = jnp.array([1e-5, 1.0, 1000.0])
        y = jnp.array([1e-5, 1.0, 1000.0])
        
        sie = SIE(theta_E=1.5, e1=0.1, e2=0.05, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        alpha_x, alpha_y = sie.deriv(x, y)
        
        assert not jnp.any(jnp.isnan(alpha_x))
        assert not jnp.any(jnp.isnan(alpha_y))
