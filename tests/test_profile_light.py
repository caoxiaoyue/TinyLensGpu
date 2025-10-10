"""
Unit tests for light profile classes.
"""
import pytest
import jax.numpy as jnp
from TinyLensGpu.Profile.Light.Sersic import Sersic, SersicEllipse
from TinyLensGpu.Profile.Light.Gaussian import Gaussian, GaussianEllipse


@pytest.mark.unit
class TestSersicProfile:
    """Test Sersic light profile."""
    
    def test_sersic_initialization(self):
        """Test Sersic profile initialization."""
        profile = Sersic()
        assert profile.name == "SERSIC"
        assert len(profile.params) == 5
        assert "R_sersic" in profile.params
        assert "n_sersic" in profile.params
        assert "Ie" in profile.params
    
    def test_sersic_light_at_center(self):
        """Test Sersic light profile at center."""
        profile = Sersic()
        x = 0.0
        y = 0.0
        R_sersic = 1.0
        n_sersic = 4.0
        center_x = 0.0
        center_y = 0.0
        Ie = 1.0
        
        light = profile.light(x, y, R_sersic, n_sersic, center_x, center_y, Ie)
        
        # At center (r=0), the profile should equal Ie * exp(bn)
        # where bn ≈ 7.67 for n=4
        bn = 1.9992 * n_sersic - 0.3271
        expected = Ie * jnp.exp(bn)
        assert jnp.allclose(light, expected, atol=1e-5)
    
    def test_sersic_light_decreases_with_radius(self):
        """Test that Sersic profile decreases with radius."""
        profile = Sersic()
        R_sersic = 1.0
        n_sersic = 4.0
        center_x = 0.0
        center_y = 0.0
        Ie = 1.0
        
        light_center = profile.light(0.0, 0.0, R_sersic, n_sersic, center_x, center_y, Ie)
        light_radius = profile.light(2.0, 0.0, R_sersic, n_sersic, center_x, center_y, Ie)
        
        # Light should decrease with radius
        assert light_radius < light_center
    
    def test_sersic_symmetry(self):
        """Test Sersic profile is radially symmetric."""
        profile = Sersic()
        R_sersic = 1.0
        n_sersic = 2.0
        center_x = 0.0
        center_y = 0.0
        Ie = 1.0
        
        # Check symmetry at same radius but different angles
        light1 = profile.light(1.0, 0.0, R_sersic, n_sersic, center_x, center_y, Ie)
        light2 = profile.light(0.0, 1.0, R_sersic, n_sersic, center_x, center_y, Ie)
        light3 = profile.light(0.707, 0.707, R_sersic, n_sersic, center_x, center_y, Ie)
        
        assert jnp.allclose(light1, light2, atol=1e-5)
        assert jnp.allclose(light1, light3, atol=1e-3)


@pytest.mark.unit
class TestSersicEllipseProfile:
    """Test elliptical Sersic light profile."""
    
    def test_sersic_ellipse_initialization(self):
        """Test SersicEllipse profile initialization."""
        profile = SersicEllipse()
        assert profile.name == "SERSIC_ELLIPSE"
        assert len(profile.params) == 7
        assert "e1" in profile.params
        assert "e2" in profile.params
    
    def test_sersic_ellipse_circular_case(self):
        """Test that SersicEllipse reduces to Sersic for circular case."""
        profile = SersicEllipse()
        x = 1.0
        y = 0.0
        R_sersic = 1.0
        n_sersic = 4.0
        e1 = 0.0
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0
        Ie = 1.0
        
        # For circular case (e1=e2=0), elliptical and circular should be similar
        light_ellipse = profile.light(x, y, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie)
        
        circular_profile = Sersic()
        light_circular = circular_profile.light(x, y, R_sersic, n_sersic, center_x, center_y, Ie)
        
        assert jnp.allclose(light_ellipse, light_circular, rtol=0.1)
    
    def test_sersic_ellipse_asymmetry(self):
        """Test that elliptical Sersic is not radially symmetric."""
        profile = SersicEllipse()
        R_sersic = 1.0
        n_sersic = 4.0
        e1 = 0.3
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0
        Ie = 1.0
        
        # Check that light differs along major and minor axes
        light_x = profile.light(1.0, 0.0, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie)
        light_y = profile.light(0.0, 1.0, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie)
        
        # Should be different due to ellipticity
        assert not jnp.allclose(light_x, light_y, rtol=0.01)


@pytest.mark.unit
class TestGaussianProfile:
    """Test Gaussian light profile."""
    
    def test_gaussian_initialization(self):
        """Test Gaussian profile initialization."""
        profile = Gaussian()
        assert profile.name == "Gaussian"
        assert len(profile.params) == 4
        assert "flux" in profile.params
        assert "sigma" in profile.params
    
    def test_gaussian_light_at_center(self):
        """Test Gaussian light profile at center."""
        profile = Gaussian()
        x = 0.0
        y = 0.0
        flux = 10.0
        sigma = 1.0
        center_x = 0.0
        center_y = 0.0
        
        light = profile.light(x, y, flux, sigma, center_x, center_y)
        
        # At center, light should be maximum
        expected = flux / (2 * jnp.pi * sigma**2)
        assert jnp.allclose(light, expected, atol=1e-5)
    
    def test_gaussian_light_decreases_with_radius(self):
        """Test that Gaussian profile decreases with radius."""
        profile = Gaussian()
        flux = 10.0
        sigma = 1.0
        center_x = 0.0
        center_y = 0.0
        
        light_center = profile.light(0.0, 0.0, flux, sigma, center_x, center_y)
        light_sigma = profile.light(sigma, 0.0, flux, sigma, center_x, center_y)
        light_far = profile.light(3*sigma, 0.0, flux, sigma, center_x, center_y)
        
        # Light should decrease with radius
        assert light_sigma < light_center
        assert light_far < light_sigma
    
    def test_gaussian_symmetry(self):
        """Test Gaussian profile is radially symmetric."""
        profile = Gaussian()
        flux = 10.0
        sigma = 1.0
        center_x = 0.0
        center_y = 0.0
        
        # Check symmetry at same radius but different angles
        light1 = profile.light(1.0, 0.0, flux, sigma, center_x, center_y)
        light2 = profile.light(0.0, 1.0, flux, sigma, center_x, center_y)
        light3 = profile.light(0.707, 0.707, flux, sigma, center_x, center_y)
        
        assert jnp.allclose(light1, light2, atol=1e-5)
        assert jnp.allclose(light1, light3, atol=1e-3)


@pytest.mark.unit
class TestGaussianEllipseProfile:
    """Test elliptical Gaussian light profile."""
    
    def test_gaussian_ellipse_initialization(self):
        """Test GaussianEllipse profile initialization."""
        profile = GaussianEllipse()
        assert profile.name == "GaussianEllipse"
        assert len(profile.params) == 6
        assert "e1" in profile.params
        assert "e2" in profile.params
    
    def test_gaussian_ellipse_circular_case(self):
        """Test that GaussianEllipse reduces to Gaussian for circular case."""
        profile = GaussianEllipse()
        x = 1.0
        y = 0.0
        flux = 10.0
        sigma = 1.0
        e1 = 0.0
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0
        
        # For circular case (e1=e2=0), elliptical and circular should be similar
        light_ellipse = profile.light(x, y, flux, sigma, e1, e2, center_x, center_y)
        
        circular_profile = Gaussian()
        light_circular = circular_profile.light(x, y, flux, sigma, center_x, center_y)
        
        assert jnp.allclose(light_ellipse, light_circular, rtol=0.1)
    
    def test_gaussian_ellipse_asymmetry(self):
        """Test that elliptical Gaussian is not radially symmetric."""
        profile = GaussianEllipse()
        flux = 10.0
        sigma = 1.0
        e1 = 0.3
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0
        
        # Check that light differs along major and minor axes
        light_x = profile.light(1.0, 0.0, flux, sigma, e1, e2, center_x, center_y)
        light_y = profile.light(0.0, 1.0, flux, sigma, e1, e2, center_x, center_y)
        
        # Should be different due to ellipticity
        assert not jnp.allclose(light_x, light_y, rtol=0.01)

