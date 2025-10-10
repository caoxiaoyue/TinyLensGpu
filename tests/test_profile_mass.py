"""
Unit tests for mass profile classes.
"""
import pytest
import jax.numpy as jnp
from TinyLensGpu.Profile.Mass.Sie import SIE
from TinyLensGpu.Profile.Mass.Shear import Shear


@pytest.mark.unit
class TestSIEProfile:
    """Test SIE (Singular Isothermal Ellipsoid) mass profile."""
    
    def test_sie_initialization(self):
        """Test SIE profile initialization."""
        profile = SIE()
        assert profile.name == "SIE"
        assert len(profile.params) == 5
        assert "theta_E" in profile.params
        assert "e1" in profile.params
        assert "e2" in profile.params
        assert "center_x" in profile.params
        assert "center_y" in profile.params
    
    def test_sie_deriv_at_center(self):
        """Test SIE deflection at center (handled by numerical regularization)."""
        profile = SIE()
        x = 0.0
        y = 0.0
        theta_E = 1.0
        e1 = 0.0
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0
        
        alpha_x, alpha_y = profile.deriv(x, y, theta_E, e1, e2, center_x, center_y)
        
        # At center, deflection is handled by numerical regularization
        # The code relocates the origin to avoid singularity, so deflection is finite
        assert jnp.isfinite(alpha_x)
        assert jnp.isfinite(alpha_y)
        # Deflection magnitude should be on order of theta_E
        assert jnp.abs(alpha_x) < 2.0 * theta_E
        assert jnp.abs(alpha_y) < 2.0 * theta_E
    
    def test_sie_deriv_increases_with_theta_E(self):
        """Test that deflection increases with Einstein radius."""
        profile = SIE()
        x = 1.0
        y = 0.0
        e1 = 0.0
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0
        
        alpha_x1, alpha_y1 = profile.deriv(x, y, 1.0, e1, e2, center_x, center_y)
        alpha_x2, alpha_y2 = profile.deriv(x, y, 2.0, e1, e2, center_x, center_y)
        
        # Deflection should scale approximately with theta_E
        assert jnp.abs(alpha_x2) > jnp.abs(alpha_x1)
    
    def test_sie_circular_symmetry(self):
        """Test SIE with circular case (e1=e2=0)."""
        profile = SIE()
        theta_E = 1.0
        e1 = 0.0
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0
        
        # Check that deflection at same radius is similar
        r = 1.0
        alpha_x1, alpha_y1 = profile.deriv(r, 0.0, theta_E, e1, e2, center_x, center_y)
        alpha_x2, alpha_y2 = profile.deriv(0.0, r, theta_E, e1, e2, center_x, center_y)
        
        # Magnitudes should be similar for circular case
        mag1 = jnp.sqrt(alpha_x1**2 + alpha_y1**2)
        mag2 = jnp.sqrt(alpha_x2**2 + alpha_y2**2)
        assert jnp.allclose(mag1, mag2, rtol=0.1)
    
    def test_sie_ellipticity_breaks_symmetry(self):
        """Test that ellipticity breaks radial symmetry."""
        profile = SIE()
        theta_E = 1.0
        e1 = 0.3
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0
        
        # Check deflection at same radius but different directions
        r = 1.0
        alpha_x1, alpha_y1 = profile.deriv(r, 0.0, theta_E, e1, e2, center_x, center_y)
        alpha_x2, alpha_y2 = profile.deriv(0.0, r, theta_E, e1, e2, center_x, center_y)
        
        # Magnitudes should differ due to ellipticity
        mag1 = jnp.sqrt(alpha_x1**2 + alpha_y1**2)
        mag2 = jnp.sqrt(alpha_x2**2 + alpha_y2**2)
        assert not jnp.allclose(mag1, mag2, rtol=0.01)
    
    def test_sie_deflection_direction(self):
        """Test that deflection points toward center."""
        profile = SIE()
        x = 2.0
        y = 0.0
        theta_E = 1.0
        e1 = 0.0
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0
        
        alpha_x, alpha_y = profile.deriv(x, y, theta_E, e1, e2, center_x, center_y)
        
        # For point on positive x-axis, deflection should be toward center (positive x direction)
        assert alpha_x > 0
        assert jnp.abs(alpha_y) < 0.1  # Should be small for point on x-axis
    
    def test_sie_batch_computation(self):
        """Test SIE with batched inputs."""
        profile = SIE()
        x = jnp.array([1.0, 2.0, 0.5])
        y = jnp.array([0.0, 1.0, 0.5])
        theta_E = 1.0
        e1 = 0.1
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0
        
        alpha_x, alpha_y = profile.deriv(x, y, theta_E, e1, e2, center_x, center_y)
        
        # Check that output has correct shape
        assert alpha_x.shape == x.shape
        assert alpha_y.shape == y.shape


@pytest.mark.unit
class TestShearProfile:
    """Test external shear profile."""
    
    def test_shear_initialization(self):
        """Test Shear profile initialization."""
        profile = Shear()
        assert profile.name == "SHEAR"
        assert len(profile.params) == 2
        assert "gamma1" in profile.params
        assert "gamma2" in profile.params
    
    def test_shear_deriv_linear(self):
        """Test that shear deflection is linear with position."""
        profile = Shear()
        gamma1 = 0.05
        gamma2 = 0.0
        
        # Test at different positions
        alpha_x1, alpha_y1 = profile.deriv(1.0, 0.0, gamma1, gamma2)
        alpha_x2, alpha_y2 = profile.deriv(2.0, 0.0, gamma1, gamma2)
        
        # Shear deflection should scale linearly with position
        assert jnp.allclose(alpha_x2 / alpha_x1, 2.0, rtol=0.01)
    
    def test_shear_deriv_zero_shear(self):
        """Test shear with zero shear values."""
        profile = Shear()
        x = 1.0
        y = 1.0
        gamma1 = 0.0
        gamma2 = 0.0
        
        alpha_x, alpha_y = profile.deriv(x, y, gamma1, gamma2)
        
        # Zero shear should give zero deflection
        assert jnp.allclose(alpha_x, 0.0, atol=1e-8)
        assert jnp.allclose(alpha_y, 0.0, atol=1e-8)
    
    def test_shear_deriv_gamma1_only(self):
        """Test shear with only gamma1 component."""
        profile = Shear()
        x = 1.0
        y = 1.0
        gamma1 = 0.05
        gamma2 = 0.0
        
        alpha_x, alpha_y = profile.deriv(x, y, gamma1, gamma2)
        
        # With only gamma1, we expect specific pattern
        # gamma1 stretches in x-direction
        assert jnp.abs(alpha_x) > 0
    
    def test_shear_deriv_gamma2_only(self):
        """Test shear with only gamma2 component."""
        profile = Shear()
        x = 1.0
        y = 1.0
        gamma1 = 0.0
        gamma2 = 0.05
        
        alpha_x, alpha_y = profile.deriv(x, y, gamma1, gamma2)
        
        # With only gamma2, we expect specific pattern
        # gamma2 creates 45-degree shear
        assert jnp.abs(alpha_x) > 0
        assert jnp.abs(alpha_y) > 0
    
    def test_shear_batch_computation(self):
        """Test Shear with batched inputs."""
        profile = Shear()
        x = jnp.array([1.0, 2.0, 0.5])
        y = jnp.array([0.0, 1.0, 0.5])
        gamma1 = 0.05
        gamma2 = 0.02
        
        alpha_x, alpha_y = profile.deriv(x, y, gamma1, gamma2)
        
        # Check that output has correct shape
        assert alpha_x.shape == x.shape
        assert alpha_y.shape == y.shape
    
    def test_shear_symmetry_properties(self):
        """Test shear symmetry properties."""
        profile = Shear()
        gamma1 = 0.05
        gamma2 = 0.0
        
        # Test at symmetric positions
        alpha_x1, alpha_y1 = profile.deriv(1.0, 0.0, gamma1, gamma2)
        alpha_x2, alpha_y2 = profile.deriv(-1.0, 0.0, gamma1, gamma2)
        
        # Deflections should be opposite
        assert jnp.allclose(alpha_x1, -alpha_x2, rtol=0.01)
        assert jnp.allclose(alpha_y1, -alpha_y2, rtol=0.01)

