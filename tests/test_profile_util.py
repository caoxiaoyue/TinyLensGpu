"""
Unit tests for Profile utility functions.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from TinyLensGpu.Profile import util


@pytest.mark.unit
class TestEllipticityConversions:
    """Test ellipticity and phi/q conversion functions."""
    
    def test_phi_q2_ellipticity_round_trip(self):
        """Test conversion from phi/q to ellipticity and back."""
        phi = jnp.pi / 4.0  # 45 degrees
        q = 0.7
        
        e1, e2 = util.phi_q2_ellipticity(phi, q)
        phi_out, q_out = util.ellipticity2phi_q(e1, e2)
        
        assert jnp.allclose(phi, phi_out, atol=1e-5)
        assert jnp.allclose(q, q_out, atol=1e-5)
    
    def test_ellipticity2phi_q_round_trip(self):
        """Test conversion from ellipticity to phi/q and back."""
        e1 = 0.2
        e2 = 0.1
        
        phi, q = util.ellipticity2phi_q(e1, e2)
        e1_out, e2_out = util.phi_q2_ellipticity(phi, q)
        
        assert jnp.allclose(e1, e1_out, atol=1e-5)
        assert jnp.allclose(e2, e2_out, atol=1e-5)
    
    def test_zero_ellipticity(self):
        """Test conversion with zero ellipticity (circular case)."""
        e1 = 0.0
        e2 = 0.0
        
        phi, q = util.ellipticity2phi_q(e1, e2)
        
        # For circular case, q should be close to 1
        assert jnp.allclose(q, 1.0, atol=1e-3)
    
    def test_ellipticity_range(self):
        """Test that ellipticity stays within valid range."""
        phi_values = jnp.linspace(0, jnp.pi, 10)
        q_values = jnp.linspace(0.3, 1.0, 10)
        
        for phi in phi_values:
            for q in q_values:
                e1, e2 = util.phi_q2_ellipticity(phi, q)
                e = jnp.sqrt(e1**2 + e2**2)
                assert e < 1.0  # Ellipticity should be less than 1


@pytest.mark.unit
class TestShearConversions:
    """Test shear conversion functions."""
    
    def test_shear_polar2cartesian_round_trip(self):
        """Test conversion from polar to cartesian shear and back."""
        phi = jnp.pi / 6.0
        gamma = 0.05
        
        gamma1, gamma2 = util.shear_polar2cartesian(phi, gamma)
        phi_out, gamma_out = util.shear_cartesian2polar(gamma1, gamma2)
        
        assert jnp.allclose(phi, phi_out, atol=1e-5)
        assert jnp.allclose(gamma, gamma_out, atol=1e-5)
    
    def test_shear_cartesian2polar_round_trip(self):
        """Test conversion from cartesian to polar shear and back."""
        gamma1 = 0.03
        gamma2 = 0.04
        
        phi, gamma = util.shear_cartesian2polar(gamma1, gamma2)
        gamma1_out, gamma2_out = util.shear_polar2cartesian(phi, gamma)
        
        assert jnp.allclose(gamma1, gamma1_out, atol=1e-5)
        assert jnp.allclose(gamma2, gamma2_out, atol=1e-5)
    
    def test_zero_shear(self):
        """Test zero shear case."""
        gamma1 = 0.0
        gamma2 = 0.0
        
        phi, gamma = util.shear_cartesian2polar(gamma1, gamma2)
        
        assert jnp.allclose(gamma, 0.0, atol=1e-8)


@pytest.mark.unit
class TestCoordinateTransforms:
    """Test coordinate transformation functions."""
    
    def test_cart2polar_round_trip(self):
        """Test cartesian to polar conversion and back."""
        x = 1.5
        y = 2.0
        
        r, phi = util.cart2polar(x, y)
        x_out, y_out = util.polar2cart(r, phi)
        
        assert jnp.allclose(x, x_out, atol=1e-5)
        assert jnp.allclose(y, y_out, atol=1e-5)
    
    def test_polar2cart_round_trip(self):
        """Test polar to cartesian conversion and back."""
        r = 2.5
        phi = jnp.pi / 3.0
        
        x, y = util.polar2cart(r, phi)
        r_out, phi_out = util.cart2polar(x, y)
        
        assert jnp.allclose(r, r_out, atol=1e-5)
        assert jnp.allclose(phi, phi_out, atol=1e-5)
    
    def test_xy_transform_rotation(self):
        """Test xy coordinate transformation with rotation."""
        x = 1.0
        y = 0.0
        xc = 0.0
        yc = 0.0
        phi = jnp.pi / 2.0  # 90 degree rotation
        
        x_rot, y_rot = util.xy_transform(x, y, xc, yc, phi)
        
        # After 90 degree rotation, (1,0) should become (0, -1)
        assert jnp.allclose(x_rot, 0.0, atol=1e-5)
        assert jnp.allclose(y_rot, -1.0, atol=1e-5)
    
    def test_xy_transform_translation(self):
        """Test xy coordinate transformation with translation."""
        x = 2.0
        y = 3.0
        xc = 1.0
        yc = 1.5
        phi = 0.0  # No rotation
        
        x_shift, y_shift = util.xy_transform(x, y, xc, yc, phi)
        
        assert jnp.allclose(x_shift, 1.0, atol=1e-5)
        assert jnp.allclose(y_shift, 1.5, atol=1e-5)
    
    def test_ellipse2circle_transform(self):
        """Test ellipse to circle transformation."""
        # Create an elliptical grid
        x = jnp.array([1.0, 0.0])
        y = jnp.array([0.0, 1.0])
        e1 = 0.3
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0
        
        x_circle, y_circle = util.ellipse2circle_transform(
            x, y, e1, e2, center_x, center_y
        )
        
        # The transformation should make the ellipse circular
        # Check that the output has the expected shape
        assert x_circle.shape == x.shape
        assert y_circle.shape == y.shape
    
    def test_relocate_radii(self):
        """Test relocate_radii handles origin singularity."""
        x = jnp.array([0.0, 1.0, 0.0])
        y = jnp.array([0.0, 0.0, 1.0])
        
        x_out, y_out, r_out = util.relocate_radii(x, y)
        
        # Check that radius at origin is not zero
        assert r_out[0] > 0.0
        assert jnp.allclose(r_out[0], 1e-5, atol=1e-8)
        
        # Check that other points are unchanged
        assert jnp.allclose(r_out[1], 1.0, atol=1e-5)
        assert jnp.allclose(r_out[2], 1.0, atol=1e-5)


@pytest.mark.unit
class TestHypergeometric:
    """Test hypergeometric function."""
    
    def test_hyp2f1_basic(self):
        """Test hypergeometric function with simple inputs."""
        a = 1.0
        b = 1.0
        c = 2.0
        z = 0.5
        
        result = util.hyp2f1_series(a, b, c, z, max_terms=10)
        
        # Check that result is a complex number
        assert jnp.iscomplexobj(result)
        
        # For these parameters, result should be approximately 1.405
        # This is based on known values of hypergeometric function
        assert abs(result.real - 1.405) < 0.1
    
    def test_hyp2f1_zero(self):
        """Test hypergeometric function at z=0."""
        a = 1.0
        b = 1.0
        c = 2.0
        z = 0.0
        
        result = util.hyp2f1_series(a, b, c, z, max_terms=10)
        
        # At z=0, hypergeometric function should be 1
        assert jnp.allclose(result.real, 1.0, atol=1e-5)

