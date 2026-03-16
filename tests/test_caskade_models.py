"""
Unit tests for physical models.

This module tests the implementations in Models.
"""

import pytest
import numpy as np
import jax.numpy as jnp

# Import model implementations
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import Shear
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse as Sersic
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import GaussianEllipse as Gaussian
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel


class TestSIE:
    """Test SIE mass profile"""

    def test_sie_deflection(self):
        """Test that SIE deflection works correctly"""
        # Test parameters
        theta_E = 1.5
        e1, e2 = 0.05, -0.03
        center_x, center_y = 0.1, -0.2

        # Create test grid
        x = jnp.linspace(-2, 2, 50)
        y = jnp.linspace(-2, 2, 50)
        X, Y = jnp.meshgrid(x, y)

        # Create SIE model
        sie = SIE(
            theta_E=theta_E, e1=e1, e2=e2,
            center_x=center_x, center_y=center_y
        )
        # Set parameters to static
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()

        alpha_x, alpha_y = sie.deriv(X, Y)

        # Verify results are valid
        assert not jnp.any(jnp.isnan(alpha_x)), "SIE deflection x has NaN values"
        assert not jnp.any(jnp.isnan(alpha_y)), "SIE deflection y has NaN values"
        assert alpha_x.shape == X.shape, "SIE deflection x shape mismatch"
        assert alpha_y.shape == X.shape, "SIE deflection y shape mismatch"

        # Check that deflection magnitude is reasonable
        deflection_mag = jnp.sqrt(alpha_x**2 + alpha_y**2)
        assert jnp.max(deflection_mag) > 0, "SIE deflection should be non-zero"
        assert jnp.max(deflection_mag) < 100, "SIE deflection magnitude seems too large"


class TestShear:
    """Test Shear mass profile"""

    def test_shear_deflection(self):
        """Test that Shear deflection works correctly"""
        # Test parameters
        gamma1, gamma2 = 0.05, 0.03

        # Create test grid
        x = jnp.linspace(-2, 2, 50)
        y = jnp.linspace(-2, 2, 50)
        X, Y = jnp.meshgrid(x, y)

        # Create Shear model
        shear = Shear(gamma1=gamma1, gamma2=gamma2)
        shear.gamma1.to_static()
        shear.gamma2.to_static()

        alpha_x, alpha_y = shear.deriv(X, Y)

        # Verify results are valid
        assert not jnp.any(jnp.isnan(alpha_x)), "Shear deflection x has NaN values"
        assert not jnp.any(jnp.isnan(alpha_y)), "Shear deflection y has NaN values"
        assert alpha_x.shape == X.shape, "Shear deflection x shape mismatch"
        assert alpha_y.shape == X.shape, "Shear deflection y shape mismatch"

        # Shear produces linear deflection
        assert jnp.max(jnp.abs(alpha_x)) > 0, "Shear deflection x should be non-zero"
        assert jnp.max(jnp.abs(alpha_y)) > 0, "Shear deflection y should be non-zero"


class TestSersic:
    """Test Sersic light profile"""

    def test_sersic_light(self):
        """Test that Sersic brightness works correctly"""
        # Test parameters
        R_sersic, n_sersic = 1.0, 4.0
        e1, e2 = 0.1, -0.05
        center_x, center_y = 0.0, 0.0
        Ie = 1.0

        # Create test grid
        x = jnp.linspace(-3, 3, 50)
        y = jnp.linspace(-3, 3, 50)
        X, Y = jnp.meshgrid(x, y)

        # Create Sersic model
        sersic = Sersic(
            R_sersic=R_sersic, n_sersic=n_sersic,
            e1=e1, e2=e2, center_x=center_x, center_y=center_y, Ie=Ie
        )
        sersic.R_sersic.to_static()
        sersic.n_sersic.to_static()
        sersic.e1.to_static()
        sersic.e2.to_static()
        sersic.center_x.to_static()
        sersic.center_y.to_static()
        sersic.Ie.to_static()

        brightness = sersic.light(X, Y)

        # Verify results are valid
        assert not jnp.any(jnp.isnan(brightness)), "Sersic brightness has NaN values"
        assert brightness.shape == X.shape, "Sersic brightness shape mismatch"
        assert jnp.max(brightness) > 0, "Sersic brightness should be positive"
        assert jnp.min(brightness) >= 0, "Sersic brightness should be non-negative"


class TestGaussian:
    """Test Gaussian light profile"""

    def test_gaussian_light(self):
        """Test that Gaussian brightness works correctly"""
        # Test parameters
        flux, sigma = 10.0, 0.5
        e1, e2 = 0.15, -0.1
        center_x, center_y = 0.5, -0.3

        # Create test grid
        x = jnp.linspace(-2, 2, 50)
        y = jnp.linspace(-2, 2, 50)
        X, Y = jnp.meshgrid(x, y)

        # Create Gaussian model
        gaussian = Gaussian(
            flux=flux, sigma=sigma, e1=e1, e2=e2,
            center_x=center_x, center_y=center_y
        )
        gaussian.flux.to_static()
        gaussian.sigma.to_static()
        gaussian.e1.to_static()
        gaussian.e2.to_static()
        gaussian.center_x.to_static()
        gaussian.center_y.to_static()

        brightness = gaussian.light(X, Y)

        # Verify results are valid
        assert not jnp.any(jnp.isnan(brightness)), "Gaussian brightness has NaN values"
        assert brightness.shape == X.shape, "Gaussian brightness shape mismatch"
        assert jnp.max(brightness) > 0, "Gaussian brightness should be positive"
        assert jnp.min(brightness) >= 0, "Gaussian brightness should be non-negative"


class TestConstantBackground:
    """Test ConstantBackground light profile."""

    def test_constant_background_public_import(self):
        """Test that ConstantBackground is re-exported from the public package API."""
        from TinyLensGpu.PhysicalModel import ConstantBackground

        background = ConstantBackground(intensity=1.0)
        background.intensity.to_static()

        brightness = background.light(jnp.ones((2, 3)), jnp.zeros((2, 3)))

        assert brightness.shape == (2, 3)
        assert jnp.allclose(brightness, 1.0)

    def test_constant_background_light(self):
        """Test that ConstantBackground returns a uniform intensity field."""
        from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import ConstantBackground

        intensity = 2.5

        x = jnp.linspace(-2, 2, 12)
        y = jnp.linspace(-1, 1, 8)
        X, Y = jnp.meshgrid(x, y)

        background = ConstantBackground(intensity=intensity)
        background.intensity.to_static()

        brightness = background.light(X, Y)

        assert brightness.shape == X.shape, "ConstantBackground brightness shape mismatch"
        assert jnp.allclose(brightness, intensity), "ConstantBackground should be spatially uniform"


class TestPhysicalModel:
    """Test PhysicalModel composite class"""

    def test_physical_model_construction(self):
        """Test that PhysicalModel can be constructed with various components"""
        sie = SIE(theta_E=1.0, e1=0.0, e2=0.0,
                  center_x=0.0, center_y=0.0)
        shear = Shear(gamma1=0.05, gamma2=0.0)
        sersic = Sersic(R_sersic=1.0, n_sersic=4.0, e1=0.0, e2=0.0,
                        center_x=0.0, center_y=0.0, Ie=1.0)

        # Create composite model
        model = PhysicalModel(
            lens_mass=[sie, shear],
            source_light=[sersic],
            lens_light=[]
        )

        # Verify structure
        assert len(model.lens_mass) == 2
        assert len(model.source_light) == 1
        assert len(model.lens_light) == 0

    def test_physical_model_deflection(self):
        """Test that PhysicalModel deflection works correctly"""
        # Create components with known parameters
        sie = SIE(theta_E=1.0, e1=0.0, e2=0.0,
                  center_x=0.0, center_y=0.0)
        shear = Shear(gamma1=0.05, gamma2=0.0)

        # Set to static
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()

        shear.gamma1.to_static()
        shear.gamma2.to_static()

        # Create composite
        model = PhysicalModel(lens_mass=[sie, shear])

        # Test grid
        x = jnp.linspace(-1, 1, 20)
        y = jnp.linspace(-1, 1, 20)
        X, Y = jnp.meshgrid(x, y)

        # Get deflections from composite
        alpha_x_total, alpha_y_total = model.deflection(X, Y)

        # Verify results are valid
        assert not jnp.any(jnp.isnan(alpha_x_total)), "Composite deflection x has NaN values"
        assert not jnp.any(jnp.isnan(alpha_y_total)), "Composite deflection y has NaN values"
        assert alpha_x_total.shape == X.shape, "Composite deflection x shape mismatch"
        assert alpha_y_total.shape == X.shape, "Composite deflection y shape mismatch"

        # Check that deflection is non-zero (SIE + Shear)
        deflection_mag = jnp.sqrt(alpha_x_total**2 + alpha_y_total**2)
        assert jnp.max(deflection_mag) > 0, "Composite deflection should be non-zero"
