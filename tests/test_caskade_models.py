"""
Unit tests for caskade-based physical models.

This module tests the caskade-based implementations in CaskadeModels.
"""

import pytest
import numpy as np
import jax.numpy as jnp

# Import caskade implementations
from TinyLensGpu.CaskadeModels.mass import SIE as SIE_caskade
from TinyLensGpu.CaskadeModels.mass import Shear as Shear_caskade
from TinyLensGpu.CaskadeModels.light import SersicEllipse as Sersic_caskade
from TinyLensGpu.CaskadeModels.light import GaussianEllipse as Gaussian_caskade
from TinyLensGpu.CaskadeModels.composite import PhysicalModel


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

        # Caskade implementation
        sie_caskade = SIE_caskade(
            theta_E=theta_E, e1=e1, e2=e2,
            center_x=center_x, center_y=center_y
        )
        # Set parameters to static
        sie_caskade.theta_E.to_static(theta_E)
        sie_caskade.e1.to_static(e1)
        sie_caskade.e2.to_static(e2)
        sie_caskade.center_x.to_static(center_x)
        sie_caskade.center_y.to_static(center_y)

        alpha_x_cask, alpha_y_cask = sie_caskade.deriv(X, Y)

        # Verify results are valid
        assert not jnp.any(jnp.isnan(alpha_x_cask)), "SIE deflection x has NaN values"
        assert not jnp.any(jnp.isnan(alpha_y_cask)), "SIE deflection y has NaN values"
        assert alpha_x_cask.shape == X.shape, "SIE deflection x shape mismatch"
        assert alpha_y_cask.shape == X.shape, "SIE deflection y shape mismatch"

        # Check that deflection magnitude is reasonable
        deflection_mag = jnp.sqrt(alpha_x_cask**2 + alpha_y_cask**2)
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

        # Caskade implementation
        shear_caskade = Shear_caskade(gamma1=gamma1, gamma2=gamma2)
        shear_caskade.gamma1.to_static(gamma1)
        shear_caskade.gamma2.to_static(gamma2)

        alpha_x_cask, alpha_y_cask = shear_caskade.deriv(X, Y)

        # Verify results are valid
        assert not jnp.any(jnp.isnan(alpha_x_cask)), "Shear deflection x has NaN values"
        assert not jnp.any(jnp.isnan(alpha_y_cask)), "Shear deflection y has NaN values"
        assert alpha_x_cask.shape == X.shape, "Shear deflection x shape mismatch"
        assert alpha_y_cask.shape == X.shape, "Shear deflection y shape mismatch"

        # Shear produces linear deflection
        assert jnp.max(jnp.abs(alpha_x_cask)) > 0, "Shear deflection x should be non-zero"
        assert jnp.max(jnp.abs(alpha_y_cask)) > 0, "Shear deflection y should be non-zero"


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

        # Caskade implementation
        sersic_caskade = Sersic_caskade(
            R_sersic=R_sersic, n_sersic=n_sersic,
            e1=e1, e2=e2, center_x=center_x, center_y=center_y, Ie=Ie
        )
        sersic_caskade.R_sersic.to_static(R_sersic)
        sersic_caskade.n_sersic.to_static(n_sersic)
        sersic_caskade.e1.to_static(e1)
        sersic_caskade.e2.to_static(e2)
        sersic_caskade.center_x.to_static(center_x)
        sersic_caskade.center_y.to_static(center_y)
        sersic_caskade.Ie.to_static(Ie)

        brightness_cask = sersic_caskade.light(X, Y)

        # Verify results are valid
        assert not jnp.any(jnp.isnan(brightness_cask)), "Sersic brightness has NaN values"
        assert brightness_cask.shape == X.shape, "Sersic brightness shape mismatch"
        assert jnp.max(brightness_cask) > 0, "Sersic brightness should be positive"
        assert jnp.min(brightness_cask) >= 0, "Sersic brightness should be non-negative"


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

        # Caskade implementation
        gaussian_caskade = Gaussian_caskade(
            flux=flux, sigma=sigma, e1=e1, e2=e2,
            center_x=center_x, center_y=center_y
        )
        gaussian_caskade.flux.to_static(flux)
        gaussian_caskade.sigma.to_static(sigma)
        gaussian_caskade.e1.to_static(e1)
        gaussian_caskade.e2.to_static(e2)
        gaussian_caskade.center_x.to_static(center_x)
        gaussian_caskade.center_y.to_static(center_y)

        brightness_cask = gaussian_caskade.light(X, Y)

        # Verify results are valid
        assert not jnp.any(jnp.isnan(brightness_cask)), "Gaussian brightness has NaN values"
        assert brightness_cask.shape == X.shape, "Gaussian brightness shape mismatch"
        assert jnp.max(brightness_cask) > 0, "Gaussian brightness should be positive"
        assert jnp.min(brightness_cask) >= 0, "Gaussian brightness should be non-negative"


class TestPhysicalModel:
    """Test PhysicalModel composite class"""

    def test_physical_model_construction(self):
        """Test that PhysicalModel can be constructed with various components"""
        sie = SIE_caskade(theta_E=1.0, e1=0.0, e2=0.0,
                          center_x=0.0, center_y=0.0)
        shear = Shear_caskade(gamma1=0.05, gamma2=0.0)
        sersic = Sersic_caskade(R_sersic=1.0, n_sersic=4.0, e1=0.0, e2=0.0,
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
        sie = SIE_caskade(theta_E=1.0, e1=0.0, e2=0.0,
                          center_x=0.0, center_y=0.0)
        shear = Shear_caskade(gamma1=0.05, gamma2=0.0)

        # Set to static
        sie.theta_E.to_static(1.0)
        sie.e1.to_static(0.0)
        sie.e2.to_static(0.0)
        sie.center_x.to_static(0.0)
        sie.center_y.to_static(0.0)

        shear.gamma1.to_static(0.05)
        shear.gamma2.to_static(0.0)

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
