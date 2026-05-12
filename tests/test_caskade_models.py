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


class TestPhotometry:
    """Test photometry utility functions for magnitude conversions."""

    def test_mag2cps_zero_point(self):
        """At zero point, magnitude should give 1 count per second."""
        from TinyLensGpu.utils.photometry import mag2cps

        result = mag2cps(22.0, 22.0)
        assert jnp.isclose(result, 1.0), "mag2cps at zero point should be 1.0"

    def test_mag2cps_brighter(self):
        """Brighter object (smaller magnitude) should give more counts."""
        from TinyLensGpu.utils.photometry import mag2cps

        result = mag2cps(21.0, 22.0)
        expected = 10.0 ** 0.4  # 1 magnitude brighter = 10^(0.4) ~ 2.512x
        assert jnp.isclose(result, expected, rtol=1e-6), \
            f"mag2cps(21, 22) should be ~{expected:.4f}, got {result}"

    def test_mag2cps_fainter(self):
        """Fainter object (larger magnitude) should give fewer counts."""
        from TinyLensGpu.utils.photometry import mag2cps

        result = mag2cps(23.0, 22.0)
        expected = 10.0 ** (-0.4)  # 1 magnitude fainter = 10^(-0.4) ~ 0.398x
        assert jnp.isclose(result, expected, rtol=1e-6), \
            f"mag2cps(23, 22) should be ~{expected:.4f}, got {result}"

    def test_cps2mag_roundtrip(self):
        """cps2mag should invert mag2cps exactly."""
        from TinyLensGpu.utils.photometry import mag2cps, cps2mag

        test_mags = jnp.array([20.0, 22.0, 24.0, 18.5, 25.3])
        zp = 22.0
        cps = mag2cps(test_mags, zp)
        recovered = cps2mag(cps, zp)
        assert jnp.allclose(test_mags, recovered, rtol=1e-6), \
            "cps2mag(mag2cps(m, zp), zp) should recover m exactly"

    def test_cps2mag_zero_point(self):
        """1 cps at zero point should equal the zero point magnitude."""
        from TinyLensGpu.utils.photometry import cps2mag

        result = cps2mag(1.0, 22.0)
        assert jnp.isclose(result, 22.0), "cps2mag(1.0, 22.0) should be 22.0"


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

    def test_sersic_magnitude_constructor(self):
        """Test that Sersic can be constructed with magnitude instead of Ie."""
        R_sersic, n_sersic = 1.0, 4.0
        e1, e2 = 0.1, -0.05
        center_x, center_y = 0.0, 0.0
        magnitude = 22.0
        mag_zero_point = 22.0

        # Should not raise
        sersic = Sersic(
            R_sersic=R_sersic, n_sersic=n_sersic,
            e1=e1, e2=e2, center_x=center_x, center_y=center_y,
            magnitude=magnitude, mag_zero_point=mag_zero_point
        )
        assert hasattr(sersic, 'magnitude'), "Sersic should have magnitude attribute"
        assert hasattr(sersic, 'mag_zero_point'), "Sersic should have mag_zero_point attribute"

    def test_sersic_mutual_exclusion(self):
        """Test that Ie and magnitude cannot both be provided."""
        with pytest.raises(ValueError, match="exactly one"):
            Sersic(
                R_sersic=1.0, n_sersic=4.0,
                e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
                Ie=1.0, magnitude=22.0
            )

    def test_sersic_neither_brightness_param(self):
        """Test that at least one brightness param (Ie or magnitude) must be given."""
        with pytest.raises(ValueError, match="exactly one"):
            Sersic(
                R_sersic=1.0, n_sersic=4.0,
                e1=0.0, e2=0.0, center_x=0.0, center_y=0.0
            )

    def test_sersic_magnitude_brightness_equivalence(self):
        """Test that magnitude-based Sersic gives same brightness as equivalent Ie."""
        # Parameters
        R_sersic, n_sersic = 1.0, 4.0
        e1, e2 = 0.1, -0.05
        center_x, center_y = 0.0, 0.0
        magnitude = 22.0
        mag_zero_point = 22.0

        # Create test grid
        x = jnp.linspace(-3, 3, 50)
        y = jnp.linspace(-3, 3, 50)
        X, Y = jnp.meshgrid(x, y)

        # Build magnitude-based Sersic
        sersic_mag = Sersic(
            R_sersic=R_sersic, n_sersic=n_sersic,
            e1=e1, e2=e2, center_x=center_x, center_y=center_y,
            magnitude=magnitude, mag_zero_point=mag_zero_point
        )
        sersic_mag.R_sersic.to_static()
        sersic_mag.n_sersic.to_static()
        sersic_mag.e1.to_static()
        sersic_mag.e2.to_static()
        sersic_mag.center_x.to_static()
        sersic_mag.center_y.to_static()
        sersic_mag.magnitude.to_static()

        brightness_mag = sersic_mag.light(X, Y)

        # Compute equivalent Ie using the class method
        Ie_equivalent = Sersic.Ie_from_magnitude(
            magnitude, mag_zero_point, R_sersic, n_sersic
        )

        # Build Ie-based Sersic with the equivalent Ie
        sersic_ie = Sersic(
            R_sersic=R_sersic, n_sersic=n_sersic,
            e1=e1, e2=e2, center_x=center_x, center_y=center_y, Ie=Ie_equivalent
        )
        sersic_ie.R_sersic.to_static()
        sersic_ie.n_sersic.to_static()
        sersic_ie.e1.to_static()
        sersic_ie.e2.to_static()
        sersic_ie.center_x.to_static()
        sersic_ie.center_y.to_static()
        sersic_ie.Ie.to_static()

        brightness_ie = sersic_ie.light(X, Y)

        assert jnp.allclose(brightness_mag, brightness_ie, rtol=1e-5), \
            "Magnitude-based and Ie-based Sersic should produce identical brightness"

    def test_sersic_total_flux_analytic(self):
        """Test that total_flux_analytic is consistent with the Sersic formula."""
        R_sersic, n_sersic = 1.0, 4.0
        Ie = 1.0

        # Compute total flux analytically
        F_analytic = Sersic.total_flux_analytic_from(
            R_sersic=R_sersic, Ie=Ie, n_sersic=n_sersic
        )

        # The result should be positive and finite
        assert F_analytic > 0, "Total flux should be positive"
        assert jnp.isfinite(F_analytic), "Total flux should be finite"

        # Test linear scaling: doubling Ie should double total flux
        F_doubled = Sersic.total_flux_analytic_from(
            R_sersic=R_sersic, Ie=2.0, n_sersic=n_sersic
        )
        assert jnp.isclose(F_doubled, 2.0 * F_analytic, rtol=1e-5), \
            "Total flux should scale linearly with Ie"

    def test_sersic_magnitude_roundtrip(self):
        """Test that converting magnitude -> Ie -> total_flux is consistent."""
        R_sersic, n_sersic = 1.0, 4.0
        magnitude = 22.0
        mag_zero_point = 22.0

        # Convert magnitude to Ie
        Ie = Sersic.Ie_from_magnitude(
            magnitude, mag_zero_point, R_sersic, n_sersic
        )

        # Compute total flux from this Ie
        F_from_Ie = Sersic.total_flux_analytic_from(
            R_sersic=R_sersic, Ie=Ie, n_sersic=n_sersic
        )

        # Direct magnitude -> cps
        from TinyLensGpu.utils.photometry import mag2cps
        F_direct = mag2cps(magnitude, mag_zero_point)

        assert jnp.isclose(F_from_Ie, F_direct, rtol=1e-5), \
            "Flux from Ie should match direct mag2cps conversion"


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
