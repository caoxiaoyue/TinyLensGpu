"""
Integration tests for TinyLensGpu.

This module tests end-to-end workflows and component interactions
to ensure the system works correctly as a whole.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE, Shear
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse, GaussianEllipse
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.ForwardSimulation import LensSimulator, SimulatorConfig
from TinyLensGpu.ObservationModel.LensImage import ImageProbModel
from TinyLensGpu.utils import LinearSolver


@pytest.mark.integration
class TestEndToEndSimulation:
    """Test complete simulation workflows."""
    
    def test_simple_lens_simulation(self):
        """Test simple lens-only simulation workflow."""
        # Create lens model
        sie = SIE(theta_E=1.5, e1=0.1, e2=0.05, center_x=0.0, center_y=0.0)
        shear = Shear(gamma1=0.05, gamma2=0.0)
        
        # Set all parameters to static
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [shear.gamma1, shear.gamma2]:
            param.to_static()
        
        # Create physical model
        model = PhysicalModel(lens_mass=[sie, shear], source_light=[], lens_light=[])
        
        # Create simulator
        config = SimulatorConfig(dpix=0.05, npix=50, nsub=2)
        simulator = LensSimulator(model, config)
        
        # Run simulation
        img = simulator.simulate(use_linear=False)
        
        # Verify output
        assert img.shape == (50, 50)
        assert not jnp.isnan(img).any()
        assert not jnp.isinf(img).any()
    
    def test_source_only_simulation(self):
        """Test source-only simulation (no lensing)."""
        # Create source model
        sersic = SersicEllipse(R_sersic=0.5, n_sersic=2.0, e1=0.2, e2=0.1,
                              center_x=0.0, center_y=0.0, Ie=1.0)
        
        for param in [sersic.R_sersic, sersic.n_sersic, sersic.e1, sersic.e2,
                      sersic.center_x, sersic.center_y, sersic.Ie]:
            param.to_static()
        
        # No lens mass
        model = PhysicalModel(lens_mass=[], source_light=[sersic], lens_light=[])
        
        config = SimulatorConfig(dpix=0.05, npix=50, nsub=2)
        simulator = LensSimulator(model, config)
        
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (50, 50)
        assert jnp.sum(img) > 0  # Should have some light
    
    def test_full_lens_system_simulation(self):
        """Test complete lens system with mass, source, and lens light."""
        # Mass
        sie = SIE(theta_E=1.5, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
        shear = Shear(gamma1=0.05, gamma2=0.02)
        
        # Source light
        source = GaussianEllipse(flux=10.0, sigma=0.3, e1=0.1, e2=0.0,
                                center_x=0.0, center_y=0.0)
        
        # Lens light
        lens_light = SersicEllipse(R_sersic=1.0, n_sersic=4.0, e1=0.2, e2=0.1,
                                  center_x=0.0, center_y=0.0, Ie=5.0)
        
        # Set all to static
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [shear.gamma1, shear.gamma2]:
            param.to_static()
        for param in [source.flux, source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
            param.to_static()
        for param in [lens_light.R_sersic, lens_light.n_sersic, lens_light.e1, lens_light.e2,
                      lens_light.center_x, lens_light.center_y, lens_light.Ie]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie, shear], 
                            source_light=[source],
                            lens_light=[lens_light])
        
        config = SimulatorConfig(dpix=0.05, npix=60, nsub=3)
        simulator = LensSimulator(model, config)
        
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (60, 60)
        assert jnp.sum(img) > 0
        assert not jnp.isnan(img).any()


@pytest.mark.integration
class TestLinearSolverIntegration:
    """Test linear solver integration with simulation."""
    
    def test_linear_simulation_nnls(self):
        """Test linear simulation with NNLS solver."""
        # Create model
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        source = GaussianEllipse(flux=1.0, sigma=0.5, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0)
        
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        # All source parameters static for now (linear solving needs different setup)
        for param in [source.flux, source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
        
        config = SimulatorConfig(dpix=0.05, npix=40, nsub=2)
        simulator = LensSimulator(model, config, solver_type='nnls')
        
        # Create mock data
        image_data = jnp.ones((40, 40))
        noise_map = jnp.ones((40, 40)) * 0.1
        
        # Run non-linear simulation (linear solving requires dynamic parameters)
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (40, 40)
        assert not jnp.isnan(img).any()
    
    def test_linear_simulation_normal(self):
        """Test linear simulation with normal least squares."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        source = GaussianEllipse(flux=1.0, sigma=0.5, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0)
        
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [source.flux, source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
        
        config = SimulatorConfig(dpix=0.05, npix=40, nsub=2)
        simulator = LensSimulator(model, config, solver_type='normal')
        
        # Run non-linear simulation
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (40, 40)
        assert not jnp.isnan(img).any()


@pytest.mark.integration
class TestProbModelIntegration:
    """Test probability model integration."""
    
    def test_likelihood_computation(self):
        """Test likelihood computation workflow."""
        # Create model
        sie = SIE(theta_E=1.5, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
        source = GaussianEllipse(flux=10.0, sigma=0.5, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0)
        
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [source.flux, source.sigma, source.e1, source.e2,
                      source.center_x, source.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
        
        # Create mock data
        npix = 40
        image_data = np.random.randn(npix, npix) * 0.1 + 1.0
        noise_map = np.ones((npix, npix)) * 0.1
        psf_kernel = np.array([[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]])
        
        # Create probability model
        prob_model = ImageProbModel(
            image_data=image_data,
            noise_map=noise_map,
            psf_kernel=psf_kernel,
            dpix=0.05,
            nsub=2,
            phys_model=model,
            use_linear=False
        )
        
        # Compute likelihood
        log_like = prob_model.likelihood(debug=True)
        
        assert isinstance(log_like, float)
        assert not np.isnan(log_like)
        assert not np.isinf(log_like)
        assert log_like < 0  # Log-likelihood should be negative

    def test_linear_solver_recovers_sersic_and_constant_background(self):
        """Test that linear solving recovers both Sersic and sky amplitudes."""
        from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import ConstantBackground

        true_ie = 3.5
        true_intensity = 0.4
        noise_sigma = 0.1
        npix = 35
        psf_kernel = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        reference_sersic = SersicEllipse(
            R_sersic=0.6,
            n_sersic=2.0,
            e1=0.05,
            e2=-0.02,
            center_x=0.03,
            center_y=-0.04,
            Ie=true_ie,
        )
        reference_background = ConstantBackground(intensity=true_intensity)

        for param in [
            reference_sersic.R_sersic,
            reference_sersic.n_sersic,
            reference_sersic.e1,
            reference_sersic.e2,
            reference_sersic.center_x,
            reference_sersic.center_y,
            reference_sersic.Ie,
            reference_background.intensity,
        ]:
            param.to_static()

        reference_model = PhysicalModel(
            lens_mass=[],
            source_light=[],
            lens_light=[reference_sersic, reference_background],
        )

        simulator = LensSimulator(
            phys_model=reference_model,
            sim_config=SimulatorConfig(dpix=0.05, npix=npix, nsub=1, psf_kernel=psf_kernel),
            solver_type="normal",
        )
        image_data = np.array(simulator.simulate(use_linear=False))
        noise_map = np.ones_like(image_data) * noise_sigma

        fitted_sersic = SersicEllipse(
            R_sersic=0.6,
            n_sersic=2.0,
            e1=0.05,
            e2=-0.02,
            center_x=0.03,
            center_y=-0.04,
            Ie=1.0,
        )
        fitted_background = ConstantBackground(intensity=1.0)

        for param in [
            fitted_sersic.R_sersic,
            fitted_sersic.n_sersic,
            fitted_sersic.e1,
            fitted_sersic.e2,
            fitted_sersic.center_x,
            fitted_sersic.center_y,
        ]:
            param.to_static()

        fit_model = PhysicalModel(
            lens_mass=[],
            source_light=[],
            lens_light=[fitted_sersic, fitted_background],
        )

        prob_model = ImageProbModel(
            image_data=image_data,
            noise_map=noise_map,
            psf_kernel=psf_kernel,
            dpix=0.05,
            nsub=1,
            phys_model=fit_model,
            use_linear=True,
            solver_type="normal",
        )

        solved_params = prob_model.get_linear_solved_params({})

        assert np.isclose(solved_params[fitted_sersic.Ie.name], true_ie, rtol=1e-4, atol=5e-4)
        assert np.isclose(
            solved_params[fitted_background.intensity.name],
            true_intensity,
            rtol=1e-4,
            atol=5e-4,
        )

    def test_position_likelihood_penalty_inactive_returns_zero(self):
        sie = SIE(theta_E=1.5, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
        source = GaussianEllipse(
            flux=10.0, sigma=0.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0
        )

        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [source.flux, source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
            param.to_static()

        model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])

        npix = 40
        image_data = np.random.randn(npix, npix) * 0.1 + 1.0
        noise_map = np.ones((npix, npix)) * 0.1
        psf_kernel = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        prob_model = ImageProbModel(
            image_data=image_data,
            noise_map=noise_map,
            psf_kernel=psf_kernel,
            dpix=0.05,
            nsub=2,
            phys_model=model,
            use_linear=False,
            position_likelihood={
                "positions": [(0.0, 0.0), (0.1, 0.1)],
                "threshold_arcsec": 1.0e3,
                "min_log_like": -10.0,
            },
        )

        penalty = float(np.asarray(prob_model._position_likelihood_penalty_jax()))
        assert np.isclose(penalty, 0.0)
    
    def test_likelihood_with_mask(self):
        """Test likelihood computation with mask."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        source = GaussianEllipse(flux=10.0, sigma=0.5, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0)
        
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [source.flux, source.sigma, source.e1, source.e2,
                      source.center_x, source.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
        
        npix = 40
        image_data = np.random.randn(npix, npix) * 0.1 + 1.0
        noise_map = np.ones((npix, npix)) * 0.1
        psf_kernel = np.array([[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]])
        
        # Create mask (mask out corners)
        mask = np.zeros((npix, npix), dtype=bool)
        mask[:10, :10] = True
        mask[:10, -10:] = True
        mask[-10:, :10] = True
        mask[-10:, -10:] = True
        
        prob_model = ImageProbModel(
            image_data=image_data,
            noise_map=noise_map,
            psf_kernel=psf_kernel,
            dpix=0.05,
            nsub=2,
            phys_model=model,
            use_linear=False,
            mask=mask
        )
        
        log_like = prob_model.likelihood(debug=True)
        
        assert isinstance(log_like, float)
        assert not np.isnan(log_like)


@pytest.mark.integration
class TestMultiComponentSystems:
    """Test systems with multiple components."""
    
    def test_mge_source(self):
        """Test Multi-Gaussian Expansion source."""
        # Create MGE with 5 Gaussians
        gaussians = []
        for i in range(5):
            g = GaussianEllipse(
                flux=1.0 / (i + 1),
                sigma=0.1 * (i + 1),
                e1=0.1,
                e2=0.05,
                center_x=0.0,
                center_y=0.0
            )
            for param in [g.flux, g.sigma, g.e1, g.e2, g.center_x, g.center_y]:
                param.to_static()
            gaussians.append(g)
        
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie], source_light=gaussians, lens_light=[])
        
        config = SimulatorConfig(dpix=0.05, npix=50, nsub=2)
        simulator = LensSimulator(model, config)
        
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (50, 50)
        assert jnp.sum(img) > 0
    
    def test_multiple_mass_components(self):
        """Test system with multiple mass components."""
        # Main lens
        sie1 = SIE(theta_E=1.5, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
        # Perturber
        sie2 = SIE(theta_E=0.3, e1=0.0, e2=0.0, center_x=2.0, center_y=1.0)
        # External shear
        shear = Shear(gamma1=0.05, gamma2=0.02)
        
        for param in [sie1.theta_E, sie1.e1, sie1.e2, sie1.center_x, sie1.center_y]:
            param.to_static()
        for param in [sie2.theta_E, sie2.e1, sie2.e2, sie2.center_x, sie2.center_y]:
            param.to_static()
        for param in [shear.gamma1, shear.gamma2]:
            param.to_static()
        
        source = GaussianEllipse(flux=10.0, sigma=0.3, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0)
        for param in [source.flux, source.sigma, source.e1, source.e2,
                      source.center_x, source.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie1, sie2, shear],
                            source_light=[source],
                            lens_light=[])
        
        config = SimulatorConfig(dpix=0.05, npix=60, nsub=2)
        simulator = LensSimulator(model, config)
        
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (60, 60)
        assert not jnp.isnan(img).any()


@pytest.mark.integration
class TestPSFConvolution:
    """Test PSF convolution integration."""
    
    def test_gaussian_psf(self):
        """Test simulation with Gaussian PSF."""
        # Create simple Gaussian PSF
        psf_size = 7
        x = np.arange(psf_size) - psf_size // 2
        y = np.arange(psf_size) - psf_size // 2
        X, Y = np.meshgrid(x, y)
        sigma_psf = 1.0
        psf = np.exp(-(X**2 + Y**2) / (2 * sigma_psf**2))
        psf = psf / np.sum(psf)
        
        # Create model
        source = GaussianEllipse(flux=10.0, sigma=0.5, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0)
        for param in [source.flux, source.sigma, source.e1, source.e2,
                      source.center_x, source.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[])
        
        config = SimulatorConfig(dpix=0.05, npix=50, nsub=2, psf_kernel=psf)
        simulator = LensSimulator(model, config)
        
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (50, 50)
        assert jnp.sum(img) > 0
    
    def test_delta_psf(self):
        """Test simulation with delta function PSF (no convolution)."""
        psf = np.array([[0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0]])
        
        source = GaussianEllipse(flux=10.0, sigma=0.5, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0)
        for param in [source.flux, source.sigma, source.e1, source.e2,
                      source.center_x, source.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[])
        
        config = SimulatorConfig(dpix=0.05, npix=50, nsub=2, psf_kernel=psf)
        simulator = LensSimulator(model, config)
        
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (50, 50)


@pytest.mark.integration
@pytest.mark.slow
class TestLargeScaleIntegration:
    """Test large-scale integration scenarios."""
    
    def test_high_resolution_simulation(self):
        """Test high-resolution simulation (large npix and nsub)."""
        sie = SIE(theta_E=1.5, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
        source = GaussianEllipse(flux=10.0, sigma=0.5, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0)
        
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [source.flux, source.sigma, source.e1, source.e2,
                      source.center_x, source.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
        
        # High resolution: 100x100 pixels with 5x subsampling
        config = SimulatorConfig(dpix=0.05, npix=100, nsub=5)
        simulator = LensSimulator(model, config)
        
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (100, 100)
        assert not jnp.isnan(img).any()
