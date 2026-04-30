# pyright: reportMissingImports=false

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
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d_transformed
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.ObservationModel.LensImage import ImageProbModel
from TinyLensGpu.ObservationModel.LensImage.multi_band_image_model import (
    BandImageData,
    BandObservationGeometry,
    MultiBandImageProbModel,
)
from TinyLensGpu.utils import LinearSolver


def test_forward_simulation_exports_simulation_result():
    from TinyLensGpu.ForwardSimulation import SimulationResult
    from TinyLensGpu.ForwardSimulation.LensImage import SimulationResult as LensImageSimulationResult

    assert LensImageSimulationResult is SimulationResult

    result = SimulationResult(model_image=np.zeros((2, 2)))

    assert result.model_image.shape == (2, 2)
    assert result.source_image is None


def test_lens_simulator_forward_matches_simulate_output():
    source = GaussianEllipse(flux=4.0, sigma=0.3, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for param in [source.flux, source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
        param.to_static()

    model = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[])
    config = SimulatorConfig(
        dpix=0.05,
        npix=24,
        nsub=1,
        psf_kernel=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    simulator = LensSimulator(model, config)

    old_image = np.asarray(simulator.simulate(use_linear=False))
    result = simulator.forward()

    assert np.allclose(np.asarray(result.model_image), old_image)
    assert result.linear_params is None
    assert result.source_image is None
    assert result.lens_image is None


def test_lens_simulator_forward_returns_components_and_linear_params():
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import ConstantBackground

    source = SersicEllipse(
        R_sersic=0.5,
        n_sersic=1.5,
        e1=0.0,
        e2=0.0,
        center_x=0.0,
        center_y=0.0,
        Ie=2.0,
    )
    background = ConstantBackground(intensity=0.3)
    for param in [source.R_sersic, source.n_sersic, source.e1, source.e2, source.center_x, source.center_y]:
        param.to_static()

    model = PhysicalModel(lens_mass=[], source_light=[], lens_light=[source, background])
    config = SimulatorConfig(
        dpix=0.05,
        npix=20,
        nsub=1,
        psf_kernel=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    simulator = LensSimulator(model, config, solver_type="normal")
    data = np.asarray(simulator.simulate(use_linear=False))
    noise = np.ones_like(data) * 0.1

    old_source, old_lens, old_linear = simulator.simulate(
        use_linear=True,
        return_intensity=True,
        ret_each_plane=True,
        image_map=data,
        noise_map=noise,
    )
    result = simulator.forward(
        data=data,
        noise_map=noise,
        return_components=True,
    )

    assert np.allclose(np.asarray(result.source_image), np.asarray(old_source))
    assert np.allclose(np.asarray(result.lens_image), np.asarray(old_lens))
    assert np.allclose(np.asarray(result.linear_params), np.asarray(old_linear))
    assert result.model_image.shape == data.shape


def test_lens_simulator_forward_rejects_return_solver():
    source = GaussianEllipse(flux=4.0, sigma=0.3, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for param in [source.flux, source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
        param.to_static()

    model = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[])
    config = SimulatorConfig(dpix=0.05, npix=24, nsub=1)
    simulator = LensSimulator(model, config)

    with pytest.raises(ValueError, match="does not return solver objects"):
        simulator.forward(return_solver=True)


def test_lens_simulator_forward_rejects_non_2d_images():
    source = GaussianEllipse(flux=4.0, sigma=0.3, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for param in [source.flux, source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
        param.to_static()

    model = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[])
    config = SimulatorConfig(dpix=0.05, npix=24, nsub=1)
    simulator = LensSimulator(model, config)

    with pytest.raises(ValueError, match="always returns 2D images"):
        simulator.forward(return_image_2d=False)


def test_image_prob_model_forward_model_matches_simulator_forward():
    sie = SIE(theta_E=1.2, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    source = GaussianEllipse(flux=8.0, sigma=0.3, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
        param.to_static()
    for param in [source.flux, source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
        param.to_static()

    image_data = np.ones((18, 18), dtype=float)
    noise_map = np.ones((18, 18), dtype=float) * 0.2
    psf_kernel = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
    prob_model = ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=0.05,
        nsub=1,
        phys_model=model,
        use_linear=False,
    )

    image_model, intensity_list = prob_model.forward_model()
    sim_result = prob_model.sim_obj.forward()

    assert np.allclose(np.asarray(image_model), np.asarray(sim_result.model_image))
    assert intensity_list is None


def test_image_prob_model_forward_model_uses_simulator_forward(monkeypatch):
    from TinyLensGpu.ForwardSimulation import SimulationResult

    source = GaussianEllipse(flux=8.0, sigma=0.3, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for param in [source.flux, source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
        param.to_static()

    image_data = np.ones((18, 18), dtype=float)
    noise_map = np.ones((18, 18), dtype=float) * 0.2
    model = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[])
    prob_model = ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        dpix=0.05,
        nsub=1,
        phys_model=model,
        use_linear=False,
    )

    sentinel = np.full((18, 18), 7.0)

    def fake_forward(**kwargs):
        assert kwargs["data"] is None
        assert kwargs["noise_map"] is None
        assert kwargs["return_components"] is False
        return SimulationResult(model_image=sentinel)

    def fail_simulate(*args, **kwargs):
        raise AssertionError("forward_model should call sim_obj.forward")

    monkeypatch.setattr(prob_model.sim_obj, "forward", fake_forward)
    monkeypatch.setattr(prob_model.sim_obj, "simulate", fail_simulate)

    image_model, intensity_list = prob_model.forward_model()

    assert np.allclose(np.asarray(image_model), sentinel)
    assert intensity_list is None


def test_image_prob_model_forward_model_ignores_linear_inputs_when_forced_nonlinear(monkeypatch):
    from TinyLensGpu.ForwardSimulation import SimulationResult

    source = GaussianEllipse(flux=8.0, sigma=0.3, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for param in [source.flux, source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
        param.to_static()

    image_data = np.ones((18, 18), dtype=float)
    noise_map = np.ones((18, 18), dtype=float) * 0.2
    model = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[])
    prob_model = ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        dpix=0.05,
        nsub=1,
        phys_model=model,
        use_linear=True,
    )

    sentinel = np.full((18, 18), 3.0)

    def fake_forward(**kwargs):
        assert kwargs["data"] is None
        assert kwargs["noise_map"] is None
        return SimulationResult(model_image=sentinel)

    monkeypatch.setattr(prob_model.sim_obj, "forward", fake_forward)

    image_model, intensity_list = prob_model.forward_model(
        use_linear=False,
        image_map=np.full((18, 18), 9.0),
        noise_map=np.full((18, 18), 4.0),
    )

    assert np.allclose(np.asarray(image_model), sentinel)
    assert intensity_list is None


def test_image_prob_model_forward_model_linear_matches_simulator_forward():
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import ConstantBackground

    source = SersicEllipse(
        R_sersic=0.5,
        n_sersic=1.5,
        e1=0.0,
        e2=0.0,
        center_x=0.0,
        center_y=0.0,
        Ie=2.0,
    )
    background = ConstantBackground(intensity=0.3)
    for param in [source.R_sersic, source.n_sersic, source.e1, source.e2, source.center_x, source.center_y]:
        param.to_static()

    image_data = np.ones((20, 20), dtype=float)
    noise_map = np.ones((20, 20), dtype=float) * 0.1
    psf_kernel = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    model = PhysicalModel(lens_mass=[], source_light=[], lens_light=[source, background])
    prob_model = ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=0.05,
        nsub=1,
        phys_model=model,
        use_linear=True,
        solver_type="normal",
    )

    image_model, intensity_list = prob_model.forward_model(
        use_linear=True,
        image_map=image_data,
        noise_map=noise_map,
    )
    sim_result = prob_model.sim_obj.forward(data=image_data, noise_map=noise_map)

    assert np.allclose(np.asarray(image_model), np.asarray(sim_result.model_image))
    assert np.allclose(np.asarray(intensity_list), np.asarray(sim_result.linear_params))


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
        img = np.asarray(simulator.simulate(use_linear=False))
        
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
        
        img = np.asarray(simulator.simulate(use_linear=False))
        
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
        
        img = np.asarray(simulator.simulate(use_linear=False))
        
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
        img = np.asarray(simulator.simulate(use_linear=False))
        
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
        img = np.asarray(simulator.simulate(use_linear=False))
        
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
class TestMultiBandIntegration:
    """Test multiband wrapper integration against single-band models."""

    @staticmethod
    def _build_sie_sersic_model(theta_e: float, source_ie: float, lens_ie: float) -> PhysicalModel:
        sie = SIE(theta_E=theta_e, e1=0.08, e2=-0.03, center_x=0.01, center_y=-0.02)
        source = SersicEllipse(
            R_sersic=0.55,
            n_sersic=2.1,
            e1=0.04,
            e2=-0.02,
            center_x=-0.03,
            center_y=0.02,
            Ie=source_ie,
        )
        lens_light = SersicEllipse(
            R_sersic=0.9,
            n_sersic=3.2,
            e1=-0.05,
            e2=0.03,
            center_x=0.0,
            center_y=0.0,
            Ie=lens_ie,
        )

        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [
            source.R_sersic,
            source.n_sersic,
            source.e1,
            source.e2,
            source.center_x,
            source.center_y,
            source.Ie,
        ]:
            param.to_static()
        for param in [
            lens_light.R_sersic,
            lens_light.n_sersic,
            lens_light.e1,
            lens_light.e2,
            lens_light.center_x,
            lens_light.center_y,
            lens_light.Ie,
        ]:
            param.to_static()

        return PhysicalModel(
            lens_mass=[sie],
            source_light=[source],
            lens_light=[lens_light],
        )

    @staticmethod
    def _simulate_image(
        phys_model: PhysicalModel,
        npix: int,
        psf_kernel: np.ndarray,
        *,
        dpix: float = 0.05,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        rotation: float = 0.0,
    ) -> np.ndarray:
        simulator = LensSimulator(
            phys_model=phys_model,
            sim_config=SimulatorConfig(dpix=dpix, npix=npix, nsub=2, psf_kernel=psf_kernel),
        )
        if np.isclose(shift_x, 0.0) and np.isclose(shift_y, 0.0) and np.isclose(rotation, 0.0):
            return np.asarray(simulator.simulate(use_linear=False), dtype=float)

        xgrid_sub_2d, ygrid_sub_2d = make_grid_2d_transformed(
            npix=npix,
            dpix=dpix,
            nsub=2,
            shift_x=jnp.asarray(shift_x),
            shift_y=jnp.asarray(shift_y),
            rotation=jnp.asarray(rotation),
        )
        flat_indices = simulator.sim_config.flat_indices
        return np.asarray(
            simulator.simulate(
                use_linear=False,
                xgrid_sub=xgrid_sub_2d.reshape(-1)[flat_indices],
                ygrid_sub=ygrid_sub_2d.reshape(-1)[flat_indices],
            ),
            dtype=float,
        )

    def test_multiband_joint_likelihood_parity(self):
        """Joint multiband likelihood should match explicit single-band sum."""
        noise_sigma = 0.15
        psf_kernel = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        npix_g, dpix_g = 32, 0.05
        npix_r, dpix_r = 40, 0.07
        npix_i, dpix_i = 48, 0.09

        phys_model_g = self._build_sie_sersic_model(theta_e=1.25, source_ie=1.8, lens_ie=2.6)
        phys_model_r = self._build_sie_sersic_model(theta_e=1.33, source_ie=2.1, lens_ie=2.2)
        phys_model_i = self._build_sie_sersic_model(theta_e=1.41, source_ie=2.3, lens_ie=2.8)

        image_g = self._simulate_image(phys_model_g, npix=npix_g, dpix=dpix_g, psf_kernel=psf_kernel)
        image_r = self._simulate_image(phys_model_r, npix=npix_r, dpix=dpix_r, psf_kernel=psf_kernel)
        image_i = self._simulate_image(phys_model_i, npix=npix_i, dpix=dpix_i, psf_kernel=psf_kernel)
        noise_map_g = np.ones((npix_g, npix_g), dtype=float) * noise_sigma
        noise_map_r = np.ones((npix_r, npix_r), dtype=float) * noise_sigma
        noise_map_i = np.ones((npix_i, npix_i), dtype=float) * noise_sigma

        band_g = BandImageData("g", image_g, noise_map_g, psf_kernel, dpix_g, 2, None)
        band_r = BandImageData("r", image_r, noise_map_r, psf_kernel, dpix_r, 2, None)
        band_i = BandImageData("i", image_i, noise_map_i, psf_kernel, dpix_i, 2, None)

        multi_band_model = MultiBandImageProbModel(
            bands=[band_g, band_r, band_i],
            phys_models=[phys_model_g, phys_model_r, phys_model_i],
            use_linear=False,
        )

        single_g = ImageProbModel(
            image_data=image_g,
            noise_map=noise_map_g,
            psf_kernel=psf_kernel,
            dpix=dpix_g,
            nsub=2,
            phys_model=phys_model_g,
            use_linear=False,
        )
        single_r = ImageProbModel(
            image_data=image_r,
            noise_map=noise_map_r,
            psf_kernel=psf_kernel,
            dpix=dpix_r,
            nsub=2,
            phys_model=phys_model_r,
            use_linear=False,
        )
        single_i = ImageProbModel(
            image_data=image_i,
            noise_map=noise_map_i,
            psf_kernel=psf_kernel,
            dpix=dpix_i,
            nsub=2,
            phys_model=phys_model_i,
            use_linear=False,
        )

        joint_like = multi_band_model.likelihood()
        explicit_sum = single_g.likelihood() + single_r.likelihood() + single_i.likelihood()

        assert np.isclose(joint_like, explicit_sum, rtol=1e-6, atol=1e-10)

    def test_multiband_alignment_enabled_linear_fit_recovers_band_amplitudes(self):
        """Alignment-enabled linear fit should recover per-band amplitudes."""
        psf_kernel = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        noise_sigma = 0.1
        npix_g, dpix_g = 33, 0.05
        npix_r, dpix_r = 41, 0.08
        shift_x, shift_y, rotation = 0.03, -0.02, float(np.degrees(0.08))

        true_ie_g = 2.6
        true_ie_r = 3.1

        true_light_g = SersicEllipse(
            R_sersic=0.7,
            n_sersic=2.2,
            e1=0.03,
            e2=-0.01,
            center_x=0.01,
            center_y=-0.02,
            Ie=ParamU("g_Ie_fit", true_ie_g),
        )
        true_light_r = SersicEllipse(
            R_sersic=0.7,
            n_sersic=2.2,
            e1=0.03,
            e2=-0.01,
            center_x=0.01,
            center_y=-0.02,
            Ie=ParamU("r_Ie_fit", true_ie_r),
        )
        for param in [
            true_light_g.R_sersic,
            true_light_g.n_sersic,
            true_light_g.e1,
            true_light_g.e2,
            true_light_g.center_x,
            true_light_g.center_y,
            true_light_g.Ie,
            true_light_r.R_sersic,
            true_light_r.n_sersic,
            true_light_r.e1,
            true_light_r.e2,
            true_light_r.center_x,
            true_light_r.center_y,
            true_light_r.Ie,
        ]:
            param.to_static()

        true_model_g = PhysicalModel(lens_mass=[], source_light=[], lens_light=[true_light_g])
        true_model_r = PhysicalModel(lens_mass=[], source_light=[], lens_light=[true_light_r])

        image_g = self._simulate_image(true_model_g, npix=npix_g, dpix=dpix_g, psf_kernel=psf_kernel)

        simulator_r = LensSimulator(
            phys_model=true_model_r,
            sim_config=SimulatorConfig(dpix=dpix_r, npix=npix_r, nsub=2, psf_kernel=psf_kernel),
        )
        xgrid_sub_2d, ygrid_sub_2d = make_grid_2d_transformed(
            npix=npix_r,
            dpix=dpix_r,
            nsub=2,
            shift_x=jnp.asarray(shift_x),
            shift_y=jnp.asarray(shift_y),
            rotation=jnp.asarray(rotation),
        )
        flat_indices = simulator_r.sim_config.flat_indices
        image_r = np.asarray(
            simulator_r.simulate(
                use_linear=False,
                xgrid_sub=xgrid_sub_2d.reshape(-1)[flat_indices],
                ygrid_sub=ygrid_sub_2d.reshape(-1)[flat_indices],
            ),
            dtype=float,
        )

        fit_light_g = SersicEllipse(
            R_sersic=0.7,
            n_sersic=2.2,
            e1=0.03,
            e2=-0.01,
            center_x=0.01,
            center_y=-0.02,
            Ie=ParamU("g_Ie_fit", 1.0),
        )
        fit_light_r = SersicEllipse(
            R_sersic=0.7,
            n_sersic=2.2,
            e1=0.03,
            e2=-0.01,
            center_x=0.01,
            center_y=-0.02,
            Ie=ParamU("r_Ie_fit", 1.0),
        )
        for param in [
            fit_light_g.R_sersic,
            fit_light_g.n_sersic,
            fit_light_g.e1,
            fit_light_g.e2,
            fit_light_g.center_x,
            fit_light_g.center_y,
            fit_light_r.R_sersic,
            fit_light_r.n_sersic,
            fit_light_r.e1,
            fit_light_r.e2,
            fit_light_r.center_x,
            fit_light_r.center_y,
        ]:
            param.to_static()

        fit_model_g = PhysicalModel(lens_mass=[], source_light=[], lens_light=[fit_light_g])
        fit_model_r = PhysicalModel(lens_mass=[], source_light=[], lens_light=[fit_light_r])

        noise_map_g = np.ones((npix_g, npix_g), dtype=float) * noise_sigma
        noise_map_r = np.ones((npix_r, npix_r), dtype=float) * noise_sigma

        multi_model = MultiBandImageProbModel(
            bands=[
                BandImageData("g", image_g, noise_map_g, psf_kernel, dpix_g, 2, None),
                BandImageData(
                    "r",
                    image_r,
                    noise_map_r,
                    psf_kernel,
                    dpix_r,
                    2,
                    None,
                    geometry=BandObservationGeometry(
                        shift_x=shift_x,
                        shift_y=shift_y,
                        rotation=rotation,
                        is_reference=False,
                    ),
                ),
            ],
            phys_models=[fit_model_g, fit_model_r],
            use_linear=True,
            solver_type="normal",
        )

        solved = multi_model.get_linear_solved_params({})
        joint_like = multi_model.likelihood()

        assert np.isfinite(joint_like)
        assert np.isclose(solved["g"]["g_Ie_fit"], true_ie_g, rtol=1e-4, atol=5e-4)
        assert np.isclose(solved["r"]["r_Ie_fit"], true_ie_r, rtol=1e-4, atol=5e-4)

    def test_multiband_identical_bands_double_count(self):
        """Adding identical bands should double the likelihood contribution."""
        npix = 34
        noise_sigma = 0.12
        psf_kernel = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        phys_model_single = self._build_sie_sersic_model(theta_e=1.28, source_ie=1.9, lens_ie=2.4)
        image_data = self._simulate_image(phys_model_single, npix=npix, psf_kernel=psf_kernel)
        noise_map = np.ones((npix, npix), dtype=float) * noise_sigma

        single_band_model = ImageProbModel(
            image_data=image_data,
            noise_map=noise_map,
            psf_kernel=psf_kernel,
            dpix=0.05,
            nsub=2,
            phys_model=phys_model_single,
            use_linear=False,
        )

        band_g = BandImageData("g", image_data, noise_map, psf_kernel, 0.05, 2, None)
        band_r = BandImageData("r", image_data, noise_map, psf_kernel, 0.05, 2, None)
        phys_model_dup = self._build_sie_sersic_model(theta_e=1.28, source_ie=1.9, lens_ie=2.4)

        multi_band_model = MultiBandImageProbModel(
            bands=[band_g, band_r],
            phys_models=[phys_model_single, phys_model_dup],
            use_linear=False,
        )

        single_like = single_band_model.likelihood()
        joint_like = multi_band_model.likelihood()

        assert np.isclose(joint_like, 2.0 * single_like, rtol=1e-10, atol=1e-10)


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
        
        img = np.asarray(simulator.simulate(use_linear=False))
        
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
        
        img = np.asarray(simulator.simulate(use_linear=False))
        
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
        
        img = np.asarray(simulator.simulate(use_linear=False))
        
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
        
        img = np.asarray(simulator.simulate(use_linear=False))
        
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
        
        img = np.asarray(simulator.simulate(use_linear=False))
        
        assert img.shape == (100, 100)
        assert not jnp.isnan(img).any()
