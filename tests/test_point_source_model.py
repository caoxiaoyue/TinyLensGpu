"""Tests for point-source position likelihood modeling."""

import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE, Shear


def _build_phys_model(theta_e: float = 1.1, gamma1: float = 0.04, gamma2: float = -0.02):
    sie = SIE(
        theta_E=theta_e,
        e1=0.0,
        e2=0.0,
        center_x=0.0,
        center_y=0.0,
    )
    shear = Shear(gamma1=gamma1, gamma2=gamma2)

    sie.theta_E.to_static()
    sie.e1.to_static()
    sie.e2.to_static()
    sie.center_x.to_static()
    sie.center_y.to_static()
    shear.gamma1.to_static()
    shear.gamma2.to_static()

    return PhysicalModel(lens_mass=[sie, shear], source_light=[], lens_light=[])


def _build_observed_positions(
    phys_model: PhysicalModel,
    source_pos: jnp.ndarray,
    solver: str = "optimization",
):
    if solver == "amr":
        solver_cfg = {
            "initial_range": 3.0,
            "n_x": 60,
            "n_y": 60,
            "k_keep": 24,
            "subgrid_res": 12,
            "depth": 8,
            "search_factor": 2.0,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        }
    else:
        solver_cfg = {
            "initial_range": 3.0,
            "n_x": 60,
            "n_y": 60,
            "k_keep": 24,
            "num_iters": 18,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        }

    model = PointSourceProbModel(
        phys_model=phys_model,
        observed_positions=[[0.0, 0.0]],
        position_sigma=[0.01],
        source_x=float(source_pos[0]),
        source_y=float(source_pos[1]),
        source_position_fixed=True,
        solver=solver,
        solver_config=solver_cfg,
    )
    images, _ = model.solve_image_positions()
    if images.shape[0] < 2:
        raise RuntimeError("Failed to generate at least two image positions for fixture")
    return np.asarray(images[:2])


@pytest.mark.unit
def test_point_source_model_optimization_solver_returns_finite_loglike():
    phys_model = _build_phys_model()
    source_true = jnp.array([0.03, -0.01], dtype=jnp.float32)
    obs = _build_observed_positions(phys_model, source_true)

    model = PointSourceProbModel(
        phys_model=phys_model,
        observed_positions=obs,
        position_sigma=[0.01, 0.012],
        source_x=float(source_true[0]),
        source_y=float(source_true[1]),
        source_position_fixed=True,
        solver="optimization",
        solver_config={
            "initial_range": 3.0,
            "n_x": 60,
            "n_y": 60,
            "k_keep": 24,
            "num_iters": 18,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        },
    )

    log_like = model.likelihood()
    assert np.isfinite(log_like)
    assert log_like > -100.0


@pytest.mark.unit
def test_point_source_model_amr_solver_returns_finite_loglike():
    phys_model = _build_phys_model()
    source_true = jnp.array([0.03, -0.01], dtype=jnp.float32)
    obs = _build_observed_positions(phys_model, source_true, solver="amr")

    model = PointSourceProbModel(
        phys_model=phys_model,
        observed_positions=obs,
        position_sigma=[0.01, 0.012],
        source_x=float(source_true[0]),
        source_y=float(source_true[1]),
        source_position_fixed=True,
        solver="amr",
        solver_config={
            "initial_range": 3.0,
            "n_x": 60,
            "n_y": 60,
            "k_keep": 24,
            "subgrid_res": 10,
            "depth": 6,
            "search_factor": 2.0,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        },
    )

    log_like = model.likelihood()
    assert np.isfinite(log_like)


@pytest.mark.unit
def test_point_source_matching_invariant_to_observed_order():
    phys_model = _build_phys_model()
    source_true = jnp.array([0.03, -0.01], dtype=jnp.float32)
    obs = _build_observed_positions(phys_model, source_true)

    model_a = PointSourceProbModel(
        phys_model=phys_model,
        observed_positions=obs,
        position_sigma=[0.01, 0.012],
        source_x=float(source_true[0]),
        source_y=float(source_true[1]),
        source_position_fixed=True,
        solver="optimization",
        solver_config={
            "initial_range": 3.0,
            "n_x": 60,
            "n_y": 60,
            "k_keep": 24,
            "num_iters": 18,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        },
    )
    model_b = PointSourceProbModel(
        phys_model=phys_model,
        observed_positions=obs[::-1],
        position_sigma=[0.012, 0.01],
        source_x=float(source_true[0]),
        source_y=float(source_true[1]),
        source_position_fixed=True,
        solver="optimization",
        solver_config={
            "initial_range": 3.0,
            "n_x": 60,
            "n_y": 60,
            "k_keep": 24,
            "num_iters": 18,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        },
    )

    assert np.isclose(model_a.likelihood(), model_b.likelihood(), atol=1e-5)


@pytest.mark.unit
def test_point_source_sigma_controls_penalty_strength():
    phys_model = _build_phys_model()
    source_true = jnp.array([0.03, -0.01], dtype=jnp.float32)
    obs = _build_observed_positions(phys_model, source_true)

    source_shift = jnp.array([0.06, -0.05], dtype=jnp.float32)

    model_small_sigma = PointSourceProbModel(
        phys_model=phys_model,
        observed_positions=obs,
        position_sigma=[0.005, 0.005],
        source_x=float(source_shift[0]),
        source_y=float(source_shift[1]),
        source_position_fixed=True,
        solver="optimization",
        solver_config={
            "initial_range": 3.0,
            "n_x": 60,
            "n_y": 60,
            "k_keep": 24,
            "num_iters": 18,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        },
    )
    model_large_sigma = PointSourceProbModel(
        phys_model=phys_model,
        observed_positions=obs,
        position_sigma=[0.05, 0.05],
        source_x=float(source_shift[0]),
        source_y=float(source_shift[1]),
        source_position_fixed=True,
        solver="optimization",
        solver_config={
            "initial_range": 3.0,
            "n_x": 60,
            "n_y": 60,
            "k_keep": 24,
            "num_iters": 18,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        },
    )

    ll_small = model_small_sigma.likelihood()
    ll_large = model_large_sigma.likelihood()
    assert ll_large > ll_small


@pytest.mark.unit
def test_point_source_returns_min_log_like_when_insufficient_images():
    phys_model = _build_phys_model()
    source_true = jnp.array([0.03, -0.01], dtype=jnp.float32)
    obs = _build_observed_positions(phys_model, source_true)

    model = PointSourceProbModel(
        phys_model=phys_model,
        # Stack many fake points to ensure we exceed the number of possible images
        observed_positions=np.vstack([obs, np.zeros((6, 2))]),
        position_sigma=0.01,
        source_x=float(source_true[0]),
        source_y=float(source_true[1]),
        source_position_fixed=True,
        solver="optimization",
        min_log_like=-12345.0,
        solver_config={
            "initial_range": 3.0,
            "n_x": 60,
            "n_y": 60,
            "k_keep": 24,
            "num_iters": 18,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        },
    )
    assert np.isclose(model.likelihood(), -12345.0)


@pytest.mark.integration
def test_point_source_model_prior_and_likelihood_integration():
    sie = SIE(
        theta_E=ParamU("theta_E", 1.1, prior_type="uniform", prior_settings=[0.8, 1.5], limits=[0.1, 3.0]),
        e1=0.0,
        e2=0.0,
        center_x=0.0,
        center_y=0.0,
    )
    shear = Shear(
        gamma1=ParamU("gamma1", 0.04, prior_type="uniform", prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
        gamma2=ParamU("gamma2", -0.02, prior_type="uniform", prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
    )

    for param in [sie.e1, sie.e2, sie.center_x, sie.center_y]:
        param.to_static()
    sie.theta_E.to_dynamic()
    shear.gamma1.to_dynamic()
    shear.gamma2.to_dynamic()

    phys_model = PhysicalModel(lens_mass=[sie, shear], source_light=[], lens_light=[])

    obs_builder = PointSourceProbModel(
        phys_model=phys_model,
        observed_positions=[[0.0, 0.0]],
        position_sigma=[0.01],
        source_x=0.03,
        source_y=-0.01,
        source_position_fixed=True,
        solver="optimization",
        solver_config={
            "initial_range": 3.0,
            "n_x": 60,
            "n_y": 60,
            "k_keep": 24,
            "num_iters": 18,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        },
    )
    obs, _ = obs_builder.solve_image_positions()
    obs = np.asarray(obs[:2])

    model = PointSourceProbModel(
        phys_model=phys_model,
        observed_positions=obs,
        position_sigma=[0.01, 0.012],
        source_x=ParamU("source_x", 0.03, prior_type="uniform", prior_settings=[-0.2, 0.2], limits=[-1.0, 1.0]),
        source_y=ParamU("source_y", -0.01, prior_type="uniform", prior_settings=[-0.2, 0.2], limits=[-1.0, 1.0]),
        source_position_fixed=False,
        solver="optimization",
        solver_config={
            "initial_range": 3.0,
            "n_x": 60,
            "n_y": 60,
            "k_keep": 24,
            "num_iters": 18,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        },
    )

    prior_transform, prior_specs = make_prior_transformation(model)
    names = [spec.name for spec in prior_specs]

    assert "theta_E" in names
    assert "gamma1" in names
    assert "gamma2" in names
    assert "source_x" in names
    assert "source_y" in names

    u = jnp.full((len(prior_specs),), 0.5)
    theta = prior_transform(u)

    loglike_fn = make_likelihood(model, vectorized=False)
    ll = loglike_fn(theta)
    assert np.isfinite(ll)
