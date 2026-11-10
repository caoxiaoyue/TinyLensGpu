"""Gradients of reconstructions and evidence on a fixed source grid."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.ForwardSimulation import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel import SIE, SersicEllipse, PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light.pixelized_source import (
    PixelizedSourceModel,
)


def _build_model(solver, joint=False, adaptive=False, interpolation="bilinear"):
    sie = SIE(theta_E=0.12, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
    sie.theta_E.to_dynamic()
    lam = ParamU("log_lambda_reg", 0.0)
    lam.to_dynamic()
    rho = 0.0
    if adaptive:
        rho = ParamU(
            "adaptive_reg_rho", 1.0, prior_type="uniform",
            prior_settings=[0.0, 3.0], limits=[0.0, 3.0],
        )
        rho.to_dynamic()
    source = PixelizedSourceModel(
        n=5, log_lambda_reg=lam, regularization_type="first-order",
        adaptive_reg_rho=rho, interpolation=interpolation,
    )
    lens_light = []
    if joint:
        lens = SersicEllipse(
            R_sersic=0.5, n_sersic=1.0, Ie=1.0,
            e1=0.0, e2=0.0, center_x=0.02, center_y=0.0,
        )
        lens.Ie.to_static(1.0)
        lens.center_x.to_dynamic()
        lens_light = [lens]
    phys = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=lens_light)
    psf = jnp.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    config = SimulatorConfig(dpix=0.08, npix=10, nsub=1, psf_kernel=psf)
    simulator = PixelizedLensSimulator(phys, config)
    mock = simulator.simulate(
        jnp.abs(jnp.linspace(-1.0, 1.0, 25)),
        lens_light_amplitudes=jnp.array([0.5]) if joint else None,
    )
    model = PixelizedImageProbModelOperator(
        mock, jnp.full_like(mock, 0.05), psf, 0.08, phys,
        solver_type=solver, fixed_source_bbox=(-0.3, 0.3, -0.3, 0.3),
        fixed_reg_template=jnp.linspace(0.1, 1.0, 25) if adaptive else None,
        position_likelihood=dict(
            positions=[[-0.2, 0.0], [0.2, 0.0]],
            threshold_arcsec=1.0, min_log_like=-10.0,
        ),
    )
    if solver == "pcg":
        # Float32 finite differences at 1e-3 amplify forward-solve error.
        # Tighten both the forward and implicit-adjoint solves in this
        # verification fixture, without changing model defaults or assertions.
        model.pcg_rtol = 1e-8
    # Use caskade's public vector interface, not mutation of parameters in a
    # traced Python closure. Its dynamic parameter ordering is authoritative.
    steps = [
        1e-2 if p is lam else 1e-3 for p in model.get_dynamic_params()
    ]
    return model, model.get_values(), jnp.asarray(steps)


def _check_reconstruction_and_evidence_jacobian(solver, joint, adaptive, interpolation):
    model, theta, steps = _build_model(solver, joint, adaptive, interpolation)
    weights = jnp.linspace(0.5, 1.5, 25)

    def outputs(parameters):
        result = model.evaluate(parameters)
        values = [weights @ result.source_pixels, result.log_evidence]
        if joint:
            values.append(result.lens_amplitudes[0])
        return jnp.stack(values)

    assert model.evaluate(theta).solver_info.converged
    actual = jax.jit(jax.jacrev(outputs))(theta)
    columns = []
    for i, step in enumerate(steps):
        delta = jnp.zeros_like(theta).at[i].set(step)
        assert model.evaluate(theta + delta).solver_info.converged
        assert model.evaluate(theta - delta).solver_info.converged
        columns.append((outputs(theta + delta) - outputs(theta - delta)) / (2 * step))
    expected = jnp.stack(columns, axis=1)
    assert jnp.all(jnp.isfinite(actual))
    # Both the source derivative and the likelihood derivative are checked:
    # the envelope theorem can hide a missing solution derivative in evidence.
    assert jnp.max(jnp.abs(expected[0])) > 1.0
    np.testing.assert_allclose(actual, expected, rtol=0.05, atol=0.03)


@pytest.mark.integration
@pytest.mark.parametrize("solver", ["pcg", "pnpg"])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_operator_reconstruction_and_evidence_jacobian(solver, joint, adaptive):
    _check_reconstruction_and_evidence_jacobian(solver, joint, adaptive, "bilinear")


@pytest.mark.integration
@pytest.mark.parametrize("solver", ["pcg", "pnpg"])
def test_hermite_operator_reconstruction_and_evidence_jacobian(solver):
    with jax.experimental.enable_x64():
        _check_reconstruction_and_evidence_jacobian(solver, True, True, "cubic_hermite")


def _build_overlap_model(solver, joint):
    n, size, dpix, nsub = 5, 7, 0.125, 2
    bbox = tuple(np.float64(value) for value in (-0.375, 0.375, -0.375, 0.375))
    lens_mass = SIE(theta_E=0.12, e1=0.1, e2=0.02, center_x=0.015, center_y=-0.025)
    lens_mass.theta_E.to_dynamic()
    rho = ParamU("adaptive_reg_rho", 1.2, limits=(0., 3.))
    rho.to_dynamic()
    source = PixelizedSourceModel(
        n=n, log_lambda_reg=np.log(0.025), adaptive_reg_rho=rho,
        regularization_type="first-order", interpolation="cubic_hermite",
    )
    source.log_lambda_reg.to_dynamic()
    lens_light = []
    if joint:
        lens = SersicEllipse(
            R_sersic=0.35, n_sersic=1.0, Ie=1.0, e1=0.05, e2=0.02, center_x=0.02, center_y=0.0,
        )
        lens.Ie.to_static(1.0)
        lens.center_x.to_dynamic()
        lens_light = [lens]
    physical = PhysicalModel(lens_mass=[lens_mass], source_light=[source], lens_light=lens_light)
    y, x = np.indices((size, size))
    mask = ((x == 0) & (y == 1)) | ((x == 3) & (y == 5))
    psf = np.array([[0., 0.04, 0.], [0.09, 0.65, 0.12], [0.06, 0.04, 0.]])
    config = SimulatorConfig(dpix=dpix, npix=size, nsub=nsub, psf_kernel=psf, mask=mask)
    simulator = PixelizedLensSimulator(physical, config)
    mock = simulator.simulate(
        jnp.linspace(-0.4, 0.8, n**2), source_bbox=bbox,
        lens_light_amplitudes=jnp.array([0.4]) if joint else None,
    )
    source_y, source_x = np.indices((n, n))
    reference = simulator.simulate(
        jnp.exp(-((source_x - 2.1)**2 + (source_y - 1.7)**2) / 2).ravel(),
        source_bbox=bbox, lens_light_amplitudes=jnp.zeros(1) if joint else None,
    )
    model = PixelizedImageProbModelOperator(
        mock, 0.05 + 0.003 * x + 0.002 * y, psf, dpix, physical, mask=mask, nsub=nsub,
        solver_type=solver, fixed_source_bbox=bbox, adapt_image=reference,
        block_size=2, evidence_logdet="overlap", evidence_block_size=2,
        lens_light_regularization=0.001, pnpg_max_iter=3000, pnpg_rtol=1e-10,
        position_likelihood=dict(
            positions=[(-0.2, 0.), (0.2, 0.)], threshold_arcsec=0.1, min_log_like=-10.,
        ),
    )
    model.pcg_rtol = 1e-12
    return model


@pytest.mark.integration
@pytest.mark.parametrize("solver,joint", [("pcg", False), ("pcg", True), ("pnpg", True)])
def test_hermite_overlap_derivatives_follow_mass_and_adaptive_strengths(solver, joint):
    with jax.experimental.enable_x64():
        model = _build_overlap_model(solver, joint)
        theta = jnp.asarray(model.get_values(), dtype=jnp.float64)
        source_weights = jnp.linspace(0.5, 1.5, 25)
        image_weights = jnp.linspace(0.3, 1.2, 49).reshape(7, 7)

        def observables(result):
            values = [
                result.log_evidence,
                source_weights @ result.source_pixels,
                jnp.sum(image_weights * result.model_image),
            ]
            if joint:
                values.append(result.lens_amplitudes[0])
            return jnp.stack(values)

        def evaluate_valid(parameters):
            result, diagnostics = model.evaluate_with_diagnostics(parameters)
            assert result.log_evidence.dtype == jnp.float64
            assert diagnostics.valid and diagnostics.evidence_valid
            assert diagnostics.position.max_separation > 0.11
            assert diagnostics.position.penalty < 0.0
            return result

        center = evaluate_valid(theta)
        if solver == "pnpg":
            active = center.source_pixels == 0
            assert jnp.any(active) and jnp.any(~active)
            assert center.solver_info.min_active_residual > 0.0

        step = 2e-5
        columns = []
        for offset in step * jnp.eye(theta.size):
            upper, lower = evaluate_valid(theta + offset), evaluate_valid(theta - offset)
            if solver == "pnpg":
                np.testing.assert_array_equal(upper.source_pixels == 0, active)
                np.testing.assert_array_equal(lower.source_pixels == 0, active)
            columns.append((observables(upper) - observables(lower)) / (2 * step))
        expected = jnp.stack(columns, axis=1)
        actual = jax.jit(jax.jacrev(lambda p: observables(model.evaluate(p))))(theta)
        assert jnp.all(jnp.isfinite(actual))
        assert jnp.all(jnp.abs(expected[0]) > 1e-3)
        np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=3e-5)


@pytest.mark.integration
@pytest.mark.parametrize("solver", ["pcg", "pnpg"])
def test_operator_batched_value_and_grad_matches_individual_calls(solver):
    model, theta, steps = _build_model(solver, joint=True)
    points = jnp.stack([theta, theta + 0.1 * steps])
    value_and_grad = jax.jit(jax.value_and_grad(model))
    expected = [value_and_grad(point) for point in points]
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(model)))(points)
    assert jnp.all(jnp.isfinite(gradients))
    np.testing.assert_allclose(values, jnp.stack([v for v, _ in expected]), rtol=2e-5)
    np.testing.assert_allclose(
        gradients, jnp.stack([g for _, g in expected]), rtol=2e-3, atol=2e-3,
    )


@pytest.mark.integration
@pytest.mark.parametrize("solver", ["pcg", "pnpg"])
def test_operator_failed_forward_keeps_penalty_and_rejects_gradient(solver):
    model, theta, _ = _build_model(solver)
    if solver == "pcg":
        model.pcg_max_iter = 0
    else:
        setattr(model, solver + "_max_iter", 0)
        if solver == "pnpg":
            # A projected PCG warm start can already pass PNPG's forward gate.
            model.pnpg_rtol = 0.0
    result = model.evaluate(theta)
    assert not result.solver_info.converged
    assert float(result.log_evidence) == -1e10
    np.testing.assert_array_equal(result.source_pixels, 0.0)
    np.testing.assert_array_equal(result.model_image, 0.0)
    gradient = jax.jit(jax.grad(model))(theta)
    assert jnp.all(jnp.isnan(gradient))
