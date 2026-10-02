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


def _build_model(solver, joint=False):
    sie = SIE(theta_E=0.12, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
    sie.theta_E.to_dynamic()
    lam = ParamU("log_lambda_reg", 0.0)
    lam.to_dynamic()
    source = PixelizedSourceModel(
        n=5, log_lambda_reg=lam, regularization_type="first-order",
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
    )
    # Use caskade's public vector interface, not mutation of parameters in a
    # traced Python closure. Its dynamic parameter ordering is authoritative.
    steps = [
        1e-2 if p is lam else 1e-3 for p in model.get_dynamic_params()
    ]
    return model, model.get_values(), jnp.asarray(steps)


@pytest.mark.integration
@pytest.mark.parametrize("solver", ["pcg", "fista"])
@pytest.mark.parametrize("joint", [False, True])
def test_operator_reconstruction_and_evidence_jacobian(solver, joint):
    model, theta, steps = _build_model(solver, joint)
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
@pytest.mark.parametrize("solver", ["pcg", "fista"])
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
@pytest.mark.parametrize("solver", ["pcg", "fista"])
def test_operator_failed_forward_keeps_penalty_and_rejects_gradient(solver):
    model, theta, _ = _build_model(solver)
    if solver == "pcg":
        model.pcg_max_iter = 0
    else:
        model.fista_max_iter = 0
    result = model.evaluate(theta)
    assert not result.solver_info.converged
    assert float(result.log_evidence) == -1e10
    np.testing.assert_array_equal(result.source_pixels, 0.0)
    np.testing.assert_array_equal(result.model_image, 0.0)
    gradient = jax.jit(jax.grad(model))(theta)
    assert jnp.all(jnp.isnan(gradient))
