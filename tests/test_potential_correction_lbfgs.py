"""Public fixed-strength L-BFGS-B fits against small explicit GN matrices."""

import jax
import numpy as np
import pytest

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.PhysicalModel import PhysicalModel, Shear


def _renderer(interpolation="cubic_hermite", dtype="float64"):
    with jax.experimental.enable_x64(dtype == "float64"):
        return PotentialCorrectionRenderer(
            PhysicalModel(
                lens_mass=[Shear(gamma1=0.0, gamma2=0.0)],
                source_light=[],
                lens_light=[],
            ),
            SimulatorConfig(0.1, 4, np.ones((1, 1))),
            3,
            (-0.17, 0.19, -0.16, 0.20),
            dtype=dtype,
            potential_spacing=0.1,
            source_interpolation=interpolation,
        )


@pytest.fixture
def coupled_case():
    renderer = _renderer()
    rng = np.random.default_rng(2026)
    source = rng.normal(size=renderer.n_source)
    potential = rng.normal(scale=2e-3, size=renderer.potential_grid.n_unknown)
    data = np.asarray(renderer.render(source, potential).model_image)
    noise = rng.uniform(0.3, 0.5, size=renderer.image_shape)
    initial_source = source * 0.8 + 0.15
    scale = rng.uniform(0.4, 1.6, size=renderer.n_source)
    return renderer, data, noise, initial_source, scale


def _prior(renderer, scale, strengths):
    """Independently enumerate all complete potential windows."""
    n_source = renderer.n_source
    nodes = np.asarray(renderer.potential_grid.node_indices)
    lookup = {tuple(node): index for index, node in enumerate(nodes)}
    matrix = np.diag(
        np.r_[
            strengths.lambda_source * scale,
            np.full(len(nodes), strengths.lambda_amplitude),
        ]
    )
    for x, y in nodes:
        for dx, dy in ((1, 0), (0, 1)):
            window = [(x + k * dx, y + k * dy) for k in range(5)]
            if all(node in lookup for node in window):
                indices = [n_source + lookup[node] for node in window]
                coefficients = [1, -4, 6, -4, 1]
                matrix[
                    np.ix_(indices, indices)
                ] += strengths.lambda_potential * np.outer(coefficients, coefficients)
    return matrix


def _weighted_render(renderer, noise):
    def forward(values):
        n = renderer.n_source
        mapping = renderer.source_mapping(values[n:])
        return np.asarray(renderer.source_forward(values[:n], mapping)) / noise.ravel()

    return forward


def _jacobian(forward, values, step=2e-7):
    return np.column_stack(
        [
            (forward(values + direction) - forward(values - direction)) / (2 * step)
            for direction in step * np.eye(values.size)
        ]
    )


def _joint(result):
    return np.r_[result.source_values, result.potential_values]


def test_dense_fixed_strengths_qualify_against_complete_gn_reference(coupled_case):
    from TinyLensGpu.ObservationModel import (
        LBFGSPotentialCorrection,
        PotentialCorrectionStrengths,
    )

    renderer, data, noise, initial_source, scale = coupled_case
    strengths = PotentialCorrectionStrengths(0.13, 0.5, 120.0)
    fit = LBFGSPotentialCorrection(
        renderer,
        data,
        noise,
        initial_source,
        source_regularization="zero-order",
        source_scale=scale,
    )
    result = fit.fit(strengths)

    assert result.success, result.failure_reason
    assert result.solver_info is None
    assert result.backend == "dense"
    assert result.algorithm == "lbfgs"
    assert result.dtype == "float64"
    assert result.evidence_method == "local_gauss_newton"
    assert result.determinant_method == "full_data_space"
    assert not result.evidence_is_exact
    assert result.strengths == strengths
    assert np.any(np.asarray(result.source_values) < 0)
    assert np.any(np.asarray(result.potential_values) < 0)
    assert result.linear_model_image is result.linear_residual is None

    values = _joint(result)
    forward = _weighted_render(renderer, noise)
    prior = _prior(renderer, scale, strengths)
    jacobian = _jacobian(forward, values)
    residual = forward(values) - data.ravel() / noise.ravel()
    gradient = jacobian.T @ residual + prior @ values
    curvature = jacobian.T @ jacobian + prior
    assert np.linalg.norm(curvature[: renderer.n_source, renderer.n_source :]) > 1
    chi2 = residual @ residual
    penalty = values @ prior @ values
    objective = 0.5 * (chi2 + penalty)
    prior_logdet = np.linalg.slogdet(prior)[1]
    posterior_logdet = np.linalg.slogdet(curvature)[1]
    noise_logdet = 2 * np.log(noise).sum()
    log_evidence = -objective - 0.5 * (
        posterior_logdet - prior_logdet + noise_logdet + noise.size * np.log(2 * np.pi)
    )

    np.testing.assert_allclose(result.nonlinear_normalized_residual.ravel(), -residual)
    np.testing.assert_allclose(result.chi2, chi2)
    np.testing.assert_allclose(result.regularization_penalty, penalty)
    np.testing.assert_allclose(result.objective, objective)
    np.testing.assert_allclose(result.prior_logdet, prior_logdet)
    np.testing.assert_allclose(result.posterior_logdet, posterior_logdet, atol=1e-7)
    np.testing.assert_allclose(result.noise_logdet, noise_logdet)
    np.testing.assert_allclose(result.log_evidence, log_evidence, atol=1e-7)
    info = result.optimizer_info
    assert info.native_success and info.native_status == 0
    assert 0 < info.iterations <= info.function_evaluations
    assert info.gradient_evaluations == info.function_evaluations
    np.testing.assert_allclose(info.gradient_inf, np.max(np.abs(gradient)), atol=2e-7)
    np.testing.assert_allclose(
        info.gn_stationarity,
        0.5 * gradient @ np.linalg.solve(curvature, gradient),
        atol=1e-12,
    )
    assert info.gn_stationarity <= info.gn_stationarity_tol == 1e-3


def test_dense_whitening_can_use_a_distinct_bilinear_reference(coupled_case):
    from scipy.optimize import minimize

    from TinyLensGpu.ObservationModel import (
        LBFGSPotentialCorrection,
        PotentialCorrectionStrengths,
        PotentialCorrectionWhiteningReference,
    )

    renderer, data, noise, initial_source, scale = coupled_case
    strengths = PotentialCorrectionStrengths(0.13, 0.5, 120.0)
    reference_source = 0.3 - 1.7 * initial_source
    fit = LBFGSPotentialCorrection(
        renderer,
        data,
        noise,
        initial_source,
        source_regularization="zero-order",
        source_scale=scale,
        whitening_reference=PotentialCorrectionWhiteningReference(
            reference_source, "bilinear"
        ),
    )

    zeros = np.zeros(renderer.potential_grid.n_unknown)
    base = np.r_[initial_source, zeros]
    reference = np.r_[reference_source, zeros]
    prior = _prior(renderer, scale, strengths)
    reference_jacobian = _jacobian(
        _weighted_render(_renderer("bilinear"), noise), reference
    )
    reference_curvature = reference_jacobian.T @ reference_jacobian + prior
    inverse_root = np.linalg.solve(
        np.linalg.cholesky(reference_curvature).T, np.eye(base.size)
    )
    forward = _weighted_render(renderer, noise)
    data_weighted = data.ravel() / noise.ravel()

    def independent_white_objective(white):
        values = base + inverse_root @ white
        residual = forward(values) - data_weighted
        prior_values = prior @ values
        physical_gradient = _jacobian(forward, values).T @ residual + prior_values
        return (
            0.5 * (residual @ residual + values @ prior_values),
            inverse_root.T @ physical_gradient,
        )

    # At one iteration the optimizer trajectory checks the whitening geometry,
    # without requiring the library's particular square root orientation.
    expected = minimize(
        independent_white_objective,
        np.zeros_like(base),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1, "maxls": 80, "ftol": 1e-15, "gtol": 1e-8},
    )
    result = fit.fit(strengths, max_iter=1)
    np.testing.assert_allclose(
        _joint(result), base + inverse_root @ expected.x, atol=1e-8
    )
    np.testing.assert_allclose(result.objective, expected.fun, atol=1e-8)
    np.testing.assert_allclose(
        result.optimizer_info.initial_objective,
        independent_white_objective(np.zeros_like(base))[0],
    )
    assert (
        result.initialization_provenance["whitening_source_interpolation"] == "bilinear"
    )
    assert result.configuration["source_interpolation"] == "cubic_hermite"
    assert not result.success and result.log_evidence is None
    assert result.failure_reason == "native_optimizer_did_not_succeed"
    assert result.optimizer_info.native_success is False
    assert result.optimizer_info.native_status == expected.status
    assert result.optimizer_info.iterations == 1
    assert np.isfinite(result.optimizer_info.gn_stationarity)


def test_dense_candidates_restart_and_use_the_supplied_amplitude_strength(coupled_case):
    from TinyLensGpu.ObservationModel import (
        LBFGSPotentialCorrection,
        PotentialCorrectionStrengths,
    )

    renderer, data, noise, initial_source, scale = coupled_case
    first = PotentialCorrectionStrengths(0.13, 0.5, 120.0)
    second = PotentialCorrectionStrengths(0.13, 0.5, 1100.0)
    fit = LBFGSPotentialCorrection(
        renderer,
        data,
        noise,
        initial_source,
        source_regularization="zero-order",
        source_scale=scale,
    )
    results = [fit.fit(strengths) for strengths in (first, second, first)]
    for result, strengths in zip(results, (first, second, first)):
        assert result.success, result.failure_reason
        assert result.strengths == strengths
        values = _joint(result)
        prior = _prior(renderer, scale, strengths)
        jacobian = _jacobian(_weighted_render(renderer, noise), values)
        prior_logdet = np.linalg.slogdet(prior)[1]
        posterior_logdet = np.linalg.slogdet(jacobian.T @ jacobian + prior)[1]
        np.testing.assert_allclose(
            result.regularization_penalty, values @ prior @ values
        )
        np.testing.assert_allclose(result.prior_logdet, prior_logdet)
        np.testing.assert_allclose(
            result.log_evidence,
            -result.objective
            - 0.5
            * (
                posterior_logdet
                - prior_logdet
                + result.noise_logdet
                + noise.size * np.log(2 * np.pi)
            ),
            atol=1e-7,
        )

    a, b, repeated_a = results
    np.testing.assert_allclose(_joint(a), _joint(repeated_a), atol=2e-7)
    np.testing.assert_allclose(
        a.log_evidence, repeated_a.log_evidence, atol=1e-7, rtol=0
    )
    assert a.optimizer_info.initial_objective == b.optimizer_info.initial_objective
    assert np.linalg.norm(a.potential_values - b.potential_values) > 1e-3


def test_dense_numerical_failure_keeps_fields_without_evidence(coupled_case):
    from TinyLensGpu.ObservationModel import (
        LBFGSPotentialCorrection,
        PotentialCorrectionStrengths,
    )

    renderer, data, noise, initial_source, scale = coupled_case
    fit = LBFGSPotentialCorrection(
        renderer,
        data,
        noise,
        initial_source,
        source_regularization="zero-order",
        source_scale=scale,
    )
    stationary = fit.fit(
        PotentialCorrectionStrengths(0.13, 0.5, 120), gn_stationarity_tol=0
    )
    assert not stationary.success and stationary.log_evidence is None
    assert stationary.optimizer_info.native_success
    assert stationary.optimizer_info.gn_stationarity > 0
    assert stationary.failure_reason == "gn_stationarity_exceeds_tolerance"

    overflow = fit.fit(PotentialCorrectionStrengths(0.13, 1e308, 120))
    assert not overflow.success and overflow.log_evidence is None
    assert overflow.optimizer_info.native_status is None
    assert overflow.optimizer_info.function_evaluations == 0
    assert "Potential prior" in overflow.failure_reason
    np.testing.assert_array_equal(overflow.source_values, initial_source)
    np.testing.assert_array_equal(overflow.potential_values, 0)
    assert np.isfinite(overflow.nonlinear_model_image).all()


def test_dense_invalid_configuration_raises(coupled_case):
    from TinyLensGpu.ObservationModel import (
        LBFGSPotentialCorrection,
        PotentialCorrectionStrengths,
        PotentialCorrectionWhiteningReference,
    )

    renderer, data, noise, initial_source, _ = coupled_case
    for invalid_renderer in (_renderer("bilinear"), _renderer(dtype="float32")):
        with pytest.raises(ValueError, match="Float64 cubic_hermite"):
            LBFGSPotentialCorrection(invalid_renderer, data, noise, initial_source)
    with pytest.raises(ValueError, match="initial_source"):
        LBFGSPotentialCorrection(renderer, data, noise, initial_source[:-1])
    with pytest.raises(ValueError, match="whitening_reference.source_values"):
        LBFGSPotentialCorrection(
            renderer,
            data,
            noise,
            initial_source,
            whitening_reference=PotentialCorrectionWhiteningReference(
                np.full(renderer.n_source, np.nan), "bilinear"
            ),
        )

    fit = LBFGSPotentialCorrection(renderer, data, noise, initial_source)
    with pytest.raises(ValueError, match="lambda_amplitude"):
        fit.fit(PotentialCorrectionStrengths(1, 1, np.inf))
    with pytest.raises(ValueError, match="max_iter"):
        fit.fit(PotentialCorrectionStrengths(1, 1, 1), max_iter=0)
