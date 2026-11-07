"""Public fixed-strength L-BFGS-B fits against small explicit GN matrices."""

import jax
import numpy as np
import pytest

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.PhysicalModel import PhysicalModel, Shear


def _renderer(
    interpolation="cubic_hermite", dtype="float64", *, bbox=(-0.17, 0.19, -0.16, 0.20)
):
    with jax.experimental.enable_x64(dtype == "float64"):
        return PotentialCorrectionRenderer(
            PhysicalModel(
                lens_mass=[Shear(gamma1=0.0, gamma2=0.0)],
                source_light=[],
                lens_light=[],
            ),
            SimulatorConfig(0.1, 4, np.ones((1, 1))),
            3,
            bbox,
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


def _first_white_step(base, inverse_root, forward, data_weighted, prior):
    from scipy.optimize import minimize

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
    return minimize(
        independent_white_objective,
        np.zeros_like(base),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1, "maxls": 80, "ftol": 1e-15, "gtol": 1e-8},
    )


def test_dense_whitening_can_use_a_distinct_bilinear_reference(coupled_case):
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

    expected = _first_white_step(base, inverse_root, forward, data_weighted, prior)
    result = fit.fit(strengths, max_iter=1)
    np.testing.assert_allclose(
        _joint(result), base + inverse_root @ expected.x, atol=1e-8
    )
    np.testing.assert_allclose(result.objective, expected.fun, atol=1e-8)
    np.testing.assert_allclose(
        result.optimizer_info.initial_objective,
        0.5 * (np.sum((forward(base) - data_weighted) ** 2) + base @ prior @ base),
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


@pytest.mark.parametrize("backend", ["dense", "operator"])
def test_candidates_restart_and_use_the_supplied_amplitude_strength(
    coupled_case, backend
):
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
    results = [
        fit.fit(strengths, backend=backend) for strengths in (first, second, first)
    ]
    for result, strengths in zip(results, (first, second, first)):
        assert result.success, result.failure_reason
        assert result.strengths == strengths
        values = _joint(result)
        prior = _prior(renderer, scale, strengths)
        jacobian = _jacobian(_weighted_render(renderer, noise), values)
        curvature = jacobian.T @ jacobian + prior
        if backend == "operator":
            assert (
                result.configuration["block_layout"] == "field_separated_contiguous_v1"
            )
            assert result.configuration["block_size"] == 64
            blocks = result.configuration["block_indices"]
            assert [index for block in blocks for index in block] == list(
                range(values.size)
            )
            assert all(
                block[-1] < renderer.n_source or block[0] >= renderer.n_source
                for block in blocks
            )
            prior_logdet = _principal_logdet(prior, blocks)
            posterior_logdet = _principal_logdet(curvature, blocks)
        else:
            prior_logdet = np.linalg.slogdet(prior)[1]
            posterior_logdet = np.linalg.slogdet(curvature)[1]
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


def _principal_logdet(matrix, blocks):
    return sum(np.linalg.slogdet(matrix[np.ix_(block, block)])[1] for block in blocks)


def _mixed_partition(n_source, n_joint, width):
    source = list(range(n_source))
    potential = list(range(n_source, n_joint))
    mixed = [index for pair in zip(source, potential) for index in pair]
    mixed.extend(potential[n_source:])
    return tuple(tuple(mixed[i : i + width]) for i in range(0, n_joint, width))


def test_operator_fixed_strengths_qualify_against_principal_block_reference(
    coupled_case,
):
    from TinyLensGpu.ObservationModel import (
        LBFGSPotentialCorrection,
        PotentialCorrectionStrengths,
    )

    renderer, data, noise, initial_source, scale = coupled_case
    strengths = PotentialCorrectionStrengths(0.13, 0.5, 120.0)
    n_joint = renderer.n_source + renderer.potential_grid.n_unknown
    blocks = _mixed_partition(renderer.n_source, n_joint, 5)
    fit = LBFGSPotentialCorrection(
        renderer,
        data,
        noise,
        initial_source,
        source_regularization="zero-order",
        source_scale=scale,
    )
    result = fit.fit(
        strengths,
        backend="operator",
        block_size=5,
        block_indices=blocks,
    )

    assert result.success, result.failure_reason
    assert result.solver_info is None
    assert result.backend == "operator"
    assert result.evidence_method == "local_gauss_newton"
    assert result.determinant_method == "principal_block_cholesky"
    assert not result.evidence_is_exact
    assert result.configuration["block_indices"] == blocks
    assert result.configuration["block_layout"] == "explicit_joint_partition_v1"
    assert result.configuration["pcg_max_iter"] == 400

    values = _joint(result)
    forward = _weighted_render(renderer, noise)
    prior = _prior(renderer, scale, strengths)
    jacobian = _jacobian(forward, values)
    residual = forward(values) - data.ravel() / noise.ravel()
    gradient = jacobian.T @ residual + prior @ values
    curvature = jacobian.T @ jacobian + prior
    prior_logdet = _principal_logdet(prior, blocks)
    posterior_logdet = _principal_logdet(curvature, blocks)

    objective = 0.5 * (residual @ residual + values @ prior @ values)
    np.testing.assert_allclose(result.objective, objective)
    np.testing.assert_allclose(result.prior_logdet, prior_logdet)
    np.testing.assert_allclose(result.posterior_logdet, posterior_logdet, atol=1e-7)
    np.testing.assert_allclose(
        result.log_evidence,
        -objective
        - 0.5
        * (
            posterior_logdet
            - prior_logdet
            + 2 * np.log(noise).sum()
            + noise.size * np.log(2 * np.pi)
        ),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.optimizer_info.gn_stationarity,
        0.5 * gradient @ np.linalg.solve(curvature, gradient),
        atol=1e-12,
    )
    certificate = result.optimizer_info.diagnostics["stationarity_solver"]
    assert certificate["method"] == "pcg"
    assert certificate["converged"]
    assert certificate["true_residual_norm"] <= certificate["true_residual_threshold"]


def test_operator_whitening_and_coupled_stationarity_at_nonstationary_endpoint(
    coupled_case,
):
    from TinyLensGpu.ObservationModel import (
        LBFGSPotentialCorrection,
        PotentialCorrectionStrengths,
        PotentialCorrectionWhiteningReference,
    )

    renderer, data, noise, initial_source, scale = coupled_case
    strengths = PotentialCorrectionStrengths(0.13, 0.5, 120.0)
    reference_source = 0.3 - 1.7 * initial_source
    base = np.r_[initial_source, np.zeros(renderer.potential_grid.n_unknown)]
    blocks = _mixed_partition(renderer.n_source, base.size, 5)
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
    result = fit.fit(
        strengths,
        backend="operator",
        block_size=5,
        block_indices=blocks,
        max_iter=1,
    )

    prior = _prior(renderer, scale, strengths)
    reference = np.r_[reference_source, base[renderer.n_source :]]
    jacobian = _jacobian(_weighted_render(_renderer("bilinear"), noise), reference)
    reference_curvature = jacobian.T @ jacobian + prior
    inverse_root = np.zeros_like(reference_curvature)
    for block in blocks:
        panel = np.ix_(block, block)
        inverse_root[panel] = np.linalg.solve(
            np.linalg.cholesky(reference_curvature[panel]).T,
            np.eye(len(block)),
        )
    forward = _weighted_render(renderer, noise)
    data_weighted = data.ravel() / noise.ravel()
    expected = _first_white_step(base, inverse_root, forward, data_weighted, prior)
    values = _joint(result)
    np.testing.assert_allclose(values, base + inverse_root @ expected.x, atol=1e-8)
    np.testing.assert_allclose(result.objective, expected.fun, atol=1e-8)

    jacobian = _jacobian(forward, values)
    gradient = jacobian.T @ (forward(values) - data_weighted) + prior @ values
    curvature = jacobian.T @ jacobian + prior
    stationarity = 0.5 * gradient @ np.linalg.solve(curvature, gradient)
    block_only = sum(
        0.5
        * gradient[list(block)]
        @ np.linalg.solve(curvature[np.ix_(block, block)], gradient[list(block)])
        for block in blocks
    )
    assert abs(stationarity - block_only) > 1e-3
    np.testing.assert_allclose(
        result.optimizer_info.gn_stationarity, stationarity, atol=1e-8
    )
    certificate = result.optimizer_info.diagnostics["stationarity_solver"]
    assert certificate["converged"]
    np.testing.assert_allclose(
        certificate["rhs_norm"], np.linalg.norm(gradient), rtol=1e-7
    )
    assert certificate["true_residual_norm"] <= certificate["true_residual_threshold"]
    assert not result.success and result.log_evidence is None
    assert result.failure_reason == "native_optimizer_did_not_succeed"

    # The first PCG direction and true residual have a direct matrix reference.
    limited = fit.fit(
        strengths,
        backend="operator",
        block_size=5,
        block_indices=blocks,
        max_iter=1,
        pcg_max_iter=1,
    )
    np.testing.assert_allclose(_joint(limited), values, atol=1e-10, rtol=0)
    direction = np.empty_like(gradient)
    for block in blocks:
        direction[list(block)] = np.linalg.solve(
            curvature[np.ix_(block, block)], gradient[list(block)]
        )
    first_step = (
        (gradient @ direction) / (direction @ curvature @ direction) * direction
    )
    limited_certificate = limited.optimizer_info.diagnostics["stationarity_solver"]
    assert not limited_certificate["converged"]
    assert limited_certificate["iterations"] == 1
    np.testing.assert_allclose(
        limited_certificate["true_residual_norm"],
        np.linalg.norm(gradient - curvature @ first_step),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        limited.optimizer_info.gn_stationarity,
        0.5 * gradient @ first_step,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "pcg_controls, reason",
    [
        ({"pcg_max_iter": 1}, "pcg_iteration_limit"),
        ({"pcg_rtol": 1e-6}, "linear_residual_exceeds_tolerance"),
    ],
)
def test_operator_rejects_uncertified_coupled_solves(
    coupled_case, pcg_controls, reason
):
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
        solve_rtol=1e-16,
    )
    result = fit.fit(strengths, backend="operator", **pcg_controls)
    assert result.optimizer_info.native_success
    assert not result.success and result.log_evidence is None
    assert result.failure_reason == reason
    assert result.solver_info is None
    certificate = result.optimizer_info.diagnostics["stationarity_solver"]
    assert certificate["failed"] and certificate["reason"] == reason
    assert certificate["recursive_converged"] == ("pcg_rtol" in pcg_controls)
    assert certificate["true_residual_norm"] > certificate["true_residual_threshold"]
    np.testing.assert_allclose(
        certificate["true_residual_threshold"],
        1e-16 * certificate["rhs_norm"],
    )
    assert np.isfinite(result.objective) and np.isfinite(_joint(result)).all()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"block_size": 0}, "block_size"),
        ({"block_indices": ((0, 1),)}, "exactly once"),
        ({"pcg_max_iter": 0}, "pcg_max_iter"),
        ({"pcg_rtol": -1}, "pcg_rtol"),
        ({"pcg_atol": np.nan}, "pcg_atol"),
    ],
)
def test_operator_invalid_controls_raise(coupled_case, options, message):
    from TinyLensGpu.ObservationModel import (
        LBFGSPotentialCorrection,
        PotentialCorrectionStrengths,
    )

    renderer, data, noise, initial_source, _ = coupled_case
    fit = LBFGSPotentialCorrection(renderer, data, noise, initial_source)
    with pytest.raises(ValueError, match=message):
        fit.fit(PotentialCorrectionStrengths(1, 1, 1), backend="operator", **options)


def test_operator_qualifies_at_tiny_positive_noise_with_zero_response():
    from TinyLensGpu.ObservationModel import (
        LBFGSPotentialCorrection,
        PotentialCorrectionStrengths,
    )

    # The macro rays fall outside the compact source support, so the data
    # response is zero and the exact GN system consists of the physical prior.
    renderer = _renderer(bbox=(10.0, 11.0, 10.0, 11.0))
    fitter = LBFGSPotentialCorrection(
        renderer,
        np.zeros(renderer.image_shape),
        np.full(renderer.image_shape, 1e-180),
        np.ones(renderer.n_source),
        source_regularization="zero-order",
    )
    strengths = PotentialCorrectionStrengths(1.0, 1.0, 1.0)
    dense = fitter.fit(strengths)
    operator = fitter.fit(strengths, backend="operator")

    assert dense.success, dense.failure_reason
    assert operator.success, operator.failure_reason
    assert operator.optimizer_info.diagnostics["stationarity_solver"]["converged"]
    np.testing.assert_allclose(_joint(operator), _joint(dense), atol=1e-14)
    np.testing.assert_allclose(operator.log_evidence, dense.log_evidence, rtol=1e-14)
