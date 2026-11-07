"""Current coupled operator updates through the shared nonlinear fit boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_potential_correction_iterative import _coupled_problem, _renderer


@pytest.mark.parametrize(
    "interpolation, boundary, gradient_policy",
    [
        ("bilinear", "hard_zero_outside_bbox", "current_analytic_bilinear_branch"),
        (
            "cubic_hermite",
            "C1_cardinal_bicubic_fixed_zero_exterior_nodes",
            "current_analytic_cubic_hermite",
        ),
    ],
)
def test_operator_iteration_matches_dense_coupled_fields_images_and_objective(
    interpolation, boundary, gradient_policy
):
    problem = _coupled_problem(source_interpolation=interpolation)
    dense = problem.fit(0.3, 0.02)
    result = problem.fit(0.3, 0.02, backend="operator", block_size=7, pcg_rtol=1e-12)
    for fit in (dense, result):
        assert fit.success, fit.termination_reason
        assert fit.configuration["source_interpolation"] == interpolation
        assert fit.configuration["source_boundary"] == boundary
        assert fit.configuration["gradient_policy"] == gradient_policy
        assert fit.initialization_provenance["gradient_policy"] == gradient_policy
        np.testing.assert_array_equal(fit.source_bbox, problem.renderer.source_bbox)
        accepted = [item for item in fit.history if item["accepted"]]
        assert len(accepted) > 1
        assert all(
            item["objective_after"] < item["objective_before"] for item in accepted
        )
        assert all(item["solver_info"]["relative_residual"] < 1e-8 for item in accepted)
    assert result.algorithm == "iterative" and result.backend == "operator"
    assert result.solver_info.method == "pcg"
    assert result.evidence_method == "local_gauss_newton"
    assert result.determinant_method == "principal_block_cholesky"
    assert not result.evidence_is_exact
    assert result.linear_model_image is result.linear_residual is None
    for field in (
        "source_values",
        "potential_values",
        "nonlinear_model_image",
        "objective",
        "deflection_x",
        "deflection_y",
        "convergence",
    ):
        np.testing.assert_allclose(
            getattr(result, field),
            getattr(dense, field),
            atol=2e-8,
            rtol=2e-7,
        )
    assert np.linalg.norm(result.potential_values) > 1e-4
    assert np.linalg.norm(result.source_values - problem.initial_values[:4]) > 1e-3


@pytest.mark.parametrize("backend", ["dense", "operator"])
def test_convergence_records_the_refreshed_physical_gradient(backend):
    problem = _coupled_problem()
    result = problem.fit(0.3, 0.02, backend=backend, block_size=7, pcg_rtol=1e-12)
    assert result.success, result.termination_reason
    with jax.experimental.enable_x64():
        values = jnp.concatenate((result.source_values, result.potential_values))
        linearization = problem.linearize(values)
        design = np.asarray(
            jax.vmap(lambda vector: problem.jacobian_forward(vector, linearization))(
                jnp.eye(problem.n_joint)
            )
        ).T
        prior = np.asarray(problem.dense_prior(0.3, 0.02))
        noise = np.asarray(problem.noise_active)
        curvature = design.T @ (design / noise[:, None] ** 2) + prior
        gradient = design.T @ (
            (
                np.asarray(problem.data_active)
                - np.asarray(result.nonlinear_image_active)
            )
            / noise**2
        ) - prior @ np.asarray(values)
        expected = np.max(np.abs(gradient) / np.sqrt(np.diag(curvature)))
    np.testing.assert_allclose(
        result.configuration["final_scaled_gradient"], expected, atol=2e-13
    )
    if result.termination_reason == "converged_gradient":
        assert expected <= result.configuration["gradient_atol"] + (
            result.configuration["gradient_rtol"]
            * result.configuration["initial_scaled_gradient"]
        )


@pytest.mark.parametrize("lambda_potential", [0.02, 0.1])
def test_final_current_block_evidence_contains_amplitude_prior_without_step_damping(
    lambda_potential,
):
    from tests.test_potential_correction_iterative import _independent_problem

    problem = _coupled_problem()
    order = np.r_[np.arange(problem.n_joint)[::2], np.arange(problem.n_joint)[1::2]]
    blocks = tuple(tuple(order[i : i + 7]) for i in range(0, len(order), 7))
    results = []
    for damping in (0.01, 10.0):
        result = problem.fit(
            0.3,
            lambda_potential,
            backend="operator",
            block_size=7,
            block_indices=blocks,
            pcg_rtol=1e-12,
            initial_damping=damping,
        )
        assert result.success, result.termination_reason
        forward, _, prior = _independent_problem(
            problem.renderer,
            problem.image_data,
            problem.noise_map,
            0.3,
            lambda_potential,
            20.0,
        )
        values = np.r_[result.source_values, result.potential_values]
        h = 1e-6
        design = np.column_stack(
            [
                (forward(values + h * unit) - forward(values - h * unit)) / (2 * h)
                for unit in np.eye(problem.n_joint)
            ]
        )
        noise = np.asarray(problem.noise_active)
        posterior = design.T @ (design / noise[:, None] ** 2) + prior
        prior_logdet = sum(np.linalg.slogdet(prior[np.ix_(b, b)])[1] for b in blocks)
        posterior_logdet = sum(
            np.linalg.slogdet(posterior[np.ix_(b, b)])[1] for b in blocks
        )
        expected = -0.5 * (
            2 * float(result.objective)
            + posterior_logdet
            - prior_logdet
            + np.log(noise**2).sum()
            + len(noise) * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(result.prior_logdet, prior_logdet, atol=3e-12)
        np.testing.assert_allclose(result.posterior_logdet, posterior_logdet, atol=3e-8)
        np.testing.assert_allclose(result.log_evidence, expected, atol=2e-8)
        assert result.configuration["block_indices"] == blocks
        assert sum(result.configuration["block_sizes"]) == problem.n_joint
        assert result.strengths.lambda_amplitude == 20.0
        results.append(result)
    np.testing.assert_allclose(
        results[0].potential_values, results[1].potential_values, atol=2e-7
    )
    np.testing.assert_allclose(
        results[0].log_evidence, results[1].log_evidence, atol=2e-6
    )


def test_one_operator_step_reports_true_coupled_residual_and_undamped_prediction():
    from tests.test_potential_correction_iterative import _independent_problem

    problem = _coupled_problem()
    mu = 0.2
    result = problem.fit(
        0.3,
        0.02,
        backend="operator",
        block_size=7,
        pcg_rtol=1e-12,
        max_iter=1,
        initial_damping=mu,
    )
    assert not result.success and result.termination_reason == "iteration_limit"
    assert result.history[1]["accepted"] and result.solver_info.converged
    with jax.experimental.enable_x64():
        initial = problem.initial_values
        point = problem.linearize(initial)
        design = np.asarray(
            jax.vmap(lambda vector: problem.jacobian_forward(vector, point))(
                jnp.eye(problem.n_joint)
            )
        ).T
        _, _, prior = _independent_problem(
            problem.renderer, problem.image_data, problem.noise_map, 0.3, 0.02, 20.0
        )
        noise = np.asarray(problem.noise_active)
        weighted = design / noise[:, None]
        curvature = weighted.T @ weighted + prior
        rhs = design.T @ (
            (np.asarray(problem.data_active) - np.asarray(problem.forward(initial)))
            / noise**2
        ) - prior @ np.asarray(initial)
    delta = np.r_[result.source_values, result.potential_values] - np.asarray(initial)
    matrix = curvature + mu * np.diag(np.diag(curvature))
    assert np.linalg.norm(curvature[:4, 4:]) > 1
    np.testing.assert_allclose(delta, np.linalg.solve(matrix, rhs), atol=2e-12)
    residual = np.linalg.norm(matrix @ delta - rhs)
    np.testing.assert_allclose(result.solver_info.residual_norm, residual, atol=3e-13)
    np.testing.assert_allclose(
        result.solver_info.relative_residual, residual / np.linalg.norm(rhs), atol=3e-13
    )
    physical = delta @ curvature @ delta
    predicted = rhs @ delta - 0.5 * physical
    damped = mu * (delta * np.diag(curvature)) @ delta
    np.testing.assert_allclose(
        result.history[1]["predicted_reduction"], predicted, atol=2e-13
    )
    np.testing.assert_allclose(
        result.history[1]["damping_fraction"], damped / (physical + damped), atol=2e-13
    )


@pytest.mark.parametrize(
    "options, solve_rtol, reason",
    [
        ({"pcg_max_iter": 1}, None, "pcg_iteration_limit"),
        ({"pcg_rtol": 1e-2}, 1e-12, "linear_residual_exceeds_tolerance"),
        ({"pcg_atol": 1e6}, None, "linear_residual_exceeds_tolerance"),
    ],
)
def test_unsuccessful_pcg_retains_last_state_and_has_no_evidence(
    options, solve_rtol, reason
):
    problem = _coupled_problem(solve_rtol=solve_rtol)
    result = problem.fit(0.3, 0.02, backend="operator", block_size=7, **options)
    assert not result.success and result.log_evidence is None
    assert result.termination_reason == result.failure_reason == "linear_solve_failed"
    assert result.solver_info.reason == result.history[-1]["reason"] == reason
    assert result.solver_info.failed and not result.solver_info.converged
    assert not any(item["accepted"] for item in result.history)
    np.testing.assert_array_equal(
        result.source_values, problem.initial_values[: problem.n_source]
    )
    np.testing.assert_array_equal(result.potential_values, 0)


def _changing_cells_problem():
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection
    from TinyLensGpu.PhysicalModel import PhysicalModel, SIS, Shear

    with jax.experimental.enable_x64():
        x, y = np.meshgrid((np.arange(12) - 5.5) * 0.12, (np.arange(12) - 5.5) * 0.12)
        radius = np.hypot(x, y)
        renderer = PotentialCorrectionRenderer(
            PhysicalModel(
                lens_mass=[
                    SIS(theta_E=0.4, center_x=0.0, center_y=0.0),
                    Shear(gamma1=0.03, gamma2=0.01),
                ],
                source_light=[],
                lens_light=[],
            ),
            SimulatorConfig(
                0.12,
                12,
                jnp.ones((1, 1)),
                nsub=2,
                mask=(radius < 0.18) | (radius > 0.65),
            ),
            5,
            (-0.6, 0.6, -0.6, 0.6),
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        source = (
            jnp.exp(-0.5 * ((sx - 0.03) / 0.22) ** 2 - 0.5 * ((sy + 0.02) / 0.17) ** 2)
            + 0.6
            * jnp.exp(-0.5 * ((sx + 0.16) / 0.12) ** 2 - 0.5 * ((sy - 0.13) / 0.1) ** 2)
        ).ravel()
        grid = renderer.potential_grid
        potential = 0.02 * jnp.exp(
            -((grid.x - 0.35) ** 2 + (grid.y - 0.07) ** 2) / 0.04
        )
        image = renderer.render(source, potential).model_image
        return IterativePotentialCorrection(
            renderer,
            image,
            jnp.full_like(image, 0.01),
            source,
            lambda_amplitude=100.0,
        )


def test_current_cell_changes_match_dense_and_repeat_without_recompiling_stale_data(
    caplog,
):
    problem = _changing_cells_problem()
    options = dict(backend="operator", block_size=16, pcg_rtol=1e-12, pcg_max_iter=800)
    with jax.log_compiles(True):
        first = problem.fit(1e-4, 100.0, **options)
        assert first.success, first.termination_reason
        caplog.clear()
        changed = problem.fit(4e-4, 70.0, initial_damping=0.1, **options)
        repeat = problem.fit(1e-4, 100.0, **options)
    assert not [r.message for r in caplog.records if "Compiling jit(" in r.message]
    fresh = _changing_cells_problem().fit(4e-4, 70.0, initial_damping=0.1, **options)
    dense = problem.fit(1e-4, 100.0)
    for result in (first, changed, repeat, fresh, dense):
        assert result.success, result.termination_reason
    for left, right in ((first, dense), (first, repeat), (changed, fresh)):
        for field in (
            "source_values",
            "potential_values",
            "nonlinear_model_image",
            "objective",
            "deflection_x",
            "deflection_y",
            "convergence",
        ):
            np.testing.assert_allclose(
                getattr(left, field), getattr(right, field), atol=2e-8, rtol=2e-7
            )
    with jax.experimental.enable_x64():
        values = jnp.concatenate((first.source_values, first.potential_values))
        initial, current = problem.linearize(problem.initial_values), problem.linearize(
            values
        )
        assert (
            np.count_nonzero(
                np.any(
                    np.asarray(initial.mapping.indices)
                    != np.asarray(current.mapping.indices),
                    axis=1,
                )
            )
            >= 5
        )
        assert np.linalg.norm(np.asarray(initial.gradient_x - current.gradient_x)) > 0.1
        rendered = problem.renderer.render(first.source_values, first.potential_values)
        np.testing.assert_allclose(
            first.nonlinear_model_image, rendered.model_image, atol=2e-13
        )
    mask = np.asarray(problem.renderer.sim_config.mask)
    assert np.isnan(np.asarray(first.nonlinear_residual)[mask]).all()
    np.testing.assert_array_equal(first.derivative_valid, dense.derivative_valid)
    assert np.isnan(
        np.asarray(first.convergence)[~np.asarray(first.derivative_valid)]
    ).all()
    assert (
        changed.configuration["block_partition_sha256"]
        == first.configuration["block_partition_sha256"]
    )


def test_operator_float32_matches_float64_coupled_reference_without_changing_defaults():
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection

    previous = jax.config.x64_enabled
    matmul = jax.config.jax_default_matmul_precision
    reference_problem = _coupled_problem()
    reference = reference_problem.fit(0.3, 0.02)
    renderer = _renderer(dtype="float32", source_n=2, bbox=(-0.5, 0.5, -0.5, 0.5))
    problem = IterativePotentialCorrection(
        renderer,
        np.asarray(reference_problem.image_data),
        np.asarray(reference_problem.noise_map),
        np.asarray(reference_problem.initial_values[:4]),
        lambda_amplitude=20.0,
        source_regularization="zero-order",
        solve_rtol=1e-5,
        solve_atol=1e-6,
    )
    # Named coupled37 precision configuration. Global-objective float32 ULP
    # exceeds the remaining decrease below this gradient; six observed
    # cold/warm observations and strict/matched-control F64 comparisons retain
    # the original field guard below. Public solver defaults are unchanged.
    # See docs/validation/potential-correction-precision-performance.md.
    result = problem.fit(
        0.3,
        0.02,
        backend="operator",
        block_size=7,
        pcg_rtol=1e-8,
        gradient_atol=1e-4,
        gradient_rtol=1e-5,
        objective_rtol=1e-6,
        step_rtol=1e-4,
    )
    assert reference.success and result.success, result.termination_reason
    assert jax.config.x64_enabled == previous
    assert jax.config.jax_default_matmul_precision == matmul
    for field in (
        "source_values",
        "potential_values",
        "nonlinear_model_image",
        "objective",
        "log_evidence",
        "prior_logdet",
        "posterior_logdet",
    ):
        assert getattr(result, field).dtype == np.dtype("float32")
    for field in (
        "source_values",
        "potential_values",
        "nonlinear_model_image",
        "objective",
    ):
        np.testing.assert_allclose(
            getattr(result, field), getattr(reference, field), atol=2e-4, rtol=2e-4
        )


def test_operator_nonfinite_current_state_cannot_produce_evidence():
    result = _coupled_problem().fit(1e308, 0.02, backend="operator", block_size=7)
    assert not result.success and result.log_evidence is None
    assert (
        result.termination_reason == result.failure_reason == "nonfinite_current_state"
    )
    assert not any(item["accepted"] for item in result.history)


@pytest.mark.parametrize(
    "options, match",
    [
        ({"block_size": 0}, "block_size"),
        ({"block_indices": [(0, 1), (1, 2)]}, "exactly once"),
        ({"pcg_max_iter": 0}, "pcg_max_iter"),
        ({"pcg_rtol": np.nan}, "pcg_rtol"),
        ({"pcg_atol": -1.0}, "pcg_atol"),
    ],
)
def test_invalid_operator_iteration_controls_raise_configuration_errors(options, match):
    with pytest.raises(ValueError, match=match):
        _coupled_problem().fit(0.3, 0.02, backend="operator", **options)
