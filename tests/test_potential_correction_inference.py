"""Public log-strength inference on real source–potential reconstructions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _problem(*, dtype="float64", lambda_amplitude=0.4, solve_rtol=None):
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    state = {"gradient_scale": 1.0}

    def gradient(bx, by):
        return (
            state["gradient_scale"] * jnp.full_like(bx, 1.2),
            state["gradient_scale"] * jnp.full_like(by, -0.8),
        )

    with jax.experimental.enable_x64(dtype == "float64"):
        renderer = PotentialCorrectionRenderer(
            PhysicalModel(
                lens_mass=[Shear(gamma1=0.0, gamma2=0.0)],
                source_light=[],
                lens_light=[],
            ),
            SimulatorConfig(0.1, 3, jnp.ones((1, 1))),
            2,
            (-0.3, 0.3, -0.3, 0.3),
            dtype=dtype,
        )
    image = np.array([[-1.4, -0.8, -1.1], [-0.3, -0.6, -0.2], [-0.9, -0.1, -0.5]])
    return (
        SingleStepPotentialCorrection(
            renderer,
            image,
            np.linspace(0.2, 0.4, 9).reshape(3, 3),
            gradient,
            lambda_amplitude=lambda_amplitude,
            source_regularization="zero-order",
            initialization_provenance={"source": "analytic mock gradient"},
            solve_rtol=solve_rtol,
        ),
        state,
    )


def test_callable_refits_reordered_candidates_with_fixed_gradient_and_amplitude():
    from TinyLensGpu.Inference import PotentialCorrectionEvidence

    problem, state = _problem()
    objective = PotentialCorrectionEvidence(problem)
    points = [np.log([1.3, 0.03]), np.log([0.2, 2.0])]
    reference = [problem.fit(*np.exp(point)) for point in points]
    state["gradient_scale"] = 10.0

    for index in (0, 1, 1, 0):
        value = objective(points[index])
        result = objective.last_reconstruction
        assert result.success
        np.testing.assert_allclose(value, -reference[index].log_evidence, atol=1e-12)
        np.testing.assert_array_equal(
            result.source_values, reference[index].source_values
        )
        np.testing.assert_array_equal(
            result.potential_values, reference[index].potential_values
        )
        assert result.strengths.lambda_amplitude == 0.4
        assert result.potential_grid is problem.renderer.potential_grid
        assert (
            result.initialization_provenance
            == reference[index].initialization_provenance
        )

    assert len(objective.evaluations) == 4
    assert all(record.success for record in objective.evaluations)
    assert all(
        record.strengths.lambda_amplitude == 0.4 for record in objective.evaluations
    )
    assert objective.best_reconstruction.success
    np.testing.assert_allclose(
        objective.best_evaluation.objective,
        min(-float(result.log_evidence) for result in reference),
    )


def _independent_negative_log_evidence(problem):
    """Gaussian data covariance assembled from coordinates and complete windows."""
    renderer = problem.renderer
    x, y = np.asarray(renderer.query_x), np.asarray(renderer.query_y)
    fx, fy = (x + 0.3) / 0.6, (y + 0.3) / 0.6
    source = np.array([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]).T
    grid = renderer.potential_grid
    nodes = np.asarray(grid.node_indices)
    lookup = {tuple(node): index for index, node in enumerate(nodes)}
    potential = np.zeros((9, len(nodes)))
    windows = []
    for row, (qx, qy) in enumerate(zip(x, y)):
        ux, uy = qx / grid.spacing, qy / grid.spacing
        ix, iy = int(np.floor(ux)), int(np.floor(uy))
        tx, ty = ux - ix, uy - iy
        for cx, cy, weight in (
            (ix, iy, (1 - tx) * (1 - ty)),
            (ix + 1, iy, tx * (1 - ty)),
            (ix, iy + 1, (1 - tx) * ty),
            (ix + 1, iy + 1, tx * ty),
        ):
            for dx, dy, factor in (
                (1, 0, -1.2),
                (-1, 0, 1.2),
                (0, 1, 0.8),
                (0, -1, -0.8),
            ):
                potential[row, lookup[(cx + dx, cy + dy)]] += (
                    weight * factor / (2 * grid.spacing)
                )
    for cx, cy in nodes:
        for dx, dy in ((1, 0), (0, 1)):
            window = [(cx + step * dx, cy + step * dy) for step in range(-2, 3)]
            if all(node in lookup for node in window):
                row = np.zeros(len(nodes))
                row[[lookup[node] for node in window]] = [1, -4, 6, -4, 1]
                windows.append(row)
    differences = np.asarray(windows)
    eigenvalues, eigenvectors = np.linalg.eigh(differences.T @ differences)
    projected = potential @ eigenvectors
    data, noise = np.asarray(problem.data_active), np.asarray(problem.noise_active)

    def negative_log_evidence(log_strengths):
        lambda_source, lambda_potential = np.exp(log_strengths)
        covariance = (
            np.diag(noise**2)
            + source @ source.T / lambda_source
            + (projected / (lambda_potential * eigenvalues + problem.lambda_amplitude))
            @ projected.T
        )
        return 0.5 * (
            data @ np.linalg.solve(covariance, data)
            + np.linalg.slogdet(covariance)[1]
            + len(data) * np.log(2 * np.pi)
        )

    return negative_log_evidence


def test_dense_direct_matches_independent_bounded_gaussian_scan_and_returns_fit():
    from TinyLensGpu.Inference import optimize_potential_correction

    problem, _ = _problem()
    bounds = np.log([[0.05, 30.0], [0.001, 30.0]])
    reference = _independent_negative_log_evidence(problem)
    scan = np.array(
        [
            reference((x, y))
            for x in np.linspace(*bounds[0], 31)
            for y in np.linspace(*bounds[1], 31)
        ]
    )
    result = optimize_potential_correction(
        problem,
        log_bounds=bounds,
        maxfun=180,
        maxiter=100,
        len_tol=1e-3,
        vol_tol=1e-12,
    )

    assert result.reconstruction.success
    assert result.has_valid_fit
    assert result.objective <= scan.min() + 1e-3
    np.testing.assert_allclose(
        result.objective, reference(result.log_strengths), atol=2e-10
    )
    np.testing.assert_allclose(
        result.objective, -result.reconstruction.log_evidence, atol=1e-12
    )
    np.testing.assert_array_equal(result.log_bounds, bounds)
    assert result.nfev == len(result.evaluations)
    assert result.configuration["optimizer"] == "scipy.optimize.direct"
    assert result.configuration["coordinate_system"] == "natural_log_strengths"
    assert result.configuration["maxfun"] == 180
    assert result.configuration["maxiter"] == 100
    assert result.configuration["len_tol"] == 1e-3
    assert result.configuration["vol_tol"] == 1e-12
    assert result.strengths == result.reconstruction.strengths
    for evaluation in result.evaluations:
        assert np.all(np.asarray(evaluation.log_strengths) >= bounds[:, 0])
        assert np.all(np.asarray(evaluation.log_strengths) <= bounds[:, 1])
        assert evaluation.strengths.lambda_amplitude == 0.4


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_callable_preserves_problem_precision_and_does_not_change_process_defaults(
    dtype,
):
    from TinyLensGpu.Inference import PotentialCorrectionEvidence

    problem, _ = _problem(dtype=dtype)
    logs = np.log([1.234567890123, 0.034567890123])
    with jax.experimental.enable_x64(dtype != "float64"):
        process_x64 = jax.config.x64_enabled
        objective = PotentialCorrectionEvidence(problem)
        objective(logs)
        result = objective.last_reconstruction
        assert result.success
        assert result.source_values.dtype == np.dtype(dtype)
        assert result.potential_values.dtype == np.dtype(dtype)
        assert result.log_evidence.dtype == np.dtype(dtype)
        assert result.dtype == dtype
        assert result.strengths.lambda_source == np.exp(logs[0])
        assert result.strengths.lambda_potential == np.exp(logs[1])
        np.testing.assert_array_equal(objective.evaluations[0].log_strengths, logs)
        assert jax.config.x64_enabled == process_x64


def test_failed_numerical_candidates_are_infinite_and_retain_distinct_reasons():
    from TinyLensGpu.Inference import PotentialCorrectionEvidence

    problem, _ = _problem(solve_rtol=1e-30)
    objective = PotentialCorrectionEvidence(problem)
    assert objective(np.log([1.3, 0.03])) == np.inf
    assert not objective.last_reconstruction.success
    assert (
        objective.evaluations[-1].failure_reason == "linear_residual_exceeds_tolerance"
    )
    assert objective.evaluations[-1].log_evidence is None

    for point in ([1000.0, 0.0], [-1000.0, 0.0]):
        assert objective(point) == np.inf
        assert objective.last_reconstruction is None
        assert objective.evaluations[-1].failure_reason == "strength_out_of_range"
        assert not objective.evaluations[-1].success
    assert objective.best_reconstruction is None
    assert objective.best_evaluation is None


def test_configuration_errors_propagate_instead_of_becoming_numerical_rejections():
    from TinyLensGpu.Inference import PotentialCorrectionEvidence

    problem, _ = _problem()
    objective = PotentialCorrectionEvidence(problem, backend="unsupported")
    with pytest.raises(ValueError, match="backend"):
        objective([0.0, 0.0])
    assert objective.evaluations == ()
    for point in ([np.nan, 0.0], [np.inf, 0.0], [0.0], [[0.0, 0.0]]):
        with pytest.raises(ValueError, match="log_strengths"):
            objective(point)


def test_direct_reports_log_range_boundary_with_explicit_proximity_tolerance():
    from TinyLensGpu.Inference import optimize_potential_correction

    problem, _ = _problem()
    bounds = np.log([[0.001, 0.01], [0.01, 1.0]])
    reference = _independent_negative_log_evidence(problem)
    # Independent evidence is still improving at the upper source boundary.
    assert reference((bounds[0, 1], 0.0)) < reference((bounds[0, 1] - 0.05, 0.0))
    result = optimize_potential_correction(
        problem,
        log_bounds=bounds,
        maxfun=200,
        len_tol=2e-4,
        boundary_rtol=0.01,
    )
    assert result.has_valid_fit
    assert result.at_boundary
    assert result.boundary_sides[0] == "upper"
    assert result.boundary_distances[0] <= 0.01
    assert result.configuration["boundary_rtol"] == 0.01


def test_direct_propagates_fit_configuration_error_with_original_type_and_reason():
    from TinyLensGpu.Inference import optimize_potential_correction

    problem, _ = _problem()
    with pytest.raises(ValueError, match="backend"):
        optimize_potential_correction(
            problem,
            log_bounds=[[-1.0, 1.0], [-2.0, 0.0]],
            backend="unsupported",
            maxfun=10,
        )


def test_search_configuration_rejects_invalid_bounds_controls_and_prior_overrides():
    from TinyLensGpu.Inference import (
        PotentialCorrectionEvidence,
        optimize_potential_correction,
    )

    problem, _ = _problem()
    bounds = [[-1.0, 1.0], [-2.0, 0.0]]
    for invalid in (
        [[0.0, 0.0], [-2.0, 0.0]],
        [[0.0, 1.0]],
        [[0.0, np.nan], [-2.0, 0.0]],
        [[0.0, np.inf], [-2.0, 0.0]],
    ):
        with pytest.raises(ValueError, match="log_bounds"):
            optimize_potential_correction(problem, log_bounds=invalid)
    for name, value in (
        ("maxfun", True),
        ("maxfun", 0),
        ("maxiter", 1.5),
        ("eps", np.nan),
        ("locally_biased", 1),
        ("vol_tol", 1.1),
        ("len_tol", -0.2),
        ("boundary_rtol", -0.1),
        ("boundary_rtol", np.nan),
        ("boundary_rtol", 0.5),
    ):
        with pytest.raises(ValueError, match=name):
            optimize_potential_correction(problem, log_bounds=bounds, **{name: value})
    with pytest.raises(ValueError, match="fixed"):
        PotentialCorrectionEvidence(problem, fit_options={"lambda_amplitude": 1.0})
    with pytest.raises(ValueError, match="representable"):
        optimize_potential_correction(problem, log_bounds=[[700.0, 800.0], [-2.0, 0.0]])


def test_all_failed_candidates_leave_no_selected_reconstruction_or_strengths():
    from TinyLensGpu.Inference import optimize_potential_correction

    problem, _ = _problem(solve_rtol=1e-30)
    result = optimize_potential_correction(
        problem,
        log_bounds=np.log([[0.1, 2.0], [0.02, 1.0]]),
        maxfun=10,
    )
    assert not result.success
    assert not result.has_valid_fit
    assert result.reconstruction is None
    assert result.strengths is None
    assert result.log_strengths is None
    assert result.objective == np.inf
    assert not result.at_boundary
    assert result.boundary_distances is None
    assert result.nfev == len(result.evaluations) > 0
    assert all(not record.success for record in result.evaluations)
    assert all(
        record.failure_reason == "linear_residual_exceeds_tolerance"
        for record in result.evaluations
    )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_direct_preserves_log_coordinates_and_fit_dtype(dtype):
    from TinyLensGpu.Inference import optimize_potential_correction

    problem, _ = _problem(dtype=dtype)
    bounds = np.array([[-0.94567890123, 1.123456789], [-3.2345678901, -1.2345678901]])
    with jax.experimental.enable_x64(dtype != "float64"):
        before = jax.config.x64_enabled
        result = optimize_potential_correction(problem, log_bounds=bounds, maxfun=5)
        assert result.has_valid_fit
        assert result.reconstruction.source_values.dtype == np.dtype(dtype)
        assert result.reconstruction.log_evidence.dtype == np.dtype(dtype)
        assert result.configuration["dtype"] == dtype
        np.testing.assert_allclose(
            result.evaluations[0].log_strengths, bounds.mean(axis=1), atol=2e-16, rtol=0
        )
        assert result.evaluations[0].strengths.lambda_source == np.exp(bounds[0].mean())
        assert jax.config.x64_enabled == before
