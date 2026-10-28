"""Cold-started nested nonlinear evidence through the public inference boundary."""

import numpy as np
import pytest
import jax

from TinyLensGpu.Inference import PotentialCorrectionEvidence
from tests.test_potential_correction_iterative import (
    _coupled_problem,
    _independent_problem,
)


@pytest.mark.parametrize("backend", ["dense", "operator"])
def test_iterative_candidates_retain_algorithm_curvature_and_termination(backend):
    problem = _coupled_problem()
    objective = PotentialCorrectionEvidence(
        problem, backend=backend, fit_options={"block_size": 7, "pcg_rtol": 1e-12}
    )
    value = objective(np.log([0.3, 0.02]))
    record = objective.evaluations[-1]
    result = objective.last_reconstruction
    assert result.success, result.termination_reason
    assert np.isfinite(value) and value == -float(result.log_evidence)
    assert record.algorithm == "iterative"
    assert record.termination_reason == result.termination_reason
    assert record.evidence_method == "local_gauss_newton"
    assert record.determinant_method == (
        "full_cholesky" if backend == "dense" else "principal_block_cholesky"
    )
    assert record.evidence_is_exact is False
    assert record.solver_reason == result.solver_info.reason
    assert objective.best_reconstruction is result


@pytest.mark.parametrize("backend", ["dense", "operator"])
def test_nested_candidates_reset_fields_damping_and_fixed_model_in_any_order(backend):
    problem = _coupled_problem()
    options = {"initial_damping": 0.01, "block_size": 7, "pcg_rtol": 1e-12}
    objective = PotentialCorrectionEvidence(
        problem, backend=backend, fit_options=options
    )
    # Repeat A around B, then reverse the order. Fresh problems are independent
    # cold-start controls; final fields alone would not detect damping leakage.
    candidates = ((0.3, 0.02), (0.7, 0.1))
    initial = np.asarray(problem.initial_values).copy()
    geometry = np.asarray(problem.renderer.potential_grid.node_indices).copy()
    references = {}
    for pair in candidates:
        fresh = _coupled_problem().fit(*pair, backend=backend, **options)
        assert fresh.success, fresh.termination_reason
        references[pair] = fresh
    for pair in (
        candidates[0],
        candidates[1],
        candidates[0],
        candidates[1],
        candidates[0],
    ):
        objective(np.log(pair))
        result = objective.last_reconstruction
        expected = references[pair]
        assert result.success, result.termination_reason
        forward, residual, prior = _independent_problem(
            problem.renderer, problem.image_data, problem.noise_map, *pair, 20.0
        )
        initial_objective = 0.5 * np.sum(residual(initial) ** 2)
        np.testing.assert_allclose(
            result.history[0]["objective_before"], initial_objective, atol=2e-13
        )
        assert result.history[0]["damping"] == options["initial_damping"]
        np.testing.assert_array_equal(initial[4:], 0.0)
        fitted = np.r_[result.source_values, result.potential_values]
        for name in (
            "source_values",
            "potential_values",
            "nonlinear_image_active",
            "objective",
            "log_evidence",
        ):
            np.testing.assert_allclose(
                getattr(result, name), getattr(expected, name), atol=2e-8, rtol=2e-7
            )
        np.testing.assert_allclose(
            result.regularization_penalty, fitted @ prior @ fitted, atol=2e-12
        )
        np.testing.assert_allclose(
            result.nonlinear_image_active, forward(fitted), atol=2e-13
        )
        assert result.strengths.lambda_source == np.exp(np.log(pair[0]))
        assert result.strengths.lambda_potential == np.exp(np.log(pair[1]))
        assert result.strengths.lambda_amplitude == 20.0
        np.testing.assert_array_equal(result.potential_grid.node_indices, geometry)
        np.testing.assert_array_equal(result.data_active, problem.data_active)
        np.testing.assert_array_equal(result.noise_active, problem.noise_active)
        np.testing.assert_array_equal(
            result.source_x_axis, problem.renderer.source_x_axis
        )
        assert result.source_bbox == problem.renderer.source_bbox
        np.testing.assert_array_equal(problem.initial_values, initial)


@pytest.mark.parametrize(
    "backend,options,point,termination,solver_reason",
    [
        ("dense", {"max_iter": 0}, (0.3, 0.02), "iteration_limit", None),
        (
            "operator",
            {"pcg_max_iter": 1},
            (0.3, 0.02),
            "linear_solve_failed",
            "pcg_iteration_limit",
        ),
        (
            "dense",
            {
                "initial_damping": 1e8,
                "objective_rtol": 1e-6,
                "step_rtol": 1e-6,
                "stall_max_iter": 1,
            },
            (0.3, 0.02),
            "stalled",
            None,
        ),
        (
            "operator",
            {"initial_damping": 1e30, "damping_max": 1e35, "max_rejections": 2},
            (0.3, 0.02),
            "rejection_limit",
            None,
        ),
        ("dense", {}, (1e308, 0.02), "nonfinite_current_state", None),
    ],
)
def test_unsuccessful_inner_candidates_never_supply_evidence_or_best_fields(
    backend, options, point, termination, solver_reason
):
    objective = PotentialCorrectionEvidence(
        _coupled_problem(), backend=backend, fit_options=options
    )
    assert objective(np.log(point)) == np.inf
    record = objective.evaluations[-1]
    result = objective.last_reconstruction
    assert not record.success and record.log_evidence is None
    assert record.termination_reason == record.failure_reason == termination
    assert record.solver_reason == solver_reason
    assert result.log_evidence is None
    assert objective.best_evaluation is objective.best_reconstruction is None
    if termination == "stalled":
        assert result.solver_info.converged
        assert result.history[-1]["damping_fraction"] > 0.99


def test_unrepresentable_candidate_has_no_invented_algorithm_or_curvature_method():
    objective = PotentialCorrectionEvidence(_coupled_problem())
    assert objective([1000.0, 0.0]) == np.inf
    record = objective.evaluations[-1]
    assert record.termination_reason == "strength_out_of_range"
    assert (
        record.algorithm is record.evidence_method is record.determinant_method is None
    )
    assert record.evidence_is_exact is record.solver_reason is None
    assert objective.last_reconstruction is objective.best_reconstruction is None


@pytest.mark.parametrize("backend", ["dense", "operator"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_nested_iterative_evaluation_preserves_explicit_precision_and_defaults(
    backend, dtype
):
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection
    from tests.test_potential_correction_iterative import _renderer

    renderer = _renderer(dtype=dtype, source_n=2, bbox=(-0.5, 0.5, -0.5, 0.5))
    problem = IterativePotentialCorrection(
        renderer,
        np.full((4, 4), -2.0),
        np.ones((4, 4)),
        np.full(4, 5.0),
        lambda_amplitude=1e8,
        source_regularization="zero-order",
    )
    with jax.experimental.enable_x64(dtype != "float64"):
        before = jax.config.x64_enabled
        matmul = jax.config.jax_default_matmul_precision
        objective = PotentialCorrectionEvidence(
            problem, backend=backend, fit_options={"initial_damping": 1.0}
        )
        logs = np.log([3.0, 0.4])
        assert np.isfinite(objective(logs))
        result = objective.last_reconstruction
        assert result.success, result.termination_reason
        for values in (
            result.source_values,
            result.potential_values,
            result.log_evidence,
            result.prior_logdet,
            result.posterior_logdet,
        ):
            assert values.dtype == np.dtype(dtype)
        np.testing.assert_allclose(
            result.source_values, -8 / 7, atol=2e-7 if dtype == "float64" else 2e-4
        )
        np.testing.assert_array_equal(objective.evaluations[-1].log_strengths, logs)
        assert objective.evaluations[-1].dtype == dtype
        assert jax.config.x64_enabled == before
        assert jax.config.jax_default_matmul_precision == matmul


@pytest.mark.parametrize("backend", ["dense", "operator"])
def test_all_iteration_limited_direct_candidates_leave_no_selected_success(backend):
    from TinyLensGpu.Inference import optimize_potential_correction

    result = optimize_potential_correction(
        _coupled_problem(),
        log_bounds=np.log([[0.1, 2.0], [0.001, 1.0]]),
        backend=backend,
        fit_options={"max_iter": 0},
        maxfun=8,
        maxiter=10,
    )
    assert not result.success and not result.has_valid_fit
    assert result.reconstruction is result.strengths is result.log_strengths is None
    assert result.objective == np.inf
    assert result.boundary_distances is None and not result.at_boundary
    assert result.nfev == len(result.evaluations) > 0
    assert result.configuration["maxfun"] == 8
    assert result.configuration["lambda_amplitude"] == 20.0
    assert all(
        row.termination_reason == "iteration_limit" for row in result.evaluations
    )
    assert all(
        row.log_evidence is None and not row.success for row in result.evaluations
    )


def test_interleaved_backend_candidates_share_no_updated_fields_or_damping():
    problem = _coupled_problem()
    dense = PotentialCorrectionEvidence(problem, fit_options={"initial_damping": 0.01})
    operator = PotentialCorrectionEvidence(
        problem,
        backend="operator",
        fit_options={"initial_damping": 10.0, "block_size": 7, "pcg_rtol": 1e-12},
    )
    dense(np.log([0.3, 0.02]))
    first = dense.last_reconstruction
    assert np.isfinite(operator(np.log([0.7, 0.1])))
    dense(np.log([0.3, 0.02]))
    repeat = dense.last_reconstruction
    for name in (
        "source_values",
        "potential_values",
        "nonlinear_image_active",
        "objective",
        "log_evidence",
    ):
        np.testing.assert_allclose(
            getattr(first, name), getattr(repeat, name), atol=2e-8, rtol=2e-7
        )
    assert repeat.history[0]["damping"] == first.history[0]["damping"] == 0.01
    assert operator.last_reconstruction.history[0]["damping"] == 10.0
    assert repeat.history[0]["objective_before"] == first.history[0]["objective_before"]


def _run_nested_comparison():
    """Also used by the retained reproduction script to record actual numbers."""
    from dataclasses import asdict
    from scipy.optimize import minimize_scalar
    from TinyLensGpu.Inference import optimize_potential_correction
    from tests.potential_correction_nested_reference import (
        SEARCH_CONTROLS,
        STRENGTH_BOUNDS,
        evaluate_reference,
        fit_options,
        independent_fit,
        mixed_partition,
        recovery_metrics,
    )

    problem = _coupled_problem()
    bounds = np.log(STRENGTH_BOUNDS)
    blocks = mixed_partition()
    assert len(blocks[0]) == 20 and len(blocks[1]) == 17
    assert sum(i < 4 for i in blocks[0]) == 4
    assert all(0 < sum(i >= 4 for i in block) < 33 for block in blocks)
    options = fit_options()
    axes = [np.geomspace(*interval, 5) for interval in STRENGTH_BOUNDS]
    scan = []
    max_field_error = max_formula_error = max_map_evidence_error = 0.0

    def certified_reference(ls, lp):
        reference = independent_fit(problem, ls, lp)
        assert reference["least_squares_success"]
        assert reference["scaled_gradient"] < 1e-8
        assert reference["derivative_step_error"] < 1e-8
        return reference

    def check_fit(result, ls, lp, reference):
        nonlocal max_field_error, max_formula_error, max_map_evidence_error
        assert result.success, result.termination_reason
        fitted = np.r_[result.source_values, result.potential_values]
        np.testing.assert_allclose(fitted, reference["values"], atol=4e-7, rtol=3e-6)
        max_field_error = max(
            max_field_error, float(np.max(np.abs(fitted - reference["values"])))
        )
        actual = evaluate_reference(problem, ls, lp, fitted)
        np.testing.assert_allclose(result.objective, actual["objective"], atol=2e-12)
        np.testing.assert_allclose(
            result.regularization_penalty, actual["penalty"], atol=2e-12
        )
        key = "full_log_evidence" if result.backend == "dense" else "block_log_evidence"
        # This strict formula check is evaluated at the actual accepted fields.
        np.testing.assert_allclose(result.log_evidence, actual[key], atol=3e-8, rtol=0)
        # Independent MAP fields stop via different numerical criteria; retain
        # the already-declared iterative damping-schedule evidence tolerance.
        np.testing.assert_allclose(
            result.log_evidence, reference[key], atol=2e-6, rtol=0
        )
        max_formula_error = max(
            max_formula_error, abs(float(result.log_evidence) - actual[key])
        )
        max_map_evidence_error = max(
            max_map_evidence_error, abs(float(result.log_evidence) - reference[key])
        )
        assert result.strengths.lambda_amplitude == 20.0
        assert result.evidence_method == "local_gauss_newton"
        assert not result.evidence_is_exact
        assert result.linear_model_image is None
        return fitted

    for ls in axes[0]:
        for lp in axes[1]:
            reference = certified_reference(ls, lp)
            fits = {
                backend: problem.fit(ls, lp, backend=backend, **options)
                for backend in ("dense", "operator")
            }
            row = dict(
                lambda_source=float(ls),
                lambda_potential=float(lp),
                full_log_evidence=float(reference["full_log_evidence"]),
                block_log_evidence=float(reference["block_log_evidence"]),
                backends={},
            )
            for backend, result in fits.items():
                row["backends"][backend] = dict(
                    success=result.success, termination_reason=result.termination_reason
                )
                if result.success:
                    check_fit(result, ls, lp, reference)
                    row["backends"][backend]["log_evidence"] = float(
                        result.log_evidence
                    )
                else:
                    assert result.log_evidence is None
            if all(result.success for result in fits.values()):
                for field in (
                    "source_values",
                    "potential_values",
                    "nonlinear_image_active",
                    "objective",
                ):
                    np.testing.assert_allclose(
                        getattr(fits["dense"], field),
                        getattr(fits["operator"], field),
                        atol=2e-8,
                        rtol=2e-7,
                    )
            scan.append(row)

    searches = {}
    records = {}
    for backend, key in (
        ("dense", "full_log_evidence"),
        ("operator", "block_log_evidence"),
    ):
        # The independent scan is increasing in lp at every sampled ls. Refine
        # its upper edge independently; no library fit or DIRECT candidate is
        # supplied to this reference optimization.
        surface = np.asarray([row[key] for row in scan]).reshape(5, 5)
        assert np.all(np.diff(surface, axis=1) > 0)
        oracle = minimize_scalar(
            lambda log_ls: -certified_reference(np.exp(log_ls), 1.0)[key],
            bounds=bounds[0],
            method="bounded",
            options={"xatol": 1e-5},
        )
        assert oracle.success
        search = optimize_potential_correction(
            problem,
            log_bounds=bounds,
            backend=backend,
            fit_options=options,
            **SEARCH_CONTROLS
        )
        assert search.has_valid_fit
        assert search.nfev == len(search.evaluations)
        assert search.configuration["maxfun"] == 100
        assert search.configuration["lambda_amplitude"] == 20.0
        assert search.configuration["dtype"] == "float64"
        assert search.configuration["fit_options"] == options
        assert search.success == search.optimizer_success
        assert search.boundary_sides[1] == "upper"
        assert search.boundary_distances[1] <= search.configuration["boundary_rtol"]
        ls, lp = search.strengths.lambda_source, search.strengths.lambda_potential
        reference = certified_reference(ls, lp)
        fitted = check_fit(search.reconstruction, ls, lp, reference)
        # A 0.005 log-evidence gap is the declared budget/resolution guard,
        # separate from the strict fixed-strength numerical checks above.
        assert reference[key] >= np.max(surface) - 0.005
        assert -reference[key] <= float(oracle.fun) + 0.005
        assert all(row.strengths.lambda_amplitude == 20.0 for row in search.evaluations)
        assert all(
            row.algorithm == "iterative" and row.evidence_method == "local_gauss_newton"
            for row in search.evaluations
        )
        assert all(
            row.log_evidence is None for row in search.evaluations if not row.success
        )
        records[backend] = dict(
            strengths=asdict(search.strengths),
            log_evidence=float(search.reconstruction.log_evidence),
            full_log_evidence=float(reference["full_log_evidence"]),
            block_log_evidence=float(reference["block_log_evidence"]),
            recovery=recovery_metrics(problem, fitted),
            optimizer_success=search.optimizer_success,
            optimizer_status=search.optimizer_status,
            message=search.message,
            nfev=search.nfev,
            nit=search.nit,
            boundary_sides=search.boundary_sides,
            boundary_distances=search.boundary_distances,
            termination_reason=search.reconstruction.termination_reason,
            independent_scan_best=float(np.max(surface)),
            independent_edge_strengths=(float(np.exp(oracle.x)), 1.0),
            independent_edge_log_evidence=float(-oracle.fun),
            independent_edge_evidence_gap=float(reference[key] + oracle.fun),
            candidate_terminations={
                reason: sum(
                    row.termination_reason == reason for row in search.evaluations
                )
                for reason in sorted(
                    {row.termination_reason for row in search.evaluations}
                )
            },
        )
        searches[backend] = search

    dense, operator = records["dense"], records["operator"]
    difference = dense["full_log_evidence"] - operator["full_log_evidence"]
    regret = dense["independent_edge_log_evidence"] - operator["full_log_evidence"]
    assert 0 < regret < 0.005
    assert abs(difference) < 0.005
    # These are recovery-effect guards at the different selected strengths,
    # not a claim that the two evidence surfaces or optima are interchangeable.
    for name in (
        "source_rmse",
        "potential_rmse",
        "deflection_rmse",
        "convergence_rmse",
    ):
        assert operator["recovery"][name] <= 1.1 * dense["recovery"][name]
    image_difference = float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(searches["dense"].reconstruction.nonlinear_image_active)
                    - np.asarray(
                        searches["operator"].reconstruction.nonlinear_image_active
                    )
                )
                ** 2
            )
        )
    )
    assert image_difference < 0.1 * 0.07
    return dict(
        strength_bounds=STRENGTH_BOUNDS,
        log_bounds=bounds.tolist(),
        fixed_lambda_amplitude=20.0,
        fit_options=options,
        search_controls=SEARCH_CONTROLS,
        scan_shape=(5, 5),
        scan=scan,
        selected=records,
        full_evidence_regret_of_operator_selection=regret,
        dense_minus_operator_selection_full_log_evidence=difference,
        selected_image_rms_difference=image_difference,
        maximum_field_error=max_field_error,
        maximum_fixed_state_evidence_formula_error=max_formula_error,
        maximum_independent_map_evidence_error=max_map_evidence_error,
    )


@pytest.mark.integration
def test_iterative_direct_matches_independent_scan_and_bounds_recovery_effect():
    _run_nested_comparison()
