"""Current-source derivatives and public fixed-strength nonlinear reconstructions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _renderer(
    *,
    dtype="float64",
    npix=4,
    source_n=3,
    bbox=(-0.5, 0.7, -0.5, 0.7),
    nsub=1,
    source_interpolation="bilinear",
):
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    with jax.experimental.enable_x64(dtype == "float64"):
        config = SimulatorConfig(0.1, npix, jnp.ones((1, 1)), nsub=nsub)
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.0, gamma2=0.0)], source_light=[], lens_light=[]
        )
        return PotentialCorrectionRenderer(
            macro, config, source_n, bbox, dtype=dtype,
            source_interpolation=source_interpolation,
        )


def test_nonaffine_source_gradient_differentiates_current_bilinear_branch():
    from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import (
        MacroMapping,
    )

    renderer = _renderer(npix=3, bbox=(0.0, 2.0, 0.0, 2.0))
    with jax.experimental.enable_x64():
        source = jnp.array([0.0, 2.0, 5.0, -1.0, 4.0, 3.0, 7.0, -2.0, 6.0])
        points = jnp.array(
            [
                [0.3, 0.6],
                [1.0, 0.4],
                [0.0, 0.4],
                [2.0, 0.4],
                [0.4, 2.0],
                [-0.1, 1.0],
                [2.1, 1.0],
                [2.0, 2.0],
                [0.4, 0.0],
            ]
        )

        def mapping(beta):
            return renderer.source_mapping(
                macro_mapping=MacroMapping(beta[:, 0], beta[:, 1])
            )

        gx, gy = renderer.source_spatial_gradient(source, mapping(points))
        np.testing.assert_allclose(gx, [3.8, 1.4, 3.2, 0.0, -9.0, 0.0, 0.0, 0.0, 2.0])
        np.testing.assert_allclose(gy, [-0.1, 2.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.2])
        h = 1e-6
        for axis, expected in ((0, gx[0]), (1, gy[0])):
            step = jnp.zeros_like(points).at[0, axis].set(h)
            plus = renderer.source_forward(source, mapping(points + step))[0]
            minus = renderer.source_forward(source, mapping(points - step))[0]
            np.testing.assert_allclose((plus - minus) / (2 * h), expected, atol=2e-9)


def test_iterative_accumulated_source_prior_converges_to_ridge_solution():
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection

    renderer = _renderer(source_n=2, bbox=(-0.5, 0.5, -0.5, 0.5))
    problem = IterativePotentialCorrection(
        renderer,
        np.full((4, 4), 2.0),
        np.ones((4, 4)),
        np.full(4, 5.0),
        lambda_amplitude=1e8,
        source_regularization="zero-order",
        initialization_provenance={"source": "constant truth-node intensities"},
    )
    result = problem.fit(3.0, 0.4, initial_damping=1.0)
    assert result.success, result.termination_reason
    np.testing.assert_allclose(result.source_values, 8 / 7, atol=2e-7)
    np.testing.assert_allclose(result.potential_values, 0.0, atol=2e-12)
    assert result.algorithm == "iterative" and result.backend == "dense"
    assert result.linear_model_image is result.linear_residual is None
    assert result.evidence_method == "local_gauss_newton"
    assert not result.evidence_is_exact
    accepted = [item for item in result.history if item["accepted"]]
    assert len(accepted) >= 3
    assert all(item["objective_after"] < item["objective_before"] for item in accepted)
    np.testing.assert_allclose(
        result.objective,
        0.5 * (np.sum((2 - 8 / 7) ** 2 * np.ones(16)) + 3 * 4 * (8 / 7) ** 2),
        atol=2e-10,
    )


def _independent_problem(renderer, data, noise, ls, lp, la):
    """Coordinate/stencil host reference; no reconstruction actions or Jacobians."""
    grid = renderer.potential_grid
    nodes = np.asarray(grid.node_indices)
    lookup = {tuple(node): i for i, node in enumerate(nodes)}
    x, y = np.asarray(renderer.query_x), np.asarray(renderer.query_y)
    dx = np.zeros((len(x), len(nodes)))
    dy = np.zeros_like(dx)
    for row, (qx, qy) in enumerate(zip(x, y)):
        u, v = (qx - grid.origin[0]) / grid.spacing, (
            qy - grid.origin[1]
        ) / grid.spacing
        i, j = int(np.floor(u)), int(np.floor(v))
        a, b = u - i, v - j
        for cx, cy, weight in (
            (i, j, (1 - a) * (1 - b)),
            (i + 1, j, a * (1 - b)),
            (i, j + 1, (1 - a) * b),
            (i + 1, j + 1, a * b),
        ):
            dx[row, lookup[(cx + 1, cy)]] += weight / (2 * grid.spacing)
            dx[row, lookup[(cx - 1, cy)]] -= weight / (2 * grid.spacing)
            dy[row, lookup[(cx, cy + 1)]] += weight / (2 * grid.spacing)
            dy[row, lookup[(cx, cy - 1)]] -= weight / (2 * grid.spacing)
    windows = []
    for cx, cy in nodes:
        for sx, sy in ((1, 0), (0, 1)):
            run = [(cx + k * sx, cy + k * sy) for k in range(-2, 3)]
            if all(node in lookup for node in run):
                vector = np.zeros(len(nodes))
                vector[[lookup[node] for node in run]] = [1, -4, 6, -4, 1]
                windows.append(vector)
    fourth = np.asarray(windows).reshape(-1, len(nodes))
    n = renderer.source_n
    xmin, xmax, ymin, ymax = map(float, renderer.source_bbox)

    def forward(values):
        source, potential = values[: n * n], values[n * n :]
        u = (x - dx @ potential - xmin) / (xmax - xmin) * (n - 1)
        v = (y - dy @ potential - ymin) / (ymax - ymin) * (n - 1)
        i, j = np.floor(u).astype(int), np.floor(v).astype(int)
        a, b = u - i, v - j
        i0, j0 = np.clip(i, 0, n - 1), np.clip(j, 0, n - 1)
        i1, j1 = np.clip(i0 + 1, 0, n - 1), np.clip(j0 + 1, 0, n - 1)
        brightness = (
            (1 - a) * (1 - b) * source[j0 * n + i0]
            + a * (1 - b) * source[j0 * n + i1]
            + (1 - a) * b * source[j1 * n + i0]
            + a * b * source[j1 * n + i1]
        )
        return np.where(
            (u >= 0) & (u <= n - 1) & (v >= 0) & (v <= n - 1), brightness, 0.0
        )

    prior = np.zeros((n * n + len(nodes),) * 2)
    prior[: n * n, : n * n] = ls * np.eye(n * n)
    prior[n * n :, n * n :] = lp * fourth.T @ fourth + la * np.eye(len(nodes))

    def residual(values):
        return np.r_[
            (np.asarray(data).ravel() - forward(values)) / np.asarray(noise).ravel(),
            np.sqrt(ls) * values[: n * n],
            np.sqrt(lp) * fourth @ values[n * n :],
            np.sqrt(la) * values[n * n :],
        ]

    return forward, residual, prior


def _coupled_problem(
    *, dtype="float64", solve_rtol=None, source_interpolation="bilinear"
):
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection

    renderer = _renderer(
        dtype=dtype,
        source_n=2,
        bbox=(-0.5, 0.5, -0.5, 0.5),
        source_interpolation=source_interpolation,
    )
    source = np.array([0.4, 1.1, 0.2, 1.4])
    with jax.experimental.enable_x64(dtype == "float64"):
        grid = renderer.potential_grid
        potential = 0.003 * jnp.exp(
            -((grid.x - 0.07) ** 2 + (grid.y + 0.03) ** 2) / 0.02
        )
        image = np.asarray(renderer.render(source, potential).model_image)
    image = image + 0.002 * np.sin(np.arange(16)).reshape(4, 4)
    noise = np.full((4, 4), 0.07)
    return IterativePotentialCorrection(
        renderer,
        image,
        noise,
        source,
        lambda_amplitude=20.0,
        source_regularization="zero-order",
        solve_rtol=solve_rtol,
    )


def test_dense_iteration_matches_independent_nonlinear_objective_and_local_evidence():
    from scipy.optimize import least_squares

    problem = _coupled_problem()
    ls, lp = 0.3, 0.02
    forward, residual, prior = _independent_problem(
        problem.renderer, problem.image_data, problem.noise_map, ls, lp, 20.0
    )
    initial = np.asarray(problem.initial_values)
    reference = least_squares(
        residual,
        initial,
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-11,
        diff_step=2e-6,
        max_nfev=300,
    )
    result = problem(ls, lp)
    assert result.success, result.termination_reason
    fitted = np.r_[result.source_values, result.potential_values]
    np.testing.assert_allclose(fitted, reference.x, atol=4e-7, rtol=3e-6)
    np.testing.assert_allclose(
        result.nonlinear_image_active, forward(fitted), atol=2e-13
    )
    np.testing.assert_allclose(
        result.objective, 0.5 * np.sum(residual(fitted) ** 2), atol=2e-12
    )
    assert np.linalg.norm(fitted[4:]) > 1e-4
    assert np.linalg.norm(fitted[:4] - initial[:4]) > 1e-3
    h = 1e-6
    jacobian = np.column_stack(
        [
            (forward(fitted + h * unit) - forward(fitted - h * unit)) / (2 * h)
            for unit in np.eye(len(fitted))
        ]
    )
    curvature = (
        jacobian.T @ (jacobian / np.asarray(problem.noise_active)[:, None] ** 2) + prior
    )
    expected = -0.5 * (
        2 * float(result.objective)
        + np.linalg.slogdet(curvature)[1]
        - np.linalg.slogdet(prior)[1]
        + np.sum(np.log(np.asarray(problem.noise_active) ** 2))
        + 16 * np.log(2 * np.pi)
    )
    np.testing.assert_allclose(result.log_evidence, expected, atol=3e-8)
    np.testing.assert_allclose(
        result.posterior_logdet, np.linalg.slogdet(curvature)[1], atol=6e-8
    )
    accepted = [item for item in result.history if item["accepted"]]
    assert all(item["objective_after"] < item["objective_before"] for item in accepted)


@pytest.mark.parametrize(
    "interpolation, x_shift", [("bilinear", 0.073), ("cubic_hermite", 0.453)]
)
def test_current_coupled_jacobian_refreshes_source_and_shifted_cells_inside_jit(
    interpolation, x_shift
):
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection

    renderer = _renderer(
        bbox=(-0.2, 0.2, -0.2, 0.2), nsub=3, source_interpolation=interpolation
    )
    source = np.array([0.0, 2.0, 5.0, -1.0, 4.0, 3.0, 7.0, -2.0, 6.0])
    problem = IterativePotentialCorrection(
        renderer, np.zeros((4, 4)), np.ones((4, 4)), source, lambda_amplitude=1.0
    )
    with jax.experimental.enable_x64():
        first = problem.initial_values
        changed = first.at[:9].set(
            jnp.array([0.2, 1.4, 0.7, -0.5, 0.4, 2.0, 0.1, -0.1, 1.1])
        )
        changed = changed.at[9:].set(
            x_shift * renderer.potential_grid.x + 0.021 * renderer.potential_grid.y
        )
        initial_linearization = problem.linearize(first)
        current_linearization = problem.linearize(changed)
        direction = 0.01 * jnp.sin(jnp.arange(problem.n_joint))
        apply = jax.jit(problem.jacobian_forward)
        initial_action = apply(direction, initial_linearization)
        current_action = apply(direction, current_linearization)
        h = 1e-6
        difference = (
            problem.forward(changed + h * direction)
            - problem.forward(changed - h * direction)
        ) / (2 * h)
        np.testing.assert_allclose(current_action, difference, atol=2e-9, rtol=2e-7)
        _, tangent = jax.jvp(problem.forward, (changed,), (direction,))
        np.testing.assert_allclose(current_action, tangent, atol=2e-13, rtol=2e-13)
        image = jnp.cos(jnp.arange(renderer.n_active))
        transpose = jax.jit(problem.jacobian_adjoint)(image, current_linearization)
        np.testing.assert_allclose(
            current_action @ image, direction @ transpose, atol=2e-13
        )
        assert np.linalg.norm(np.asarray(initial_action - current_action)) > 0.01
        assert not np.array_equal(
            initial_linearization.mapping.valid, current_linearization.mapping.valid
        )


def test_fit_resets_initialization_and_final_evidence_is_independent_of_step_damping():
    problem = _coupled_problem()
    first = problem.fit(0.3, 0.02, initial_damping=0.01)
    changed = problem.fit(0.7, 0.1)
    repeat = problem.fit(0.3, 0.02, initial_damping=0.01)
    other_damping = problem.fit(0.3, 0.02, initial_damping=10.0)
    for result in (first, changed, repeat, other_damping):
        assert result.success, result.termination_reason
    np.testing.assert_allclose(first.source_values, repeat.source_values, atol=2e-11)
    np.testing.assert_allclose(
        first.potential_values, repeat.potential_values, atol=2e-11
    )
    np.testing.assert_allclose(
        first.source_values, other_damping.source_values, atol=2e-6
    )
    np.testing.assert_allclose(
        first.potential_values, other_damping.potential_values, atol=2e-6
    )
    np.testing.assert_allclose(
        first.log_evidence, other_damping.log_evidence, atol=2e-6
    )
    np.testing.assert_allclose(
        first.history[0]["objective_before"], repeat.history[0]["objective_before"]
    )
    assert first.initialization_provenance["reset_each_fit"]


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"max_iter": 0}, "iteration_limit"),
        (
            {"initial_damping": 1e30, "damping_max": 1e35, "max_rejections": 2},
            "rejection_limit",
        ),
        (
            {
                "initial_damping": 1e8,
                "objective_rtol": 1e-6,
                "step_rtol": 1e-6,
                "stall_max_iter": 1,
            },
            "stalled",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["dense", "operator"])
def test_nonconvergence_and_damping_tiny_steps_never_produce_success(
    options, reason, backend
):
    result = _coupled_problem().fit(0.3, 0.02, backend=backend, **options)
    assert not result.success and result.status == "failed"
    assert result.termination_reason == result.failure_reason == reason
    assert result.log_evidence is None
    if reason == "stalled":
        assert result.solver_info.converged and not result.solver_info.failed
        assert result.history[-1]["damping_fraction"] > 0.99
    if options.get("initial_damping", 0) == 1e30:
        assert not any(item["accepted"] for item in result.history)


@pytest.mark.parametrize("backend", ["dense", "operator"])
def test_kink_minimum_with_nonzero_branch_gradient_reports_honest_rejection(backend):
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection

    renderer = _renderer(npix=1, bbox=(-1.0, 1.0, -1.0, 1.0))
    source = np.zeros(9)
    source[4] = 1.0
    problem = IterativePotentialCorrection(
        renderer,
        [[2.0]],
        [[1.0]],
        source,
        lambda_amplitude=1.0,
        source_regularization="zero-order",
    )
    result = problem.fit(1.0, 0.1, backend=backend, max_rejections=2)
    assert not result.success and result.termination_reason == "rejection_limit"
    assert result.log_evidence is None
    np.testing.assert_allclose(result.objective, 1.0, atol=2e-14)
    np.testing.assert_array_equal(result.source_values, source)
    np.testing.assert_array_equal(result.potential_values, 0.0)
    assert result.history[-1]["scaled_gradient"] > 0.1
    assert not any(item["accepted"] for item in result.history)


def test_valid_overflow_and_unsuccessful_dense_solve_have_distinct_failure_records():
    overflow = _coupled_problem().fit(1e308, 0.02)
    assert not overflow.success and overflow.log_evidence is None
    assert overflow.termination_reason == "nonfinite_current_state"
    inaccurate = _coupled_problem(solve_rtol=1e-30).fit(0.3, 0.02)
    assert not inaccurate.success and inaccurate.log_evidence is None
    assert inaccurate.termination_reason == "linear_solve_failed"
    assert inaccurate.solver_info.reason == "linear_residual_exceeds_tolerance"
    assert inaccurate.history[-1]["reason"] == inaccurate.solver_info.reason


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("backend", ["dense", "operator"])
def test_explicit_iteration_precision_preserves_defaults_and_signed_fields(
    dtype, backend
):
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection

    previous = jax.config.x64_enabled
    previous_matmul = jax.config.jax_default_matmul_precision
    renderer = _renderer(dtype=dtype, source_n=2, bbox=(-0.5, 0.5, -0.5, 0.5))
    problem = IterativePotentialCorrection(
        renderer,
        np.full((4, 4), -2.0),
        np.ones((4, 4)),
        np.full(4, 5.0),
        lambda_amplitude=1e8,
        source_regularization="zero-order",
    )
    result = problem.fit(3.0, 0.4, backend=backend, initial_damping=1.0)
    assert result.success, result.termination_reason
    assert jax.config.x64_enabled == previous
    assert jax.config.jax_default_matmul_precision == previous_matmul
    for values in (
        result.source_values,
        result.potential_values,
        result.nonlinear_model_image,
        result.log_evidence,
        result.prior_logdet,
        result.posterior_logdet,
    ):
        assert values.dtype == np.dtype(dtype)
    tolerance = 2e-7 if dtype == "float64" else 2e-4
    np.testing.assert_allclose(result.source_values, -8 / 7, atol=tolerance)
    assert np.max(result.source_values) < 0


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "unsupported"},
        {"max_iter": -1},
        {"max_rejections": True},
        {"damping_min": 1.0},
        {"damping_increase": 1.0},
        {"step_rtol": float("nan")},
        {"damping_fraction_max": 1.0},
    ],
)
def test_invalid_iteration_configuration_raises_instead_of_failed_fit(options):
    with pytest.raises(ValueError):
        _coupled_problem().fit(0.3, 0.02, **options)


@pytest.mark.parametrize("backend", ["dense", "operator"])
def test_nonfinite_trial_is_rejected_without_replacing_last_finite_fields(backend):
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection

    class ConstantDeflection:
        """Exact macro rays for a constant deflection, at extreme numeric scale."""

        def deflection(self, x, y):
            return x + 3e-160, y + 3e-160

    # Keep the prior in the normal float64 range on CPU. With unit noise,
    # this signal still overflows the candidate gradient after a valid solve.
    signal_amplitude = 1e30
    with jax.experimental.enable_x64():
        renderer = PotentialCorrectionRenderer(
            ConstantDeflection(),
            SimulatorConfig(1e-160, 2, jnp.ones((1, 1))),
            2,
            (0.0, 1.0, 0.0, 1.0),
        )
        problem = IterativePotentialCorrection(
            renderer,
            signal_amplitude * np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            np.ones((2, 2)),
            np.zeros(4),
            lambda_amplitude=1.0,
            source_regularization="zero-order",
            solve_atol=1e-10,
        )
        result = problem.fit(1e-250, 0.1, backend=backend, max_rejections=1)
    assert not result.success and result.log_evidence is None
    assert result.termination_reason == "nonfinite_candidate_limit"
    assert result.solver_info.converged
    assert len(result.history) > 1
    assert all(item["reason"] == "nonfinite_candidate" for item in result.history[1:])
    assert all(not item["accepted"] for item in result.history[1:])
    np.testing.assert_array_equal(result.source_values, 0.0)
    np.testing.assert_array_equal(result.potential_values, 0.0)
    np.testing.assert_allclose(result.objective, 2.0 * signal_amplitude**2)


def test_iteration_kernels_reuse_compilation_for_changed_strengths_and_damping(caplog):
    problem = _coupled_problem()
    with jax.log_compiles(True):
        assert problem.fit(0.3, 0.02).success
        caplog.clear()
        result = problem.fit(0.7, 0.1, initial_damping=0.1)
        assert result.success, result.termination_reason
    assert not [
        record.message
        for record in caplog.records
        if "Compiling jit(" in record.message
    ]
