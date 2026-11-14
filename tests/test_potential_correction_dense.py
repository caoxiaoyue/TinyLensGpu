"""Public fixed-strength source–potential fits and independent numerical checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.utils.lensing.potential import PotentialGrid


def _small_renderer(*, dtype="float64", nsub=1, source_interpolation="bilinear"):
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    with jax.experimental.enable_x64(dtype == "float64"):
        config = SimulatorConfig(0.1, 3, jnp.ones((1, 1)), nsub=nsub)
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.0, gamma2=0.0)], source_light=[], lens_light=[]
        )
        return PotentialCorrectionRenderer(
            macro,
            config,
            2,
            (-0.3, 0.3, -0.3, 0.3),
            dtype=dtype,
            source_interpolation=source_interpolation,
        )


def test_renderer_interpolation_is_validated_and_fixed():
    with pytest.raises(ValueError, match="source interpolation"):
        _small_renderer(source_interpolation="cubic")
    renderer = _small_renderer(source_interpolation="cubic_hermite")
    with pytest.raises(AttributeError):
        renderer.source_interpolation = "bilinear"


def _independent_design_and_prior(
    renderer, lambda_source, lambda_potential, lambda_amplitude
):
    """Explicit host construction, using coordinates rather than library actions."""
    x, y = np.asarray(renderer.query_x), np.asarray(renderer.query_y)
    fx, fy = (x + 0.3) / 0.6, (y + 0.3) / 0.6
    source = np.array([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]).T
    grid = renderer.potential_grid
    nodes = np.asarray(grid.node_indices)
    lookup = {tuple(node): i for i, node in enumerate(nodes)}
    response = np.zeros((len(x), grid.n_unknown))
    for k, (qx, qy) in enumerate(zip(x, y)):
        ux, uy = qx / grid.spacing, qy / grid.spacing
        ix, iy = int(np.floor(ux)), int(np.floor(uy))
        a, b = ux - ix, uy - iy
        for cx, cy, weight in (
            (ix, iy, (1 - a) * (1 - b)),
            (ix + 1, iy, a * (1 - b)),
            (ix, iy + 1, (1 - a) * b),
            (ix + 1, iy + 1, a * b),
        ):
            for dx, dy, factor in (
                (1, 0, -1.2),
                (-1, 0, 1.2),
                (0, 1, 0.8),
                (0, -1, -0.8),
            ):
                response[k, lookup[(cx + dx, cy + dy)]] += (
                    weight * factor / (2 * grid.spacing)
                )
    potential = lambda_amplitude * np.eye(grid.n_unknown)
    # Independent center-label enumeration: every center with all ±2 axial
    # neighbors contributes one outer product, including support-only centers.
    for cx, cy in nodes:
        for dx, dy in ((1, 0), (0, 1)):
            window = [(cx + j * dx, cy + j * dy) for j in range(-2, 3)]
            if all(node in lookup for node in window):
                row = np.zeros(grid.n_unknown)
                row[[lookup[node] for node in window]] = [1, -4, 6, -4, 1]
                potential += lambda_potential * np.outer(row, row)
    precision = np.zeros((4 + grid.n_unknown, 4 + grid.n_unknown))
    precision[:4, :4] = lambda_source * np.eye(4)
    precision[4:, 4:] = potential
    return np.column_stack((source, response)), precision


def test_complete_fourth_windows_include_support_only_runs_and_do_not_bridge_holes():
    from TinyLensGpu.utils.inversion.potential_regularization import (
        PotentialRegularization,
    )

    with jax.experimental.enable_x64():
        base = PotentialGrid.from_queries([0.25], [0.25], spacing=1.0)
        extra = [(x, 8) for x in range(6)] + [(x, 10) for x in (0, 1, 2, 4, 5)]
        grid = PotentialGrid.from_queries(
            [0.25],
            [0.25],
            spacing=1.0,
            node_indices=np.vstack((base.node_indices, extra)),
        )
        regularization = PotentialRegularization(grid)
        x = grid.node_indices[:, 0].astype(jnp.float64)
        y = grid.node_indices[:, 1].astype(jnp.float64)
        cubic = x**3 + 3 * x * y + 2 * y**2 + 7
        np.testing.assert_allclose(regularization.forward(cubic), 0.0, atol=1e-12)

        # Four core-support horizontal windows plus two wholly support-only
        # windows. The broken row at y=10 contributes no artificial window.
        dx, dy = regularization.horizontal_windows, regularization.vertical_windows
        assert dx.shape == (6, 5)
        assert dy.shape == (4, 5)
        np.testing.assert_allclose(regularization.forward(x**4)[:6], 24.0)
        np.testing.assert_allclose(regularization.forward(x**4)[6:], 0.0)
        assert int(jnp.sum(regularization.window_coverage)) == 50
        assert np.all(
            np.asarray(regularization.window_coverage)[np.asarray(y) == 10] == 0
        )

        # Pixel-unit smoothing has no spacing factors, and the independent
        # positive amplitude term applies to every unknown, including holes.
        p = x**4
        expected = 2.5 * 6 * 24**2 + 0.3 * float(jnp.vdot(p, p))
        q = regularization.to_dense(2.5, 0.3)
        np.testing.assert_allclose(p @ q @ p, expected, rtol=2e-13)
        np.testing.assert_allclose(
            regularization.matvec(p, 2.5, 0.3), q @ p, atol=2e-11
        )
        assert np.linalg.eigvalsh(np.asarray(q)).min() > 0.29999999999


@pytest.mark.parametrize("lambda_potential", [0.03, 2.0])
def test_public_dense_fit_is_coupled_signed_and_has_normalized_gaussian_evidence(
    lambda_potential,
):
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection

    renderer = _small_renderer()
    image = np.array([[-1.4, -0.8, -1.1], [-0.3, -0.6, -0.2], [-0.9, -0.1, -0.5]])
    noise = np.linspace(0.2, 0.4, 9).reshape(3, 3)
    problem = SingleStepPotentialCorrection(
        renderer,
        image,
        noise,
        lambda bx, by: (jnp.full_like(bx, 1.2), jnp.full_like(by, -0.8)),
        lambda_amplitude=0.4,
        source_regularization="zero-order",
        initialization_provenance={"source": "analytic fixed linear source gradient"},
    )
    result = problem(1.3, lambda_potential)
    design, prior = _independent_design_and_prior(renderer, 1.3, lambda_potential, 0.4)
    covariance = np.diag(noise.ravel() ** 2) + design @ np.linalg.solve(prior, design.T)
    expected_logz = -0.5 * (
        image.ravel() @ np.linalg.solve(covariance, image.ravel())
        + np.linalg.slogdet(covariance)[1]
        + 9 * np.log(2 * np.pi)
    )
    curvature = design.T @ (design / noise.ravel()[:, None] ** 2) + prior
    rhs = design.T @ (image.ravel() / noise.ravel() ** 2)
    expected = np.linalg.solve(curvature, rhs)
    fitted = np.r_[result.source_values, result.potential_values]

    assert result.success and result.status == "success"
    assert result.failure_reason is None
    np.testing.assert_allclose(fitted, expected, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(
        result.log_evidence, expected_logz, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        result.linear_model_image.ravel(), design @ expected, atol=2e-12
    )
    np.testing.assert_allclose(curvature @ fitted, rhs, rtol=2e-12, atol=2e-12)
    assert np.linalg.norm(curvature[:4, 4:]) > 1
    decoupled_source = np.linalg.solve(curvature[:4, :4], rhs[:4])
    assert np.linalg.norm(np.asarray(result.source_values) - decoupled_source) > 0.2
    assert np.min(result.source_values) < -0.1
    assert np.min(result.potential_values) < 0 < np.max(result.potential_values)
    assert result.determinant_method == "full_cholesky"
    assert result.evidence_method == "conditional_linear_gaussian"
    assert result.evidence_is_exact is True
    assert result.strengths.lambda_amplitude == 0.4
    assert result.dtype == "float64"
    assert result.solver_info.relative_residual < 1e-11
    assert result.initialization_provenance["source"].startswith("analytic")
    assert result.potential_grid is renderer.potential_grid
    assert result.potential_window_coverage.shape == result.potential_values.shape
    assert np.isnan(
        np.asarray(result.convergence)[~np.asarray(result.derivative_valid)]
    ).all()
    assert (
        np.linalg.norm(
            np.asarray(result.linear_model_image - result.nonlinear_model_image)
        )
        > 1e-3
    )


@pytest.mark.parametrize(
    "interpolation, expected_support",
    [
        ("bilinear", [True, False, False, False, False, True, False, False, False]),
        ("cubic_hermite", [True, True, True, False, False, True, True, False, False]),
    ],
)
def test_single_step_uses_frozen_callback_inside_selected_support(
    interpolation, expected_support
):
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import (
        MacroMapping,
    )
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    with jax.experimental.enable_x64():
        renderer = PotentialCorrectionRenderer(
            PhysicalModel(
                lens_mass=[Shear(gamma1=0.0, gamma2=0.0)],
                source_light=[],
                lens_light=[],
            ),
            SimulatorConfig(0.1, 3, jnp.ones((1, 1))),
            3,
            (0.0, 1.0, 0.0, 1.0),
            source_interpolation=interpolation,
        )
        # Hermite's first virtual line is x=-0.5. Its outer edges are -1 and 2.
        bx = jnp.array([0.25, -0.25, -0.5, -1.0, -1.01, 1.0, 1.25, 2.0, 2.01])
        by = jnp.full_like(bx, 0.5)
        calls = []

        def gradient(x, y):
            calls.append((x, y))
            return 3 + x, -2 + y

        problem = SingleStepPotentialCorrection(
            renderer,
            np.zeros((3, 3)),
            np.ones((3, 3)),
            gradient,
            lambda_amplitude=1.0,
            macro_mapping=MacroMapping(bx, by),
        )
        grid = renderer.potential_grid
        potential = 2 * grid.x - 3 * grid.y
        joint = jnp.concatenate((jnp.zeros(renderer.n_source), potential))
        image = problem.forward(joint)
        problem.forward(2 * joint)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], (bx, by))
    bx, by = np.asarray(bx), np.asarray(by)
    expected = np.where(expected_support, -2 * (3 + bx) + 3 * (-2 + by), 0)
    np.testing.assert_allclose(image, expected, atol=1e-13, rtol=1e-13)


def test_frozen_hermite_callback_excludes_exact_float64_support_edges():
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
    )
    from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import (
        MacroMapping,
    )
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection

    base = _small_renderer()
    lower, upper = -0.022404768054233637, 0.3577528867209083
    n = 11
    width = 2 * (upper - lower) / (n - 1)
    left, right = lower - width, upper + width
    bx = np.array([
        np.nextafter(left, -np.inf), left, np.nextafter(left, np.inf),
        lower, (lower + upper) / 2, upper,
        np.nextafter(right, -np.inf), right, np.nextafter(right, np.inf),
    ])
    by = np.full_like(bx, (lower + upper) / 2)
    with jax.experimental.enable_x64():
        renderer = PotentialCorrectionRenderer(
            base.macro_model, base.sim_config, n, (lower, upper, lower, upper),
            source_interpolation="cubic_hermite",
        )
        problem = SingleStepPotentialCorrection(
            renderer, np.zeros((3, 3)), np.ones((3, 3)),
            lambda x, y: (jnp.ones_like(x), jnp.zeros_like(y)),
            macro_mapping=MacroMapping(jnp.asarray(bx), jnp.asarray(by)),
            lambda_amplitude=1.0,
        )
        np.testing.assert_array_equal(
            problem.mapping.valid, [False, True, True, True, True, True, True, True, False]
        )
        # A linear potential has unit x deflection on the entire query grid.
        joint = jnp.concatenate((jnp.zeros(n**2), renderer.potential_grid.x))
        image = problem.forward(joint)
    np.testing.assert_allclose(image, [0, 0, -1, -1, -1, -1, -1, 0, 0], atol=2e-14, rtol=0)


def test_frozen_response_has_lens_equation_sign_and_quadratic_linearization_error():
    from scipy.signal import convolve2d

    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    with jax.experimental.enable_x64():
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        kernel = jnp.array([[0.0, 0.1, 0.0], [0.05, 0.6, 0.1], [0.0, 0.15, 0.0]])
        config = SimulatorConfig(0.1, 5, kernel, nsub=3, mask=mask)
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.03, gamma2=-0.02)], source_light=[], lens_light=[]
        )
        renderer = PotentialCorrectionRenderer(macro, config, 3, (-0.6, 0.6, -0.6, 0.6))
        state = {"scale": 1.0}

        def gradient(bx, by):
            return state["scale"] * (0.7 + 0.4 * by), state["scale"] * (-0.3 + 0.4 * bx)

        problem = SingleStepPotentialCorrection(
            renderer,
            np.zeros((5, 5)),
            np.ones((5, 5)),
            gradient,
            lambda_amplitude=0.2,
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        source = (1 + 0.7 * sx - 0.3 * sy + 0.4 * sx * sy).ravel()
        grid = renderer.potential_grid
        potential = 0.03 * grid.x**2 - 0.02 * grid.x * grid.y + 0.01 * grid.y**2
        base = renderer.render(source, jnp.zeros_like(potential)).image_active
        # Compute both expectations across the full cutout, including the
        # masked emitter at (2, 2), then convolve independently of the renderer.
        x, y = config.xgrid_sub, config.ygrid_sub
        bx, by = macro.deflection(x, y)
        ax, ay = 0.06 * x - 0.02 * y, -0.02 * x + 0.02 * y

        def convolve_and_select(values):
            native = np.asarray(values).reshape(5, 3, 5, 3).mean(axis=(1, 3))
            return convolve2d(native, np.asarray(kernel), mode="same")[~mask]

        expected_response = convolve_and_select(-(0.7 + 0.4 * by) * ax - (-0.3 + 0.4 * bx) * ay)
        response = problem.forward(jnp.r_[jnp.zeros_like(source), potential])
        np.testing.assert_allclose(response, expected_response, atol=2e-16)
        # Changing the external provider after preparation must not change G0.
        state["scale"] = 5.0
        np.testing.assert_allclose(
            problem.forward(jnp.r_[jnp.zeros_like(source), potential]), response
        )

        errors = []
        for scale in (0.1, 0.05):
            linear = problem.forward(jnp.r_[source, scale * potential])
            nonlinear = renderer.render(source, scale * potential).image_active
            cx, cy = bx - scale * ax, by - scale * ay
            expected_nonlinear = convolve_and_select(1 + 0.7 * cx - 0.3 * cy + 0.4 * cx * cy)
            np.testing.assert_allclose(nonlinear, expected_nonlinear, atol=8e-16, rtol=0)
            np.testing.assert_allclose(
                linear, base + scale * expected_response, atol=3e-16
            )
            errors.append(np.linalg.norm(np.asarray(nonlinear - linear)))
        assert errors[0] > 1e-9
        np.testing.assert_allclose(errors[0] / errors[1], 4, rtol=2e-7)

        vector = jnp.linspace(-0.2, 0.4, problem.n_joint)
        data_vector = jnp.linspace(-0.7, 0.3, renderer.n_active)
        np.testing.assert_allclose(
            jnp.vdot(problem.forward(vector), data_vector),
            jnp.vdot(vector, problem.adjoint(data_vector)),
            atol=2e-14,
        )
        np.testing.assert_allclose(
            problem.dense_mapping() @ vector, problem.forward(vector), atol=2e-15
        )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(
    "source_regularization", ["zero-order", "first-order", "second-order"]
)
def test_source_prior_and_fit_precision_preserve_process_defaults(
    dtype, source_regularization
):
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection

    previous_default = jax.config.x64_enabled
    previous_matmul = jax.config.jax_default_matmul_precision
    # Run each requested dtype while the surrounding context requests the
    # opposite one, so hidden reliance on the process setting is observable.
    with jax.experimental.enable_x64(dtype == "float32"):
        renderer = _small_renderer(dtype=dtype)
        image = np.array([[-1.4, -0.8, -1.1], [-0.3, -0.6, -0.2], [-0.9, -0.1, -0.5]])
        noise = np.linspace(0.2, 0.4, 9).reshape(3, 3)
        problem = SingleStepPotentialCorrection(
            renderer,
            image,
            noise,
            lambda bx, by: (jnp.full_like(bx, 1.2), jnp.full_like(by, -0.8)),
            lambda_amplitude=0.4,
            source_regularization=source_regularization,
        )
        result = problem.fit(1.3, 0.3)
        dense_prior = problem.dense_prior(1.3, 0.3)
        joint = np.r_[result.source_values, result.potential_values]
        prior_product = problem.prior_matvec(joint, 1.3, 0.3)
        assert jax.config.x64_enabled == (dtype == "float32")
        assert jax.config.jax_default_matmul_precision == previous_matmul
    assert jax.config.x64_enabled == previous_default
    assert jax.config.jax_default_matmul_precision == previous_matmul

    design, prior = _independent_design_and_prior(renderer, 1.3, 0.3, 0.4)
    if source_regularization != "zero-order":
        # Existing 2x2 source-prior boundary behavior, as a worked literal:
        # four adjacent differences and the last row/column amplitude terms.
        source_precision = np.array(
            [[2, -1, -1, 0], [-1, 3, 0, -1], [-1, 0, 3, -1], [0, -1, -1, 4]]
        )
        order = 2 if source_regularization == "first-order" else 4
        prior[:4, :4] = 1.3 * source_precision / 0.6**order
    expected = np.linalg.solve(
        design.T @ (design / noise.ravel()[:, None] ** 2) + prior,
        design.T @ (image.ravel() / noise.ravel() ** 2),
    )
    tolerance = 3e-12 if dtype == "float64" else 3e-5
    assert result.success
    assert result.dtype == dtype
    for array in (
        result.source_values,
        result.potential_values,
        result.linear_model_image,
        result.nonlinear_model_image,
        result.log_evidence,
        dense_prior,
        prior_product,
    ):
        assert array.dtype == np.dtype(dtype)
    np.testing.assert_allclose(joint, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(dense_prior, prior, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        prior_product, prior @ joint, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize(
    "lambda_amplitude,lambda_potential,solve_rtol,reason",
    [
        (0.4, 1e308, None, "nonfinite_system"),
        (1e-30, 1.0, None, "prior_factorization_failed"),
        (0.4, 0.3, 1e-30, "linear_residual_exceeds_tolerance"),
    ],
)
def test_numerical_failure_has_status_and_no_evidence_or_prior_rescue(
    lambda_amplitude,
    lambda_potential,
    solve_rtol,
    reason,
):
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection

    problem = SingleStepPotentialCorrection(
        _small_renderer(),
        np.arange(9).reshape(3, 3),
        np.ones((3, 3)),
        lambda bx, by: (jnp.ones_like(bx), jnp.full_like(by, -0.3)),
        lambda_amplitude=lambda_amplitude,
        source_regularization="zero-order",
        solve_rtol=solve_rtol,
    )
    result = problem.fit(0.9, lambda_potential)
    assert not result.success
    assert result.status == "failed"
    assert result.log_evidence is None
    assert result.failure_reason == result.termination_reason == reason
    assert result.solver_info.failed and not result.solver_info.converged
    assert result.solver_info.reason == reason
    assert result.strengths.lambda_amplitude == lambda_amplitude
    assert result.strengths.lambda_potential == lambda_potential
    # Invalid configuration is a different boundary from numerical failure.
    for invalid in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="lambda_source.*finite positive"):
            problem.fit(invalid, 0.3)
    with pytest.raises(ValueError, match="backend"):
        problem.fit(0.9, 0.3, backend="missing")


def test_masked_observation_result_preserves_data_and_both_image_diagnostics():
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    with jax.experimental.enable_x64():
        mask = np.ones((5, 5), dtype=bool)
        mask[1, 1:3] = False
        mask[3, 3] = mask[0, 4] = False
        config = SimulatorConfig(
            0.1,
            5,
            jnp.array([[0, 0.1, 0], [0.05, 0.6, 0.1], [0, 0.15, 0]]),
            nsub=3,
            mask=mask,
        )
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.0, gamma2=0.0)], source_light=[], lens_light=[]
        )
        renderer = PotentialCorrectionRenderer(macro, config, 3, (-0.4, 0.4, -0.4, 0.4))
        image = np.full((5, 5), np.nan)
        image[~mask] = [0.7, -0.2, 0.4, 0.1]
        noise = np.full((5, 5), np.inf)
        noise[~mask] = [0.2, 0.3, 0.25, 0.4]
        problem = SingleStepPotentialCorrection(
            renderer,
            image,
            noise,
            lambda bx, by: (0.7 + by, -0.3 + bx),
            lambda_amplitude=0.4,
            initialization_provenance={"source": "mock truth derivative"},
        )
        result = problem.fit(0.1, 0.3)
        rerendered = renderer.render(result.source_values, result.potential_values)

    assert result.success
    assert result.configuration["n_data"] == 4
    assert result.configuration["n_potential"] == renderer.potential_grid.n_unknown
    assert result.initialization_provenance["source"] == "mock truth derivative"
    np.testing.assert_array_equal(result.data_active, image[~mask])
    # Compiled and eager FFT paths can differ by roundoff at zero-brightness pixels.
    np.testing.assert_allclose(result.nonlinear_model_image, rerendered.model_image, atol=1e-15)
    np.testing.assert_allclose(
        result.linear_image_active, np.asarray(result.linear_model_image)[~mask]
    )
    for kind in ("linear", "nonlinear"):
        active = np.asarray(getattr(result, f"{kind}_image_active"))
        residual = np.asarray(getattr(result, f"{kind}_residual"))
        normalized = np.asarray(getattr(result, f"{kind}_normalized_residual"))
        np.testing.assert_allclose(residual[~mask], image[~mask] - active)
        np.testing.assert_allclose(
            normalized[~mask], (image[~mask] - active) / noise[~mask]
        )
        assert np.isnan(residual[mask]).all() and np.isnan(normalized[mask]).all()
    np.testing.assert_allclose(result.noise_logdet, 2 * np.log(noise[~mask]).sum())


def test_fixed_adaptive_source_scale_enters_dense_operator_penalty_and_evidence():
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
    from TinyLensGpu.utils.inversion.regularization import (
        FiniteDifferenceRegularization,
    )

    renderer = _small_renderer()
    with jax.experimental.enable_x64():
        image = jnp.arange(9, dtype=jnp.float64).reshape(3, 3) / 20
        noise = jnp.full((3, 3), 0.3, dtype=jnp.float64)
        scale = np.array([1.0, 2.0, 4.0, 8.0])

        def gradient(x, y):
            return jnp.ones_like(x), -jnp.ones_like(y)

        problem = SingleStepPotentialCorrection(
            renderer,
            image,
            noise,
            gradient,
            lambda_amplitude=20.0,
            source_regularization="first-order",
            source_scale=scale,
        )
        scale[:] = 100.0
        source_prior = FiniteDifferenceRegularization(2, "first-order").matrix(
            *renderer.source_bbox,
            scale=jnp.array([1.0, 2.0, 4.0, 8.0]),
        )
        q = np.asarray(problem.dense_prior(0.3, 0.02))
        np.testing.assert_allclose(q[:4, :4], 0.3 * source_prior, rtol=1e-12)
        vector = jnp.linspace(-0.2, 0.3, problem.n_joint)
        np.testing.assert_allclose(
            problem.prior_matvec(vector, 0.3, 0.02), q @ vector, rtol=1e-12, atol=1e-12
        )
        fit = problem.fit(0.3, 0.02)
        assert fit.success
        a = np.asarray(problem.dense_mapping())
        n = np.asarray(problem.noise_active)
        d = np.asarray(problem.data_active)
        covariance = np.diag(n**2) + a @ np.linalg.solve(q, a.T)
        expected = -0.5 * (
            d @ np.linalg.solve(covariance, d)
            + np.linalg.slogdet(covariance)[1]
            + d.size * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(fit.log_evidence, expected, atol=1e-10)
        operator = problem.fit(
            0.3,
            0.02,
            backend="operator",
            block_size=7,
            pcg_max_iter=500,
            pcg_rtol=1e-12,
        )
        assert operator.success
        np.testing.assert_allclose(
            operator.source_values, fit.source_values, rtol=1e-7, atol=1e-9
        )
        assert fit.configuration["source_adaptive_scale_fixed"] is True


@pytest.mark.parametrize(
    "scale",
    [np.ones(3), [1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 1.0, 1.0], [1.0, np.nan, 1.0, 1.0]],
)
def test_invalid_fixed_source_scale_rejected(scale):
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection

    renderer = _small_renderer()
    with pytest.raises(ValueError, match="source_scale"):
        SingleStepPotentialCorrection(
            renderer,
            np.ones((3, 3)),
            np.ones((3, 3)),
            lambda x, y: (x, y),
            lambda_amplitude=20.0,
            source_scale=scale,
        )
