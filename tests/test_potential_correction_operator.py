"""Fixed-strength operator reconstruction at the public numerical boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_potential_correction_dense import (
    _independent_design_and_prior,
    _small_renderer,
)


def _problem(dtype="float64", **kwargs):
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection

    return SingleStepPotentialCorrection(
        _small_renderer(dtype=dtype),
        np.array([[-1.4, -0.8, -1.1], [-0.3, -0.6, -0.2], [-0.9, -0.1, -0.5]]),
        np.linspace(0.2, 0.4, 9).reshape(3, 3),
        lambda bx, by: (jnp.full_like(bx, 1.2), jnp.full_like(by, -0.8)),
        lambda_amplitude=0.4,
        source_regularization="zero-order",
        **kwargs,
    )


def test_operator_fit_matches_full_coupled_dense_solution_at_fixed_strengths():
    problem = _problem()
    dense = problem.fit(1.3, 0.3)
    fitted = problem.fit(
        1.3,
        0.3,
        backend="operator",
        block_size=8,
        pcg_rtol=1e-11,
    )

    assert fitted.success, fitted.failure_reason
    assert fitted.backend == "operator"
    assert fitted.determinant_method == "principal_block_cholesky"
    assert fitted.evidence_method == "conditional_linear_gaussian"
    assert fitted.evidence_is_exact is False
    assert fitted.solver_info.method == "pcg"
    assert fitted.solver_info.relative_residual < 1e-10
    for field in (
        "source_values",
        "potential_values",
        "linear_model_image",
        "nonlinear_model_image",
        "objective",
    ):
        np.testing.assert_allclose(
            getattr(fitted, field),
            getattr(dense, field),
            rtol=2e-9,
            atol=2e-10,
        )
    # The reference fixture has nonzero cross curvature. Its source moves by
    # more than 0.2 from the source-only solve, so a decoupled fit cannot pass.
    with jax.experimental.enable_x64():
        design = np.asarray(problem.dense_mapping())
        prior = np.asarray(problem.dense_prior(1.3, 0.3))
        curvature = (
            design.T @ (design / np.asarray(problem.noise_active)[:, None] ** 2) + prior
        )
        rhs = design.T @ (
            np.asarray(problem.data_active) / np.asarray(problem.noise_active) ** 2
        )
        decoupled = np.linalg.solve(curvature[:4, :4], rhs[:4])
        assert np.linalg.norm(curvature[:4, 4:]) > 1
        assert np.linalg.norm(np.asarray(fitted.source_values) - decoupled) > 0.2
        joint = np.r_[fitted.source_values, fitted.potential_values]
        relative = np.linalg.norm(curvature @ joint - rhs) / np.linalg.norm(rhs)
        np.testing.assert_allclose(
            fitted.solver_info.relative_residual, relative, atol=2e-14
        )


@pytest.mark.parametrize("lambda_potential", [0.03, 2.0])
def test_block_evidence_uses_complete_prior_on_every_support_coefficient(
    lambda_potential,
):
    problem = _problem()
    # Deliberately mixed, irregular, padded blocks, including support-only
    # coefficients: the partition is an explicit public numerical control.
    order = np.r_[np.arange(problem.n_joint)[::2], np.arange(problem.n_joint)[1::2]]
    blocks = tuple(tuple(order[start : start + 7]) for start in range(0, len(order), 7))
    fitted = problem.fit(
        1.3,
        lambda_potential,
        backend="operator",
        block_size=7,
        block_indices=blocks,
        pcg_rtol=1e-11,
    )
    design, prior = _independent_design_and_prior(
        problem.renderer,
        1.3,
        lambda_potential,
        0.4,
    )
    noise = np.asarray(problem.noise_active)
    posterior = design.T @ (design / noise[:, None] ** 2) + prior
    prior_logdet = sum(
        np.linalg.slogdet(prior[np.ix_(block, block)])[1] for block in blocks
    )
    posterior_logdet = sum(
        np.linalg.slogdet(posterior[np.ix_(block, block)])[1] for block in blocks
    )
    expected = -0.5 * (
        2 * float(fitted.objective)
        + posterior_logdet
        - prior_logdet
        + 2 * np.log(noise).sum()
        + noise.size * np.log(2 * np.pi)
    )
    assert fitted.success, fitted.failure_reason
    np.testing.assert_allclose(fitted.prior_logdet, prior_logdet, atol=2e-12)
    np.testing.assert_allclose(fitted.posterior_logdet, posterior_logdet, atol=2e-12)
    np.testing.assert_allclose(fitted.log_evidence, expected, atol=2e-12)
    assert fitted.strengths.lambda_amplitude == 0.4
    assert fitted.configuration["block_indices"] == blocks
    assert sum(fitted.configuration["block_sizes"]) == problem.n_joint
    assert np.any(~np.asarray(fitted.derivative_valid))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_operator_precision_adjoint_and_repeated_strengths_preserve_process_defaults(
    dtype,
):
    previous_x64 = jax.config.x64_enabled
    previous_matmul = jax.config.jax_default_matmul_precision
    with jax.experimental.enable_x64(dtype == "float32"):
        problem = _problem(dtype)
        first = problem.fit(1.3, 0.3, backend="operator", block_size=8)
        changed = problem.fit(0.7, 2.0, backend="operator", block_size=8)
        dense = problem.fit(0.7, 2.0)
        repeated = problem.fit(1.3, 0.3, backend="operator", block_size=8)
        assert jax.config.x64_enabled == (dtype == "float32")
        assert jax.config.jax_default_matmul_precision == previous_matmul
        with jax.experimental.enable_x64(dtype == "float64"):
            values = jnp.linspace(-0.2, 0.4, problem.n_joint, dtype=dtype)
            image = jnp.linspace(-0.7, 0.3, problem.renderer.n_active, dtype=dtype)
            forward = jax.jit(problem.forward)(values)
            transpose = jax.jit(problem.adjoint)(image)
            tolerance = 2e-12 if dtype == "float64" else 2e-5
            np.testing.assert_allclose(
                jnp.vdot(forward, image), jnp.vdot(values, transpose), atol=tolerance
            )
    assert jax.config.x64_enabled == previous_x64
    assert jax.config.jax_default_matmul_precision == previous_matmul
    for result in (first, changed, repeated):
        assert result.success, result.failure_reason
        for array in (
            result.source_values,
            result.potential_values,
            result.linear_model_image,
            result.nonlinear_model_image,
            result.log_evidence,
            result.prior_logdet,
        ):
            assert array.dtype == np.dtype(dtype)
    tolerance = 2e-7 if dtype == "float64" else 8e-4
    np.testing.assert_allclose(
        changed.source_values, dense.source_values, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        changed.potential_values, dense.potential_values, rtol=tolerance, atol=tolerance
    )
    # GPU scatter reductions can differ slightly between invocations; the
    # same initialization must reproduce the solve at its declared tolerance.
    np.testing.assert_allclose(
        first.source_values, repeated.source_values, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        first.potential_values,
        repeated.potential_values,
        rtol=tolerance,
        atol=tolerance,
    )
    assert (
        first.configuration["block_partition_sha256"]
        == changed.configuration["block_partition_sha256"]
    )


def test_strength_candidates_reuse_compiled_kernels(caplog):
    problem = _problem()
    with jax.log_compiles(True):
        first = problem.fit(1.3, 0.3, backend="operator", block_size=8)
        assert first.success
        caplog.clear()
        for source_strength, potential_strength in ((0.7, 2.0), (1.3, 0.3)):
            fitted = problem.fit(
                source_strength, potential_strength, backend="operator", block_size=8
            )
            assert fitted.success
    compilations = [
        record.message
        for record in caplog.records
        if "Compiling jit(" in record.message
    ]
    assert not compilations


@pytest.mark.parametrize(
    "strength,options,solve_rtol,reason",
    [
        (0.3, {"pcg_max_iter": 1}, None, "pcg_iteration_limit"),
        (1e308, {}, None, "nonfinite_system"),
        (0.3, {"pcg_rtol": 1e-3}, 1e-12, "linear_residual_exceeds_tolerance"),
    ],
)
def test_operator_numerical_failure_has_status_and_never_valid_evidence(
    strength,
    options,
    solve_rtol,
    reason,
):
    problem = _problem(solve_rtol=solve_rtol)
    result = problem.fit(1.3, strength, backend="operator", block_size=8, **options)
    assert not result.success
    assert result.status == "failed"
    assert result.log_evidence is None
    assert result.failure_reason == result.termination_reason == reason
    assert result.solver_info.failed and not result.solver_info.converged
    assert result.solver_info.reason == reason
    assert result.strengths.lambda_amplitude == 0.4
    assert result.strengths.lambda_potential == strength


@pytest.mark.parametrize(
    "options,match",
    [
        ({"block_size": 0}, "block_size"),
        ({"block_size": np.inf}, "block_size"),
        ({"block_indices": [(0, 1), (1, 2)]}, "exactly once"),
        ({"block_indices": [(0.0, 1.0)]}, "integer index"),
        ({"pcg_max_iter": 0}, "pcg_max_iter"),
        ({"pcg_max_iter": np.inf}, "pcg_max_iter"),
        ({"pcg_rtol": np.nan}, "pcg_rtol"),
        ({"pcg_atol": -1}, "pcg_atol"),
    ],
)
def test_invalid_operator_controls_raise_configuration_errors(options, match):
    with pytest.raises(ValueError, match=match):
        _problem().fit(1.3, 0.3, backend="operator", **options)


def test_zero_data_converges_to_zero_with_zero_absolute_tolerance():
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection

    problem = SingleStepPotentialCorrection(
        _small_renderer(),
        np.zeros((3, 3)),
        np.ones((3, 3)),
        lambda bx, by: (jnp.ones_like(bx), -jnp.ones_like(by)),
        lambda_amplitude=0.4,
    )
    fitted = problem.fit(1.3, 0.3, backend="operator", pcg_atol=0.0)
    assert fitted.success, fitted.failure_reason
    assert fitted.solver_info.iterations == 0
    assert fitted.solver_info.residual_norm == 0
    np.testing.assert_array_equal(fitted.source_values, 0)
    np.testing.assert_array_equal(fitted.potential_values, 0)
    assert np.isfinite(fitted.log_evidence)


def test_operator_matches_dense_with_mask_holes_subpixels_and_asymmetric_psf():
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
        renderer = PotentialCorrectionRenderer(
            PhysicalModel(
                lens_mass=[Shear(gamma1=0.03, gamma2=-0.02)],
                source_light=[],
                lens_light=[],
            ),
            config,
            3,
            (-0.4, 0.4, -0.4, 0.4),
        )
        data = np.full((5, 5), np.nan)
        noise = np.full((5, 5), np.inf)
        data[~mask], noise[~mask] = [0.7, -0.2, 0.4, 0.1], [0.2, 0.3, 0.25, 0.4]
        problem = SingleStepPotentialCorrection(
            renderer,
            data,
            noise,
            lambda bx, by: (0.7 + by, -0.3 + bx),
            lambda_amplitude=0.4,
        )
        dense = problem.fit(0.1, 0.3)
        fitted = problem.fit(0.1, 0.3, backend="operator", block_size=9, pcg_rtol=1e-12)
        vector = jnp.linspace(-0.1, 0.7, problem.n_joint)
        image = jnp.linspace(0.2, -0.4, renderer.n_active)
        np.testing.assert_allclose(
            jnp.vdot(problem.forward(vector), image),
            jnp.vdot(vector, problem.adjoint(image)),
            atol=2e-14,
        )
    assert dense.success and fitted.success, fitted.failure_reason
    for field in (
        "source_values",
        "potential_values",
        "linear_model_image",
        "nonlinear_model_image",
        "objective",
    ):
        np.testing.assert_allclose(
            getattr(fitted, field), getattr(dense, field), rtol=2e-9, atol=2e-10
        )
    assert np.isnan(np.asarray(fitted.linear_residual)[mask]).all()
    np.testing.assert_allclose(
        np.asarray(fitted.linear_residual)[~mask],
        data[~mask] - np.asarray(fitted.linear_image_active),
    )
