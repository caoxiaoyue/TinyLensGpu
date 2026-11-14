"""Stiff physical source–potential fits retain a bounded true-residual certificate."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# The larger render footprint changes the conditioning. These test strengths
# exercise the same true-residual gates on CPU and CUDA with the updated grid.
_LAMBDA_SOURCE = 6e-4
_LAMBDA_POTENTIAL = 3e10
_FULL_BUDGET = 20000
_LIMITED_BUDGET = 6000


def _scientific_control():
    """Independent analytic two-Gaussian control, not a representable nodal mock."""
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection
    from TinyLensGpu.PhysicalModel import PhysicalModel, SIS, Shear

    def brightness(x, y):
        return jnp.exp(
            -0.5 * ((x - 0.05) / 0.16) ** 2 - 0.5 * ((y - 0.03) / 0.12) ** 2
        ) + 0.35 * jnp.exp(
            -0.5 * ((x + 0.08) / (0.16 * 0.7)) ** 2
            - 0.5 * ((y - 0.13) / (0.12 * 0.7)) ** 2
        )

    with jax.experimental.enable_x64():
        axis = (jnp.arange(64) - 31.5) * 0.0375
        x, y = jnp.meshgrid(axis, axis)
        radius = jnp.hypot(x, y)
        px, py = jnp.meshgrid(jnp.arange(-2, 3) * 0.0375, jnp.arange(-2, 3) * 0.0375)
        psf = jnp.exp(-(px**2 + py**2) / (2 * 0.05**2))
        psf = psf / psf.sum()
        renderer = PotentialCorrectionRenderer(
            PhysicalModel(
                lens_mass=[
                    SIS(theta_E=0.8, center_x=0.0, center_y=0.0),
                    Shear(gamma1=0.05, gamma2=0.02),
                ],
                source_light=[],
                lens_light=[],
            ),
            SimulatorConfig(0.0375, 64, psf, 2, (radius < 0.45) | (radius > 1.1)),
            49,
            (-0.4, 0.4, -0.4, 0.4),
            potential_spacing=0.075,
            dtype="float64",
        )
        image = renderer.image_forward(
            brightness(*renderer.macro_mapping), full_image=True
        )
        image = image + jnp.asarray(
            np.random.default_rng(17).standard_normal((64, 64)) * 0.01
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        return IterativePotentialCorrection(
            renderer,
            image,
            jnp.full((64, 64), 0.01),
            brightness(sx, sy).ravel(),
            lambda_amplitude=1e4,
            solve_rtol=1e-8,
            solve_atol=2e-5,
        )


def _fixed_partition(problem):
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures/potential_correction_control_partition.json"
        ).read_text()
    )
    renderer = problem.renderer
    nodes = np.asarray(renderer.potential_grid.node_indices)
    assert fixture["psf_response"] == renderer.psf_response.convention
    assert fixture["source_n"] == renderer.source_n
    assert fixture["potential_node_indices_sha256"] == hashlib.sha256(nodes.tobytes()).hexdigest()
    return fixture


@pytest.mark.integration
@pytest.mark.slow
def test_stiff_control_operator_converges_at_unchanged_true_residual_limits(caplog):
    problem = _scientific_control()
    fixture = _fixed_partition(problem)
    dense = problem.fit(_LAMBDA_SOURCE, _LAMBDA_POTENTIAL, gradient_atol=3e-5)
    with jax.log_compiles(True):
        result = problem.fit(
            _LAMBDA_SOURCE,
            _LAMBDA_POTENTIAL,
            backend="operator",
            gradient_atol=3e-5,
            block_size=fixture["block_size"],
            block_indices=fixture["block_indices"],
            pcg_max_iter=_FULL_BUDGET,
            pcg_rtol=1e-12,
        )
    assert (
        len([row for row in caplog.records if "Compiling jit(solve)" in row.message])
        == 1
    )
    assert dense.success, dense.termination_reason
    assert result.success, (result.termination_reason, result.solver_info)
    # Physical agreement is measured independently of block evidence.
    np.testing.assert_allclose(
        result.source_values, dense.source_values, atol=2e-7, rtol=2e-6
    )
    np.testing.assert_allclose(
        result.potential_values, dense.potential_values, atol=2e-8, rtol=2e-6
    )
    np.testing.assert_allclose(
        result.convergence, dense.convergence, atol=2e-6, rtol=2e-6, equal_nan=True
    )
    assert (
        np.sqrt(
            np.mean(
                (
                    (
                        np.asarray(result.nonlinear_image_active)
                        - np.asarray(dense.nonlinear_image_active)
                    )
                    / 0.01
                )
                ** 2
            )
        )
        < 2e-6
    )
    np.testing.assert_allclose(result.objective, dense.objective, atol=2e-7, rtol=0)
    solves = [row["solver_info"] for row in result.history if "solver_info" in row]
    assert any(row["diagnostics"]["refinement_attempted"] for row in solves)
    for info in solves:
        diagnostic = info["diagnostics"]
        assert info["iterations"] == sum(
            phase["iterations"] for phase in diagnostic["phases"]
        )
        assert info["iterations"] <= diagnostic["iteration_budget"] == _FULL_BUDGET
        assert info["residual_norm"] <= diagnostic["true_residual_threshold"]


@pytest.mark.integration
@pytest.mark.slow
def test_residual_correction_uses_remaining_budget_and_failed_steps_stay_failed():
    problem = _scientific_control()
    fixture = _fixed_partition(problem)
    result = problem.fit(
        _LAMBDA_SOURCE,
        _LAMBDA_POTENTIAL,
        backend="operator",
        gradient_atol=3e-5,
        block_size=fixture["block_size"],
        block_indices=fixture["block_indices"],
        pcg_max_iter=_LIMITED_BUDGET,
        pcg_rtol=1e-12,
    )
    assert not result.success and result.log_evidence is None
    assert result.termination_reason == "linear_solve_failed"
    assert result.solver_info.reason == "pcg_iteration_limit"
    diagnostic = result.solver_info.diagnostics
    assert diagnostic["refinement_attempted"]
    assert result.solver_info.iterations == diagnostic["iteration_budget"] == _LIMITED_BUDGET
    assert sum(phase["iterations"] for phase in diagnostic["phases"]) == _LIMITED_BUDGET
    assert diagnostic["phases"][0]["recursive_converged"]
    assert not diagnostic["phases"][1]["recursive_converged"]
