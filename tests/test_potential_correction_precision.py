"""Explicit correction precision at public fit and numerical-operator seams."""

import jax
import numpy as np
import pytest

from tests.test_potential_correction_iterative import _coupled_problem, _renderer


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("algorithm", ["single_step", "iterative"])
def test_requested_precision_covers_psf_mapping_gradient_actions_and_evidence(
    dtype, algorithm
):
    from TinyLensGpu.ObservationModel import (
        IterativePotentialCorrection,
        SingleStepPotentialCorrection,
    )

    original = _coupled_problem()
    image = np.asarray(original.image_data)
    noise = np.asarray(original.noise_map)
    initial = np.asarray(original.initial_values[:4])
    frozen_gradient = original.renderer.source_spatial_gradient(
        initial, original.renderer.source_mapping()
    )
    before = (jax.config.x64_enabled, jax.config.jax_default_matmul_precision)
    with jax.experimental.enable_x64(dtype == "float32"):
        renderer = _renderer(dtype=dtype, source_n=2, bbox=(-0.5, 0.5, -0.5, 0.5))
        options = dict(
            lambda_amplitude=20.0, source_regularization="zero-order", solve_rtol=1e-5
        )
        if algorithm == "single_step":
            problem = SingleStepPotentialCorrection(
                renderer, image, noise, lambda bx, by: frozen_gradient, **options
            )
            result = problem.fit(
                0.3, 0.02, backend="operator", block_size=7, pcg_rtol=1e-8
            )
        else:
            problem = IterativePotentialCorrection(
                renderer, image, noise, initial, solve_atol=1e-6, **options
            )
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
        assert result.success, result.termination_reason
        joint = np.r_[
            np.asarray(result.source_values), np.asarray(result.potential_values)
        ]
        mapping = renderer.source_mapping(result.potential_values)
        gx, gy = renderer.source_spatial_gradient(result.source_values, mapping)
        if algorithm == "single_step":
            forward = problem.forward(joint)
            transpose = problem.adjoint(problem.data_active)
        else:
            current = problem.linearize(joint)
            forward = problem.jacobian_forward(joint, current)
            transpose = problem.jacobian_adjoint(problem.data_active, current)
        for values in (
            renderer.query_x,
            renderer.source_x_axis,
            mapping.weights,
            gx,
            gy,
            forward,
            transpose,
            problem.prior_matvec(joint, 0.3, 0.02),
            result.source_values,
            result.potential_values,
            result.nonlinear_image_active,
            result.objective,
            result.prior_logdet,
            result.posterior_logdet,
            result.log_evidence,
        ):
            assert values.dtype == np.dtype(dtype)
        assert renderer.psf_fft.dtype == np.dtype(
            "complex128" if dtype == "float64" else "complex64"
        )
        assert np.issubdtype(mapping.indices.dtype, np.integer)
        assert mapping.valid.dtype == np.dtype("bool")
        assert jax.config.x64_enabled == (dtype == "float32")
        assert jax.config.jax_default_matmul_precision == before[1]
    assert (jax.config.x64_enabled, jax.config.jax_default_matmul_precision) == before
