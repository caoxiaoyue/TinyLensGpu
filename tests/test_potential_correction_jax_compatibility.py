"""Dense public fits retain the declared older JAX map API contract."""

import jax
import numpy as np
import pytest

from tests.test_potential_correction_iterative import _coupled_problem


@pytest.mark.parametrize("algorithm", ["single_step", "iterative"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_dense_fit_matches_native_map_with_legacy_map_signature(
    monkeypatch, algorithm, dtype
):
    from TinyLensGpu.ObservationModel import (
        IterativePotentialCorrection,
        SingleStepPotentialCorrection,
    )

    def problem():
        current = _coupled_problem(dtype=dtype, solve_rtol=1e-5)
        if algorithm == "iterative":
            return IterativePotentialCorrection(
                current.renderer,
                current.image_data,
                current.noise_map,
                np.asarray(current.initial_values[:4]),
                lambda_amplitude=20.0,
                source_regularization="zero-order",
                solve_rtol=1e-5,
                solve_atol=1e-6,
            )
        return SingleStepPotentialCorrection(
            current.renderer,
            current.image_data,
            current.noise_map,
            lambda bx, by: (0.95 + 0.5 * by, 0.05 + 0.5 * bx),
            lambda_amplitude=20.0,
            source_regularization="zero-order",
            solve_rtol=1e-5,
        )

    controls = {}
    if algorithm == "iterative":
        controls.update(
            gradient_atol=1e-4,
            gradient_rtol=1e-5,
            objective_rtol=1e-6,
            step_rtol=1e-4,
        )
    with jax.experimental.enable_x64(dtype == "float64"):
        reference = problem().fit(0.3, 0.02, **controls)
        native_map = jax.lax.map

        # External API adapter: JAX 0.4.20 accepts only map(f, xs). Delegate
        # its numerical work to this installed JAX without accepting new kwargs.
        def legacy_map(f, xs):
            return native_map(f, xs)

        monkeypatch.setattr(jax.lax, "map", legacy_map)
        compatible = problem().fit(0.3, 0.02, **controls)
    assert reference.success and compatible.success, compatible.termination_reason
    tolerance = 2e-9 if dtype == "float64" else 2e-4
    for name in (
        "source_values",
        "potential_values",
        "nonlinear_model_image",
        "objective",
        "prior_logdet",
        "posterior_logdet",
        "log_evidence",
    ):
        values = getattr(compatible, name)
        assert values.dtype == np.dtype(dtype)
        np.testing.assert_allclose(
            values, getattr(reference, name), atol=tolerance, rtol=tolerance
        )
    if algorithm == "single_step":
        np.testing.assert_allclose(
            compatible.linear_model_image,
            reference.linear_model_image,
            atol=tolerance,
            rtol=tolerance,
        )
    # The 37 fitted coefficients require two full panels and a five-column
    # remainder. Full public fields catch dropped/reordered remainder columns.
    assert compatible.source_values.size + compatible.potential_values.size == 37
    assert compatible.determinant_method == "full_cholesky"
