"""Small fixed-strength iterative assembly through the public result."""

import numpy as np


def test_fixed_strength_dense_iteration_runs_without_outer_optimization():
    from examples.potential_correction.mock.iterative.demos.fixed_strength import main

    result = main([])
    assert result.success, result.termination_reason
    assert result.algorithm == "iterative" and result.backend == "dense"
    assert result.objective < result.history[0]["objective_before"]
    assert np.linalg.norm(result.potential_values) > 0
    assert result.linear_model_image is None
    assert result.log_evidence is not None


def test_fixed_strength_operator_iteration_uses_shared_result_and_history():
    from examples.potential_correction.mock.iterative.demos.fixed_strength import main

    result = main(
        [
            "--backend",
            "operator",
            "--block-size",
            "16",
            "--pcg-max-iter",
            "800",
            "--pcg-rtol",
            "1e-12",
        ]
    )
    assert result.success, result.termination_reason
    assert result.algorithm == "iterative" and result.backend == "operator"
    assert result.determinant_method == "principal_block_cholesky"
    assert result.evidence_method == "local_gauss_newton"
    assert result.objective < result.history[0]["objective_before"]
    assert all(
        item["objective_after"] < item["objective_before"]
        for item in result.history
        if item["accepted"]
    )
    assert result.log_evidence is not None
    assert result.configuration["pcg_max_iter"] == 800
