"""Small fixed-strength iterative assembly through the public result."""

import numpy as np


def test_fixed_strength_dense_iteration_runs_without_outer_optimization():
    from examples.potential_correction.iterate_fixed_strength import main

    result = main([])
    assert result.success, result.termination_reason
    assert result.algorithm == "iterative" and result.backend == "dense"
    assert result.objective < result.history[0]["objective_before"]
    assert np.linalg.norm(result.potential_values) > 0
    assert result.linear_model_image is None
    assert result.log_evidence is not None
