"""The small tuning entry point returns and saves its public reconstruction."""

import json

import numpy as np
import pytest


@pytest.mark.integration
def test_tuning_example_saves_bounds_candidates_and_best_reconstruction(tmp_path):
    from examples.potential_correction.tune_single_step import main

    result = main(
        [
            "--lambda-amplitude",
            "0.4",
            "--log-source-bounds",
            "-2",
            "2",
            "--log-potential-bounds",
            "-4",
            "2",
            "--maxfun",
            "12",
            "--output",
            str(tmp_path),
        ]
    )
    assert result.has_valid_fit
    assert result.reconstruction.success
    assert not result.optimizer_success
    assert not result.success
    metadata = json.loads((tmp_path / "tuning.json").read_text())
    assert metadata["has_valid_fit"] is True
    assert metadata["optimizer_success"] is False
    assert metadata["configuration"]["lambda_amplitude"] == 0.4
    assert metadata["configuration"]["maxfun"] == 12
    assert metadata["initialization_provenance"]["source"] == "mock analytic gradient"
    assert len(metadata["evaluations"]) == result.nfev
    np.testing.assert_array_equal(metadata["log_bounds"], [[-2.0, 2.0], [-4.0, 2.0]])
    with np.load(tmp_path / "reconstruction.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            saved["source_values"], result.reconstruction.source_values
        )
        np.testing.assert_array_equal(
            saved["potential_values"], result.reconstruction.potential_values
        )
        np.testing.assert_array_equal(
            saved["linear_model_image"], result.reconstruction.linear_model_image
        )
        np.testing.assert_array_equal(
            saved["nonlinear_model_image"], result.reconstruction.nonlinear_model_image
        )
        assert saved["source_values"].dtype == np.float64
