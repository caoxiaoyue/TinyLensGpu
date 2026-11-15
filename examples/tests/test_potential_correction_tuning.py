"""The small tuning entry point returns and saves its public reconstruction."""

import json

import numpy as np
import pytest


@pytest.mark.integration
def test_tuning_example_saves_bounds_candidates_and_best_reconstruction(tmp_path):
    from examples.potential_correction.mock.single_step.tune import main

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


@pytest.mark.integration
def test_tuning_reruns_remove_all_invalid_products_and_recover(tmp_path):
    from examples.potential_correction.mock.single_step.tune import run_tuning

    unrelated = tmp_path / "observer-notes.txt"
    unrelated.write_text("Retain notes across strength searches.\n")
    arguments = dict(lambda_amplitude=0.4, maxfun=1, maxiter=10, output=tmp_path)
    valid_bounds = ((-2.0, 2.0), (-4.0, 2.0))
    first = run_tuning(log_bounds=valid_bounds, **arguments)
    assert first.has_valid_fit and first.reconstruction.success
    assert not first.optimizer_success
    assert (tmp_path / "reconstruction.npz").exists()

    # Bounds themselves are representable: failure comes from the real coupled
    # system at these source strengths, rather than a mocked numerical result.
    invalid_bounds = (tuple(np.log([1e307, 1e308])), (-4.0, 2.0))
    failed = run_tuning(log_bounds=invalid_bounds, **arguments)
    assert not failed.has_valid_fit
    assert failed.reconstruction is None
    metadata = json.loads((tmp_path / "tuning.json").read_text())
    assert metadata["has_valid_fit"] is False
    assert metadata["selected_strengths"] is None
    assert metadata["reconstruction_configuration"] is None
    assert metadata["initialization_provenance"]["source"] == "mock analytic gradient"
    assert {row["failure_reason"] for row in metadata["evaluations"]} == {
        "nonfinite_system"
    }
    assert not (tmp_path / "reconstruction.npz").exists()

    recovered = run_tuning(log_bounds=valid_bounds, **arguments)
    assert recovered.has_valid_fit and recovered.reconstruction.success
    # A valid reconstruction is still retained when the outer budget expires.
    assert not recovered.optimizer_success
    metadata = json.loads((tmp_path / "tuning.json").read_text())
    assert metadata["has_valid_fit"] is True
    assert metadata["optimizer_success"] is False
    assert metadata["configuration"]["maxfun"] == 1
    with np.load(tmp_path / "reconstruction.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            saved["source_values"], recovered.reconstruction.source_values
        )
        np.testing.assert_array_equal(
            saved["potential_values"], recovered.reconstruction.potential_values
        )
    assert unrelated.read_text() == "Retain notes across strength searches.\n"
