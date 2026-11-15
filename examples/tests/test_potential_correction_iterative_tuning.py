"""Nested numerical example hands its retained public result to diagnostics."""

import json
import numpy as np
import pytest


@pytest.mark.parametrize("backend", ["dense", "operator"])
def test_small_nested_example_exports_the_retained_success_and_search(
    tmp_path, backend
):
    from examples.potential_correction.mock.iterative.tune import main

    search = main(["--backend", backend, "--maxfun", "5", "--output", str(tmp_path)])
    result = search.reconstruction
    assert search.has_valid_fit and result.success
    assert result.algorithm == "iterative"
    assert result.linear_model_image is None
    metadata = json.loads((tmp_path / "nested_result.json").read_text())
    record = json.loads((tmp_path / "search.json").read_text())
    assert metadata["termination_reason"] == result.termination_reason
    assert metadata["evidence_method"] == "local_gauss_newton"
    assert metadata["determinant_method"] == result.determinant_method
    assert metadata["history"] == list(result.history)
    assert metadata["strengths"]["lambda_amplitude"] == 20.0
    assert record["nfev"] == len(record["evaluations"]) == search.nfev
    assert record["optimizer_success"] == search.optimizer_success
    assert record["optimizer_status"] == search.optimizer_status
    assert record["configuration"]["maxfun"] == 5
    assert len(record["log_bounds"]) == 2
    assert all(row["algorithm"] == "iterative" for row in record["evaluations"])
    with np.load(tmp_path / "nested_result.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["nonlinear_model_image"], result.nonlinear_model_image
        )
        np.testing.assert_array_equal(arrays["source_values"], result.source_values)
        np.testing.assert_array_equal(
            arrays["potential_values"], result.potential_values
        )
        assert "linear_model_image" not in arrays
        assert arrays["truth_source_values"].shape == (4,)
        assert arrays["potential_node_indices"].shape == (33, 2)
    assert not (tmp_path / "failure.json").exists()


def test_all_invalid_nested_example_keeps_reasons_and_removes_stale_success(tmp_path):
    from examples.potential_correction.mock.iterative.tune import main

    main(["--maxfun", "5", "--output", str(tmp_path)])
    search = main(["--maxfun", "5", "--max-iter", "0", "--output", str(tmp_path)])
    assert not search.has_valid_fit and search.reconstruction is None
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert not failure["success"]
    assert failure["failure_reason"] == "no_successful_inner_reconstruction"
    assert all(
        row["termination_reason"] == "iteration_limit"
        for row in failure["search"]["evaluations"]
    )
    assert all(row["log_evidence"] is None for row in failure["search"]["evaluations"])
    assert not (tmp_path / "nested_result.json").exists()
    assert not (tmp_path / "nested_result.npz").exists()
