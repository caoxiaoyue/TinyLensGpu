"""The small fixed-strength operator example publishes public result products."""

import json

import numpy as np
import pytest


def test_fixed_operator_example_saves_fields_both_images_and_method_metadata(tmp_path):
    from examples.potential_correction.single_step_operator import main

    output = main(["--output", str(tmp_path)])
    assert output == tmp_path
    metadata = json.loads((output / "configuration.json").read_text())
    assert metadata["success"] is True
    assert metadata["backend"] == "operator"
    assert metadata["algorithm"] == "single_step"
    assert metadata["evidence_is_exact"] is False
    assert metadata["determinant_method"] == "principal_block_cholesky"
    assert metadata["configuration"]["pcg_max_iter"] > 1
    with np.load(output / "reconstruction.npz", allow_pickle=False) as products:
        assert products["source_values"].shape == (25,)
        assert products["potential_values"].shape == products["derivative_valid"].shape
        assert products["linear_model_image"].shape == (12, 12)
        assert products["nonlinear_model_image"].shape == (12, 12)
        assert np.isfinite(products["linear_model_image"]).all()
        assert np.isfinite(products["nonlinear_model_image"]).all()
        assert np.isnan(products["convergence"][~products["derivative_valid"]]).all()


@pytest.mark.integration
def test_fixed_operator_reruns_replace_success_with_failure_and_recover(tmp_path):
    from examples.potential_correction.single_step_operator import main

    unrelated = tmp_path / "observer-notes.txt"
    unrelated.write_text("Retain notes across reconstruction attempts.\n")
    arguments = ["--output", str(tmp_path)]
    main(arguments)
    assert json.loads((tmp_path / "configuration.json").read_text())["success"]
    assert (tmp_path / "reconstruction.npz").exists()

    with pytest.raises(RuntimeError, match="pcg_iteration_limit"):
        main([*arguments, "--pcg-max-iter", "1"])
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["success"] is False
    assert failure["failure_reason"] == "pcg_iteration_limit"
    assert failure["configuration"]["pcg_max_iter"] == 1
    assert (
        failure["initialization_provenance"]["source"] == "analytic mock truth gradient"
    )
    assert not (tmp_path / "configuration.json").exists()
    assert not (tmp_path / "reconstruction.npz").exists()

    main(arguments)
    assert not (tmp_path / "failure.json").exists()
    assert json.loads((tmp_path / "configuration.json").read_text())["success"]
    with np.load(tmp_path / "reconstruction.npz", allow_pickle=False) as saved:
        assert np.isfinite(saved["source_values"]).all()
        assert np.isfinite(saved["potential_values"]).all()
    assert unrelated.read_text() == "Retain notes across reconstruction attempts.\n"
