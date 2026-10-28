"""The small fixed-strength operator example publishes public result products."""

import json

import numpy as np


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
