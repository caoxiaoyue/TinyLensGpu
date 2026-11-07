"""Matched L-BFGS-B mock fits through the common result products."""

from dataclasses import asdict
import json

import numpy as np
import pytest


def _configuration():
    from examples.potential_correction.mock import MockConfig, SourceTruth

    return {
        "mock": asdict(
            MockConfig(
                npix=12,
                dpix=0.2,
                nsub=2,
                theta_E=0.7,
                b_sub=0.03,
                subhalo_x=0.65,
                subhalo_y=0.1,
                mask_inner=0.4,
                mask_outer=1.0,
                psf_size=3,
                noise_sigma=0.03,
                source=SourceTruth(sigma_x=0.16, sigma_y=0.12, clump_scale=0.7),
            )
        ),
        "source_n": 9,
        "source_bbox": [-0.4, 0.4, -0.4, 0.4],
        "potential_spacing": 0.4,
        "source_regularization": "first-order",
        "strengths": {
            "lambda_source": 0.001,
            "lambda_potential": 10000.0,
            "lambda_amplitude": 10000.0,
        },
        "make_plots": False,
    }


def test_matched_dense_mock_round_trips_fields_and_qualification(tmp_path):
    from examples.potential_correction.lbfgs_mock import run_lbfgs_mock
    from examples.potential_correction.result_products import load_result

    config = _configuration()
    run = run_lbfgs_mock(config, output=tmp_path)

    assert run.record["success"]
    assert set(run.results) == {"subhalo/dense", "control/dense"}
    assert json.loads((tmp_path / "configuration.json").read_text()) == config
    for key, result in run.results.items():
        output = tmp_path / key
        record = json.loads((output / "result.json").read_text())
        assert record["algorithm"] == "lbfgs"
        assert record["evidence_method"] == "local_gauss_newton"
        assert record["determinant_method"] == "full_data_space"
        assert record["evidence_image"] == "nonlinear_model_image"
        assert record["evidence_is_exact"] is False
        assert record["solver_info"] is None
        assert record["strengths"] == config["strengths"]
        assert record["optimizer_info"]["native_success"]
        assert record["optimizer_info"]["gn_stationarity"] <= 1e-3

        restored = load_result(output)
        assert restored.solver_info is None
        assert restored.optimizer_info == result.optimizer_info
        assert restored.configuration == result.configuration
        assert restored.strengths == result.strengths
        assert restored.log_evidence == result.log_evidence
        assert restored.linear_model_image is restored.linear_residual is None
        assert restored.determinant_method == result.determinant_method
        for name in (
            "source_values",
            "potential_values",
            "deflection_x",
            "deflection_y",
            "convergence",
            "derivative_valid",
            "nonlinear_model_image",
            "nonlinear_residual",
        ):
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(result, name)
            )
        np.testing.assert_array_equal(
            restored.potential_grid.node_indices, result.potential_grid.node_indices
        )

        with np.load(output / "reconstruction.npz", allow_pickle=False) as arrays:
            valid, selected = arrays["derivative_valid"], ~arrays["mask"]
            assert np.isnan(arrays["convergence"][~valid]).all()
            delta = (
                arrays["convergence"][valid]
                - arrays["truth_discrete_convergence"][valid]
            )
            recovery = record["summary"]["convergence"]["all_core"]
            assert recovery["count"] == valid.sum()
            np.testing.assert_allclose(
                recovery["rmse_discrete_truth"], np.sqrt(np.mean(delta**2))
            )
            normalized = (arrays["image_data"] - arrays["nonlinear_model_image"])[
                selected
            ]
            normalized /= arrays["noise_map"][selected]
            np.testing.assert_allclose(
                record["summary"]["nonlinear_residual_rms_noise"],
                np.sqrt(np.mean(normalized**2)),
            )
    with (
        np.load(
            tmp_path / "subhalo/dense/reconstruction.npz", allow_pickle=False
        ) as subhalo,
        np.load(
            tmp_path / "control/dense/reconstruction.npz", allow_pickle=False
        ) as control,
    ):
        for name in (
            "noise_map",
            "noise_realization",
            "mask",
            "psf_kernel",
            "derivative_valid",
        ):
            np.testing.assert_array_equal(subhalo[name], control[name])
        assert np.any(
            subhalo["truth_discrete_convergence"][subhalo["derivative_valid"]] != 0
        )
        assert np.all(
            control["truth_discrete_convergence"][control["derivative_valid"]] == 0
        )


def test_unqualified_mock_exports_native_failure_and_withholds_evidence(tmp_path):
    from examples.potential_correction.lbfgs_mock import run_lbfgs_mock
    from examples.potential_correction.result_products import load_result

    config = _configuration()
    config["fit_options"] = {"max_iter": 1}
    run = run_lbfgs_mock(config, output=tmp_path)

    assert run.record["success"] is False
    for key, result in run.results.items():
        output = tmp_path / key
        failure = json.loads((output / "failure.json").read_text())
        assert failure["success"] is False
        assert failure["failure_reason"] == result.failure_reason
        assert failure["failure_reason"] == "native_optimizer_did_not_succeed"
        assert failure["log_evidence"] is None
        assert failure["solver_info"] is None
        assert failure["optimizer_info"] == asdict(result.optimizer_info)
        assert failure["optimizer_info"]["native_success"] is False
        assert failure["optimizer_info"]["iterations"] == 1
        assert not (output / "reconstruction.npz").exists()
        with pytest.raises(ValueError, match="failure record"):
            load_result(output)
