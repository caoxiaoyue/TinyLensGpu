"""The calibration workflow preserves public fits and honest inference records."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

CONSUMER_FIXTURE = Path(__file__).parent / "fixtures/potential_correction_consumers"


def _assert_metrics(actual, expected):
    """Compare public diagnostic outputs with pre-refactor saved values."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            _assert_metrics(actual[name], value)
    elif isinstance(expected, (bool, str)) or expected is None:
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


@pytest.mark.integration
@pytest.mark.parametrize("layout", ["morton", "source_incidence"])
def test_small_calibration_exports_raw_fields_domains_and_candidate_status(
    tmp_path, layout
):
    from examples.potential_correction.calibrate_single_step import run_calibration
    from examples.potential_correction.mock import (
        MockConfig,
        SourceTruth,
        generate_mock_pair,
    )
    from examples.potential_correction.result_products import summarize_result

    configuration = {
        "mock": asdict(
            MockConfig(
                npix=12,
                dpix=0.2,
                nsub=1,
                theta_E=0.7,
                subhalo_x=0.65,
                subhalo_y=0.1,
                b_sub=0.4,
                mask_inner=0.4,
                mask_outer=1.0,
                psf_size=3,
                noise_sigma=0.03,
                source=SourceTruth(clump_scale=0.8),
            )
        ),
        "source_n": 9,
        "source_bbox": [-0.4, 0.4, -0.4, 0.4],
        "potential_spacing": 0.4,
        "lambda_amplitudes": [100.0, 1000.0],
        "source_strengths": [0.001, 0.01],
        "potential_strengths": [10.0, 1000.0],
        "log_bounds": np.log([[0.001, 0.01], [10.0, 1000.0]]).tolist(),
        "block_size": 16,
        "block_layout": layout,
        "source_tile": 4,
        "pcg_max_iter": 2000,
        "pcg_rtol": 1e-11,
        "solve_rtol": 1e-8,
        "direct_maxfun": 8,
        "direct_len_tol": 1e-2,
        "boundary_rtol": 0.1,
    }
    baseline = json.loads((CONSUMER_FIXTURE / "baseline.json").read_text())
    observations = generate_mock_pair(MockConfig.from_dict(configuration["mock"]))
    run = run_calibration(configuration, output=tmp_path)
    record = json.loads((tmp_path / "calibration.json").read_text())
    assert record["configuration"] == configuration
    assert record["block_partition"]["method"] == layout
    assert len(record["runs"]) == 4
    with np.load(tmp_path / "inputs.npz", allow_pickle=False) as inputs:
        assert np.count_nonzero(inputs["control_potential_truth"]) == 0
        np.testing.assert_array_equal(inputs["subhalo_noise"], inputs["control_noise"])
    with np.load(tmp_path / "reconstructions.npz", allow_pickle=False) as saved:
        for key, fits in run.reconstructions.items():
            row = next(row for row in record["runs"] if row["key"] == key)
            expected = baseline["calibration"][key]
            assert sorted(row["selected"]["dense"]) == expected["schema"]
            _assert_metrics(
                {name: row["selected"]["dense"][name] for name in expected["metrics"]},
                expected["metrics"],
            )
            assert row["scan"]
            assert row["order_check"]["passed"]
            assert row["failure_check"]["objective"] is None
            assert row["failure_check"]["success"] is False
            assert row["failure_check"]["failure_reason"] == "pcg_iteration_limit"
            for backend in ("dense", "operator"):
                fit = fits[backend]
                assert fit.success
                assert row["selected"][backend]["optimizer_success"] is False
                assert row["selected"][backend]["boundary_sides"]
                assert row["selected"][backend]["determinant_method"] == (
                    "full_cholesky"
                    if backend == "dense"
                    else "principal_block_cholesky"
                )
                assert fit.strengths.lambda_amplitude == row["lambda_amplitude"]
                assert (
                    row["selected"][backend]["initialization_provenance"]
                    == fit.initialization_provenance
                )
                for field in (
                    "source_values",
                    "potential_values",
                    "convergence",
                    "linear_model_image",
                    "nonlinear_model_image",
                    "derivative_valid",
                ):
                    np.testing.assert_array_equal(
                        saved[f"{key}_{backend}_{field}"], getattr(fit, field)
                    )
                assert np.isnan(
                    saved[f"{key}_{backend}_convergence"][~fit.derivative_valid]
                ).all()
                case = key.split("_", 1)[0]
                summary, _ = summarize_result(
                    fit, getattr(observations, case), affine_diagnostics=True
                )
                calibration = row["selected"][backend]
                for name in (
                    "source_relative_l2",
                    "source_rmse",
                    "potential_rmse_raw_center_zero_truth",
                    "potential_rmse_mean_centered_truth",
                    "potential_rms",
                    "potential_mean",
                    "deflection_vector_rmse_discrete_truth",
                ):
                    assert calibration[name] == pytest.approx(summary[name], abs=1e-14)
                for domain, metrics in calibration["convergence"].items():
                    _assert_metrics(
                        {
                            name: summary["convergence"][domain][name]
                            for name in metrics
                        },
                        metrics,
                    )


@pytest.mark.parametrize("case", ["subhalo", "control"])
def test_saved_public_results_keep_pre_review_truth_domains_and_recovery(case):
    from examples.potential_correction.mock import MockConfig, generate_mock_pair
    from examples.potential_correction.result_products import (
        load_result,
        summarize_result,
        truth_products,
    )

    baseline = json.loads((CONSUMER_FIXTURE / "baseline.json").read_text())
    observation = getattr(
        generate_mock_pair(MockConfig.from_dict(baseline["configuration"]["mock"])),
        case,
    )
    result = load_result(CONSUMER_FIXTURE / case)
    summary, truth = summarize_result(result, observation, affine_diagnostics=True)
    _assert_metrics(summary, baseline["scientific"][case])
    products, local_radius = truth_products(result, observation)
    assert local_radius == pytest.approx(1.058300524425836)
    assert local_radius > 2 * result.potential_grid.spacing
    assert {name: row["count"] for name, row in summary["convergence"].items()} == {
        "all_core": 37,
        "illuminated_core": 15,
        "local_core": 18,
    }
    if case == "control":
        assert observation.subhalo_truth.b_sub == 0.0
        assert observation.config.b_sub == 0.4
    with np.load(
        CONSUMER_FIXTURE / case / "reconstruction.npz", allow_pickle=False
    ) as saved:
        for name, values in truth.items():
            np.testing.assert_allclose(values, saved[name], rtol=2e-12, atol=2e-14)
        for name, values in products.items():
            np.testing.assert_array_equal(values, truth[name])
        valid = np.asarray(result.derivative_valid)
        for name in (
            "truth_discrete_deflection_x",
            "truth_discrete_deflection_y",
            "truth_discrete_convergence",
            "truth_analytic_deflection_x",
            "truth_analytic_deflection_y",
            "truth_analytic_convergence",
        ):
            assert np.isnan(products[name][~valid]).all()
        # Consumer diagnostics preserve the signed fit and its image/evidence.
        assert np.any(result.source_values < 0)
        assert np.any(result.potential_values < 0)
        for name in (
            "source_values",
            "potential_values",
            "linear_model_image",
            "nonlinear_model_image",
        ):
            np.testing.assert_array_equal(getattr(result, name), saved[name])
    metadata = json.loads((CONSUMER_FIXTURE / case / "result.json").read_text())
    assert float(result.log_evidence) == metadata["log_evidence"]
