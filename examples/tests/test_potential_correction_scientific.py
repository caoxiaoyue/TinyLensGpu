"""Scientific single-step runs export public fields and honest diagnostics."""

from dataclasses import asdict
import json

import numpy as np


def small_configuration():
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
        "inversion_nsub": 1,
        "source_n": 9,
        "source_bbox": [-0.4, 0.4, -0.4, 0.4],
        "potential_spacing": 0.4,
        "lambda_amplitude": 1000.0,
        "fixed_strengths": {
            "subhalo": {"lambda_source": 0.001, "lambda_potential": 1000.0},
            "control": {"lambda_source": 0.001, "lambda_potential": 1000.0},
        },
        "backends": ["dense"],
        "mode": "fixed",
        "source_tile": 4,
        "block_size": 32,
        "pcg_max_iter": 2000,
        "pcg_rtol": 1e-11,
        "solve_rtol": 1e-8,
        "make_plots": True,
        "affine_diagnostics": True,
    }


def test_scientific_pair_preserves_raw_results_truth_domains_and_two_renderings(
    tmp_path,
):
    from examples.potential_correction.single_step_scientific import run_scientific

    config = small_configuration()
    run = run_scientific(config, output=tmp_path)
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["configuration"] == config
    assert record["success"] is True
    assert set(run.results) == {"subhalo/dense", "control/dense"}
    assert record["supplied_truth"]["subhalo"]["sampled_truth_vs_mock_rms_noise"] > 0
    assert (
        record["supplied_truth"]["control"][
            "analytic_source_discrete_potential_vs_inversion_truth_rms_noise"
        ]
        == 0
    )
    for key, result in run.results.items():
        output = tmp_path / key
        metadata = json.loads((output / "result.json").read_text())
        assert metadata["success"] is True
        assert metadata["evidence_image"] == "linear_model_image"
        assert metadata["evidence_method"] == "conditional_linear_gaussian"
        assert metadata["determinant_method"] == "full_cholesky"
        assert metadata["initialization_provenance"] == result.initialization_provenance
        assert metadata["integration"] == {"mock_nsub": 2, "inversion_nsub": 1}
        assert (
            metadata["summary"]["convergence"]["all_core"]["count"]
            == result.potential_grid.n_core
        )
        assert (
            metadata["summary"]["affine_diagnostic_only"][
                "applied_to_raw_fields_or_images_or_evidence"
            ]
            is False
        )
        with np.load(output / "reconstruction.npz", allow_pickle=False) as saved:
            for field in (
                "source_values",
                "potential_values",
                "convergence",
                "deflection_x",
                "deflection_y",
                "linear_model_image",
                "nonlinear_model_image",
            ):
                np.testing.assert_array_equal(saved[field], getattr(result, field))
            np.testing.assert_array_equal(
                saved["potential_core_indices"], result.potential_grid.core_indices
            )
            assert np.isnan(saved["convergence"][~saved["derivative_valid"]]).all()
            assert np.isnan(
                saved["truth_discrete_convergence"][~saved["derivative_valid"]]
            ).all()
            assert np.isnan(saved["linear_residual"][saved["mask"]]).all()
            assert (
                np.linalg.norm(
                    saved["nonlinear_model_image"] - saved["linear_model_image"]
                )
                > 1e-9
            )
            if key.startswith("control/"):
                assert np.count_nonzero(saved["truth_potential_values"]) == 0
        assert (output / "images.png").is_file()
        assert (output / "fields.png").is_file()
        assert not (output / "failure.json").exists()
    with np.load(tmp_path / "subhalo/dense/reconstruction.npz") as sub, np.load(
        tmp_path / "control/dense/reconstruction.npz"
    ) as control:
        np.testing.assert_array_equal(
            sub["noise_realization"], control["noise_realization"]
        )


def test_failed_tuning_replaces_previous_success_with_diagnostic_records_only(tmp_path):
    from examples.potential_correction.single_step_scientific import run_scientific

    config = small_configuration()
    config.update(backends=["operator"], make_plots=False)
    first = run_scientific(config, output=tmp_path)
    assert first.record["success"]
    assert (tmp_path / "subhalo/operator/reconstruction.npz").is_file()
    config.update(
        mode="tune",
        pcg_max_iter=1,
        direct_maxfun=8,
        log_bounds=np.log([[0.001, 0.01], [10.0, 1000.0]]).tolist(),
    )
    failed = run_scientific(config, output=tmp_path)
    assert failed.record["success"] is False
    for case in ("subhalo", "control"):
        destination = tmp_path / case / "operator"
        failure = json.loads((destination / "failure.json").read_text())
        assert failure["success"] is False
        assert failure["failure_reason"] == "no_successful_candidate"
        assert failure["search"]["has_valid_fit"] is False
        assert all(
            not row["success"] and row["failure_reason"] == "pcg_iteration_limit"
            for row in failure["search"]["evaluations"]
        )
        assert not (destination / "reconstruction.npz").exists()
        assert not (destination / "result.json").exists()
        assert not (destination / "images.png").exists()
