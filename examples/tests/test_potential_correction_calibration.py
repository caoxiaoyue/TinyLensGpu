"""The calibration workflow preserves public fits and honest inference records."""

from dataclasses import asdict
import json

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.parametrize("layout", ["morton", "source_incidence"])
def test_small_calibration_exports_raw_fields_domains_and_candidate_status(
    tmp_path, layout
):
    from examples.potential_correction.calibrate_single_step import run_calibration
    from examples.potential_correction.mock import MockConfig, SourceTruth

    configuration = {
        "mock": asdict(
            MockConfig(
                npix=12,
                dpix=0.2,
                nsub=1,
                theta_E=0.7,
                subhalo_x=0.65,
                subhalo_y=0.1,
                b_sub=0.01,
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
