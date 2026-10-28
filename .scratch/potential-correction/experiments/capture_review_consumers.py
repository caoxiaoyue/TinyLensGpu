"""Capture small public consumer artifacts at an explicitly selected source tree.

The checked-in consumer fixtures were captured before the review refactor at
e14ea508. This script runs only a small 12-pixel calibration, then exports its
already selected public results; it never repeats the scientific campaign.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))

    import numpy as np
    from examples.potential_correction.calibrate_single_step import run_calibration
    from examples.potential_correction.mock import (
        MockConfig,
        SourceTruth,
        generate_mock_pair,
    )
    from examples.potential_correction.result_products import export_result

    metric_names = (
        "source_relative_l2",
        "source_rmse",
        "potential_rmse_raw_center_zero_truth",
        "potential_rmse_mean_centered_truth",
        "potential_rms",
        "potential_mean",
        "deflection_vector_rmse_discrete_truth",
        "affine_diagnostic_only",
        "convergence",
        "convergence_peak_xy",
        "peak_distance_to_subhalo",
        "corrected_ray_source_box",
        "linear_residual_rms_noise",
        "nonlinear_residual_rms_noise",
        "fitted_linearization_rms_noise",
    )
    config = {
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
        "block_layout": "morton",
        "source_tile": 4,
        "pcg_max_iter": 2000,
        "pcg_rtol": 1e-11,
        "solve_rtol": 1e-8,
        "direct_maxfun": 8,
        "direct_len_tol": 1e-2,
        "boundary_rtol": 0.1,
    }
    run = run_calibration(config, output=args.output / "calibration")
    pair = generate_mock_pair(MockConfig.from_dict(config["mock"]))
    source_paths = (
        "examples/potential_correction/calibrate_single_step.py",
        "examples/potential_correction/result_products.py",
        "examples/potential_correction/mock.py",
    )
    summary = dict(
        source_commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256={
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        configuration=config,
        calibration={
            row["key"]: {
                "schema": sorted(row["selected"]["dense"]),
                "metrics": {
                    name: row["selected"]["dense"][name] for name in metric_names
                },
            }
            for row in run.record["runs"]
        },
        scientific={},
    )
    for case in ("subhalo", "control"):
        result = run.reconstructions[f"{case}_amplitude1000"]["dense"]
        metadata = export_result(
            result,
            getattr(pair, case),
            args.output / case,
            configuration={"case": case, "mock": config["mock"]},
            make_plots=False,
            affine_diagnostics=True,
        )
        summary["scientific"][case] = metadata["summary"]
    (args.output / "baseline.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
