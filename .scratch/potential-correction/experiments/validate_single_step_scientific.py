"""Reproduce the 200x200 fixed-strength reference and independent prior probes.

Run in tinylens_gpu with XLA_PYTHON_CLIENT_PREALLOCATE=false, from repository
root. Only compact JSON summaries/configs belong in version control; output
artifacts default to the explicitly retained root worktree output location.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from examples.potential_correction.single_step_scientific import run_scientific
from examples.potential_correction.result_products import write_json
from examples.potential_correction.result_products import (
    load_result,
    plot_result_products,
)
from examples.potential_correction.mock import MockConfig, generate_mock_pair


def replot_reference(output):
    """Replot saved public results, verifying the regenerated observation context."""
    run = json.loads((output / "reference/run.json").read_text())
    pair = generate_mock_pair(MockConfig.from_dict(run["configuration"]["mock"]))
    for key in run["runs"]:
        observation = getattr(pair, key.split("/")[0])
        path = output / "reference" / key
        result = load_result(path)
        with np.load(path / "reconstruction.npz", allow_pickle=False) as saved:
            for name in ("image_data", "noise_realization", "mask", "psf_kernel"):
                np.testing.assert_array_equal(saved[name], getattr(observation, name))
        plot_result_products(result, observation, path)


def consolidate(output, destination):
    reference = json.loads((output / "reference/run.json").read_text())
    record = {
        "scope": "actual 200x200 fixed-strength reconstruction and independently fixed amplitude sensitivity; no optimized 200x200 strength or detection/mass claim",
        "configuration": reference["configuration"],
        "software": reference["software"],
        "sizes": reference["sizes"],
        "block_partition": {
            k: v for k, v in reference["block_partition"].items() if k != "indices"
        },
        "supplied_truth": reference["supplied_truth"],
        "selected_fixed_results": reference["runs"],
        "backend_comparison": {},
        "amplitude_sensitivity": [],
        "artifact_root": str(output.resolve()),
    }
    for case in ("subhalo", "control"):
        with np.load(
            output / f"reference/{case}/dense/reconstruction.npz"
        ) as dense, np.load(
            output / f"reference/{case}/operator/reconstruction.npz"
        ) as operator:
            valid = dense["derivative_valid"]
            noise = dense["noise_active"]
            differences = {
                "source_max_abs_difference": np.max(
                    np.abs(dense["source_values"] - operator["source_values"])
                ),
                "potential_max_abs_difference": np.max(
                    np.abs(dense["potential_values"] - operator["potential_values"])
                ),
                "convergence_max_abs_difference_valid_core": np.max(
                    np.abs(dense["convergence"][valid] - operator["convergence"][valid])
                ),
                "linear_model_rms_difference_noise": np.sqrt(
                    np.mean(
                        (
                            (
                                dense["linear_image_active"]
                                - operator["linear_image_active"]
                            )
                            / noise
                        )
                        ** 2
                    )
                ),
                "nonlinear_model_rms_difference_noise": np.sqrt(
                    np.mean(
                        (
                            (
                                dense["nonlinear_image_active"]
                                - operator["nonlinear_image_active"]
                            )
                            / noise
                        )
                        ** 2
                    )
                ),
            }
        dr, op = (
            reference["runs"][f"{case}/{backend}"] for backend in ("dense", "operator")
        )
        differences["operator_minus_complete_log_evidence"] = (
            op["log_evidence"] - dr["log_evidence"]
        )
        record["backend_comparison"][case] = differences
    for amplitude in (1000, 10000, 100000):
        path = (
            output
            / ("reference" if amplitude == 10000 else f"amplitude-{amplitude}")
            / "run.json"
        )
        run = json.loads(path.read_text())
        if not run["success"] or run["sizes"] != reference["sizes"]:
            raise ValueError("Incomplete or changed-geometry amplitude probe")
        expected = dict(reference["configuration"], lambda_amplitude=float(amplitude))
        for key in ("backends", "make_plots"):
            expected[key] = run["configuration"][key]
        if expected != run["configuration"]:
            raise ValueError(
                "Amplitude probe changed another physical/numerical setting"
            )
        for case in ("subhalo", "control"):
            with np.load(
                path.parent / case / "operator/reconstruction.npz", allow_pickle=False
            ) as probe, np.load(
                output / "reference" / case / "operator/reconstruction.npz",
                allow_pickle=False,
            ) as baseline:
                for name in (
                    "image_data",
                    "noise_map",
                    "noise_realization",
                    "psf_kernel",
                    "mask",
                    "source_x_axis",
                    "source_y_axis",
                    "potential_node_indices",
                    "potential_core_indices",
                    "potential_query_x",
                    "potential_query_y",
                ):
                    np.testing.assert_array_equal(probe[name], baseline[name])
        record["amplitude_sensitivity"].append(
            {
                "lambda_amplitude": amplitude,
                "configuration": run["configuration"],
                "software": run["software"],
                "sizes": run["sizes"],
                "results": {
                    key: value
                    for key, value in run["runs"].items()
                    if key.endswith("operator")
                },
            }
        )
    record["reference_recovery_checks"] = {}
    for backend in ("dense", "operator"):
        sub = reference["runs"][f"subhalo/{backend}"]["summary"]
        control = reference["runs"][f"control/{backend}"]["summary"]
        checks = {
            "peak_within_one_potential_spacing": sub[
                "peak_distance_to_injected_position_arcsec"
            ]
            < reference["configuration"]["potential_spacing"],
            "local_kappa_rmse_below_zero_field_baseline": sub["convergence"][
                "local_core"
            ]["rmse_discrete_truth"]
            < sub["convergence"]["local_core"]["truth_discrete_rms"],
            "all_core_kappa_rmse_below_zero_field_baseline": sub["convergence"][
                "all_core"
            ]["rmse_discrete_truth"]
            < sub["convergence"]["all_core"]["truth_discrete_rms"],
            "matched_control_kappa_rms_below_subhalo": control["convergence"][
                "all_core"
            ]["recovered_rms"]
            < sub["convergence"]["all_core"]["recovered_rms"],
        }
        if not all(checks.values()):
            raise ValueError(
                f"Recovery/control validation failed for {backend}: {checks}"
            )
        record["reference_recovery_checks"][backend] = checks
    record["artifact_sha256"] = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("reference", "amplitude-1000", "amplitude-100000")
        for path in sorted((output / directory).rglob("*"))
        if path.is_file()
    }
    record["figure_replot"] = {
        "source": "safe load_result from retained NPZ/JSON plus identical regenerated observation context; no refit",
        "plotting_source_sha256": hashlib.sha256(
            (ROOT / "examples/potential_correction/result_products.py").read_bytes()
        ).hexdigest(),
        "normalization": "shared observed/linear/nonlinear intensity range and shared truth/recovered source range",
        "field_domains": "psi on U; deflection and convergence on core C; cross = injected position",
    }
    write_json(destination, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "examples/potential_correction/scientific_configuration.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT
        / ".scratch/potential-correction/research/single-step-scientific/summary.json",
    )
    parser.add_argument("--reference-already-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--replot-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.replot_only:
        replot_reference(args.output)
    elif not args.summarize_only:
        if not args.reference_already_run:
            result = run_scientific(config, output=args.output / "reference")
            if not result.record["success"]:
                raise RuntimeError(
                    "Reference reconstruction failed; diagnostic failure retained"
                )
        for amplitude in (1000, 100000):
            probe = dict(
                config,
                lambda_amplitude=float(amplitude),
                backends=["operator"],
                make_plots=False,
            )
            result = run_scientific(
                probe, output=args.output / f"amplitude-{amplitude}"
            )
            if not result.record["success"]:
                raise RuntimeError(
                    "Amplitude sensitivity reconstruction failed; diagnostic failure retained"
                )
    consolidate(args.output, args.summary)


if __name__ == "__main__":
    main()
