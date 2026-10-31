"""Measure one complete source/potential fit plus evidence on the current mock.

Reuse the isolated, synchronized benchmark worker. Each variant runs in its
own fresh GPU process with persistent compilation caching disabled. Setup,
compile-inclusive first fit, and repeated warm fits remain separate phases.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "benchmarks/potential_correction_workflow.py"
CANONICAL = ROOT / ".scratch/potential-correction/output"
REFERENCES = {
    "original_single_step": (
        "single_step",
        CANONICAL / "single-step-scientific/reference/subhalo/dense",
    ),
    "original_iterative": (
        "iterative",
        CANONICAL / "iterative-scientific/validated-reference/subhalo/dense",
    ),
    "selected_single_step": (
        "single_step",
        CANONICAL / "peak-recovery/selected-single-step-final/subhalo/dense",
    ),
    "selected_iterative": (
        "iterative",
        CANONICAL / "peak-recovery/selected-iterative-final/subhalo/dense",
    ),
}
ITERATION_OPTIONS = (
    "max_iter", "initial_damping", "damping_min", "damping_max",
    "damping_increase", "damping_decrease", "max_rejections",
    "objective_rtol", "step_rtol", "gradient_atol", "gradient_rtol",
    "stall_max_iter", "damping_fraction_max", "acceptance_ratio",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-repeats", type=int, default=5)
    parser.add_argument("--iterative-repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.single_repeats, args.iterative_repeats) < 1:
        parser.error("At least one warm repeat is required")
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = args.output / "fixtures"
    fixtures.mkdir()
    base = fixtures / "base.npz"
    subprocess.run(
        [
            sys.executable, str(WORKER), "--prepare", "--case", "scientific200",
            "--observation", "subhalo", "--reference",
            str(CANONICAL / "single-step-scientific/reference"),
            "--output", str(base),
        ],
        cwd=ROOT, check=True,
    )
    with np.load(base, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    base_configuration = json.loads(str(arrays["metadata"]))
    report = {
        "scope": "One fixed-strength joint fit plus evidence; dense GPU float64; subhalo observation",
        "timing_contract": {
            "timed_call": "problem.fit(lambda_source, lambda_potential, backend='dense', **options)",
            "synchronization": "All JAX arrays in the public result and grid are synchronized before stopping each timer",
            "includes": "Joint source/potential solve, prior and posterior determinants, evidence normalization and returned numerical fields; complete nonlinear convergence for iteration",
            "outside_fit_timer": "File input, imports/device/geometry setup (reported separately), plotting and file export; input simulation/partition generation is outside all worker timings",
            "warm_reuse": "Fixed geometry, frozen single-step design and compiled kernels are reusable; each fit resets initial source/potential and computes evidence again",
            "cold_limit": "Compile-inclusive first fit also includes lazy numerical preparation; first minus warm is not pure compiler time",
            "isolation": "Serial independent GPU processes; persistent JAX compilation cache disabled",
        },
        "driver_sha256": sha(Path(__file__)),
        "benchmark_worker_sha256": sha(WORKER),
        "rows": {},
    }
    for name, (algorithm, reference) in REFERENCES.items():
        metadata = json.loads((reference / "result.json").read_text())
        assert metadata["success"] and metadata["dtype"] == "float64"
        cfg = copy.deepcopy(base_configuration)
        cfg["benchmark_variant"] = name
        cfg.update(metadata["strengths"])
        cfg["fixed_strengths"]["subhalo"] = {
            key: metadata["strengths"][key]
            for key in ("lambda_source", "lambda_potential")
        }
        controls = metadata["result_configuration"]
        cfg["source_regularization"] = controls["source_regularization"]
        cfg["solve_rtol"] = controls["solve_rtol"]
        if algorithm == "iterative":
            cfg["iteration_options"] = {
                key: controls[key] for key in ITERATION_OPTIONS
            }
            cfg["solve_atol"] = controls["solve_atol"]
        cfg["case_iteration_options"] = {}
        fixture = fixtures / f"{name}.npz"
        values = {**arrays, "metadata": np.asarray(json.dumps(cfg))}
        with np.load(reference / "reconstruction.npz", allow_pickle=False) as saved:
            for key, original_key in (
                ("image_data", "image_data"), ("noise_map", "noise_map"),
                ("mask", "mask"), ("psf", "psf_kernel"),
                ("initial_source", "truth_source_values"),
            ):
                np.testing.assert_array_equal(values[key], saved[original_key])
        np.savez_compressed(fixture, **values)
        destination = args.output / f"{name}.json"
        repeats = (
            args.single_repeats if algorithm == "single_step"
            else args.iterative_repeats
        )
        print(f"Measuring {name}: one first fit + {repeats} warm fits", flush=True)
        subprocess.run(
            [
                sys.executable, str(WORKER), "--inputs", str(fixture),
                "--algorithm", algorithm, "--backend", "dense", "--dtype",
                "float64", "--profile", "scientific", "--repeats", str(repeats),
                "--output", str(destination),
            ],
            cwd=ROOT, check=True,
        )
        result = json.loads(destination.read_text())
        timing = result["timing"]
        statuses = [timing["cold_status"], *timing["warm_status"]]
        assert all(row["success"] for row in statuses), "A failed fit is not a speed result"
        assert result["software"]["platform"] == "gpu"
        assert result["result"]["determinant_method"] == "full_cholesky"
        assert result["geometry"]["integer_topology_matches_fixture"]
        assert len(timing["warm_fit_seconds"]) == repeats
        field_errors = {}
        with np.load(destination.with_suffix(".npz")) as fitted:
            with np.load(reference / "reconstruction.npz") as saved:
                for key in (
                    "source_values", "potential_values", "convergence",
                    "nonlinear_image_active",
                ):
                    np.testing.assert_allclose(
                        fitted[key], saved[key], atol=1e-7, rtol=1e-7,
                    )
                    field_errors[key] = float(
                        np.nanmax(np.abs(fitted[key] - saved[key]))
                    )
        evidence_error = max(
            abs(row["log_evidence"] - metadata["log_evidence"])
            for row in statuses
        )
        assert evidence_error < 1e-3
        warm = np.asarray(timing["warm_fit_seconds"])
        row = {
            "algorithm": algorithm, "backend": "dense", "dtype": "float64",
            "strengths": metadata["strengths"],
            "effective_fit_options": result["effective_fit_options"],
            "counts": result["counts"],
            "setup_seconds": timing["setup_seconds"],
            "first_fit_compile_inclusive_seconds": timing["first_fit_compile_inclusive_seconds"],
            "cold_workflow_seconds": timing["cold_workflow_seconds"],
            "warm_fit_seconds": warm.tolist(),
            "warm_median_seconds": float(np.median(warm)),
            "warm_min_seconds": float(warm.min()),
            "warm_max_seconds": float(warm.max()),
            "warm_mean_seconds": float(warm.mean()),
            "warm_sample_sd_seconds": float(warm.std(ddof=1)) if repeats > 1 else None,
            "accepted_iterations": result["result"]["configuration"].get("accepted_iterations"),
            "termination_reason": result["result"]["termination_reason"],
            "evidence_method": result["result"]["evidence_method"],
            "determinant_method": result["result"]["determinant_method"],
            "reference": str(reference.relative_to(ROOT)),
            "reference_metadata_sha256": sha(reference / "result.json"),
            "reference_arrays_sha256": sha(reference / "reconstruction.npz"),
            "fixture_sha256": sha(fixture),
            "all_cold_warm_fits_successful": True,
            "max_evidence_error_against_reference": evidence_error,
            "field_max_errors_against_reference": field_errors,
            "software": result["software"],
        }
        report["rows"][name] = row
        write_json(args.output / "summary.json", report)
        print(json.dumps({"variant": name, "warm_median_seconds": row["warm_median_seconds"],
                          "warm_range_seconds": [row["warm_min_seconds"], row["warm_max_seconds"]],
                          "first_fit_seconds": row["first_fit_compile_inclusive_seconds"],
                          "accepted_iterations": row["accepted_iterations"]}), flush=True)


if __name__ == "__main__":
    main()
