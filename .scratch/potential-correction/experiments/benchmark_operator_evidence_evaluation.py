"""Measure operator evaluations using the identical frozen dense fixtures.

The existing worker owns synchronization, process isolation and timing.
Fields and objective are compared against the matching dense measurement;
approximate operator evidence is checked separately, never equated to dense.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from benchmark_evidence_evaluation import (
    ROOT, WORKER, REFERENCES, sha, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dense = json.loads((args.dense_run / "summary.json").read_text())
    assert set(dense["rows"]) == set(REFERENCES)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Matched operator versus dense GPU float64 evidence-evaluation timings on the same 200-pixel subhalo mock",
        "dense_summary_sha256": sha(args.dense_run / "summary.json"),
        "dense_measurement_directory": str(args.dense_run.resolve()),
        "driver_sha256": sha(Path(__file__)),
        "shared_driver_sha256": sha(Path(__file__).with_name("benchmark_evidence_evaluation.py")),
        "benchmark_worker_sha256": sha(WORKER),
        "timing_contract": {
            **dense["timing_contract"],
            "timed_call": "problem.fit(lambda_source, lambda_potential, backend='operator', **same_controls_and_recorded_partition)",
            "determinants": "Principal-block Cholesky approximation; no full coupled determinant; original recorded partition and PCG controls retained",
            "warm_reuse": "Fixed geometry, fixed block topology and compiled kernels are reusable; each fit resets original source/potential, rebuilds required blocks and recomputes approximate evidence",
        },
        "rows": {},
    }
    for name, (algorithm, reference) in REFERENCES.items():
        d = dense["rows"][name]
        fixture = args.dense_run / "fixtures" / f"{name}.npz"
        assert sha(fixture) == d["fixture_sha256"]
        repeats = len(d["warm_fit_seconds"])
        destination = args.output / f"{name}.json"
        print(f"Measuring operator {name}: one first fit + {repeats} warm fits", flush=True)
        subprocess.run(
            [
                sys.executable, str(WORKER), "--inputs", str(fixture),
                "--algorithm", algorithm, "--backend", "operator", "--dtype",
                "float64", "--profile", "scientific", "--repeats", str(repeats),
                "--output", str(destination),
            ],
            cwd=ROOT, check=True,
        )
        result = json.loads(destination.read_text())
        timing = result["timing"]
        statuses = [timing["cold_status"], *timing["warm_status"]]
        assert all(row["success"] for row in statuses), "Failed endpoints are not speed results"
        assert all(math.isfinite(row["log_evidence"]) for row in statuses)
        assert result["software"]["platform"] == "gpu"
        assert result["software"]["device"] == d["software"]["device"]
        assert result["result"]["determinant_method"] == "principal_block_cholesky"
        assert result["result"]["evidence_is_exact"] is False
        assert result["geometry"]["integer_topology_matches_fixture"]
        assert result["counts"] == d["counts"]
        assert result["result"]["strengths"] == d["strengths"]
        assert len(timing["warm_fit_seconds"]) == repeats
        differences = {}
        with np.load(destination.with_suffix(".npz")) as fitted:
            with np.load((args.dense_run / name).with_suffix(".npz")) as saved:
                for key in (
                    "source_values", "potential_values", "convergence",
                    "nonlinear_image_active",
                ):
                    np.testing.assert_allclose(
                        fitted[key], saved[key], atol=1e-7, rtol=1e-7,
                    )
                    differences[key] = float(np.nanmax(np.abs(fitted[key] - saved[key])))
        dense_result = json.loads((args.dense_run / f"{name}.json").read_text())
        objective_error = max(
            abs(row["objective"] - dense_result["result"]["objective"])
            for row in statuses
        )
        assert objective_error < 1e-3
        evidence_spread = max(
            abs(row["log_evidence"] - statuses[0]["log_evidence"])
            for row in statuses
        )
        assert evidence_spread < 1e-3
        reference_evidence_error = None
        if name.startswith("original_"):
            operator_reference = reference.with_name("operator")
            reference_metadata = json.loads((operator_reference / "result.json").read_text())
            assert reference_metadata["success"]
            assert reference_metadata["strengths"] == result["result"]["strengths"]
            reference_evidence_error = max(
                abs(row["log_evidence"] - reference_metadata["log_evidence"])
                for row in statuses
            )
            assert reference_evidence_error < 1e-3
        warm = np.asarray(timing["warm_fit_seconds"])
        row = {
            "algorithm": algorithm, "backend": "operator", "dtype": "float64",
            "strengths": result["result"]["strengths"],
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
            "dense_warm_median_seconds": d["warm_median_seconds"],
            "dense_over_operator_warm_ratio": d["warm_median_seconds"] / float(np.median(warm)),
            "accepted_iterations": result["result"]["configuration"].get("accepted_iterations"),
            "termination_reason": result["result"]["termination_reason"],
            "evidence_method": result["result"]["evidence_method"],
            "determinant_method": result["result"]["determinant_method"],
            "evidence_is_exact": result["result"]["evidence_is_exact"],
            "log_evidence": result["result"]["log_evidence"],
            "operator_minus_dense_log_evidence": result["result"]["log_evidence"] - dense_result["result"]["log_evidence"],
            "max_cold_warm_log_evidence_spread": evidence_spread,
            "max_evidence_error_against_existing_operator_reference": reference_evidence_error,
            "max_objective_error_against_dense": objective_error,
            "field_max_errors_against_dense": differences,
            "fixture_sha256": sha(fixture),
            "all_cold_warm_fits_successful": True,
            "software": result["software"],
        }
        report["rows"][name] = row
        write_json(args.output / "summary.json", report)
        print(json.dumps({
            "variant": name, "operator_warm_median_seconds": row["warm_median_seconds"],
            "warm_range_seconds": [row["warm_min_seconds"], row["warm_max_seconds"]],
            "first_fit_seconds": row["first_fit_compile_inclusive_seconds"],
            "dense_over_operator_warm_ratio": row["dense_over_operator_warm_ratio"],
            "accepted_iterations": row["accepted_iterations"],
            "operator_minus_dense_log_evidence": row["operator_minus_dense_log_evidence"],
        }), flush=True)


if __name__ == "__main__":
    main()
