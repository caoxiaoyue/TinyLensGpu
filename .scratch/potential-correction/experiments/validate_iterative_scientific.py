"""Reproduce matched iterative scientific fits and compact public-result checks.

No timing is measured. Use tinylens_gpu and preallocation=false. Run once
with the recorded64 configuration before the200 production reference.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np

from examples.potential_correction.iterative_scientific import run_iterative_scientific
from examples.potential_correction.result_products import write_json


def compact_record(run):
    record = {k: v for k, v in run.record.items() if k != "runs"}
    record["block_partition"] = {
        k: v for k, v in record["block_partition"].items() if k != "indices"
    }
    record["runs"] = {}
    for key, result in run.results.items():
        row = dict(run.record["runs"][key])
        row["configuration"] = {
            k: v for k, v in result.configuration.items() if k != "block_indices"
        }
        row["objective"] = float(result.objective)
        row["chi2"] = float(result.chi2)
        row["regularization_penalty"] = float(result.regularization_penalty)
        row["prior_logdet"] = float(result.prior_logdet)
        row["posterior_logdet"] = float(result.posterior_logdet)
        solves = [h["solver_info"] for h in result.history if "solver_info" in h]
        if result.backend == "operator":
            row["inner_solve_audit"] = {
                "residual_correction_attempts": sum(
                    s["diagnostics"]["refinement_attempted"] for s in solves
                ),
                "residual_corrections_selected": sum(
                    s["diagnostics"]["refinement_accepted"] for s in solves
                ),
                "iteration_budget_per_solve": result.configuration["pcg_max_iter"],
                "maximum_aggregate_iterations_one_solve": max(
                    (s["iterations"] for s in solves), default=0
                ),
                "total_inner_iterations_all_trials": sum(
                    s["iterations"] for s in solves
                ),
                "all_phase_iteration_sums_match": all(
                    s["iterations"]
                    == sum(p["iterations"] for p in s["diagnostics"]["phases"])
                    for s in solves
                ),
                "all_aggregate_iterations_within_budget": all(
                    s["iterations"] <= s["diagnostics"]["iteration_budget"]
                    for s in solves
                ),
                "all_true_residual_certificates_pass": all(
                    s["residual_norm"] <= s["diagnostics"]["true_residual_threshold"]
                    for s in solves
                ),
            }
        row["maximum_true_residual_norm"] = max(
            (s["residual_norm"] for s in solves), default=0
        )
        row["maximum_true_relative_residual"] = max(
            (s["relative_residual"] for s in solves), default=0
        )
        record["runs"][key] = row
    record["backend_comparison"] = {}
    for case in ("subhalo", "control"):
        dense, operator = (
            run.results.get(f"{case}/{b}") for b in ("dense", "operator")
        )
        if dense is None or operator is None:
            continue
        if not (dense.success and operator.success):
            record["backend_comparison"][case] = {"both_successful": False}
            continue
        valid = np.asarray(dense.derivative_valid)
        comparison = {"both_successful": True}
        for field in (
            "source_values",
            "potential_values",
            "deflection_x",
            "deflection_y",
            "convergence",
        ):
            left, right = np.asarray(getattr(dense, field)), np.asarray(
                getattr(operator, field)
            )
            domain = (
                valid
                if field in ("deflection_x", "deflection_y", "convergence")
                else np.ones(len(left), dtype=bool)
            )
            delta = left[domain] - right[domain]
            comparison[field] = {
                "max_absolute_difference": float(np.max(np.abs(delta))),
                "rms_difference": float(np.sqrt(np.mean(delta**2))),
            }
        comparison["nonlinear_image_rms_difference_noise"] = float(
            np.sqrt(
                np.mean(
                    (
                        (
                            np.asarray(dense.nonlinear_image_active)
                            - np.asarray(operator.nonlinear_image_active)
                        )
                        / np.asarray(dense.noise_active)
                    )
                    ** 2
                )
            )
        )
        comparison["operator_minus_dense_objective"] = float(
            operator.objective
        ) - float(dense.objective)
        comparison["operator_minus_dense_log_evidence"] = float(
            operator.log_evidence
        ) - float(dense.log_evidence)
        comparison["determinant_contribution_to_log_evidence_difference"] = -0.5 * (
            (float(operator.posterior_logdet) - float(dense.posterior_logdet))
            - (float(operator.prior_logdet) - float(dense.prior_logdet))
        )
        comparison["non_determinant_contribution_to_log_evidence_difference"] = -(
            float(operator.objective) - float(dense.objective)
        )
        record["backend_comparison"][case] = comparison
    return record


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "examples/potential_correction/iterative_scientific_configuration.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run = run_iterative_scientific(config, output=args.output)
    record = compact_record(run)
    write_json(args.summary, record)
    print(
        json.dumps(
            {
                "success": record["success"],
                "summary": str(args.summary),
                "sizes": record["sizes"],
            },
            indent=2,
        )
    )
    if not record["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
