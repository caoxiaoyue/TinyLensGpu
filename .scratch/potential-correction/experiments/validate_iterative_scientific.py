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


def terminal_gradient_audit(result, observation, configuration):
    """Independent public current-J/Q diagnostic, separate from fit/export code."""
    import jax
    import jax.numpy as jnp
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection

    with jax.experimental.enable_x64(result.dtype == "float64"):
        mock = observation.config
        renderer = PotentialCorrectionRenderer(
            observation.macro,
            SimulatorConfig(
                mock.dpix,
                mock.npix,
                observation.psf_kernel,
                configuration["inversion_nsub"],
                observation.mask,
            ),
            configuration["source_n"],
            tuple(configuration["source_bbox"]),
            potential_spacing=configuration["potential_spacing"],
            dtype=mock.dtype,
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        problem = IterativePotentialCorrection(
            renderer,
            observation.image_data,
            observation.noise_map,
            observation.source_truth.brightness(sx, sy).ravel(),
            lambda_amplitude=configuration["lambda_amplitude"],
            source_regularization=configuration.get(
                "source_regularization", "second-order"
            ),
        )
        values = jnp.concatenate((result.source_values, result.potential_values))
        current = problem.linearize(values)
        ls, lp = result.strengths.lambda_source, result.strengths.lambda_potential
        rhs = problem.jacobian_adjoint(
            (result.data_active - result.nonlinear_image_active)
            / result.noise_active**2,
            current,
        ) - problem.prior_matvec(values, ls, lp)
        n = len(values)

        @jax.jit
        def panel_diagonal(indices, linearization):
            basis = jax.nn.one_hot(indices, n, dtype=values.dtype)
            panel = jax.vmap(lambda v: problem.jacobian_forward(v, linearization))(
                basis
            )
            prior = jax.vmap(lambda v: problem.prior_matvec(v, ls, lp))(basis)
            return (
                jnp.sum((panel / result.noise_active[None, :]) ** 2, axis=1)
                + prior[jnp.arange(len(indices)), indices]
            )

        pieces = [
            panel_diagonal(
                jnp.arange(start, min(start + 16, n), dtype=jnp.int32), current
            )
            for start in range(0, n, 16)
        ]
        gradient = jnp.max(jnp.abs(rhs) / jnp.sqrt(jnp.concatenate(pieces)))
        return float(gradient)


def save_diagnostic_terminals(run):
    """Keep failed numerical endpoints explicitly diagnostic, never result exports."""
    arrays = {}
    manifest = {
        "scope": "FAILED terminal arrays for numerical stopping calibration only; no successful scientific reconstruction",
        "runs": {},
    }
    for key, result in run.results.items():
        if result.success:
            continue
        label = key.replace("/", "_")
        for field in (
            "source_values",
            "source_x_axis",
            "source_y_axis",
            "potential_values",
            "deflection_x",
            "deflection_y",
            "convergence",
            "derivative_valid",
            "nonlinear_image_active",
            "nonlinear_model_image",
            "data_active",
            "noise_active",
        ):
            arrays[f"{label}_{field}"] = np.asarray(getattr(result, field))
        manifest["runs"][key] = {
            "success": False,
            "termination_reason": result.termination_reason,
            "objective": float(result.objective),
            "log_evidence": None,
        }
    if arrays:
        np.savez_compressed(run.output / "diagnostic-failed-terminals.npz", **arrays)
        write_json(run.output / "diagnostic-failed-terminals.json", manifest)


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
    parser.add_argument("--gradient-atol", type=float)
    parser.add_argument("--solve-atol", type=float)
    parser.add_argument("--lambda-amplitude", type=float)
    parser.add_argument("--case", choices=("subhalo", "control"))
    parser.add_argument("--backend", choices=("dense", "operator", "both"))
    parser.add_argument("--single-step-reference", type=Path)
    parser.add_argument("--audit-gradients", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.gradient_atol is not None:
        config.setdefault("iteration_options", {})["gradient_atol"] = args.gradient_atol
    if args.solve_atol is not None:
        config["solve_atol"] = args.solve_atol
    if args.lambda_amplitude is not None:
        config["lambda_amplitude"] = args.lambda_amplitude
    if args.no_plots:
        config["make_plots"] = False
    if args.case:
        config["cases"] = [args.case]
    if args.backend:
        config["backends"] = (
            ["dense", "operator"] if args.backend == "both" else [args.backend]
        )
    run = run_iterative_scientific(
        config, output=args.output, single_step_reference=args.single_step_reference
    )
    record = compact_record(run)
    save_diagnostic_terminals(run)
    if args.audit_gradients:
        from examples.potential_correction.mock import MockConfig, generate_mock_pair

        pair = generate_mock_pair(MockConfig.from_dict(config["mock"]))
        for key, result in run.results.items():
            record["runs"][key]["independent_terminal_gradient"] = (
                terminal_gradient_audit(
                    result, getattr(pair, key.split("/")[0]), config
                )
            )
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
