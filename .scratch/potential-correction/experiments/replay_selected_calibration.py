"""Persist fresh public fits at the selected candidates of a completed run.

The original scalar scan/DIRECT record is preserved. This is an independent
fixed-strength replay with the same input, partition and numerical controls;
it does not claim access to unsaved original arrays or reuse fitted state.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import jax
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from examples.potential_correction.calibrate_single_step import (
    _fit_record,
    _json_safe,
    _truth,
    _compare,
)
from examples.potential_correction.mock import MockConfig, generate_mock_pair


def replay(record, run_key, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = record["configuration"]
    row = next(row for row in record["runs"] if row["key"] == run_key)
    pair = generate_mock_pair(MockConfig.from_dict(config["mock"]))
    observation = getattr(pair, row["case"])
    renderer = PotentialCorrectionRenderer(
        observation.macro,
        observation.sim_config,
        config["source_n"],
        tuple(config["source_bbox"]),
        potential_spacing=config["potential_spacing"],
        dtype="float64",
    )
    problem = SingleStepPotentialCorrection(
        renderer,
        observation.image_data,
        observation.noise_map,
        observation.source_truth.gradient,
        lambda_amplitude=row["lambda_amplitude"],
        solve_rtol=config["solve_rtol"],
        initialization_provenance={
            "source": "analytic mock truth gradient; fixed macro rays"
        },
    )
    options = dict(
        block_size=config["block_size"],
        block_indices=record["block_partition"]["indices"],
        pcg_max_iter=config["pcg_max_iter"],
        pcg_rtol=config["pcg_rtol"],
        pcg_atol=0.0,
    )
    fitted, arrays, checks = {}, {}, {}
    truth = _truth(renderer, observation)
    for backend in ("dense", "operator"):
        original = row["selected"][backend]
        strength = original["strengths"]
        result = problem.fit(
            strength["lambda_source"],
            strength["lambda_potential"],
            backend=backend,
            **(options if backend == "operator" else {}),
        )
        if not result.success:
            raise RuntimeError(f"Replay failed: {backend}: {result.failure_reason}")
        summary = _fit_record(result, truth, observation)
        checks[backend] = {
            field: summary[field] - original[field]
            for field in (
                "log_evidence",
                "chi2",
                "regularization_penalty",
                "source_relative_l2",
                "potential_rms",
                "linear_residual_rms_noise",
                "nonlinear_residual_rms_noise",
            )
        }
        row["selected"][backend].update(summary)
        fitted[backend] = result
    dense, operator = fitted["dense"], fitted["operator"]
    fitted["dense_at_operator"] = problem.fit(
        operator.strengths.lambda_source, operator.strengths.lambda_potential
    )
    fitted["operator_at_dense"] = problem.fit(
        dense.strengths.lambda_source,
        dense.strengths.lambda_potential,
        backend="operator",
        **options,
    )
    row["at_dense_selected_strengths"] = _compare(dense, fitted["operator_at_dense"])
    row["at_operator_selected_strengths"] = _compare(
        fitted["dense_at_operator"], operator
    )
    row["selected_field_comparison"] = _compare(dense, operator)
    row["dense_at_operator"] = _fit_record(
        fitted["dense_at_operator"], truth, observation
    )
    row["dense_selected_minus_complete_at_operator_log_evidence"] = float(
        dense.log_evidence - fitted["dense_at_operator"].log_evidence
    )
    row["raw_artifact_provenance"] = dict(
        method="fresh fixed-strength public fits at the original selected candidates",
        original_scan_and_search_preserved=True,
        original_unsaved_arrays_not_claimed=True,
        scalar_replay_differences=checks,
    )
    for backend, result in fitted.items():
        for field in (
            "source_values",
            "source_x_axis",
            "source_y_axis",
            "potential_values",
            "deflection_x",
            "deflection_y",
            "convergence",
            "derivative_valid",
            "potential_window_coverage",
            "linear_model_image",
            "nonlinear_model_image",
            "linear_residual",
            "nonlinear_residual",
        ):
            arrays[f"{run_key}_{backend}_{field}"] = np.asarray(getattr(result, field))
    grid = renderer.potential_grid
    inputs = {
        f'{row["case"]}_{field}': value
        for field, value in dict(
            image=np.asarray(observation.image_data),
            noiseless_image=np.asarray(observation.model_image),
            noise=np.asarray(observation.noise_realization),
            source_truth=truth["source"],
            potential_truth=truth["potential"],
            discrete_kappa_truth=truth["kappa"],
            analytic_kappa_truth=truth["analytic_kappa"],
            illuminated_core=truth["illuminated_core"],
            local_core=truth["local_core"],
        ).items()
    }
    inputs.update(
        potential_x=np.asarray(grid.x),
        potential_y=np.asarray(grid.y),
        potential_node_indices=np.asarray(grid.node_indices),
        potential_core_indices=np.asarray(grid.core_indices),
        derivative_valid=np.asarray(grid.derivative_valid),
        noise_map=np.asarray(observation.noise_map),
        mask=np.asarray(observation.mask),
        psf=np.asarray(observation.psf_kernel),
    )
    np.savez_compressed(output / "inputs.npz", **inputs)
    np.savez_compressed(output / "reconstructions.npz", **arrays)
    result = {**record, "runs": [row]}
    (output / "calibration.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(checks), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with jax.experimental.enable_x64():
        replay(json.loads(args.record.read_text()), args.run_key, args.output)


if __name__ == "__main__":
    main()
