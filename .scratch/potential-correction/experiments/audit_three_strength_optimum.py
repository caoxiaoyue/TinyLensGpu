"""Check local evidence curvature and the constant-potential gauge numerically."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from optimize_single_step_three_strengths import (
    Objective, PotentialCorrectionRenderer, SimulatorConfig,
    SingleStepPotentialCorrection, MockConfig, generate_mock_pair, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    saved = json.loads((args.run / "summary.json").read_text())
    config = saved["configuration"]
    with jax.experimental.enable_x64():
        mock = MockConfig.from_dict(config["mock"])
        pair = generate_mock_pair(mock)
        case = config["three_strength_experiment"]["case"]
        observation = getattr(pair, case)
        renderer = PotentialCorrectionRenderer(
            pair.control.macro,
            SimulatorConfig(mock.dpix, mock.npix, pair.control.psf_kernel,
                            config["inversion_nsub"], pair.control.mask),
            config["source_n"], tuple(config["source_bbox"]),
            potential_spacing=config["potential_spacing"], dtype="float64",
        )
        problem = SingleStepPotentialCorrection(
            renderer, observation.image_data, observation.noise_map,
            observation.source_truth.gradient, lambda_amplitude=config["lambda_amplitude"],
            solve_rtol=config["solve_rtol"],
        )
        trace = args.run / "local-audit-evaluations.jsonl"
        if trace.exists():
            raise FileExistsError(trace)
        objective = Objective(problem, trace)
        objective.phase = "local_curvature_audit"
        x = np.asarray(saved["selected"]["log10_strengths"])
        center = objective(x)
        report = dict(center_log10_strengths=x.tolist(),
                      repeat_log_evidence_error=float(-center-saved["selected"]["log_evidence"]),
                      finite_difference_coordinates="log10 strengths", curvatures=[])
        for step in (0.01, 0.02):
            hessian = np.zeros((3, 3))
            gradient = np.zeros(3)
            for i in range(3):
                ei = np.eye(3)[i] * step
                plus, minus = objective(x + ei), objective(x - ei)
                gradient[i] = (plus - minus) / (2*step)
                hessian[i, i] = (plus - 2*center + minus) / step**2
                for j in range(i):
                    ej = np.eye(3)[j] * step
                    hessian[i, j] = hessian[j, i] = (
                        objective(x+ei+ej) - objective(x+ei-ej)
                        - objective(x-ei+ej) + objective(x-ei-ej)
                    ) / (4*step**2)
            report["curvatures"].append(dict(
                step_dex=step, negative_log_evidence_gradient=gradient.tolist(),
                negative_log_evidence_hessian=hessian.tolist(),
                hessian_eigenvalues=np.linalg.eigvalsh(hessian).tolist(),
            ))
        constant = jnp.ones(problem.n_potential) / np.sqrt(problem.n_potential)
        joint = jnp.concatenate([jnp.zeros(problem.n_source), constant])
        response = problem.dense_mapping() @ joint
        report["constant_potential_gauge"] = dict(
            unit_norm_mode=True, max_absolute_image_response=float(jnp.max(jnp.abs(response))),
            fourth_difference_norm=float(jnp.linalg.norm(problem.potential_regularizer.forward(constant))),
            expected_prior_precision="lambda_amplitude; exact normalized evidence cancels this unobserved scalar mode",
        )
        report["evaluations"] = len(objective.rows)
        report["failed_evaluations"] = sum(not row["success"] for row in objective.rows)
        report["interpretation"] = "Local maximum diagnostic, not a global certificate or hyperparameter posterior interval"
        write_json(args.run / "local-audit.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
