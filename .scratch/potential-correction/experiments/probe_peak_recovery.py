"""Controlled public-fitter ablations on matched scientific subhalo/null data.

Run in tinylens_gpu with XLA_PYTHON_CLIENT_PREALLOCATE=false. Each candidate
starts from the same truth-derived source nodes and zero potential. Raw data
and default macro/PSF/noise/geometry match the scientific demo exactly.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import IterativePotentialCorrection, SingleStepPotentialCorrection
from examples.potential_correction.mock import MockConfig, generate_mock_pair
from examples.potential_correction.result_products import export_result, write_json


def numbers(text):
    return [float(value) for value in text.split(",")]


def label(value):
    return f"{value:.8g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["single-step", "iterative"], default="single-step")
    parser.add_argument("--potentials", type=numbers, default=[1.6e7])
    parser.add_argument("--sources", type=numbers, default=[6e-4])
    parser.add_argument("--amplitude", type=float, default=1e4)
    parser.add_argument("--source-n", type=int, default=49)
    parser.add_argument("--spacing", type=float, default=0.075)
    parser.add_argument("--data", choices=["analytic", "discrete", "linear"], default="analytic")
    parser.add_argument("--noise-factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cases", nargs="+", default=["subhalo", "control"])
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--gradient-atol", type=float, default=5e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data == "linear" and args.algorithm != "single-step":
        parser.error("Linear synthetic data are a single-step-only diagnostic")
    config = json.loads((ROOT / "examples/potential_correction/iterative_scientific_configuration.json").read_text())
    config.update(source_n=args.source_n, potential_spacing=args.spacing, make_plots=False)
    config["mock"]["seed"] = args.seed
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "arguments.json", vars(args))
    implementation = [
        ROOT / "TinyLensGpu/ObservationModel/LensImage/potential_correction.py",
        ROOT / "TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py",
        ROOT / "TinyLensGpu/ForwardSimulation/LensImage/potential_correction.py",
        ROOT / "TinyLensGpu/utils/inversion/potential_regularization.py",
        Path(__file__),
    ]
    record = {
        "configuration": config,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation},
        "fits": [],
    }
    with jax.experimental.enable_x64():
        mock = MockConfig.from_dict(config["mock"])
        pair = generate_mock_pair(mock)
        renderer = PotentialCorrectionRenderer(
            pair.control.macro,
            SimulatorConfig(mock.dpix, mock.npix, pair.control.psf_kernel, config["inversion_nsub"], pair.control.mask),
            args.source_n, tuple(config["source_bbox"]),
            potential_spacing=args.spacing, dtype="float64",
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        source = pair.control.source_truth.brightness(sx, sy).ravel()
        grid = renderer.potential_grid
        for case in args.cases:
            observation = getattr(pair, case)
            psi = observation.subhalo_truth.potential(grid.x, grid.y)
            if args.data == "analytic":
                noiseless = observation.model_image
            elif args.data == "discrete":
                noiseless = renderer.render(source, psi).model_image
            else:
                linear = SingleStepPotentialCorrection(renderer, observation.image_data, observation.noise_map,
                    observation.source_truth.gradient, lambda_amplitude=args.amplitude)
                noiseless = linear.forward_image(jnp.concatenate((source, psi)))
            observation = replace(observation, model_image=noiseless,
                                  noise_realization=observation.noise_realization * args.noise_factor,
                                  image_data=noiseless + observation.noise_realization * args.noise_factor)
            common = dict(lambda_amplitude=args.amplitude, solve_rtol=config["solve_rtol"])
            if args.algorithm == "single-step":
                problem = SingleStepPotentialCorrection(renderer, observation.image_data, observation.noise_map,
                                                       observation.source_truth.gradient, **common)
            else:
                problem = IterativePotentialCorrection(renderer, observation.image_data, observation.noise_map,
                                                       source, solve_atol=config["solve_atol"], **common)
            record.setdefault("truth_penalties", {})[case] = {
                "d4_norm2": float(jnp.vdot(problem.potential_regularizer.forward(psi), problem.potential_regularizer.forward(psi))),
                "amplitude_norm2_center_zero": float(jnp.vdot(psi, psi)),
                "amplitude_norm2_mean_zero": float(jnp.vdot(psi - jnp.mean(psi), psi - jnp.mean(psi))),
            }
            valid = np.asarray(grid.derivative_valid)
            truth = np.asarray(renderer.render(source, psi).convergence)[valid]
            peak_index = int(np.argmax(truth))
            radius = max(2 * np.sqrt(mock.b_sub * mock.theta_E), 2 * args.spacing)
            local = (np.asarray(grid.x)[valid] - mock.subhalo_x)**2 + (np.asarray(grid.y)[valid] - mock.subhalo_y)**2 <= radius**2
            for ls in args.sources:
                for lp in args.potentials:
                    key = f"ls-{label(ls)}_lp-{label(lp)}/{case}/dense"
                    options = {} if args.algorithm == "single-step" else {"max_iter": args.max_iter, "gradient_atol": args.gradient_atol}
                    start = time.perf_counter()
                    result = problem.fit(ls, lp, **options)
                    jax.block_until_ready(result.potential_values)
                    row = {
                        "key": key, "case": case, "lambda_source": ls, "lambda_potential": lp,
                        "lambda_amplitude": args.amplitude, "success": result.success,
                        "termination": result.termination_reason, "objective": float(result.objective),
                        "log_evidence": None if result.log_evidence is None else float(result.log_evidence),
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                    recovered = np.asarray(result.convergence)[valid]
                    row.update(
                        diagnostic_endpoint=not result.success,
                        kappa_peak=float(np.nanmax(recovered)),
                        kappa_max_abs=float(np.nanmax(np.abs(recovered))),
                        kappa_rms=float(np.sqrt(np.nanmean(recovered**2))),
                        kappa_local_rmse=float(np.sqrt(np.nanmean((recovered[local] - truth[local])**2))),
                        kappa_background_rms=float(np.sqrt(np.nanmean(recovered[~local]**2))),
                        residual_rms_noise=float(np.sqrt(np.mean(((np.asarray(result.nonlinear_image_active) - np.asarray(result.data_active)) / np.asarray(result.noise_active))**2))),
                    )
                    if case == "subhalo":
                        row.update(truth_kappa_peak=float(truth[peak_index]),
                                   recovery_at_truth_peak=float(recovered[peak_index] / truth[peak_index]))
                    export_result(result, observation, args.output / key, configuration={**config, "peak_recovery_probe": record["arguments"]}, make_plots=False)
                    record["fits"].append(row)
                    write_json(args.output / "summary.json", record)
                    print(json.dumps(row), flush=True)
    print(json.dumps({"summary": str(args.output / "summary.json"), "fits": len(record["fits"])}))


if __name__ == "__main__":
    main()
