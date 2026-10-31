"""Exact frozen-linear decomposition of prior, representation, and noise effects.

This diagnostic does not claim an exact nonlinear bias decomposition. It uses
the public single-step design/prior and verifies superposition against its fit.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from examples.potential_correction.mock import MockConfig, generate_mock_pair
from examples.potential_correction.result_products import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "examples/potential_correction/scientific_configuration.json").read_text())
    report = {"configuration": config, "scope": __doc__, "cases": {}}
    with jax.experimental.enable_x64():
        mock = MockConfig.from_dict(config["mock"])
        pair = generate_mock_pair(mock)
        renderer = PotentialCorrectionRenderer(pair.control.macro,
            SimulatorConfig(mock.dpix, mock.npix, pair.control.psf_kernel, 2, pair.control.mask),
            49, tuple(config["source_bbox"]), potential_spacing=0.075, dtype="float64")
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        source = pair.control.source_truth.brightness(sx, sy).ravel()
        grid = renderer.potential_grid
        core = np.asarray(grid.core_indices)
        nominal = int(np.argmin((np.asarray(grid.x)[core] - mock.subhalo_x)**2 + (np.asarray(grid.y)[core] - mock.subhalo_y)**2))
        for case in ("subhalo", "control"):
            obs = getattr(pair, case)
            problem = SingleStepPotentialCorrection(renderer, obs.image_data, obs.noise_map,
                obs.source_truth.gradient, lambda_amplitude=1e4)
            design = problem.dense_mapping()
            psi = obs.subhalo_truth.potential(grid.x, grid.y)
            truth = jnp.concatenate((source, psi))
            linear_image = design @ truth
            discrete_image = renderer.render(source, psi).image_active
            analytic_image = obs.model_image.ravel()[renderer.flat_indices]
            names = ["regularized_matched_linear_truth", "nonlinear_minus_linear_truth",
                     "analytic_minus_discrete_model", "noise"]
            components = jnp.stack([linear_image, discrete_image - linear_image,
                                    analytic_image - discrete_image,
                                    obs.noise_realization.ravel()[renderer.flat_indices]], axis=1)
            np.testing.assert_allclose(np.asarray(components.sum(axis=1)), np.asarray(problem.data_active), atol=2e-15)
            case_report = []
            for lp in (4e5, 1.6e7):
                fitted = problem.fit(6e-4, lp)
                assert fitted.success
                prior = problem.dense_prior(6e-4, lp)
                weighted = design / problem.noise_active[:, None]
                with jax.default_matmul_precision("highest"):
                    curvature = weighted.T @ weighted + prior
                    rhs = weighted.T @ (components / problem.noise_active[:, None])
                    values = jsp.linalg.cho_solve((jnp.linalg.cholesky(curvature), True), rhs)
                complete = np.concatenate((np.asarray(fitted.source_values), np.asarray(fitted.potential_values)))
                error = float(np.max(np.abs(np.asarray(values.sum(axis=1)) - complete)))
                np.testing.assert_allclose(np.asarray(values.sum(axis=1)), complete, atol=1e-8, rtol=1e-8)
                decomposition = {}
                for i, name in enumerate(names):
                    kappa = np.asarray(grid.derivatives(values[renderer.n_source:, i]).convergence)
                    decomposition[name] = {
                        "kappa_at_nominal_peak": float(kappa[nominal]),
                        "core_kappa_rms": float(np.sqrt(np.mean(kappa**2))),
                        "image_rms_noise": float(np.sqrt(np.mean((np.asarray(components[:, i]) / np.asarray(problem.noise_active))**2))),
                    }
                row = {"lambda_source": 6e-4, "lambda_potential": lp,
                       "truth_kappa_at_nominal_peak": float(grid.derivatives(psi).convergence[nominal]),
                       "fitted_kappa_at_nominal_peak": float(np.asarray(fitted.convergence)[core[nominal]]),
                       "superposition_max_coefficient_error": error,
                       "components": decomposition}
                case_report.append(row)
                print(json.dumps({"case": case, **row}), flush=True)
            report["cases"][case] = case_report
            write_json(args.output, report)


if __name__ == "__main__":
    main()
