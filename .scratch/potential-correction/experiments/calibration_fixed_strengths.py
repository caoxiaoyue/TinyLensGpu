"""Public dense/operator checks on actual ill-conditioned calibration systems.

Run before expensive searches. This deliberately uses a dense small reference;
operator preparation still contains only bounded principal blocks.
"""

import argparse
import json
from pathlib import Path
import time

import jax

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from examples.potential_correction.calibrate_single_step import (
    source_incidence_blocks,
    _truth,
    _fit_record,
    _compare,
    _json_safe,
)
from examples.potential_correction.mock import MockConfig, generate_mock_pair


def run(config):
    pair = generate_mock_pair(MockConfig.from_dict(config["mock"]))
    renderer = PotentialCorrectionRenderer(
        pair.control.macro,
        pair.control.sim_config,
        config["source_n"],
        tuple(config["source_bbox"]),
        potential_spacing=config["potential_spacing"],
        dtype="float64",
    )
    rows = []
    blocks = None
    tolerances = dict(
        source_values_max_abs_difference=2e-5,
        potential_values_max_abs_difference=2e-7,
        convergence_max_abs_difference=1e-5,
        linear_image_active_max_abs_difference=1e-7,
        nonlinear_image_active_max_abs_difference=1e-7,
    )
    for case in ("subhalo", "control"):
        observation = getattr(pair, case)
        truth = _truth(renderer, observation)
        for amplitude in config["lambda_amplitudes"]:
            problem = SingleStepPotentialCorrection(
                renderer,
                observation.image_data,
                observation.noise_map,
                observation.source_truth.gradient,
                lambda_amplitude=amplitude,
                solve_rtol=config["solve_rtol"],
            )
            if blocks is None:
                blocks = [
                    b.tolist()
                    for b in source_incidence_blocks(
                        renderer,
                        problem.mapping,
                        problem.gradient_x,
                        problem.gradient_y,
                        source_tile=config["source_tile"],
                        block_size=config["block_size"],
                    )
                ]
            options = dict(
                block_size=config["block_size"],
                block_indices=blocks,
                pcg_max_iter=config["pcg_max_iter"],
                pcg_rtol=config["pcg_rtol"],
                pcg_atol=0.0,
            )
            for source, potential in ((6e-5, 2.5e6), (1e-5, 1e11)):
                start = time.perf_counter()
                dense = problem.fit(source, potential)
                dense_time = time.perf_counter() - start
                start = time.perf_counter()
                operator = problem.fit(source, potential, backend="operator", **options)
                operator_time = time.perf_counter() - start
                comparison = _compare(dense, operator)
                passed = comparison["success"] and all(
                    comparison[key] <= value for key, value in tolerances.items()
                )
                row = dict(
                    case=case,
                    lambda_amplitude=amplitude,
                    lambda_source=source,
                    lambda_potential=potential,
                    dense_seconds=dense_time,
                    operator_seconds=operator_time,
                    dense=_fit_record(dense, truth, observation),
                    operator=_fit_record(operator, truth, observation),
                    comparison=comparison,
                    numerical_equivalence=passed,
                )
                rows.append(row)
                print(
                    json.dumps(
                        _json_safe(
                            dict(
                                case=case,
                                amplitude=amplitude,
                                potential=potential,
                                operator_seconds=operator_time,
                                solver_info=operator.solver_info.__dict__,
                                numerical_equivalence=passed,
                                comparison=comparison,
                            )
                        )
                    ),
                    flush=True,
                )
    return dict(
        configuration=config,
        scope=__doc__,
        absolute_field_tolerances=tolerances,
        results=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with jax.experimental.enable_x64():
        result = run(json.loads(args.config.read_text()))
    args.output.write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n"
    )
    if not all(row["numerical_equivalence"] for row in result["results"]):
        raise RuntimeError(
            "Fixed-strength calibration includes a failed or inaccurate solve; see output"
        )


if __name__ == "__main__":
    main()
