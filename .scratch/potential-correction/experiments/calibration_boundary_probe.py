"""Check the control's finite upper D4 bound without changing source geometry."""

import argparse
import json
from pathlib import Path

import jax

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from examples.potential_correction.mock.single_step.calibrate import (
    source_incidence_blocks,
    _truth,
    _fit_record,
    _compare,
    _json_safe,
)
from examples.potential_correction.mock import MockConfig, generate_mock_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda-amplitude", type=float, default=10000.0)
    parser.add_argument("--lambda-source", type=float, default=6.716770257778804e-5)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with jax.experimental.enable_x64():
        observation = generate_mock_pair(MockConfig.from_dict(config["mock"])).control
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
            lambda_amplitude=args.lambda_amplitude,
            solve_rtol=config["solve_rtol"],
        )
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
        truth = _truth(renderer, observation)
        rows = []
        for strength in (1e10, 1e11, 1e12, 1e13):
            dense = problem.fit(args.lambda_source, strength)
            operator = problem.fit(
                args.lambda_source, strength, backend="operator", **options
            )
            row = dict(
                lambda_potential=strength,
                dense=_fit_record(dense, truth, observation),
                operator=_fit_record(operator, truth, observation),
                comparison=_compare(dense, operator),
            )
            rows.append(row)
            print(
                json.dumps(
                    _json_safe(
                        dict(
                            lambda_potential=strength,
                            dense_success=dense.success,
                            operator_success=operator.success,
                            dense_logz=(
                                None if not dense.success else float(dense.log_evidence)
                            ),
                            operator_logz=(
                                None
                                if not operator.success
                                else float(operator.log_evidence)
                            ),
                            solver_info=operator.solver_info.__dict__,
                        )
                    )
                ),
                flush=True,
            )
    args.output.write_text(
        json.dumps(
            _json_safe(
                dict(
                    scope=__doc__,
                    configuration=config,
                    lambda_source=args.lambda_source,
                    lambda_amplitude=args.lambda_amplitude,
                    rows=rows,
                )
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
