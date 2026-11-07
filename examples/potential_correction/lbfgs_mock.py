"""Fit a matched Pseudo-Jaffe mock and control with fixed-strength L-BFGS-B."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.ObservationModel import (
    LBFGSPotentialCorrection,
    PotentialCorrectionStrengths,
)
from examples.potential_correction.mock import MockConfig, generate_mock_pair
from examples.potential_correction.result_products import export_result, write_json


@dataclass(frozen=True)
class LBFGSMockRun:
    results: dict
    record: dict


def run_lbfgs_mock(configuration, *, output):
    """Fit both observations with the same geometry, initialization and strengths."""
    mock = MockConfig.from_dict(configuration["mock"])
    if mock.dtype != "float64":
        raise ValueError("The L-BFGS-B mock requires Float64")
    strengths = PotentialCorrectionStrengths(**configuration["strengths"])
    output = Path(output)
    fit_options = configuration.get("fit_options", {})
    pair = generate_mock_pair(mock)

    with jax.experimental.enable_x64():
        renderer = PotentialCorrectionRenderer(
            pair.control.macro,
            pair.control.sim_config,
            configuration["source_n"],
            tuple(configuration["source_bbox"]),
            potential_spacing=configuration["potential_spacing"],
            dtype=mock.dtype,
            source_interpolation="cubic_hermite",
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        initial_source = mock.source.brightness(sx, sy).ravel()

    results, runs = {}, {}
    for case in ("subhalo", "control"):
        observation = getattr(pair, case)
        fitter = LBFGSPotentialCorrection(
            renderer,
            observation.image_data,
            observation.noise_map,
            initial_source,
            source_regularization=configuration.get(
                "source_regularization", "second-order"
            ),
            initialization_provenance={
                "source": "analytic mock brightness sampled on the fixed source nodes",
            },
        )
        result = fitter.fit(strengths, backend="dense", **fit_options)
        key = f"{case}/dense"
        results[key] = result
        metadata = export_result(
            result,
            observation,
            output / key,
            configuration=configuration,
            make_plots=configuration.get("make_plots", True),
        )
        runs[key] = {
            name: metadata[name]
            for name in (
                "success",
                "termination_reason",
                "failure_reason",
                "objective",
                "strengths",
                "log_evidence",
                "evidence_method",
                "determinant_method",
                "optimizer_info",
            )
        }
        if result.success:
            runs[key]["summary"] = metadata["summary"]

    record = {
        "success": all(result.success for result in results.values()),
        "runs": runs,
    }
    write_json(output / "configuration.json", configuration)
    write_json(output / "run.json", record)
    return LBFGSMockRun(results, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("lbfgs_mock_configuration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/potential_correction_lbfgs_mock"),
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    configuration = json.loads(args.config.read_text())
    if args.no_plots:
        configuration["make_plots"] = False
    run = run_lbfgs_mock(configuration, output=args.output)
    print(
        json.dumps(
            {"success": run.record["success"], "output": str(args.output.resolve())}
        )
    )
    if not run.record["success"]:
        raise SystemExit(1)
    return run


if __name__ == "__main__":
    main()
