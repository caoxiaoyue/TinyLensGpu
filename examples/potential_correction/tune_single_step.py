"""Tune a small dense mock; strengths and bounds are explicit numerical inputs.

This is a runnable inference example, without a scientific recovery or weak
amplitude-prior calibration claim. The 12-by-12 mock uses truth-derived analytic
gradients, a fixed 5-by-5 source box, and the fixed true macro lens.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.Inference import optimize_potential_correction
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from examples.potential_correction.mock import MockConfig, generate_mock_pair


def run_tuning(
    *,
    log_bounds,
    lambda_amplitude,
    dtype="float64",
    source_n=5,
    potential_spacing=None,
    maxfun=80,
    maxiter=1000,
    len_tol=1e-3,
    boundary_rtol=0.01,
    output=None,
):
    """Return selected strengths and the complete public reconstruction result."""
    mock_config = MockConfig(npix=12, dpix=0.25, nsub=1, psf_size=3, dtype=dtype)
    observation = generate_mock_pair(mock_config).subhalo
    renderer = PotentialCorrectionRenderer(
        observation.macro,
        observation.sim_config,
        source_n,
        (-0.6, 0.6, -0.6, 0.6),
        potential_spacing=potential_spacing,
        dtype=dtype,
    )
    problem = SingleStepPotentialCorrection(
        renderer,
        observation.image_data,
        observation.noise_map,
        observation.source_truth.gradient,
        lambda_amplitude=lambda_amplitude,
        initialization_provenance={"source": "mock analytic gradient"},
    )
    result = optimize_potential_correction(
        problem,
        log_bounds=log_bounds,
        maxfun=maxfun,
        maxiter=maxiter,
        len_tol=len_tol,
        boundary_rtol=boundary_rtol,
    )
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        reconstruction = result.reconstruction
        metadata = {
            "purpose": "small dense tuning example; scientific settings uncalibrated",
            "mock_configuration": asdict(mock_config),
            "configuration": result.configuration,
            "log_bounds": result.log_bounds,
            "selected_log_strengths": result.log_strengths,
            "selected_strengths": (
                None if result.strengths is None else asdict(result.strengths)
            ),
            "has_valid_fit": result.has_valid_fit,
            "optimizer_success": result.optimizer_success,
            "optimizer_status": result.optimizer_status,
            "optimizer_message": result.message,
            "nfev": result.nfev,
            "nit": result.nit,
            "boundary_sides": result.boundary_sides,
            "boundary_distances": result.boundary_distances,
            "evaluations": [
                {
                    **asdict(record),
                    "objective": (
                        record.objective if np.isfinite(record.objective) else None
                    ),
                }
                for record in result.evaluations
            ],
            "initialization_provenance": problem.initialization_provenance,
            "reconstruction_configuration": (
                None if reconstruction is None else reconstruction.configuration
            ),
        }
        if reconstruction is not None:
            metadata.update(
                log_evidence=float(reconstruction.log_evidence),
                evidence_method=reconstruction.evidence_method,
                determinant_method=reconstruction.determinant_method,
                evidence_is_exact=reconstruction.evidence_is_exact,
            )
            np.savez_compressed(
                output / "reconstruction.npz",
                source_values=np.asarray(reconstruction.source_values),
                source_x_axis=np.asarray(reconstruction.source_x_axis),
                source_y_axis=np.asarray(reconstruction.source_y_axis),
                potential_values=np.asarray(reconstruction.potential_values),
                potential_node_indices=np.asarray(
                    reconstruction.potential_grid.node_indices
                ),
                derivative_valid=np.asarray(reconstruction.derivative_valid),
                linear_model_image=np.asarray(reconstruction.linear_model_image),
                nonlinear_model_image=np.asarray(reconstruction.nonlinear_model_image),
            )
        (output / "tuning.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-amplitude", type=float, required=True)
    parser.add_argument("--log-source-bounds", type=float, nargs=2, required=True)
    parser.add_argument("--log-potential-bounds", type=float, nargs=2, required=True)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--source-n", type=int, default=5)
    parser.add_argument("--potential-spacing", type=float, default=None)
    parser.add_argument("--maxfun", type=int, default=80)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--len-tol", type=float, default=1e-3)
    parser.add_argument("--boundary-rtol", type=float, default=0.01)
    parser.add_argument(
        "--output", type=Path, default=Path("output/potential_correction_tuning")
    )
    args = parser.parse_args(argv)
    values = vars(args).copy()
    values["log_bounds"] = [
        values.pop("log_source_bounds"),
        values.pop("log_potential_bounds"),
    ]
    result = run_tuning(**values)
    summary = {
        "selected_strengths": (
            None if result.strengths is None else asdict(result.strengths)
        ),
        "has_valid_fit": result.has_valid_fit,
        "optimizer_success": result.optimizer_success,
        "optimizer_message": result.message,
        "boundary_sides": result.boundary_sides,
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return result


if __name__ == "__main__":
    main()
