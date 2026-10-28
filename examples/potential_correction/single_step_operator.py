"""Small fixed-strength operator fit, with public fields and both renderings.

Run from the repository root:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/single_step_operator.py

This is an assembly demonstration, not regularization calibration or a claim
of subhalo recovery. The fixed amplitude prior and coarse source are explicit.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from examples.potential_correction.mock import MockConfig, generate_mock_pair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("output/potential_correction_operator")
    )
    parser.add_argument("--source-n", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--lambda-source", type=float, default=0.01)
    parser.add_argument("--lambda-potential", type=float, default=100.0)
    parser.add_argument("--lambda-amplitude", type=float, default=100.0)
    parser.add_argument("--pcg-max-iter", type=int, default=600)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args(argv)
    mock_config = MockConfig(npix=12, dpix=0.24, nsub=1, psf_size=3)
    pair = generate_mock_pair(mock_config)
    observation = pair.control if args.control else pair.subhalo
    renderer = PotentialCorrectionRenderer(
        observation.macro,
        observation.sim_config,
        args.source_n,
        (-0.5, 0.5, -0.5, 0.5),
        dtype=mock_config.dtype,
    )
    problem = SingleStepPotentialCorrection(
        renderer,
        observation.image_data,
        observation.noise_map,
        observation.source_truth.gradient,
        lambda_amplitude=args.lambda_amplitude,
        initialization_provenance={"source": "analytic mock truth gradient"},
    )
    result = problem.fit(
        args.lambda_source,
        args.lambda_potential,
        backend="operator",
        block_size=args.block_size,
        pcg_max_iter=args.pcg_max_iter,
    )
    metadata = {
        "success": result.success,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "algorithm": result.algorithm,
        "backend": result.backend,
        "dtype": result.dtype,
        "evidence_method": result.evidence_method,
        "determinant_method": result.determinant_method,
        "evidence_is_exact": result.evidence_is_exact,
        "log_evidence": (
            None if result.log_evidence is None else float(result.log_evidence)
        ),
        "strengths": asdict(result.strengths),
        "solver_info": asdict(result.solver_info),
        "initialization_provenance": result.initialization_provenance,
        "configuration": result.configuration,
        "mock_configuration": asdict(mock_config),
        "observation": "control" if args.control else "subhalo",
        "scope": "fixed-strength assembly only; no tuning or recovery acceptance",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if not result.success:
        for filename in ("configuration.json", "reconstruction.npz"):
            (args.output / filename).unlink(missing_ok=True)
        (args.output / "failure.json").write_text(json.dumps(metadata, indent=2) + "\n")
        raise RuntimeError(f"Operator reconstruction failed: {result.failure_reason}")
    arrays = {
        name: np.asarray(getattr(result, name))
        for name in (
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
            "data_active",
            "noise_active",
        )
    }
    arrays.update(
        potential_node_indices=np.asarray(result.potential_grid.node_indices),
        potential_x=np.asarray(result.potential_grid.x),
        potential_y=np.asarray(result.potential_grid.y),
        potential_core_indices=np.asarray(result.potential_grid.core_indices),
        source_bbox=np.asarray(result.source_bbox),
    )
    np.savez_compressed(args.output / "reconstruction.npz", **arrays)
    (args.output / "configuration.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (args.output / "failure.json").unlink(missing_ok=True)
    print(
        f"Saved fixed-strength operator fit to {args.output.resolve()}; "
        f"PCG iterations={result.solver_info.iterations}, "
        f"relative residual={result.solver_info.relative_residual:.3g}. "
        "Evidence uses principal-block determinants."
    )
    return args.output


if __name__ == "__main__":
    main()
