"""Small dense/operator nonlinear reconstruction from truth-derived source nodes.

Run from the repository root with XLA_PYTHON_CLIENT_PREALLOCATE=false:
    python examples/potential_correction/iterate_fixed_strength.py

This representable nodal mock isolates the iterative algorithm: its noiseless
image is rendered from sampled source and potential truth on the SAME fixed
grids. It is not the independent analytic-image scientific recovery study,
regularization selection, or a detection/mass-calibration result.
"""

import argparse
from dataclasses import asdict
import json

if __package__ in (None, ""):
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from TinyLensGpu.PhysicalModel import PhysicalModel, SIS, Shear
from examples.potential_correction.mock import SourceTruth, SubhaloTruth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-n", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--lambda-source", type=float, default=1e-4)
    parser.add_argument("--lambda-potential", type=float, default=100.0)
    parser.add_argument("--lambda-amplitude", type=float, default=100.0)
    parser.add_argument("--backend", choices=("dense", "operator"), default="dense")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--pcg-max-iter", type=int, default=800)
    parser.add_argument("--pcg-rtol", type=float)
    parser.add_argument("--pcg-atol", type=float, default=0.0)
    args = parser.parse_args(argv)
    with jax.experimental.enable_x64():
        coordinates = (np.arange(12) - 5.5) * 0.12
        x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
        radius = np.hypot(x, y)
        config = SimulatorConfig(
            0.12, 12, jnp.ones((1, 1)), mask=(radius < 0.18) | (radius > 0.65)
        )
        macro = PhysicalModel(
            lens_mass=[
                SIS(theta_E=0.4, center_x=0.0, center_y=0.0),
                Shear(gamma1=0.03, gamma2=0.01),
            ],
            source_light=[],
            lens_light=[],
        )
        renderer = PotentialCorrectionRenderer(
            macro, config, args.source_n, (-0.6, 0.6, -0.6, 0.6)
        )
        sx, sy = jnp.meshgrid(
            renderer.source_x_axis, renderer.source_y_axis, indexing="xy"
        )
        source_truth = SourceTruth(sigma_x=0.22, sigma_y=0.17, clump_scale=0.8)
        initial_source = source_truth.brightness(sx, sy).ravel()
        potential_truth = SubhaloTruth(0.01, 0.4, 0.38, 0.07).potential(
            renderer.potential_grid.x, renderer.potential_grid.y
        )
        # An explicit truth reference only; no constraint is added to fitted p.
        potential_truth = potential_truth - jnp.mean(potential_truth)
        image = renderer.render(initial_source, potential_truth).model_image
        problem = IterativePotentialCorrection(
            renderer,
            image,
            jnp.full_like(image, 0.01),
            initial_source,
            lambda_amplitude=args.lambda_amplitude,
            initialization_provenance={
                "source": "analytic two-clump truth sampled on fixed source nodes",
                "mock_image": "nonlinear nodal source and nodal Pseudo-Jaffe potential",
            },
        )
        result = problem.fit(
            args.lambda_source,
            args.lambda_potential,
            max_iter=args.max_iter,
            backend=args.backend,
            block_size=args.block_size,
            pcg_max_iter=args.pcg_max_iter,
            pcg_rtol=args.pcg_rtol,
            pcg_atol=args.pcg_atol,
        )
    summary = dict(
        success=result.success,
        termination_reason=result.termination_reason,
        algorithm=result.algorithm,
        backend=result.backend,
        dtype=result.dtype,
        strengths=asdict(result.strengths),
        initial_objective=result.history[0]["objective_before"],
        final_objective=float(result.objective),
        chi2=float(result.chi2),
        accepted_iterations=result.configuration["accepted_iterations"],
        log_evidence=(
            None if result.log_evidence is None else float(result.log_evidence)
        ),
        evidence_method=result.evidence_method,
        determinant_method=result.determinant_method,
        solver_info=asdict(result.solver_info),
        final_scaled_gradient=result.configuration["final_scaled_gradient"],
        n_data=renderer.n_active,
        n_source=renderer.n_source,
        n_potential=renderer.potential_grid.n_unknown,
        scope="representable nodal fixed-strength algorithm demonstration",
    )
    print(json.dumps(summary, indent=2))
    if not result.success:
        raise RuntimeError(
            f"Iterative reconstruction failed: {result.termination_reason}"
        )
    return result


if __name__ == "__main__":
    main()
