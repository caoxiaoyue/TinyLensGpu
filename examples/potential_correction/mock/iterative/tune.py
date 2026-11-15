"""Small cold-started DIRECT search around complete nonlinear reconstructions.

Run with XLA_PYTHON_CLIENT_PREALLOCATE=false, for example:
    python examples/potential_correction/mock/iterative/tune.py --backend operator

This 4x4 numerical nodal case validates nested inference, not a scientific
subhalo recovery or detection. The independent scan and frozen block layout
are documented in docs/validation/potential-correction-iterative-inference.md.
Every candidate starts at the same truth-derived source and zero correction.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.Inference import optimize_potential_correction
from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from TinyLensGpu.PhysicalModel import PhysicalModel, Shear
from examples.potential_correction.shared.result_products import (
    RESULT_ARRAYS,
    json_safe,
    write_json,
)

# Calibrated once from initial response-column squared norms on this exact
# tiny float64 model. Runtime never builds a Jacobian to choose blocks and
# never changes the partition between strength candidates. Neither block
# contains the whole potential or joint field. This is not a layout recipe
# calibrated for larger source or scientific image grids.
BLOCK_INDICES = (
    (0, 1, 2, 3, 21, 20, 19, 28, 26, 27, 18, 22, 13, 12, 14, 29, 25, 11, 15, 9),
    (33, 8, 32, 31, 7, 4, 5, 6, 10, 16, 17, 23, 24, 30, 34, 35, 36),
)
SCOPE = "numerical nodal nested-inference validation; no scientific recovery or detection claim"


def make_problem():
    """Return the fixed float64 problem and supplied nodal truth, with no fit."""
    with jax.experimental.enable_x64():
        renderer = PotentialCorrectionRenderer(
            PhysicalModel(
                lens_mass=[Shear(gamma1=0.0, gamma2=0.0)],
                source_light=[],
                lens_light=[],
            ),
            SimulatorConfig(0.1, 4, jnp.ones((1, 1))),
            2,
            (-0.5, 0.5, -0.5, 0.5),
            dtype="float64",
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        source = (0.775 + 0.95 * sx + 0.05 * sy + 0.5 * sx * sy).ravel()
        grid = renderer.potential_grid
        potential = 0.003 * jnp.exp(
            -((grid.x - 0.07) ** 2 + (grid.y + 0.03) ** 2) / 0.02
        )
        noiseless = renderer.render(source, potential).model_image
        image = noiseless + 0.002 * jnp.sin(jnp.arange(16)).reshape(4, 4)
        problem = IterativePotentialCorrection(
            renderer,
            image,
            jnp.full((4, 4), 0.07),
            source,
            lambda_amplitude=20.0,
            source_regularization="zero-order",
            initialization_provenance={
                "source": "bilinear truth 0.775+0.95*x+0.05*y+0.5*x*y sampled at fixed nodes",
                "mock_image": "same fixed-grid nonlinear source and Gaussian nodal potential, plus deterministic 0.002*sin(pixel_index)",
                "potential_truth_reference": "raw Gaussian nodal values, no alignment or mean subtraction",
            },
        )
    return problem, dict(
        truth_source_values=np.asarray(source),
        truth_potential_values=np.asarray(potential),
        truth_nonlinear_model_image=np.asarray(noiseless),
    )


def search_record(search):
    """Small scalar optimizer record; full fields stay in its public best fit."""
    return dict(
        has_valid_fit=search.has_valid_fit,
        success=search.success,
        optimizer_success=search.optimizer_success,
        optimizer_status=search.optimizer_status,
        message=search.message,
        nfev=search.nfev,
        nit=search.nit,
        objective=search.objective,
        log_strengths=search.log_strengths,
        strengths=None if search.strengths is None else asdict(search.strengths),
        log_bounds=search.log_bounds,
        at_boundary=search.at_boundary,
        boundary_sides=search.boundary_sides,
        boundary_distances=search.boundary_distances,
        configuration=search.configuration,
        evaluations=[asdict(row) for row in search.evaluations],
        scope=SCOPE,
    )


def write_run(search, truth, output):
    """Hand the retained public result to numerical diagnostics without refitting.

    Compact nested_* products are specific to this numerical example. The
    complete scientific mock exporter/figures are a separate workflow.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = search_record(search)
    write_json(output / "search.json", record)
    if not search.has_valid_fit:
        for filename in ("nested_result.json", "nested_result.npz"):
            (output / filename).unlink(missing_ok=True)
        write_json(
            output / "failure.json",
            dict(
                success=False,
                failure_reason="no_successful_inner_reconstruction",
                search=record,
                scope=SCOPE,
            ),
        )
        return
    result = search.reconstruction
    metadata = dict(
        success=result.success,
        algorithm=result.algorithm,
        backend=result.backend,
        dtype=result.dtype,
        strengths=asdict(result.strengths),
        termination_reason=result.termination_reason,
        solver_info=asdict(result.solver_info),
        evidence_method=result.evidence_method,
        determinant_method=result.determinant_method,
        evidence_is_exact=result.evidence_is_exact,
        log_evidence=result.log_evidence,
        chi2=result.chi2,
        objective=result.objective,
        regularization_penalty=result.regularization_penalty,
        prior_logdet=result.prior_logdet,
        posterior_logdet=result.posterior_logdet,
        noise_logdet=result.noise_logdet,
        initialization_provenance=result.initialization_provenance,
        configuration=result.configuration,
        history=result.history,
        source_bbox=result.source_bbox,
        potential_spacing=result.potential_grid.spacing,
        potential_origin=result.potential_grid.origin,
        truth_reference="raw nodal Gaussian potential; no alignment",
        scope=SCOPE,
    )
    arrays = {
        name: np.asarray(getattr(result, name))
        for name in RESULT_ARRAYS
        if getattr(result, name) is not None
    }
    arrays.update(truth)
    arrays.update(
        potential_node_indices=np.asarray(result.potential_grid.node_indices),
        potential_core_indices=np.asarray(result.potential_grid.core_indices),
    )
    np.savez_compressed(output / "nested_result.npz", **arrays)
    write_json(output / "nested_result.json", metadata)
    (output / "failure.json").unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("dense", "operator"), default="dense")
    parser.add_argument("--maxfun", type=int, default=100)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=240)
    parser.add_argument(
        "--log-source-bounds", type=float, nargs=2, default=np.log([0.1, 2.0])
    )
    parser.add_argument(
        "--log-potential-bounds", type=float, nargs=2, default=np.log([0.001, 1.0])
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "tuning",
    )
    args = parser.parse_args(argv)
    problem, truth = make_problem()
    search = optimize_potential_correction(
        problem,
        log_bounds=[args.log_source_bounds, args.log_potential_bounds],
        backend=args.backend,
        fit_options=dict(
            max_iter=args.max_iter,
            block_size=20,
            block_indices=BLOCK_INDICES,
            pcg_rtol=1e-12,
        ),
        maxfun=args.maxfun,
        maxiter=args.maxiter,
        len_tol=1e-3,
    )
    write_run(search, truth, args.output)
    summary = search_record(search)
    summary.pop("evaluations")
    if search.has_valid_fit:
        result = search.reconstruction
        summary.update(
            algorithm=result.algorithm,
            termination_reason=result.termination_reason,
            evidence_method=result.evidence_method,
            determinant_method=result.determinant_method,
            evidence_is_exact=result.evidence_is_exact,
            log_evidence=result.log_evidence,
        )
    print(json.dumps(json_safe(summary), indent=2, allow_nan=False))
    return search


if __name__ == "__main__":
    if not main().has_valid_fit:
        raise SystemExit(1)
