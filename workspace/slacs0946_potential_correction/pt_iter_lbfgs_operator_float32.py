"""Search J0946 strengths with the Float32 operator L-BFGS-B fitter.

Use the local M3 handoff and rank qualified candidates by local GN evidence
with principal-block determinants. Use --plot-only to redraw a completed run.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parents[1]))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import (
    LBFGSPotentialCorrection,
    PotentialCorrectionWhiteningReference,
)
from TinyLensGpu.PhysicalModel import EPL, PhysicalModel, Shear
from TinyLensGpu.example_support import resolve_source_interpolation
from examples.potential_correction.result_products import write_json
from workspace.slacs0946_potential_correction.pt import load_handoff
from workspace.slacs0946_potential_correction.pt_iter_lbfgs import (
    DIRECT_OPTIONS,
    REFINE_RADIUS_DEX,
    STRENGTH_ORDER,
    log10_bounds,
    plot_saved_diagnostics,
    save_reconstruction,
    search,
)


def prepare_problem(macro_dir, potential_spacing):
    metadata, observation = load_handoff(macro_dir)
    parameters = metadata["macro_parameters"]
    with jax.experimental.enable_x64(False):
        simulation = SimulatorConfig(
            metadata["dpix"],
            observation["mask"].shape[0],
            jnp.asarray(observation["psf_kernel"], dtype=jnp.float32),
            nsub=metadata["nsub"],
            mask=observation["mask"],
        )
    macro = PhysicalModel(
        lens_mass=[
            EPL(
                theta_E=parameters["theta_E"],
                gamma=parameters["gamma"],
                e1=parameters["e1_mass"],
                e2=parameters["e2_mass"],
                center_x=parameters["center_x_mass"],
                center_y=parameters["center_y_mass"],
            ),
            Shear(gamma1=parameters["gamma1"], gamma2=parameters["gamma2"]),
        ],
        source_light=[],
        lens_light=[],
    )
    renderer = PotentialCorrectionRenderer(
        macro,
        simulation,
        metadata["source_n"],
        observation["source_bbox"],
        potential_spacing=potential_spacing,
        dtype="float32",
        source_interpolation="cubic_hermite",
    )
    source = observation["source_values"]
    reference_interpolation = resolve_source_interpolation(
        metadata=metadata, context="M3 handoff"
    )
    problem = LBFGSPotentialCorrection(
        renderer,
        observation["lens_subtracted_image"],
        observation["noise_map"],
        source,
        source_regularization=metadata["source_regularization"],
        source_scale=observation["source_scale"],
        initialization_provenance={
            "source": "saved M3 source intensities",
            "handoff_sha256": metadata["handoff_sha256"],
            "m3_sample_index": metadata["selected_sample_index"],
        },
        whitening_reference=PotentialCorrectionWhiteningReference(
            source, reference_interpolation
        ),
    )
    return problem, metadata, observation


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--macro-dir", type=Path, default=HERE / "macro")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "potential_lbfgs_operator_float32",
        help="Directory for the Float32 operator results",
    )
    parser.add_argument("--potential-spacing", type=float, default=0.1)
    parser.add_argument("--log10-lambda-s", type=float, nargs=2)
    parser.add_argument("--log10-lambda4", type=float, nargs=2)
    parser.add_argument("--log10-lambda0", type=float, nargs=2)
    parser.add_argument(
        "--maxfun", type=int, default=160, help="Nominal global DIRECT budget"
    )
    parser.add_argument(
        "--refine-maxfun",
        type=int,
        default=90,
        help="Nominal refinement budget (0 skips it)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=3000, help="L-BFGS-B iterations per candidate"
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Plot a completed output"
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.plot_only:
        plot_saved_diagnostics(output)
        return
    if args.maxfun < 1 or args.refine_maxfun < 0 or args.max_iter < 1:
        parser.error(
            "Use positive global and inner budgets and a nonnegative refinement budget"
        )
    macro_dir = args.macro_dir.resolve()
    if output.exists() or output.is_relative_to(macro_dir):
        parser.error("Use a new output directory outside the M3 handoff")

    fit_options = {
        "backend": "operator",
        "max_iter": args.max_iter,
        "gn_stationarity_tol": 1e-2,
        "pcg_max_iter": 2000,
    }
    with jax.experimental.enable_x64(False), jax.default_matmul_precision("highest"):
        problem, metadata, observation = prepare_problem(
            macro_dir, args.potential_spacing
        )
        bounds = log10_bounds(args, metadata)
        with np.errstate(over="ignore", under="ignore"):
            strengths = np.asarray(10.0**bounds, dtype=np.float32)
        if not np.all(np.isfinite(strengths) & (strengths > 0)):
            parser.error("Log10 bounds must give finite positive Float32 strengths")
        output.mkdir(parents=True, exist_ok=False)
        write_json(
            output / "manifest.json",
            {
                "macro_dir": macro_dir,
                "m3_handoff_json_sha256": hashlib.sha256(
                    (macro_dir / "m3_handoff.json").read_bytes()
                ).hexdigest(),
                "fixed_macro": metadata,
                "initialization_provenance": problem.initialization_provenance,
                "potential_spacing_arcsec": args.potential_spacing,
                "native_sampling_dtype": "float32",
                "fit_dtype": problem.dtype.name,
                "optimizer_state_dtype": "float64",
                "backend": fit_options["backend"],
                "determinant_method": "principal_block_cholesky",
                "strength_order": STRENGTH_ORDER,
                "log10_bounds": bounds,
                "direct_options": DIRECT_OPTIONS,
                "global_maxfun": args.maxfun,
                "refine_maxfun": args.refine_maxfun,
                "refine_radius_dex": REFINE_RADIUS_DEX,
                "fit_options": fit_options,
                "selection": "Highest qualified operator local GN evidence among visited candidates",
                "versions": {
                    "jax": jax.__version__,
                    "scipy": scipy.__version__,
                    "numpy": np.__version__,
                },
                "device": jax.devices()[0].device_kind,
            },
        )
        best, result, summary = search(
            problem,
            bounds,
            output,
            maxfun=args.maxfun,
            refine_maxfun=args.refine_maxfun,
            fit_options=fit_options,
        )

    if result is not None:
        destination = output / "maximum_evidence"
        save_reconstruction(destination, result, best, observation)
        point = np.asarray(best["log10_strengths"])
        summary["selected"] = best
        summary["distance_to_global_bounds_dex"] = dict(
            zip(STRENGTH_ORDER, np.minimum(point - bounds[:, 0], bounds[:, 1] - point))
        )
        summary["result"] = "maximum_evidence/result.json"
    write_json(output / "completion.json", summary)
    if result is None:
        raise SystemExit(
            "DIRECT found no qualified L-BFGS-B candidate; see evaluations.jsonl"
        )
    plot_saved_diagnostics(output)
    print(
        json.dumps(
            {"strengths": best["strengths"], "log_evidence": best["log_evidence"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
