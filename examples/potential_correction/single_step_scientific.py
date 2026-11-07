"""Matched scientific single-step source–potential reconstructions.

Run from the repository root with --config scientific_configuration.json.
Every candidate uses fixed source/potential geometry and the same frozen
analytic truth gradient. Both renderings and all raw fitted fields are saved.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.Inference import optimize_potential_correction
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from examples.potential_correction._scientific_partitions import prepare_partition
from examples.potential_correction.calibrate_single_step import source_incidence_blocks
from examples.potential_correction.mock import MockConfig, generate_mock_pair
from examples.potential_correction.result_products import (
    export_failure,
    export_result,
    write_json,
)


@dataclass(frozen=True)
class ScientificRun:
    """Public reconstructions and compact records from the matched experiment."""

    results: dict
    record: dict
    output: Path


def _search_record(search):
    from dataclasses import asdict

    return {
        "has_valid_fit": search.has_valid_fit,
        "optimizer_success": search.optimizer_success,
        "optimizer_status": search.optimizer_status,
        "message": search.message,
        "nfev": search.nfev,
        "nit": search.nit,
        "log_bounds": search.log_bounds,
        "selected_log_strengths": search.log_strengths,
        "boundary_sides": search.boundary_sides,
        "boundary_distances": search.boundary_distances,
        "configuration": search.configuration,
        "evaluations": [asdict(row) for row in search.evaluations],
    }


def _supplied_truth_diagnostics(renderer, observation):
    """Separate fixed-grid sampling from the analytic source Taylor remainder."""
    grid = renderer.potential_grid
    mapping = renderer.source_mapping()
    sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
    source = observation.source_truth.brightness(sx, sy).ravel()
    potential = observation.subhalo_truth.potential(grid.x, grid.y)
    sampled = renderer.render(source, potential).image_active
    ax, ay = grid.deflection(potential)
    analytic_discrete = renderer.image_forward(
        observation.source_truth.brightness(mapping.beta_x - ax, mapping.beta_y - ay)
    )
    exact_ax, exact_ay = observation.subhalo_truth.deflection(
        grid.query_x, grid.query_y
    )
    analytic_exact = renderer.image_forward(
        observation.source_truth.brightness(
            mapping.beta_x - exact_ax, mapping.beta_y - exact_ay
        )
    )
    gx, gy = observation.source_truth.gradient(mapping.beta_x, mapping.beta_y)
    analytic_taylor = renderer.image_forward(
        observation.source_truth.brightness(mapping.beta_x, mapping.beta_y)
        - gx * exact_ax
        - gy * exact_ay
    )
    mock_active = observation.model_image.ravel()[renderer.flat_indices]
    noise = observation.noise_map.ravel()[renderer.flat_indices]
    rms = lambda difference: float(jnp.sqrt(jnp.mean((difference / noise) ** 2)))
    return {
        "sampled_truth_vs_mock_rms_noise": rms(sampled - mock_active),
        "source_interpolation_at_discrete_potential_rms_noise": rms(
            sampled - analytic_discrete
        ),
        "analytic_source_discrete_potential_vs_inversion_truth_rms_noise": rms(
            analytic_discrete - analytic_exact
        ),
        "analytic_truth_inversion_vs_mock_sampling_rms_noise": rms(
            analytic_exact - mock_active
        ),
        "analytic_source_taylor_remainder_rms_noise": rms(
            analytic_taylor - analytic_exact
        ),
        "convention": "selected pixels; public supplied-field rendering; no fit; contributions need not add in quadrature",
    }


def run_scientific(configuration, *, output):
    """Generate a matched pair, explicitly fit/tune, then export public results.

    ``mode='fixed'`` uses case strengths, optionally keyed by backend.
    ``mode='tune'`` uses explicit natural-log bounds and retains DIRECT budget,
    boundary and failed-candidate status separately from successful fits.
    """
    config = dict(configuration)
    mock = MockConfig.from_dict(config["mock"])
    backends = config.get("backends", ["dense", "operator"])
    if (
        not backends
        or len(set(backends)) != len(backends)
        or any(b not in ("dense", "operator") for b in backends)
    ):
        raise ValueError("backends must be a unique nonempty dense/operator selection")
    mode = config.get("mode", "fixed")
    if mode not in ("fixed", "tune"):
        raise ValueError("mode must be fixed or tune")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pair = generate_mock_pair(mock)
    with jax.experimental.enable_x64(mock.dtype == "float64"):
        inversion_sim = SimulatorConfig(
            mock.dpix,
            mock.npix,
            pair.control.psf_kernel,
            config["inversion_nsub"],
            pair.control.mask,
        )
        renderer = PotentialCorrectionRenderer(
            pair.control.macro,
            inversion_sim,
            config["source_n"],
            tuple(config["source_bbox"]),
            potential_spacing=config["potential_spacing"],
            dtype=mock.dtype,
            source_interpolation=config.get("source_interpolation", "bilinear"),
        )

        mapping = renderer.source_mapping()
        gx, gy = pair.control.source_truth.gradient(mapping.beta_x, mapping.beta_y)

        def build_blocks():
            return source_incidence_blocks(
                renderer,
                mapping,
                gx,
                gy,
                source_tile=config["source_tile"],
                block_size=config["block_size"],
            )

        blocks, partition = prepare_partition(
            renderer,
            config,
            build_blocks,
            source="frozen_macro",
            method="source_incidence",
            incidence_gradients=(gx, gy),
        )
        options = {
            "block_size": config["block_size"],
            "block_indices": blocks,
            "pcg_max_iter": config["pcg_max_iter"],
            "pcg_rtol": config["pcg_rtol"],
            "pcg_atol": 0.0,
        }
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        root = Path(__file__).resolve().parents[2]
        files = [
            Path(__file__),
            Path(__file__).with_name("result_products.py"),
            Path(__file__).with_name("mock.py"),
            Path(__file__).with_name("calibrate_single_step.py"),
            Path(__file__).with_name("_scientific_partitions.py"),
            root / "TinyLensGpu/utils/lensing/mapping.py",
        ]
        files.extend(
            root.glob(
                "TinyLensGpu/ObservationModel/LensImage/*potential_correction*.py"
            )
        )
        record = {
            "configuration": config,
            "success": True,
            "runs": {},
            "software": {
                "python": platform.python_version(),
                "jax": jax.__version__,
                "numpy": np.__version__,
                "devices": [str(d) for d in jax.devices()],
                "git_commit": revision,
                "git_dirty": bool(
                    subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        check=False,
                    ).stdout
                ),
                "implementation_sha256": {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files
                },
            },
            "sizes": {
                "image_shape": list(renderer.image_shape),
                "data": renderer.n_active,
                "source": renderer.n_source,
                "potential_core": renderer.potential_grid.n_core,
                "potential_support": renderer.potential_grid.n_unknown,
                "query_subpixels": len(renderer.query_x),
            },
            "source_interpolation": renderer.source_interpolation,
            "source_boundary": renderer.source_boundary,
            "block_partition": partition,
            "initialization": "frozen analytic mock truth gradient at fixed true macro rays; zero potential",
            "geometry": "source box and potential support fixed throughout every fit and strength candidate",
            "scope": "reconstruction validation; no detection significance or mass confidence intervals",
            "supplied_truth": {},
        }
        results = {}
        for case in ("subhalo", "control"):
            observation = getattr(pair, case)
            record["supplied_truth"][case] = _supplied_truth_diagnostics(
                renderer, observation
            )
            problem = SingleStepPotentialCorrection(
                renderer,
                observation.image_data,
                observation.noise_map,
                observation.source_truth.gradient,
                lambda_amplitude=config["lambda_amplitude"],
                solve_rtol=config["solve_rtol"],
                initialization_provenance={
                    "source": "analytic mock truth gradient; fixed macro rays",
                    "source_geometry": "fixed explicit box; no expansion",
                },
            )
            for backend in backends:
                key = f"{case}/{backend}"
                fit_options = options if backend == "operator" else {}
                search_record = None
                if mode == "tune":
                    search = optimize_potential_correction(
                        problem,
                        backend=backend,
                        fit_options=fit_options,
                        log_bounds=config["log_bounds"],
                        maxfun=config.get("direct_maxfun", 200),
                        maxiter=config.get("direct_maxiter", 1000),
                        len_tol=config.get("direct_len_tol", 1e-3),
                        boundary_rtol=config.get("boundary_rtol", 0.01),
                    )
                    search_record = _search_record(search)
                    result = search.reconstruction
                    if result is None:
                        failure = {
                            "success": False,
                            "status": "failed",
                            "algorithm": "single_step",
                            "backend": backend,
                            "dtype": mock.dtype,
                            "stage": "strength_search",
                            "search": search_record,
                            "configuration": config,
                            "source_interpolation": renderer.source_interpolation,
                            "source_boundary": renderer.source_boundary,
                            "initialization_provenance": problem.initialization_provenance,
                            "failure_reason": "no_successful_candidate",
                        }
                        export_failure(failure, output / key)
                        record["runs"][key] = failure
                        record["success"] = False
                        continue
                else:
                    strengths = config["fixed_strengths"][case]
                    strengths = strengths.get(backend, strengths)
                    result = problem.fit(
                        strengths["lambda_source"],
                        strengths["lambda_potential"],
                        backend=backend,
                        **fit_options,
                    )
                results[key] = result
                metadata = export_result(
                    result,
                    observation,
                    output / key,
                    configuration=config,
                    search=search_record,
                    make_plots=config.get("make_plots", True),
                    affine_diagnostics=config.get("affine_diagnostics", False),
                )
                record["runs"][key] = {
                    name: metadata[name]
                    for name in (
                        "success",
                        "failure_reason",
                        "strengths",
                        "log_evidence",
                        "solver_info",
                        "evidence_method",
                        "determinant_method",
                        "search",
                    )
                }
                if result.success:
                    record["runs"][key]["summary"] = metadata["summary"]
                record["success"] &= result.success
                write_json(output / "run.json", record)
    write_json(output / "run.json", record)
    return ScientificRun(results, record, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("scientific_configuration.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/potential_correction_scientific")
    )
    parser.add_argument("--backend", choices=("dense", "operator", "both"))
    parser.add_argument("--mode", choices=("fixed", "tune"))
    parser.add_argument("--npix", type=int)
    parser.add_argument("--dpix", type=float)
    parser.add_argument("--mock-nsub", type=int)
    parser.add_argument("--inversion-nsub", type=int)
    parser.add_argument("--source-n", type=int)
    parser.add_argument("--potential-spacing", type=float)
    parser.add_argument("--dtype", choices=("float64", "float32"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    for key in ("npix", "dpix", "dtype"):
        if getattr(args, key) is not None:
            config["mock"][key] = getattr(args, key)
    if args.mock_nsub is not None:
        config["mock"]["nsub"] = args.mock_nsub
    for key in ("inversion_nsub", "source_n", "potential_spacing", "mode"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.backend is not None:
        config["backends"] = (
            ["dense", "operator"] if args.backend == "both" else [args.backend]
        )
    if args.no_plots:
        config["make_plots"] = False
    run = run_scientific(config, output=args.output)
    print(
        json.dumps(
            {
                "success": run.record["success"],
                "output": str(args.output.resolve()),
                "sizes": run.record["sizes"],
            },
            indent=2,
        )
    )
    if not run.record["success"]:
        raise SystemExit(1)
    return run


if __name__ == "__main__":
    main()
