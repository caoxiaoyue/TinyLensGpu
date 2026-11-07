"""Matched nonlinear source–potential reconstructions from truth-derived nodes.

The macro and both geometries remain fixed. Every fit starts from the same
source-node intensities and zero potential; all later gradients are current
pixelized-source derivatives. Exports consume only the public fit result.
"""

import argparse
from dataclasses import asdict, dataclass
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
from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from examples.potential_correction._scientific_partitions import (
    geometry_fingerprint,
    prepare_partition,
)
from examples.potential_correction.calibrate_single_step import source_incidence_blocks
from examples.potential_correction.mock import MockConfig, generate_mock_pair
from examples.potential_correction.result_products import (
    compare_reconstructions,
    export_failure,
    export_result,
    load_result,
    optimization_record,
    write_json,
)


@dataclass(frozen=True)
class IterativeScientificRun:
    results: dict
    record: dict
    output: Path


def _partition(renderer, initial_source, config):
    partition_source = config.get("partition_source", "initial_nodal")
    if partition_source not in ("single_step", "initial_nodal"):
        raise ValueError("partition_source must be single_step or initial_nodal")
    default_file = (
        Path(__file__).with_name("scientific_partition.json")
        if partition_source == "single_step"
        else None
    )

    mapping = renderer.source_mapping()
    gx, gy = renderer.source_spatial_gradient(initial_source, mapping)

    def build_blocks():
        return source_incidence_blocks(
            renderer,
            mapping,
            gx,
            gy,
            source_tile=config["source_tile"],
            block_size=config["block_size"],
        )

    blocks, provenance = prepare_partition(
        renderer,
        config,
        build_blocks,
        source="initial_nodal",
        method="source_incidence",
        default_file=default_file,
        incidence_gradients=(gx, gy),
    )
    if provenance["origin"] == "generated":
        provenance["gradient"] = "analytic derivative of initial pixelized source"
    return blocks, provenance


def _comparisons(results, reference_directory, mock, cases):
    """Read saved public results; comparisons never construct or refit a solver."""
    report = dict(
        algorithms={},
        iterative_backends={},
        single_step_backends={},
        operator_algorithm_comparisons={},
        primary_algorithm_comparison="dense single-step versus dense iterative, at recorded strengths",
        interpretation="raw fields; distinguish inference objectives/local linearization from determinant approximations; no detection statistic",
    )
    for case in cases:
        dense, operator = (
            results.get(f"{case}/{backend}") for backend in ("dense", "operator")
        )
        if dense is not None and operator is not None:
            report["iterative_backends"][case] = compare_reconstructions(
                dense, operator
            )
        if reference_directory is None:
            continue
        references = {}
        for backend in ("dense", "operator"):
            path = Path(reference_directory) / case / backend
            if (path / "result.json").exists() and not (path / "failure.json").exists():
                metadata = json.loads((path / "result.json").read_text())
                if metadata["mock_configuration"] != asdict(mock):
                    raise ValueError(
                        "Single-step comparison requires the identical physical mock, PSF, noise and precision"
                    )
                references[backend] = load_result(path)
        if "dense" in references and dense is not None:
            report["algorithms"][case] = compare_reconstructions(
                references["dense"], dense
            )
        if "operator" in references and operator is not None:
            report["operator_algorithm_comparisons"][case] = compare_reconstructions(
                references["operator"], operator
            )
        if len(references) == 2:
            report["single_step_backends"][case] = compare_reconstructions(
                references["dense"], references["operator"]
            )
    report["single_step_reference"] = (
        None if reference_directory is None else str(reference_directory)
    )
    return report


def run_iterative_scientific(configuration, *, output, single_step_reference=None):
    """Fit fixed strengths or optimize around complete reset-per-candidate runs."""
    config = dict(configuration)
    mode = config.get("mode", "fixed")
    if mode not in ("fixed", "tune"):
        raise ValueError("mode must be fixed or tune")
    cases = config.get("cases", ["subhalo", "control"])
    if (
        not cases
        or len(set(cases)) != len(cases)
        or any(case not in ("subhalo", "control") for case in cases)
    ):
        raise ValueError("cases must be a nonempty unique subhalo/control selection")
    backends = config.get("backends", ["dense", "operator"])
    if (
        not backends
        or len(set(backends)) != len(backends)
        or any(b not in ("dense", "operator") for b in backends)
    ):
        raise ValueError("backends must be a nonempty unique dense/operator selection")
    mock = MockConfig.from_dict(config["mock"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "configuration.json", config)
    pair = generate_mock_pair(mock)
    with jax.experimental.enable_x64(mock.dtype == "float64"):
        sim = SimulatorConfig(
            mock.dpix,
            mock.npix,
            pair.control.psf_kernel,
            config["inversion_nsub"],
            pair.control.mask,
        )
        renderer = PotentialCorrectionRenderer(
            pair.control.macro,
            sim,
            config["source_n"],
            tuple(config["source_bbox"]),
            potential_spacing=config["potential_spacing"],
            dtype=mock.dtype,
            source_interpolation=config.get("source_interpolation", "bilinear"),
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        initial_source = pair.control.source_truth.brightness(sx, sy).ravel()
        blocks, partition = _partition(renderer, initial_source, config)
        options = {
            "block_size": config["block_size"],
            "block_indices": blocks,
            "pcg_max_iter": config["pcg_max_iter"],
            "pcg_rtol": config["pcg_rtol"],
            "pcg_atol": config.get("pcg_atol", 0.0),
        }
        root = Path(__file__).resolve().parents[2]
        files = [
            Path(__file__),
            Path(__file__).with_name("result_products.py"),
            Path(__file__).with_name("mock.py"),
            Path(__file__).with_name("calibrate_single_step.py"),
            Path(__file__).with_name("_scientific_partitions.py"),
        ]
        files.extend(
            root.glob(
                "TinyLensGpu/ObservationModel/LensImage/*potential_correction*.py"
            )
        )
        files.extend(
            root / path
            for path in (
                "TinyLensGpu/utils/cg_solver.py",
                "TinyLensGpu/utils/lensing/mapping.py",
                "TinyLensGpu/ForwardSimulation/LensImage/potential_correction.py",
                "TinyLensGpu/Inference/potential_correction.py",
            )
        )
        record = {
            "configuration": config,
            "success": True,
            "runs": {},
            "source_interpolation": renderer.source_interpolation,
            "source_boundary": renderer.source_boundary,
            "block_partition": partition,
            "sizes": {
                "data": renderer.n_active,
                "source": renderer.n_source,
                "potential_core": renderer.potential_grid.n_core,
                "potential_support": renderer.potential_grid.n_unknown,
                "query_subpixels": len(renderer.query_x),
                "image_shape": renderer.image_shape,
            },
            "software": {
                "python": platform.python_version(),
                "jax": jax.__version__,
                "numpy": np.__version__,
                "devices": [d.device_kind for d in jax.devices()],
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "git_dirty": bool(
                    subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=root,
                        capture_output=True,
                        text=True,
                    ).stdout
                ),
                "implementation_sha256": {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files
                },
            },
            "initialization": "truth brightness at fixed source NODES; zero potential; each fit resets; current nodal gradients thereafter",
            "scope": "fixed true macro nonlinear reconstruction; no detection significance, mass intervals or timing claim",
        }
        results = {}
        for case in cases:
            observation = getattr(pair, case)
            problem = IterativePotentialCorrection(
                renderer,
                observation.image_data,
                observation.noise_map,
                initial_source,
                lambda_amplitude=config["lambda_amplitude"],
                solve_rtol=config["solve_rtol"],
                solve_atol=config.get("solve_atol", 0.0),
                source_regularization=config.get(
                    "source_regularization", "second-order"
                ),
                initialization_provenance={
                    "source": "analytic mock truth brightness sampled at fixed source nodes",
                    "source_geometry": "fixed explicit box; no expansion",
                },
            )
            for backend in backends:
                key = f"{case}/{backend}"
                fit_options = {
                    **config.get("iteration_options", {}),
                    **config.get("case_iteration_options", {}).get(case, {}),
                    **(options if backend == "operator" else {}),
                }
                search_metadata = None
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
                    search_metadata = optimization_record(search)
                    result = search.reconstruction
                    if result is None:
                        failure = dict(
                            success=False,
                            status="failed",
                            algorithm="iterative",
                            backend=backend,
                            dtype=mock.dtype,
                            failure_reason="no_successful_inner_reconstruction",
                            configuration=config,
                            source_interpolation=renderer.source_interpolation,
                            source_boundary=renderer.source_boundary,
                            search=search_metadata,
                            initialization_provenance=problem.initialization_provenance,
                        )
                        export_failure(failure, output / key)
                        record["runs"][key] = failure
                        record["success"] = False
                        write_json(output / "run.json", record)
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
                    search=search_metadata,
                    make_plots=config.get("make_plots", True),
                    affine_diagnostics=config.get("affine_diagnostics", False),
                )
                record["runs"][key] = {
                    k: metadata[k]
                    for k in (
                        "success",
                        "failure_reason",
                        "termination_reason",
                        "strengths",
                        "log_evidence",
                        "evidence_method",
                        "determinant_method",
                        "solver_info",
                    )
                }
                record["runs"][key]["configuration"] = result.configuration
                record["runs"][key]["search"] = search_metadata
                if result.success:
                    record["runs"][key]["summary"] = metadata["summary"]
                record["success"] &= result.success
                write_json(output / "run.json", record)
    record["comparisons"] = _comparisons(results, single_step_reference, mock, cases)
    write_json(output / "comparisons.json", record["comparisons"])
    write_json(output / "run.json", record)
    return IterativeScientificRun(results, record, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("iterative_scientific_configuration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/potential_correction_iterative_scientific"),
    )
    parser.add_argument("--backend", choices=("dense", "operator", "both"))
    parser.add_argument("--mode", choices=("fixed", "tune"))
    parser.add_argument("--single-step-reference", type=Path)
    parser.add_argument("--partition-source", choices=("single_step", "initial_nodal"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.backend:
        config["backends"] = (
            ["dense", "operator"] if args.backend == "both" else [args.backend]
        )
    if args.partition_source:
        config["partition_source"] = args.partition_source
    if args.mode:
        config["mode"] = args.mode
    if args.no_plots:
        config["make_plots"] = False
    run = run_iterative_scientific(
        config, output=args.output, single_step_reference=args.single_step_reference
    )
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
