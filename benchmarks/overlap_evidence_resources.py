"""Qualify the GPU resources of matched Euclid mock evidence evaluations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from math import prod
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

from lbfgs_operator_resources import _HLO_ARRAY


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / ".scratch/adaptive-image-overlap-evidence/experiments"
GPU_QUERY = "name,uuid,driver_version,memory.total"
SCORE_ATOL = 2e-6
OUTER_BATCH = 23
JOINT_MASK_RADIUS = 2.5
JOINT_BASIS_COUNT = 8


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, record):
    path.write_text(json.dumps(record, allow_nan=False, indent=2) + "\n")


def command_output(*command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def require_idle_gpu():
    processes = command_output(
        "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader",
    )
    if processes:
        raise RuntimeError(f"GPU compute processes are active before a measurement: {processes}")


def joint_mask(mock):
    import numpy as np

    y, x = np.ogrid[:mock.image.shape[0], :mock.image.shape[1]]
    x = (x - (mock.image.shape[1] - 1) / 2) * mock.dpix
    y = (y - (mock.image.shape[0] - 1) / 2) * mock.dpix
    return x**2 + y**2 > JOINT_MASK_RADIUS**2


def lens_bases():
    import numpy as np
    from TinyLensGpu.PhysicalModel import GaussianEllipse

    return [
        GaussianEllipse(
            flux=1.0, sigma=float(sigma), e1=0.12, e2=0.02,
            center_x=0.02, center_y=-0.01,
        )
        for sigma in np.geomspace(0.14, 1.25, JOINT_BASIS_COUNT)
    ]


def joint_image(mock, strengths, output):
    """Render the foreground once on CPU before any GPU measurement."""
    import caskade as ck
    import jax.numpy as jnp
    import numpy as np
    from TinyLensGpu.Inference.build_prior import make_prior_transformation
    from euclid_mock_replay import build_m3

    model = build_m3(
        mock, strengths, dtype=np.float64, evidence_logdet="block", lens_light=lens_bases(),
    )
    _, specs = make_prior_transformation(model)
    theta = jnp.asarray([mock.injected_mass[spec.name] for spec in specs])
    amplitudes = jnp.geomspace(0.1, 1.0, JOINT_BASIS_COUNT)
    with ck.ActiveContext(model):
        model.fill_params(theta)
        _, light, _ = model.reconstruct_component_images(jnp.zeros(mock.n**2), amplitudes)
    light = np.asarray(light)
    image = mock.image + (5.0 * np.max(mock.image) / np.max(light)) * light
    np.save(output / "joint_image.npy", image)


def allocation_audit(lowered, compiled, live_shapes, *, native, data, source, lens, window):
    """Check full matrix shapes, including transposed, batched and flattened forms."""
    forbidden = {
        "full_native_psf": (native, native),
        "full_active_psf": (data, data),
        "full_native_design": (native, source),
        "full_active_design": (data, source),
        "full_source_curvature": (source, source),
    }
    if lens:
        forbidden.update(
            full_joint_curvature=(source + lens, source + lens),
            full_joint_design=(data, source + lens),
            full_native_joint_design=(native, source + lens),
        )
    by_axes = {tuple(sorted(shape)): name for name, shape in forbidden.items()}
    by_elements = {prod(shape): name for name, shape in forbidden.items()}
    panel_shapes = set()
    counts = {}
    for phase, shapes in (
        ("lowered", {tuple(map(int, dimensions.split(","))) for _, dimensions
                     in _HLO_ARRAY.findall(lowered.compiler_ir("hlo").as_hlo_text())}),
        ("optimized", {tuple(map(int, dimensions.split(","))) for _, dimensions
                       in _HLO_ARRAY.findall(compiled.as_text())}),
        ("live", live_shapes),
    ):
        counts[phase] = len(shapes)
        for shape in shapes:
            for axes in combinations(shape, 2):
                name = by_axes.get(tuple(sorted(axes)))
                if name is not None:
                    raise AssertionError(f"{phase} contains {name} {shape}.")
            if prod(shape) in by_elements or any(axis in by_elements for axis in shape):
                raise AssertionError(f"{phase} contains a flattened complete matrix {shape}.")
            if len(shape) >= 2 and window in shape[-2:]:
                panel_shapes.add(shape)
    if counts["lowered"] == 0 or counts["optimized"] == 0:
        raise RuntimeError("The HLO audit did not observe compiled array shapes.")
    return {
        "shape_counts": counts,
        "prohibited_matrix_shapes": forbidden,
        "observed_shapes_with_window_axis": sorted(panel_shapes),
        "violations": [],
    }


def worker(args):
    sys.path[:0] = [str(ROOT), str(EXPERIMENTS)]
    import TinyLensGpu
    import jax
    import jax.numpy as jnp
    import numpy as np
    from TinyLensGpu.Inference.build_likelihood import DEFAULT_OVERLAP_CHUNK_SIZE, make_likelihood
    from TinyLensGpu.Inference.build_prior import make_prior_transformation
    from TinyLensGpu.ObservationModel.LensImage._pixelized_regularization import HERMITE_COVERAGE_OFFSET
    from euclid_mock_replay import build_m3, load_mock

    if Path(TinyLensGpu.__file__).resolve().parent != ROOT / "TinyLensGpu":
        raise RuntimeError("The worker did not import its checkout.")
    mock = load_mock(args.archive)
    replay = read_json(EXPERIMENTS / "euclid_mock_results.json")
    controls = read_json(EXPERIMENTS / "m3_optimizer_config.json")
    strengths = replay["m2"]["selected"]
    if (mock.hashes != replay["configuration"]["inputs_sha256"]
            or HERMITE_COVERAGE_OFFSET != controls["hermite_coverage_offset"]
            or HERMITE_COVERAGE_OFFSET != replay["configuration"]["m2_m3"]["hermite_coverage_offset"]
            or strengths != replay["m3"]["fixed_strengths"]):
        raise ValueError("The GPU benchmark inputs differ from the qualified mock.")
    if args.case == "joint":
        mock = replace(mock, mask=joint_mask(mock))
    if args.worker == "fixture":
        joint_image(mock, strengths, args.output)
        return

    device, = jax.devices()
    if device.platform != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("A single CUDA device with Float64 enabled is required.")
    if args.case == "joint":
        mock = replace(mock, image=np.load(args.output / "joint_image.npy"))

    started = perf_counter()
    model = build_m3(
        mock, strengths, dtype=np.float64, evidence_logdet=args.evidence,
        lens_light=lens_bases() if args.case == "joint" else (),
    )
    model.pcg_max_iter = controls["solver"]["max_iterations"]
    model.pcg_rtol = controls["solver"]["relative_tolerance"]
    _, specs = make_prior_transformation(model)
    names = [spec.name for spec in specs]
    points = replay["m3"]["points"]
    first = np.array([points["injected_free_mass"]["parameters"][name] for name in names])
    last = np.array([points["posterior_peak"]["parameters"][name] for name in names])
    point = jnp.asarray(first)
    jax.block_until_ready(jax.live_arrays())
    preparation_seconds = perf_counter() - started
    preparation_peak_bytes = device.memory_stats()["peak_bytes_in_use"]

    chunk = None if args.chunk == "auto" else int(args.chunk)
    if args.worker == "batch":
        function = make_likelihood(model, vectorized=True, vectorized_chunk_size=chunk)
        argument = jnp.asarray(np.linspace(first, last, args.outer_batch))
    else:
        function = lambda theta: model(theta)
        if args.worker == "gradient":
            function = jax.value_and_grad(function)
        argument = point

    argument.block_until_ready()
    started = perf_counter()
    lowered = jax.jit(function).lower(argument)
    compiled = lowered.compile()
    compilation_seconds = perf_counter() - started
    started = perf_counter()
    result = jax.block_until_ready(compiled(argument))
    first_execution_seconds = perf_counter() - started
    warm_seconds = []
    for _ in range(args.repeats):
        started = perf_counter()
        jax.block_until_ready(compiled(argument))
        warm_seconds.append(perf_counter() - started)
    allocator = device.memory_stats()
    memory = compiled.memory_analysis()
    live_shapes = {tuple(array.shape) for array in jax.live_arrays()}
    audit = allocation_audit(
        lowered, compiled, live_shapes, native=mock.image.size,
        data=int((~mock.mask).sum()), source=mock.n**2, lens=model.n_lens_light,
        window=(2 * model.evidence_block_size)**2,
    )

    if args.worker == "gradient":
        result, gradient = result
        gradient = np.asarray(gradient)
        if not np.isfinite(gradient).all():
            raise AssertionError("The complete score gradient is nonfinite.")
    else:
        gradient = None
    scores = np.asarray(result)
    expected_shape = (args.outer_batch,) if args.worker == "batch" else ()
    if (scores.shape != expected_shape or not np.isfinite(scores).all()
            or np.any(scores <= -1e10)):
        raise AssertionError(f"Invalid {args.worker} scores: {scores}")

    write_json(args.output / "worker.json", {
        "case": args.case,
        "method": args.evidence,
        "phase": args.worker,
        "requested_chunk": args.chunk if args.worker == "batch" else None,
        "effective_chunk": (
            (DEFAULT_OVERLAP_CHUNK_SIZE if chunk is None else chunk)
            if args.worker == "batch" else None
        ),
        "device": device.device_kind,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "dtype": str(point.dtype),
        "active_pixels": int((~mock.mask).sum()),
        "lens_basis_count": model.n_lens_light,
        "lens_light_regularization": float(model.lens_light_regularization),
        "block_size": model.block_size,
        "evidence_block_size": model.evidence_block_size,
        "pcg_max_iter": model.pcg_max_iter,
        "pcg_rtol": model.pcg_rtol,
        "image_sha256": sha256(mock.image.tobytes()).hexdigest(),
        "mask_sha256": sha256(mock.mask.tobytes()).hexdigest(),
        "preparation_seconds": preparation_seconds,
        "first_compilation_seconds": compilation_seconds,
        "first_execution_seconds": first_execution_seconds,
        "warm_seconds": warm_seconds,
        "median_warm_seconds": statistics.median(warm_seconds),
        "preparation_allocator_peak_bytes": preparation_peak_bytes,
        "allocator_peak_bytes": allocator["peak_bytes_in_use"],
        "allocator_pool_peak_bytes": allocator["peak_pool_bytes"],
        "allocator_limit_bytes": allocator["bytes_limit"],
        "compiled_temporary_bytes": memory.temp_size_in_bytes,
        "workspace_audit": audit,
        "scores": scores.tolist(),
        "gradient": None if gradient is None else gradient.tolist(),
        "parameter_names": names,
        "proposal_sha256": sha256(np.asarray(argument).tobytes()).hexdigest(),
    })


def run_worker(args, phase, case, evidence="block", chunk="1"):
    if phase != "fixture":
        require_idle_gpu()
    subprocess.run(
        [
            sys.executable, str(Path(__file__).resolve()),
            "--archive", str(args.archive), "--output", str(args.output),
            "--outer-batch", str(args.outer_batch), "--repeats", str(args.repeats),
            "--worker", phase, "--case", case, "--evidence", evidence, "--chunk", chunk,
        ],
        cwd=ROOT,
        check=True,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu" if phase == "fixture" else "cuda",
            "JAX_ENABLE_X64": "true",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
    )
    if phase != "fixture":
        return read_json(args.output / "worker.json")


def check_score_agreement(records):
    comparisons = {}
    matched_fields = (
        "device", "jax_version", "dtype", "active_pixels", "lens_basis_count",
        "lens_light_regularization", "block_size", "evidence_block_size", "pcg_max_iter",
        "pcg_rtol", "image_sha256", "mask_sha256", "proposal_sha256",
    )
    for case in ("source", "joint"):
        for phase in ("score", "gradient", "batch"):
            rows = [row for row in records if (row["case"], row["phase"]) == (case, phase)]
            if any(len({row[key] for row in rows}) != 1 for key in matched_fields):
                raise AssertionError(f"Physical or numerical inputs differ in {case}/{phase}.")
        for evidence in ("block", "overlap"):
            point, derivative, narrow, wide = [
                row for row in records if (row["case"], row["method"]) == (case, evidence)
            ]
            difference = max(abs(a - b) for a, b in zip(narrow["scores"], wide["scores"], strict=True))
            scalar_difference = max(
                abs(point["scores"] - other)
                for other in (derivative["scores"], narrow["scores"][0], wide["scores"][0])
            )
            if max(difference, scalar_difference) > SCORE_ATOL:
                raise AssertionError(f"Scores disagree in {case}/{evidence}.")
            comparisons[f"{case}/{evidence}"] = {
                "max_absolute_difference_between_chunks": difference,
                "max_absolute_difference_from_first_scalar": scalar_difference,
            }
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--outer-batch", default=OUTER_BATCH, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--worker", choices=("fixture", "score", "gradient", "batch"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=("source", "joint"), help=argparse.SUPPRESS)
    parser.add_argument("--evidence", choices=("block", "overlap"), help=argparse.SUPPRESS)
    parser.add_argument("--chunk", choices=("auto", "1", "5"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.archive = args.archive.resolve()
    args.output = args.output.resolve()
    if args.outer_batch < 6 or args.repeats < 1:
        parser.error("The outer batch needs at least six proposals and repeats must be positive.")
    if args.worker:
        worker(args)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    run_worker(args, "fixture", "joint")
    replay = read_json(EXPERIMENTS / "euclid_mock_results.json")
    report = {
        "command": [sys.executable, *sys.argv],
        "utc_start": datetime.now(timezone.utc).isoformat(),
        "library_revision": command_output("git", "rev-parse", "HEAD"),
        "gpu": command_output("nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader"),
        "outer_batch": args.outer_batch,
        "warm_repeats": args.repeats,
        "mock_input_sha256": replay["configuration"]["inputs_sha256"],
        "fixed_strengths": replay["m2"]["selected"],
        "source_grid": replay["configuration"]["source_node_bbox"],
        "source_grid_side": replay["configuration"]["source_grid_side"],
        "subpixels_per_side": replay["configuration"]["subpixels_per_side"],
        "position_likelihood": replay["configuration"]["m2_m3"]["position_likelihood"],
        "hermite_coverage_offset": replay["configuration"]["m2_m3"]["hermite_coverage_offset"],
        "environment": {
            key: os.environ[key] for key in (
                "CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION",
                "OPENBLAS_NUM_THREADS",
            ) if key in os.environ
        },
        "joint_mask_radius_arcsec": JOINT_MASK_RADIUS,
        "max_allowed_absolute_score_difference": SCORE_ATOL,
        "runs": [],
    }
    for case in ("source", "joint"):
        for evidence in ("block", "overlap"):
            narrow_chunk = "auto" if evidence == "overlap" else "1"
            for phase, chunk in (("score", "1"), ("gradient", "1"), ("batch", narrow_chunk), ("batch", "5")):
                row = run_worker(args, phase, case, evidence, chunk)
                report["runs"].append(row)
                write_json(args.output / "qualification.json", report)
                print(f"{case} {evidence} {phase} chunk={chunk}: "
                      f"{row['median_warm_seconds']:.4f} s, "
                      f"{row['allocator_peak_bytes'] / 2**20:.1f} MiB", flush=True)
    report["score_agreement"] = check_score_agreement(report["runs"])
    write_json(args.output / "qualification.json", report)
    (args.output / "worker.json").unlink()


if __name__ == "__main__":
    main()
