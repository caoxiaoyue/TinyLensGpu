"""Isolated source–potential precision and GPU workflow measurements.

The parent imports no JAX. Each worker fits one frozen physical input, one
algorithm/backend/dtype, with all lazy preparation included in the first fit.
Use --prepare once to save float64 inputs; other dtypes cast those same inputs.
"""

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def json_value(value):
    if dataclasses.is_dataclass(value):
        return {
            f.name: json_value(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2) + "\n")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    import numpy as np

    a = np.asarray(value)
    return hashlib.sha256(
        str((a.shape, a.dtype.str)).encode() + a.tobytes()
    ).hexdigest()


def synchronize(value):
    """Wait for arrays inside result/grid dataclasses without deep-copying them."""
    import jax

    arrays = []

    def visit(v):
        if isinstance(v, jax.Array):
            arrays.append(v)
        elif dataclasses.is_dataclass(v):
            for f in dataclasses.fields(v):
                visit(getattr(v, f.name))
        elif isinstance(v, dict):
            for x in v.values():
                visit(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                visit(x)

    visit(value)
    jax.block_until_ready(arrays)


def prepare(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear
    from examples.potential_correction.mock.single_step.calibrate import (
        source_incidence_blocks,
    )
    from examples.potential_correction.mock import (
        MockConfig,
        SourceTruth,
        generate_mock_pair,
    )

    with jax.experimental.enable_x64():
        if args.case == "coupled37":
            cfg = dict(
                name="coupled37",
                npix=4,
                dpix=0.1,
                inversion_nsub=1,
                source_n=2,
                source_bbox=[-0.5, 0.5, -0.5, 0.5],
                potential_spacing=0.2,
                lambda_amplitude=20.0,
                lambda_source=0.3,
                lambda_potential=0.02,
                source_regularization="zero-order",
                block_size=7,
                macro=dict(kind="zero"),
                initialization="fixed four truth source nodes; zero potential",
            )
            macro = PhysicalModel(
                lens_mass=[Shear(gamma1=0.0, gamma2=0.0)],
                source_light=[],
                lens_light=[],
            )
            psf = jnp.ones((1, 1))
            mask = np.zeros((4, 4), bool)
            sim = SimulatorConfig(0.1, 4, psf, 1, mask)
            renderer = PotentialCorrectionRenderer(
                macro, sim, 2, tuple(cfg["source_bbox"]), dtype="float64"
            )
            initial = jnp.asarray([0.4, 1.1, 0.2, 1.4])
            g = renderer.potential_grid
            potential = 0.003 * jnp.exp(-((g.x - 0.07) ** 2 + (g.y + 0.03) ** 2) / 0.02)
            image = np.asarray(
                renderer.render(initial, potential).model_image
            ) + 0.002 * np.sin(np.arange(16)).reshape(4, 4)
            noise = np.full((4, 4), 0.07)
            mapping = renderer.source_mapping()
            gx, gy = renderer.source_spatial_gradient(initial, mapping)
            # The standard field-separated bounded partition is part of this fixture.
            # Public default partitions split each field at least once.
            blocks = None
        else:
            cfg = json.loads(
                (
                    ROOT
                    / "examples/potential_correction/mock/iterative/configurations/scientific.json"
                ).read_text()
            )
            n = int(args.case.removeprefix("scientific"))
            cfg["name"] = args.case
            if n != 200:
                side, h, k, tile, cap = {
                    32: (17, 0.15, 5, 8, 128),
                    64: (33, 0.1, 9, 16, 512),
                    128: (49, 0.075, 17, 24, 1024),
                }[n]
                cfg["mock"].update(npix=n, dpix=2.4 / n, psf_size=k, dtype="float64")
                cfg.update(
                    source_n=side, potential_spacing=h, source_tile=tile, block_size=cap
                )
            if args.source_n is not None:
                cfg["source_n"] = args.source_n
                cfg["name"] += f"-source{args.source_n}"
            cfg.update(npix=n, dpix=cfg["mock"]["dpix"], macro=dict(kind="scientific"))
            strengths = cfg["fixed_strengths"][args.observation]
            cfg.update(
                lambda_source=strengths["lambda_source"],
                lambda_potential=strengths["lambda_potential"],
            )
            mock = MockConfig.from_dict(cfg["mock"])
            if args.reference_inputs or (n == 200 and args.reference):
                original = args.reference_inputs or (
                    Path(args.reference)
                    / args.observation
                    / "dense"
                    / "reconstruction.npz"
                )
                a = np.load(original)
                image, noise, psf, mask = (
                    a[k]
                    for k in (
                        "image_data",
                        "noise_map",
                        "psf" if args.reference_inputs else "psf_kernel",
                        "mask",
                    )
                )
                from TinyLensGpu.PhysicalModel import SIS

                macro = PhysicalModel(
                    lens_mass=[
                        SIS(
                            theta_E=mock.theta_E,
                            center_x=mock.macro_center_x,
                            center_y=mock.macro_center_y,
                        ),
                        Shear(gamma1=mock.gamma1, gamma2=mock.gamma2),
                    ],
                    source_light=[],
                    lens_light=[],
                )
                cfg["reference_input_sha256"] = sha_file(original)
                if args.reference_inputs:
                    original_cfg = json.loads(str(a["metadata"]))
                    if (
                        original_cfg["mock"] != cfg["mock"]
                        or original_cfg["observation"] != args.observation
                    ):
                        raise ValueError(
                            "A source-axis fixture must preserve the original observation configuration"
                        )
                    cfg["source_axis_observations_match"] = True
            else:
                pair = generate_mock_pair(mock)
                obs = getattr(pair, args.observation)
                image, noise, psf, mask, macro = (
                    obs.image_data,
                    obs.noise_map,
                    obs.psf_kernel,
                    obs.mask,
                    obs.macro,
                )
            sim = SimulatorConfig(cfg["dpix"], n, psf, cfg["inversion_nsub"], mask)
            renderer = PotentialCorrectionRenderer(
                macro,
                sim,
                cfg["source_n"],
                tuple(cfg["source_bbox"]),
                potential_spacing=cfg["potential_spacing"],
                dtype="float64",
            )
            sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
            truth = SourceTruth(**cfg["mock"]["source"])
            initial = truth.brightness(sx, sy).ravel()
            mapping = renderer.source_mapping()
            gx, gy = truth.gradient(mapping.beta_x, mapping.beta_y)
            if n == 200 and args.source_n is None:
                from examples.potential_correction.mock.iterative.run import (
                    geometry_fingerprint,
                )

                recorded = json.loads(
                    (
                        ROOT / "examples/potential_correction/shared/scientific_partition.json"
                    ).read_text()
                )
                if recorded["geometry_sha256"] != geometry_fingerprint(renderer):
                    raise ValueError(
                        "Scientific float64 fixture differs from the packaged partition geometry"
                    )
                blocks = recorded["indices"]
                cfg["packaged_partition_geometry_sha256"] = recorded["geometry_sha256"]
            else:
                blocks = [
                    b.tolist()
                    for b in source_incidence_blocks(
                        renderer,
                        mapping,
                        gx,
                        gy,
                        source_tile=cfg["source_tile"],
                        block_size=cfg["block_size"],
                    )
                ]
        cfg["observation"] = args.observation
        cfg["block_indices"] = blocks
        grid = renderer.potential_grid
        values = dict(
            image_data=np.asarray(image, dtype=np.float64),
            noise_map=np.asarray(noise, dtype=np.float64),
            psf=np.asarray(psf, dtype=np.float64),
            mask=np.asarray(mask, dtype=bool),
            initial_source=np.asarray(initial, dtype=np.float64),
            gradient_x=np.asarray(gx, dtype=np.float64),
            gradient_y=np.asarray(gy, dtype=np.float64),
            node_indices=np.asarray(grid.node_indices),
            core_indices=np.asarray(grid.core_indices),
            interpolation_indices=np.asarray(grid.interpolation_indices),
            query_x=np.asarray(grid.query_x),
            query_y=np.asarray(grid.query_y),
            metadata=np.asarray(json.dumps(cfg)),
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **values)
        write_json(
            Path(args.output).with_suffix(".json"),
            dict(
                configuration=cfg,
                input_sha256=sha_file(args.output),
                float64_fixture=True,
            ),
        )


def software(device):
    versions = {}
    for name in (
        "jax",
        "jaxlib",
        "jax-cuda12-plugin",
        "jax-cuda12-pjrt",
        "numpy",
        "scipy",
        "caskade",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    files = list(
        (ROOT / "TinyLensGpu/ObservationModel/LensImage").glob(
            "*potential_correction*.py"
        )
    )
    files += [
        Path(__file__),
        ROOT / "TinyLensGpu/ForwardSimulation/LensImage/potential_correction.py",
        ROOT / "TinyLensGpu/utils/inversion/regularization.py",
        ROOT / "TinyLensGpu/utils/cg_solver.py",
        ROOT / "TinyLensGpu/Inference/potential_correction.py",
    ]

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip()

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
    )
    return dict(
        python=platform.python_version(),
        versions=versions,
        platform=device.platform,
        device=device.device_kind,
        gpu_query=gpu.stdout.strip(),
        git_commit=git("rev-parse", "HEAD"),
        git_dirty=bool(git("status", "--porcelain")),
        implementation_sha256={str(p.relative_to(ROOT)): sha_file(p) for p in files},
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("JAX_", "XLA_", "CUDA_", "OMP_", "OPENBLAS_"))
        },
    )


def worker(args):
    entered = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.ObservationModel import (
        SingleStepPotentialCorrection,
        IterativePotentialCorrection,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear, SIS

    defaults = dict(
        x64=jax.config.x64_enabled, matmul=jax.config.jax_default_matmul_precision
    )
    device = jax.devices()[0]
    snapshots = []

    def snapshot(phase):
        stats = device.memory_stats()
        snapshots.append(
            dict(
                phase=phase,
                stats={} if stats is None else stats,
                semantics="cumulative_allocator_high_water_since_device_initialization",
            )
        )

    snapshot("device_initialized")
    fixture = np.load(args.inputs)
    cfg = json.loads(str(fixture["metadata"]))
    dtype = np.dtype(args.dtype)
    profile = args.profile or ("strict" if cfg["name"] == "coupled37" else "scientific")
    with jax.experimental.enable_x64(dtype == np.dtype("float64")):

        def cast(value):
            return jnp.asarray(value, dtype=dtype)

        if cfg["macro"]["kind"] == "zero":
            macro = PhysicalModel(
                lens_mass=[Shear(gamma1=cast(0.0), gamma2=cast(0.0))],
                source_light=[],
                lens_light=[],
            )
        else:
            m = cfg["mock"]
            macro = PhysicalModel(
                lens_mass=[
                    SIS(
                        theta_E=cast(m["theta_E"]),
                        center_x=cast(m["macro_center_x"]),
                        center_y=cast(m["macro_center_y"]),
                    ),
                    Shear(gamma1=cast(m["gamma1"]), gamma2=cast(m["gamma2"])),
                ],
                source_light=[],
                lens_light=[],
            )
        sim = SimulatorConfig(
            cfg["dpix"],
            cfg["npix"],
            cast(fixture["psf"]),
            cfg["inversion_nsub"],
            fixture["mask"],
        )
        renderer = PotentialCorrectionRenderer(
            macro,
            sim,
            cfg["source_n"],
            tuple(cfg["source_bbox"]),
            potential_spacing=cfg["potential_spacing"],
            dtype=args.dtype,
        )
        grid = renderer.potential_grid
        topology = all(
            np.array_equal(np.asarray(getattr(grid, k)), fixture[k])
            for k in ("node_indices", "core_indices")
        )
        geometry = dict(
            integer_topology_matches_fixture=topology,
            node_sha256=array_hash(grid.node_indices),
            core_sha256=array_hash(grid.core_indices),
            query_sha256=array_hash(grid.query_x),
            mask_sha256=array_hash(fixture["mask"]),
            source_axis_sha256=array_hash(renderer.source_x_axis),
            changed_query_corner_rows=(
                int(
                    np.any(
                        np.asarray(grid.interpolation_indices)
                        != fixture["interpolation_indices"],
                        axis=1,
                    ).sum()
                )
                if np.asarray(grid.interpolation_indices).shape
                == fixture["interpolation_indices"].shape
                else None
            ),
        )
        if not topology and cfg["block_indices"] is not None:
            raise ValueError(
                "Precision changed integer topology; recorded partition cannot be reused without a separately declared configuration"
            )
        initial = cast(fixture["initial_source"])
        gradient = (cast(fixture["gradient_x"]), cast(fixture["gradient_y"]))
        controls = {}
        solve_rtol = cfg.get("solve_rtol", 1e-8)
        solve_atol = cfg.get("solve_atol", 0.0)
        pcg_rtol = cfg.get("pcg_rtol", 1e-12)
        if profile == "small32":
            controls = dict(
                gradient_atol=1e-4,
                gradient_rtol=1e-5,
                objective_rtol=1e-6,
                step_rtol=1e-4,
            )
            # Tight enough for the independent field guard in the coupled
            # single-step problem; recursive 1e-5 misses it even in float64.
            solve_rtol, pcg_rtol = 1e-5, 1e-8
            # Observed float32 dense residuals near small RHS reach 6–7e-7.
            # This named absolute floor is checked against independent fields.
            solve_atol = 1e-6
        elif profile == "scientific":
            controls = {
                "gradient_rtol": 1e-8,
                "objective_rtol": 1e-10,
                "step_rtol": 1e-7,
                **cfg.get("iteration_options", {}),
                **cfg.get("case_iteration_options", {}).get(cfg["observation"], {}),
            }
        if args.gradient_atol is not None:
            controls["gradient_atol"] = args.gradient_atol
        common = dict(
            lambda_amplitude=cfg["lambda_amplitude"],
            source_regularization=cfg["source_regularization"],
            solve_rtol=solve_rtol,
        )
        if args.algorithm == "single_step":
            problem = SingleStepPotentialCorrection(
                renderer,
                cast(fixture["image_data"]),
                cast(fixture["noise_map"]),
                lambda bx, by: gradient,
                **common,
            )
            options = {}
        else:
            problem = IterativePotentialCorrection(
                renderer,
                cast(fixture["image_data"]),
                cast(fixture["noise_map"]),
                initial,
                solve_atol=solve_atol,
                **common,
            )
            options = controls
        if args.backend == "operator":
            options = {
                **options,
                "block_size": cfg["block_size"],
                "block_indices": cfg["block_indices"],
                "pcg_rtol": pcg_rtol,
                "pcg_max_iter": cfg.get("pcg_max_iter", 400),
                "pcg_atol": 0.0,
            }
        synchronize(
            (
                initial,
                gradient,
                grid,
                problem.data_active,
                problem.noise_active,
                renderer.psf_fft,
                renderer.source_x_axis,
            )
        )
        setup_end = time.perf_counter()
        snapshot("setup")
        timing = dict(
            setup_seconds=setup_end - entered,
            warm_fit_seconds=[],
            synchronization="recursive dataclass/collection array leaves inside each timed fit",
        )
        search_record = None
        if args.search:
            from TinyLensGpu.Inference import optimize_potential_correction

            bounds = np.log([[0.25, 0.35], [0.015, 0.025]])
            t = time.perf_counter()
            search = optimize_potential_correction(
                problem,
                backend=args.backend,
                fit_options=options,
                log_bounds=bounds,
                maxfun=args.search,
                maxiter=100,
                len_tol=1e-3,
            )
            result = search.reconstruction
            synchronize(result)
            timing["search_seconds"] = time.perf_counter() - t
            if result is None:
                raise RuntimeError(
                    "All search candidates failed; see retained worker log"
                )
            search_record = dict(
                has_valid_fit=search.has_valid_fit,
                optimizer_success=search.optimizer_success,
                message=search.message,
                nfev=search.nfev,
                log_bounds=search.log_bounds,
                log_strengths=search.log_strengths,
                boundary_sides=search.boundary_sides,
                evaluations=search.evaluations,
                bounds_dtype=str(bounds.dtype),
            )
            timing["cold_workflow_seconds"] = time.perf_counter() - entered
            snapshot("first_fit")
            snapshot("warm_fit")
        else:
            t = time.perf_counter()
            result = problem.fit(
                cfg["lambda_source"],
                cfg["lambda_potential"],
                backend=args.backend,
                **options,
            )
            synchronize(result)
            timing["first_fit_compile_inclusive_seconds"] = time.perf_counter() - t
            timing["cold_workflow_seconds"] = time.perf_counter() - entered
            snapshot("first_fit")

            def quality(r):
                row = dict(
                    success=r.success,
                    termination_reason=r.termination_reason,
                    solver_info=r.solver_info,
                    objective=float(r.objective),
                    log_evidence=(
                        None if r.log_evidence is None else float(r.log_evidence)
                    ),
                    final_scaled_gradient=r.configuration.get("final_scaled_gradient"),
                )
                if cfg["name"] == "coupled37":
                    row.update(
                        source_values=np.asarray(r.source_values).tolist(),
                        potential_values=np.asarray(r.potential_values).tolist(),
                        nonlinear_image_active=np.asarray(
                            r.nonlinear_image_active
                        ).tolist(),
                    )
                return row

            cold_status = quality(result)
            warm_status = []
            if result.success:
                for _ in range(args.repeats):
                    del result
                    t = time.perf_counter()
                    result = problem.fit(
                        cfg["lambda_source"],
                        cfg["lambda_potential"],
                        backend=args.backend,
                        **options,
                    )
                    synchronize(result)
                    timing["warm_fit_seconds"].append(time.perf_counter() - t)
                    warm_status.append(quality(result))
            snapshot("warm_fit")
            timing["cold_status"] = cold_status
            timing["warm_status"] = warm_status
        # Snapshot before extra audit actions: these do not pre-warm the first fit.
        live = jax.live_arrays("gpu")
        shapes = sorted(set(tuple(a.shape) for a in live))
        largest = max(a.size for a in live)
        live_bytes = sum(a.nbytes for a in live)
        del live
        forbidden = {
            (problem.n_source, problem.n_source),
            (problem.n_potential, problem.n_potential),
            (problem.n_joint, problem.n_joint),
            (problem.n_source, problem.n_potential),
            (problem.n_potential, problem.n_source),
            (renderer.n_active, problem.n_joint),
            (problem.n_joint, renderer.n_active),
        }
        suspects = sorted(set(shapes) & forbidden)
        audit_started = time.perf_counter()
        joint = jnp.concatenate((result.source_values, result.potential_values))
        mapping = renderer.source_mapping(result.potential_values)
        gx, gy = renderer.source_spatial_gradient(result.source_values, mapping)
        if args.algorithm == "single_step":
            action = problem.forward(joint)
            adjoint = problem.adjoint(problem.data_active)
        else:
            lin = problem.linearize(joint)
            action = problem.jacobian_forward(joint, lin)
            adjoint = problem.jacobian_adjoint(problem.data_active, lin)
        prior_action = problem.prior_matvec(
            joint, cfg["lambda_source"], cfg["lambda_potential"]
        )
        synchronize((action, adjoint, prior_action, mapping, gx, gy))
        audit = {
            "query": str(renderer.query_x.dtype),
            "source_axis": str(renderer.source_x_axis.dtype),
            "psf_fft": str(renderer.psf_fft.dtype),
            "mapping_weights": str(mapping.weights.dtype),
            "gradient_x": str(gx.dtype),
            "gradient_y": str(gy.dtype),
            "forward_action": str(action.dtype),
            "adjoint_action": str(adjoint.dtype),
            "prior_action": str(prior_action.dtype),
            "initial_source": str(initial.dtype),
            "frozen_gradient": str(gradient[0].dtype),
        }
        arrays = {
            f.name: np.asarray(getattr(result, f.name))
            for f in dataclasses.fields(result)
            if isinstance(getattr(result, f.name), jax.Array)
        }
        for name in (
            "node_indices",
            "core_indices",
            "interpolation_indices",
            "interpolation_weights",
            "derivative_indices",
        ):
            arrays["potential_" + name] = np.asarray(getattr(grid, name))
        for name in (
            "source_values",
            "potential_values",
            "objective",
            "chi2",
            "regularization_penalty",
            "prior_logdet",
            "posterior_logdet",
            "noise_logdet",
            "log_evidence",
        ):
            v = getattr(result, name)
            audit["result_" + name] = None if v is None else str(v.dtype)
        timing["audit_seconds"] = time.perf_counter() - audit_started
        counts = dict(
            image_shape=renderer.image_shape,
            data=renderer.n_active,
            source=renderer.n_source,
            core=grid.n_core,
            support=grid.n_unknown,
            support_only=grid.n_unknown - grid.n_core,
            joint=problem.n_joint,
            query_subpixels=len(grid.query_x),
            mock_nsub=cfg.get("mock", {}).get("nsub", 1),
            inversion_nsub=renderer.nsub,
        )
        result_record = {
            k: getattr(result, k)
            for k in (
                "success",
                "status",
                "termination_reason",
                "failure_reason",
                "solver_info",
                "strengths",
                "objective",
                "chi2",
                "regularization_penalty",
                "prior_logdet",
                "posterior_logdet",
                "log_evidence",
                "evidence_method",
                "determinant_method",
                "evidence_is_exact",
                "configuration",
                "history",
            )
        }
    after = dict(
        x64=jax.config.x64_enabled, matmul=jax.config.jax_default_matmul_precision
    )
    export_started = time.perf_counter()
    np.savez_compressed(Path(args.output).with_suffix(".npz"), **arrays)
    timing["array_export_seconds"] = time.perf_counter() - export_started
    record = dict(
        case=cfg["name"],
        observation=cfg["observation"],
        algorithm=args.algorithm,
        backend=args.backend,
        dtype=args.dtype,
        profile=profile,
        pid=os.getpid(),
        started_utc=started_utc,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        configuration=cfg,
        effective_fit_options=options,
        input_sha256=sha_file(args.inputs),
        input_path=str(Path(args.inputs).resolve()),
        result=result_record,
        search=search_record,
        geometry=geometry,
        counts=counts,
        timing=timing,
        dtype_audit=audit,
        precision_defaults=dict(before=defaults, after=after),
        allocator_snapshots=snapshots,
        live_arrays=dict(
            shapes=shapes,
            largest_elements=largest,
            bytes=live_bytes,
            prohibited_live_shapes=suspects,
            semantics="post-fit live buffers, not transient peak; image-shape coincidences possible for tiny fixtures",
        ),
        host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        software=software(device),
        array_role=(
            "successful_reconstruction"
            if result.success
            else "failure_diagnostics_null_evidence"
        ),
    )
    write_json(args.output, record)


def parent(args):
    """Run a fresh child and sample only that PID; no JAX import in this path."""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "PYTHONPATH": str(ROOT),
    }
    samples = []
    monitor = None
    try:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-compute-apps=timestamp,gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
                "--loop-ms=100",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        pass
    started = time.perf_counter()
    with output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )

        def collect():
            if monitor is None:
                return
            for line in monitor.stdout:
                fields = [s.strip() for s in line.strip().split(",")]
                if len(fields) == 4 and fields[2] == str(process.pid):
                    try:
                        samples.append(
                            dict(
                                timestamp=fields[0],
                                gpu_uuid=fields[1],
                                bytes=int(fields[3]) * 1024**2,
                            )
                        )
                    except ValueError:
                        pass

        thread = threading.Thread(target=collect, daemon=True)
        thread.start()
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = -9
        wall = time.perf_counter() - started
    if monitor is not None:
        monitor.terminate()
        monitor.wait(timeout=10)
        thread.join(timeout=2)
    if output.exists() and code == 0:
        record = json.loads(output.read_text())
    else:
        record = dict(
            worker_failure=True, returncode=code, log=str(output.with_suffix(".log"))
        )
    record["process_wall_seconds"] = wall
    record["worker_returncode"] = code
    record["process_gpu_memory"] = dict(
        semantics="sampled_worker_residency_lower_bound",
        sample_interval_ms=100,
        peak_bytes=max((s["bytes"] for s in samples), default=None),
        sample_count=len(samples),
        samples=samples,
    )
    write_json(output, record)
    print(
        json.dumps(
            dict(
                output=str(output),
                returncode=code,
                success=record.get("result", {}).get("success"),
            )
        )
    )
    return code


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument(
        "--case",
        choices=(
            "coupled37",
            "scientific32",
            "scientific64",
            "scientific128",
            "scientific200",
        ),
        default="coupled37",
    )
    p.add_argument("--observation", choices=("subhalo", "control"), default="subhalo")
    p.add_argument("--reference", type=Path)
    p.add_argument(
        "--reference-inputs",
        type=Path,
        help="Preparation only: preserve these saved observations for a source-resolution axis",
    )
    p.add_argument(
        "--source-n",
        type=int,
        help="Preparation only: override the source side and declare a new fixed-grid case",
    )
    p.add_argument("--inputs", type=Path)
    p.add_argument(
        "--algorithm", choices=("single_step", "iterative"), default="single_step"
    )
    p.add_argument("--backend", choices=("dense", "operator"), default="dense")
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--profile", choices=("strict", "small32", "scientific"))
    p.add_argument("--gradient-atol", type=float)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--search", type=int, default=0)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.prepare:
        prepare(args)
        return 0
    if args.inputs is None:
        p.error("--inputs is required for fitting")
    if args.worker:
        worker(args)
        return 0
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
