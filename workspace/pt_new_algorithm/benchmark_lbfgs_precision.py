"""Compare genuine JAX float32/float64 on the unchanged standard L-BFGS mock.

Workers run serially in fresh processes. Both receive the same double-generated
data, integer geometry and truth-node initial source (the original ~1 s timing
protocol). The optimizer remains SciPy's double-precision CPU controller; every
floating JAX fit operation uses the requested dtype. No prior changes/jitter.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

OPTIONS = dict(maxiter=3000, maxfun=120000, maxcor=30, maxls=80, gtol=1e-8, ftol=1e-15)
POLICY = dict(
    objective_excess_max=1e-3,
    model_difference_rms_in_noise_max=1e-3,
    source_relative_difference_max=0.005,
    kappa_difference_rms_atol=1e-5,
    kappa_difference_rms_rtol=0.01,
)


def write(path, obj):
    def scalar(item):
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(type(item).__name__)

    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=scalar) + "\n")


def worker(args):
    import traceback
    import jax
    import jax.numpy as jnp
    import numpy as np
    import scipy
    from scipy.optimize import minimize
    from threadpoolctl import threadpool_limits
    from benchmark_scientific import PreparedAD, sync
    from run_experiment import objective_for
    from run_scientific import (
        MockConfig,
        generate_mock_pair,
        SimulatorConfig,
        PotentialCorrectionRenderer,
        IterativePotentialCorrection,
        digest,
    )

    class TypedAD(PreparedAD):
        """Same AD metric and whitening; prevent accidental double promotion."""

        def matrix(self, point, diagonal=False):
            size = self.problem.n_joint
            output = np.empty((size,) if diagonal else (size, size), dtype=point.dtype)
            block = self.diagonal_block if diagonal else self.metric_block
            for start in range(0, size, self.panel):
                rows = np.asarray(
                    block(point, jnp.arange(start, start + self.panel, dtype=jnp.int32))
                )
                assert rows.dtype == point.dtype
                count = min(self.panel, size - start)
                if diagonal:
                    output[start : start + count] = rows[:count]
                else:
                    output[:, start : start + count] = rows[:count].T
            if not diagonal:
                output = (output + output.T) / np.asarray(2, dtype=point.dtype)
            return output

        def fit_recorded(self):
            started = perf_counter()
            row = dict(phase="cold", timings={})
            timing = row["timings"]
            stage = "metric"
            try:
                t = perf_counter()
                metric = self.matrix(self.problem.initial_values)
                self.last_metric = metric
                timing["metric_seconds"] = perf_counter() - t
                stage = "cholesky"
                t = perf_counter()
                chol = jnp.linalg.cholesky(
                    jnp.asarray(metric, dtype=self.problem.dtype)
                )
                sync(chol)
                timing["factor_seconds"] = perf_counter() - t
                row["metric_dtype"] = str(metric.dtype)
                row["cholesky_dtype"] = str(chol.dtype)
                if not np.isfinite(np.asarray(chol)).all():
                    raise RuntimeError(
                        "Initial metric Cholesky failed; no jitter added"
                    )
                stage = "gradient_setup"
                t = perf_counter()
                _, g0 = self.gradient(self.problem.initial_values)
                gamma0 = float(
                    jnp.max(jnp.abs(g0) / jnp.sqrt(jnp.diag(jnp.asarray(metric))))
                )
                # Only the old diagnostic gate; never used to change native stopping.
                threshold = self.atol + 1e-8 * gamma0
                zero = jnp.zeros(self.problem.n_joint, dtype=self.problem.dtype)
                f0, g0w = self.value_grad(zero, chol)
                sync((f0, g0w))
                assert f0.dtype == g0w.dtype == chol.dtype == self.problem.dtype
                timing["gradient_setup_seconds"] = perf_counter() - t
                row["objective_dtype"] = str(f0.dtype)
                row["white_gradient_dtype"] = str(g0w.dtype)
                history = []

                def fun(z):
                    value, grad = self.value_grad(
                        jnp.asarray(z, dtype=self.problem.dtype), chol
                    )
                    return float(value), np.asarray(grad, dtype=np.float64)

                def callback(intermediate_result):
                    history.append(float(intermediate_result.fun))

                stage = "lbfgs"
                t = perf_counter()
                result = minimize(
                    fun,
                    np.zeros(self.problem.n_joint, dtype=np.float64),
                    jac=True,
                    method="L-BFGS-B",
                    callback=callback,
                    options=OPTIONS.copy(),
                )
                values = self.decode(
                    jnp.asarray(result.x, dtype=self.problem.dtype), chol
                )
                sync(values)
                timing["solve_seconds"] = perf_counter() - t
                stage = "certificate"
                t = perf_counter()
                value, grad = self.gradient(values)
                diagonal = self.matrix(values, diagonal=True)
                gamma = float(jnp.max(jnp.abs(grad) / jnp.sqrt(jnp.asarray(diagonal))))
                timing["certificate_seconds"] = perf_counter() - t
                stage = "products"
                t = perf_counter()
                products = self.problem._result_products(values, *self.strengths)
                sync(products)
                timing["products_seconds"] = perf_counter() - t
                row.update(
                    completed=True,
                    library_success=bool(result.success),
                    status=int(result.status),
                    message=str(result.message),
                    iterations=int(result.nit),
                    nfev=int(result.nfev),
                    objective=float(value),
                    scaled_gradient=gamma,
                    gradient_threshold=threshold,
                    legacy_gradient_pass=gamma <= threshold,
                    white_gradient_inf=float(np.max(np.abs(result.jac))),
                    values_dtype=str(values.dtype),
                    history=history,
                )
                timing["total_seconds"] = perf_counter() - started
                return values, row
            except Exception as exc:
                timing["total_seconds"] = perf_counter() - started
                row.update(completed=False, failure_stage=stage, error=str(exc))
                return None, row

    output = args.output / f"{args.dtype}_{args.case}"
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        dtype=args.dtype, case=args.case, rows=[], options=OPTIONS, policy=POLICY
    )
    try:
        with threadpool_limits(limits=1), jax.default_matmul_precision("highest"):
            t = perf_counter()
            cfg = json.loads(
                (
                    ROOT
                    / "examples/potential_correction/iterative_scientific_configuration.json"
                ).read_text()
            )
            # One canonical double-generated mock. Only the inference dtype changes.
            with jax.experimental.enable_x64(True):
                mock = MockConfig.from_dict(cfg["mock"])
                observation = getattr(generate_mock_pair(mock), args.case)
                sim = SimulatorConfig(
                    mock.dpix,
                    mock.npix,
                    observation.psf_kernel,
                    cfg["inversion_nsub"],
                    observation.mask,
                )
                reference_renderer = PotentialCorrectionRenderer(
                    observation.macro,
                    sim,
                    cfg["source_n"],
                    tuple(cfg["source_bbox"]),
                    potential_spacing=cfg["potential_spacing"],
                    dtype="float64",
                )
                sx, sy = jnp.meshgrid(
                    reference_renderer.source_x_axis, reference_renderer.source_y_axis
                )
                source64 = observation.source_truth.brightness(sx, sy).ravel()
                reference_problem = IterativePotentialCorrection(
                    reference_renderer,
                    observation.image_data,
                    observation.noise_map,
                    source64,
                    lambda_amplitude=cfg["lambda_amplitude"],
                    source_regularization=cfg["source_regularization"],
                    solve_rtol=cfg["solve_rtol"],
                    solve_atol=cfg["solve_atol"],
                )
                sync(reference_problem.initial_values)
                canonical_sha = digest(
                    reference_problem.data_active,
                    reference_problem.noise_active,
                    reference_problem.initial_values,
                )
            with jax.experimental.enable_x64(args.dtype == "float64"):
                dtype = jnp.dtype(args.dtype)
                # Preserve node topology; rebuilding after rounding boundary queries
                # can change which nodes exist. Public renderer validates these rays.
                changes = {}
                for field in dataclasses.fields(reference_renderer.potential_grid):
                    value = getattr(reference_renderer.potential_grid, field.name)
                    if isinstance(value, jax.Array):
                        changes[field.name] = (
                            value.astype(dtype)
                            if np.issubdtype(value.dtype, np.floating)
                            else value
                        )
                grid = dataclasses.replace(reference_renderer.potential_grid, **changes)
                renderer = PotentialCorrectionRenderer(
                    observation.macro,
                    sim,
                    cfg["source_n"],
                    tuple(cfg["source_bbox"]),
                    potential_spacing=cfg["potential_spacing"],
                    dtype=args.dtype,
                    potential_grid=grid,
                )
                problem = IterativePotentialCorrection(
                    renderer,
                    observation.image_data,
                    observation.noise_map,
                    source64.astype(dtype),
                    lambda_amplitude=cfg["lambda_amplitude"],
                    source_regularization=cfg["source_regularization"],
                    solve_rtol=cfg["solve_rtol"],
                    solve_atol=cfg["solve_atol"],
                )
                strengths_cfg = cfg["fixed_strengths"][args.case]
                strengths = (
                    strengths_cfg["lambda_source"],
                    strengths_cfg["lambda_potential"],
                )
                atol = (
                    cfg["case_iteration_options"]
                    .get(args.case, {})
                    .get("gradient_atol", cfg["iteration_options"]["gradient_atol"])
                )
                sync(
                    (problem.initial_values, problem.data_active, problem.noise_active)
                )
                prep_seconds = perf_counter() - t
                prepared = TypedAD(problem, strengths, atol)
                # Inputs and actual forward/prior outputs, outside the fit timer.
                audit = dict(
                    initial=str(problem.initial_values.dtype),
                    data=str(problem.data_active.dtype),
                    noise=str(problem.noise_active.dtype),
                    source_axis=str(renderer.source_x_axis.dtype),
                    query=str(renderer.query_x.dtype),
                    psf_fft=str(renderer.psf_fft.dtype),
                    macro_rays=[str(x.dtype) for x in renderer.macro_mapping],
                    grid_weights=str(grid.interpolation_weights.dtype),
                )
                report.update(
                    config=cfg,
                    canonical_input_sha256=canonical_sha,
                    topology_sha256=digest(grid.node_indices, grid.core_indices),
                    dimensions=dict(
                        data=renderer.n_active,
                        source=problem.n_source,
                        potential=problem.n_potential,
                        core=grid.n_core,
                    ),
                    dtype_audit=audit,
                    preparation_seconds=prep_seconds,
                    python=platform.python_version(),
                    jax=jax.__version__,
                    scipy=scipy.__version__,
                    numpy=np.__version__,
                    devices=[d.device_kind for d in jax.devices()],
                    matmul_precision=str(jax.config.jax_default_matmul_precision),
                    persistent_compilation_cache=str(
                        jax.config.jax_compilation_cache_dir
                    ),
                    script_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    scope="Same truth-node initialization as original ~1 s benchmark. Prepared problem through fresh numeric H0/Cholesky, L-BFGS, final AD certificate and raw products. Excludes Python/CUDA startup, mock/geometry, audit and file I/O. SciPy CPU control remains float64.",
                )
                for repetition in range(args.repeats + 1):
                    values, row = prepared.fit_recorded()
                    row.update(
                        phase="cold" if repetition == 0 else "warm",
                        repetition=repetition,
                    )
                    report["rows"].append(row)
                    if values is not None:
                        np.savez(
                            output / f"endpoint_{repetition}.npz",
                            values=np.asarray(values),
                        )
                    write(output / "result.json", report)
                    print(
                        f"{args.dtype}/{args.case} {row['phase']} "
                        f"{row['timings']['total_seconds']:.4f}s "
                        f"completed={row['completed']} native={row.get('library_success')} "
                        f"iterations={row.get('iterations')} error={row.get('error')}",
                        flush=True,
                    )
                    if values is None:
                        break
                # Actual array dtypes and tracing are checked after cold timing.
                point = problem.initial_values
                forward = problem.forward(point)
                prior = problem.prior_matvec(point, *strengths)
                _, grad = prepared.gradient(point)
                audit.update(
                    forward=str(forward.dtype),
                    prior=str(prior.dtype),
                    gradient=str(grad.dtype),
                )
                jaxpr = str(
                    jax.make_jaxpr(jax.value_and_grad(prepared.objective))(point)
                )
                (output / "objective_jaxpr.txt").write_text(jaxpr)
                audit["jaxpr_contains_f64_or_c128"] = (
                    "f64[" in jaxpr or "c128[" in jaxpr
                )
                assert all(
                    audit[name] == args.dtype
                    for name in (
                        "initial",
                        "data",
                        "noise",
                        "source_axis",
                        "query",
                        "grid_weights",
                        "forward",
                        "prior",
                        "gradient",
                    )
                )
                assert (
                    args.dtype == "float64" or not audit["jaxpr_contains_f64_or_c128"]
                )
                # Save the problematic metric for independent, untimed analysis.
                np.save(output / "initial_metric.npy", prepared.last_metric)
            # Re-evaluate every endpoint in the canonical double model. This is
            # deliberately outside all timings and cannot affect stopping.
            with jax.experimental.enable_x64(True):
                objective64 = jax.jit(objective_for(reference_problem, strengths))
                for i, row in enumerate(report["rows"]):
                    path = output / f"endpoint_{i}.npz"
                    if not path.exists():
                        continue
                    value64 = jnp.asarray(np.load(path)["values"], dtype=jnp.float64)
                    products = reference_problem._result_products(value64, *strengths)
                    sync(products)
                    core = np.asarray(reference_renderer.potential_grid.core_indices)
                    kappa = np.asarray(products["convergence"])[core]
                    active = np.asarray(products["nonlinear_image_active"])
                    residual = (
                        np.asarray(reference_problem.data_active) - active
                    ) / np.asarray(reference_problem.noise_active)
                    peak = int(np.argmax(kappa))
                    row["canonical64"] = dict(
                        objective=float(objective64(value64)),
                        residual_rms_in_noise=float(np.sqrt(np.mean(residual**2))),
                        kappa_rms=float(np.sqrt(np.mean(kappa**2))),
                        kappa_peak=float(kappa[peak]),
                        peak_x=float(reference_renderer.potential_grid.x[core[peak]]),
                        peak_y=float(reference_renderer.potential_grid.y[core[peak]]),
                        finite=bool(
                            np.isfinite(np.asarray(value64)).all()
                            and np.isfinite(active).all()
                            and np.isfinite(kappa).all()
                        ),
                    )
                    np.savez(
                        output / f"canonical_fields_{i}.npz",
                        values=np.asarray(value64),
                        model=active,
                        source=np.asarray(products["source_values"]),
                        kappa=kappa,
                        noise=np.asarray(reference_problem.noise_active),
                    )
                if report["rows"] and report["rows"][0]["completed"]:
                    warm = [
                        r["timings"]["total_seconds"]
                        for r in report["rows"]
                        if r["phase"] == "warm"
                        and r["completed"]
                        and r.get("library_success")
                    ]
                    report["completed_warm_median_seconds"] = (
                        statistics.median(warm) if warm else None
                    )
    except Exception:
        report["worker_error"] = traceback.format_exc()
        print(report["worker_error"], flush=True)
    write(output / "result.json", report)


def summarize(output):
    import numpy as np

    results = {}
    for case in ("subhalo", "control"):
        reports = {}
        for dtype in ("float64", "float32"):
            path = output / f"{dtype}_{case}" / "result.json"
            if path.exists():
                reports[dtype] = json.loads(path.read_text())
        if len(reports) != 2:
            results[case] = reports
            continue
        a, b = reports["float64"], reports["float32"]
        assert a["canonical_input_sha256"] == b["canonical_input_sha256"]
        assert a["topology_sha256"] == b["topology_sha256"]
        ref = np.load(output / f"float64_{case}" / "canonical_fields_0.npz")
        checks = []
        for i, row in enumerate(b["rows"]):
            path = output / f"float32_{case}" / f"canonical_fields_{i}.npz"
            if not path.exists():
                checks.append(dict(completed=False, failure=row.get("error")))
                continue
            candidate = np.load(path)
            objective_excess = (
                row["canonical64"]["objective"]
                - a["rows"][0]["canonical64"]["objective"]
            )
            image_delta = float(
                np.sqrt(
                    np.mean(((candidate["model"] - ref["model"]) / ref["noise"]) ** 2)
                )
            )
            source_delta = float(
                np.linalg.norm(candidate["source"] - ref["source"])
                / np.linalg.norm(ref["source"])
            )
            kappa_delta = float(
                np.sqrt(np.mean((candidate["kappa"] - ref["kappa"]) ** 2))
            )
            kappa_limit = POLICY["kappa_difference_rms_atol"] + POLICY[
                "kappa_difference_rms_rtol"
            ] * float(np.sqrt(np.mean(ref["kappa"] ** 2)))
            checks.append(
                dict(
                    completed=True,
                    objective_excess=objective_excess,
                    model_difference_rms_in_noise=image_delta,
                    source_relative_difference=source_delta,
                    kappa_difference_rms=kappa_delta,
                    kappa_difference_limit=kappa_limit,
                    reference_consistency_pass=bool(
                        row["canonical64"]["finite"]
                        and objective_excess <= POLICY["objective_excess_max"]
                        and image_delta <= POLICY["model_difference_rms_in_noise_max"]
                        and source_delta <= POLICY["source_relative_difference_max"]
                        and kappa_delta <= kappa_limit
                    ),
                )
            )

        def completion_summary(report):
            rows = report["rows"]
            warm = [r for r in rows if r["phase"] == "warm"]
            completed = [r for r in warm if r["completed"] and r.get("library_success")]
            return dict(
                completed_warm_median_seconds=(
                    statistics.median(r["timings"]["total_seconds"] for r in completed)
                    if completed
                    else None
                ),
                warm_attempted=len(warm),
                warm_completed=len(completed),
                all_attempts_completed=all(r["completed"] for r in rows),
                worker_error=report.get("worker_error"),
            )

        results[case] = dict(
            float64_timing=completion_summary(a),
            float32_timing=completion_summary(b),
            float32_validation=checks,
            float32_rows=b["rows"],
            float64_rows=a["rows"],
        )
    write(output / "summary.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "lbfgs_precision")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dtype", choices=("float64", "float32"))
    parser.add_argument("--case", choices=("subhalo", "control"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    write(
        args.output / "protocol.json",
        dict(options=OPTIONS, policy=POLICY, repeats=args.repeats),
    )
    for case in ("subhalo", "control"):
        for dtype in ("float64", "float32"):
            if (args.output / f"{dtype}_{case}").exists():
                raise RuntimeError(
                    "Use a fresh output directory; existing results are preserved"
                )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--dtype",
                    dtype,
                    "--case",
                    case,
                    "--output",
                    str(args.output.resolve()),
                    "--repeats",
                    str(args.repeats),
                ],
                cwd=ROOT,
                check=True,
            )
    summary = summarize(args.output)
    print(
        json.dumps(
            {
                case: {k: v for k, v in item.items() if not k.endswith("_rows")}
                for case, item in summary.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
