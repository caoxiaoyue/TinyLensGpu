"""Mixed precision L-BFGS: form/factor H0 in double, evaluate physics in single.

full64: unchanged double reference.
mixed32: H64 -> Cholesky64 -> factor32, then whitening/physics/AD in float32.
mixed64: H64 -> Cholesky64, retain double whitening, cast physical point to
         float32; physical AD gradient32, chain-rule white gradient64.

No prior changes, jitter, new stopping rules or numerical-metric caching.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from benchmark_lbfgs_precision import OPTIONS, POLICY, write

VARIANTS = ("full64", "mixed32", "mixed64")


def compare_fields(candidate, reference, objective, reference_objective):
    import numpy as np

    excess = objective - reference_objective
    image = float(
        np.sqrt(
            np.mean(
                ((candidate["model"] - reference["model"]) / reference["noise"]) ** 2
            )
        )
    )
    source = float(
        np.linalg.norm(candidate["source"] - reference["source"])
        / np.linalg.norm(reference["source"])
    )
    kappa = float(np.sqrt(np.mean((candidate["kappa"] - reference["kappa"]) ** 2)))
    limit = POLICY["kappa_difference_rms_atol"] + POLICY[
        "kappa_difference_rms_rtol"
    ] * float(np.sqrt(np.mean(reference["kappa"] ** 2)))
    checks = dict(
        objective=excess <= POLICY["objective_excess_max"],
        image=image <= POLICY["model_difference_rms_in_noise_max"],
        source=source <= POLICY["source_relative_difference_max"],
        kappa=kappa <= limit,
    )
    return dict(
        objective_excess=excess,
        model_difference_rms_in_noise=image,
        source_relative_difference=source,
        kappa_difference_rms=kappa,
        kappa_difference_limit=limit,
        checks=checks,
        reference_consistency_pass=bool(all(checks.values())),
    )


def worker(args):
    import traceback
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jsl
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

    class MixedFit:
        def __init__(self, problem64, problem, strengths, atol):
            self.problem64, self.problem = problem64, problem
            self.strengths, self.atol = strengths, atol
            self.metric = PreparedAD(problem64, strengths, atol)
            self.physical = PreparedAD(problem, strengths, atol)
            self.white_dtype = jnp.float32 if args.variant == "mixed32" else jnp.float64

            def decode(z, factor):
                if args.variant == "mixed64":
                    point = problem64.initial_values + jsl.solve_triangular(
                        factor.T, z, lower=False
                    )
                    return point.astype(jnp.float32)
                return problem.initial_values + jsl.solve_triangular(
                    factor.T, z, lower=False
                )

            decoded = jax.jit(decode)

            def scoped_decode(z, factor):
                with jax.experimental.enable_x64(self.white_dtype == jnp.float64):
                    return decoded(z, factor)

            self.decode = scoped_decode
            composite = jax.jit(
                jax.value_and_grad(
                    lambda z, factor: self.physical.objective(decode(z, factor))
                )
            )
            # Different x64 contexts must also own AD tracing/lowering. Merely
            # placing enable_x64(False) inside a library operation is insufficient
            # for reverse-mode index lowering in this JAX version.
            pull64 = jax.jit(
                lambda g, factor: jsl.solve_triangular(
                    factor, g.astype(jnp.float64), lower=True
                )
            )

            def value_grad(z, factor):
                if args.variant == "mixed64":
                    point = self.decode(z, factor)
                    with jax.experimental.enable_x64(False):
                        value, physical_gradient = self.physical.gradient(point)
                    with jax.experimental.enable_x64(True):
                        white_gradient = pull64(physical_gradient, factor)
                    return value, white_gradient
                with jax.experimental.enable_x64(problem.dtype == jnp.float64):
                    return composite(z, factor)

            self.value_grad = value_grad

        def fit(self):
            started = perf_counter()
            timing, row = {}, {}
            stage = "metric64"
            try:
                t = perf_counter()
                matrix = self.metric.matrix(self.problem64.initial_values)
                self.last_metric = matrix
                timing["metric_seconds"] = perf_counter() - t
                assert matrix.dtype == np.float64
                stage = "cholesky64"
                t = perf_counter()
                factor64 = jnp.linalg.cholesky(jnp.asarray(matrix, dtype=jnp.float64))
                sync(factor64)
                assert factor64.dtype == jnp.float64
                if not np.isfinite(np.asarray(factor64)).all():
                    raise RuntimeError("H64 Cholesky failed; no hidden jitter")
                factor = factor64.astype(self.white_dtype)
                sync(factor)
                timing["factor_and_cast_seconds"] = perf_counter() - t
                row.update(
                    metric_dtype=str(matrix.dtype),
                    cholesky_dtype=str(factor64.dtype),
                    applied_factor_dtype=str(factor.dtype),
                    factorization_pass=True,
                )
                stage = "gradient_setup"
                t = perf_counter()
                with jax.experimental.enable_x64(self.problem.dtype == jnp.float64):
                    _, g0 = self.physical.gradient(self.problem.initial_values)
                initial_gamma = float(
                    jnp.max(
                        jnp.abs(g0)
                        / jnp.sqrt(
                            jnp.asarray(
                                np.diag(matrix).copy(), dtype=self.problem.dtype
                            )
                        )
                    )
                )
                threshold = self.atol + 1e-8 * initial_gamma
                zero = jnp.zeros(self.problem.n_joint, dtype=self.white_dtype)
                f0, gw0 = self.value_grad(zero, factor)
                sync((f0, gw0))
                assert f0.dtype == g0.dtype == self.problem.dtype
                assert gw0.dtype == self.white_dtype
                row.update(
                    objective_dtype=str(f0.dtype),
                    physical_gradient_dtype=str(g0.dtype),
                    white_gradient_dtype=str(gw0.dtype),
                )
                timing["gradient_setup_seconds"] = perf_counter() - t
                history = []

                def fun(z):
                    f, g = self.value_grad(
                        jnp.asarray(z, dtype=self.white_dtype), factor
                    )
                    return float(f), np.asarray(g, dtype=np.float64)

                def callback(intermediate_result):
                    history.append(float(intermediate_result.fun))

                stage = "lbfgs"
                t = perf_counter()
                result = minimize(
                    fun,
                    np.zeros(self.problem.n_joint, dtype=np.float64),
                    jac=True,
                    method="L-BFGS-B",
                    options=OPTIONS.copy(),
                    callback=callback,
                )
                values = self.decode(
                    jnp.asarray(result.x, dtype=self.white_dtype), factor
                )
                sync(values)
                timing["solve_seconds"] = perf_counter() - t
                stage = "certificate"
                t = perf_counter()
                with jax.experimental.enable_x64(self.problem.dtype == jnp.float64):
                    value, gradient = self.physical.gradient(values)
                    diagonal = self.physical.matrix(values, diagonal=True)
                gamma = float(
                    jnp.max(
                        jnp.abs(gradient)
                        / jnp.sqrt(jnp.asarray(diagonal, dtype=self.problem.dtype))
                    )
                )
                timing["certificate_seconds"] = perf_counter() - t
                stage = "products"
                t = perf_counter()
                with jax.experimental.enable_x64(self.problem.dtype == jnp.float64):
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
                row["timings"] = timing
                return values, row
            except Exception as exc:
                timing["total_seconds"] = perf_counter() - started
                row.update(
                    completed=False, failure_stage=stage, error=str(exc), timings=timing
                )
                if not isinstance(
                    exc, RuntimeError
                ) or "H64 Cholesky failed" not in str(exc):
                    raise
                return None, row

    output = args.output / f"seed{args.seed}_{args.variant}_{args.case}"
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        variant=args.variant,
        seed=args.seed,
        case=args.case,
        rows=[],
        options=OPTIONS,
        policy=POLICY,
    )
    try:
        with threadpool_limits(limits=1), jax.default_matmul_precision(
            "highest"
        ), jax.experimental.enable_x64(True):
            preparation = perf_counter()
            cfg = json.loads(
                (
                    ROOT
                    / "examples/potential_correction/iterative_scientific_configuration.json"
                ).read_text()
            )
            cfg["mock"]["seed"] = args.seed
            observation = getattr(
                generate_mock_pair(MockConfig.from_dict(cfg["mock"])), args.case
            )
            mock = MockConfig.from_dict(cfg["mock"])
            sim = SimulatorConfig(
                mock.dpix,
                mock.npix,
                observation.psf_kernel,
                cfg["inversion_nsub"],
                observation.mask,
            )
            renderer64 = PotentialCorrectionRenderer(
                observation.macro,
                sim,
                cfg["source_n"],
                tuple(cfg["source_bbox"]),
                potential_spacing=cfg["potential_spacing"],
                dtype="float64",
            )
            sx, sy = jnp.meshgrid(renderer64.source_x_axis, renderer64.source_y_axis)
            source64 = observation.source_truth.brightness(sx, sy).ravel()

            def make_problem(renderer, source):
                return IterativePotentialCorrection(
                    renderer,
                    observation.image_data,
                    observation.noise_map,
                    source,
                    lambda_amplitude=cfg["lambda_amplitude"],
                    source_regularization=cfg["source_regularization"],
                    solve_rtol=cfg["solve_rtol"],
                    solve_atol=cfg["solve_atol"],
                )

            problem64 = make_problem(renderer64, source64)
            if args.variant == "full64":
                problem = problem64
            else:
                with jax.experimental.enable_x64(False):
                    changes = {}
                    for field in dataclasses.fields(renderer64.potential_grid):
                        value = getattr(renderer64.potential_grid, field.name)
                        if isinstance(value, jax.Array):
                            changes[field.name] = (
                                value.astype(jnp.float32)
                                if np.issubdtype(value.dtype, np.floating)
                                else value
                            )
                    grid32 = dataclasses.replace(renderer64.potential_grid, **changes)
                    renderer32 = PotentialCorrectionRenderer(
                        observation.macro,
                        sim,
                        cfg["source_n"],
                        tuple(cfg["source_bbox"]),
                        potential_spacing=cfg["potential_spacing"],
                        dtype="float32",
                        potential_grid=grid32,
                    )
                    problem = make_problem(renderer32, source64.astype(jnp.float32))
            strength_config = cfg["fixed_strengths"][args.case]
            strengths = (
                strength_config["lambda_source"],
                strength_config["lambda_potential"],
            )
            atol = (
                cfg["case_iteration_options"]
                .get(args.case, {})
                .get("gradient_atol", cfg["iteration_options"]["gradient_atol"])
            )
            sync(
                (problem64.initial_values, problem.initial_values, problem.data_active)
            )
            report.update(
                config=cfg,
                preparation_seconds=perf_counter() - preparation,
                canonical_input_sha256=digest(
                    problem64.data_active,
                    problem64.noise_active,
                    problem64.initial_values,
                ),
                topology_sha256=digest(
                    renderer64.potential_grid.node_indices,
                    renderer64.potential_grid.core_indices,
                ),
                python=platform.python_version(),
                jax=jax.__version__,
                scipy=scipy.__version__,
                numpy=np.__version__,
                devices=[d.device_kind for d in jax.devices()],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dependencies_sha256={
                    name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                    for name in (
                        "benchmark_scientific.py",
                        "benchmark_lbfgs_precision.py",
                        "run_experiment.py",
                    )
                },
                dimensions=dict(
                    data=problem.renderer.n_active,
                    source=problem.n_source,
                    potential=problem.n_potential,
                    core=problem.renderer.potential_grid.n_core,
                ),
                scope="Prepared truth-node initial source model through recomputed H64/Cholesky64, optimization, final physical AD certificate and raw fields. Excludes mock/geometry, startup, independent canonical64 validation and file I/O. SciPy CPU control float64.",
            )
            fit = MixedFit(problem64, problem, strengths, atol)
            for i in range(args.repeats + 1):
                values, row = fit.fit()
                row.update(phase="cold" if i == 0 else "warm", repetition=i)
                if hasattr(fit, "last_metric"):
                    row["metric_sha256"] = hashlib.sha256(
                        fit.last_metric.tobytes()
                    ).hexdigest()
                    if i == 0 or not row.get("factorization_pass"):
                        np.save(output / f"metric_{i}.npy", fit.last_metric)
                if values is not None:
                    np.savez(output / f"endpoint_{i}.npz", values=np.asarray(values))
                report["rows"].append(row)
                write(output / "result.json", report)
                print(
                    f"seed{args.seed}/{args.variant}/{args.case} {row['phase']}{i}: "
                    f"{row['timings']['total_seconds']:.4f}s factor={row.get('factorization_pass',False)} "
                    f"native={row.get('library_success')} nit={row.get('iterations')} {row.get('error','')}",
                    flush=True,
                )
            # Audit physical kernels independently of mixed whitening, after timing.
            point = problem.initial_values
            with jax.experimental.enable_x64(problem.dtype == jnp.float64):
                _, gradient = fit.physical.gradient(point)
                audit = dict(
                    physical_values=str(point.dtype),
                    data=str(problem.data_active.dtype),
                    noise=str(problem.noise_active.dtype),
                    forward=str(problem.forward(point).dtype),
                    prior=str(problem.prior_matvec(point, *strengths).dtype),
                    gradient=str(gradient.dtype),
                    query=str(problem.renderer.query_x.dtype),
                    psf_fft=str(problem.renderer.psf_fft.dtype),
                )
                physical_jaxpr = str(
                    jax.make_jaxpr(jax.value_and_grad(fit.physical.objective))(point)
                )
            (output / "physical_objective_jaxpr.txt").write_text(physical_jaxpr)
            audit["physical_jaxpr_contains_f64_or_c128"] = (
                "f64[" in physical_jaxpr or "c128[" in physical_jaxpr
            )
            if args.variant != "full64":
                assert all(
                    audit[k] == "float32"
                    for k in (
                        "physical_values",
                        "data",
                        "noise",
                        "forward",
                        "prior",
                        "gradient",
                        "query",
                    )
                )
                assert not audit["physical_jaxpr_contains_f64_or_c128"]
            report["dtype_audit"] = audit
            objective64 = jax.jit(objective_for(problem64, strengths))
            ref_report_path = (
                args.output / f"seed{args.seed}_full64_{args.case}" / "result.json"
            )
            reference = None
            reference_objective = None
            if args.variant != "full64":
                ref_report = json.loads(ref_report_path.read_text())
                assert (
                    ref_report["canonical_input_sha256"]
                    == report["canonical_input_sha256"]
                )
                assert ref_report["topology_sha256"] == report["topology_sha256"]
                reference = np.load(ref_report_path.parent / "canonical_fields_0.npz")
                reference_objective = ref_report["rows"][0]["canonical64"]["objective"]
            for i, row in enumerate(report["rows"]):
                if not row["completed"]:
                    continue
                values = jnp.asarray(
                    np.load(output / f"endpoint_{i}.npz")["values"], dtype=jnp.float64
                )
                products = problem64._result_products(values, *strengths)
                sync(products)
                core = np.asarray(renderer64.potential_grid.core_indices)
                kappa = np.asarray(products["convergence"])[core]
                fields = dict(
                    values=np.asarray(values),
                    model=np.asarray(products["nonlinear_image_active"]),
                    source=np.asarray(products["source_values"]),
                    kappa=kappa,
                    noise=np.asarray(problem64.noise_active),
                )
                objective = float(objective64(values))
                peak = int(np.argmax(kappa))
                px = float(renderer64.potential_grid.x[core[peak]])
                py = float(renderer64.potential_grid.y[core[peak]])
                residual = (
                    np.asarray(problem64.data_active) - fields["model"]
                ) / fields["noise"]
                finite = bool(all(np.isfinite(v).all() for v in fields.values()))
                rms = float(np.sqrt(np.mean(residual**2)))
                kappa_rms = float(np.sqrt(np.mean(kappa**2)))
                sanity = finite and 0.9 <= rms <= 1.1
                if args.case == "subhalo":
                    sanity = (
                        sanity
                        and np.hypot(px - mock.subhalo_x, py - mock.subhalo_y)
                        <= 2**0.5 * cfg["potential_spacing"]
                    )
                else:
                    sanity = sanity and kappa_rms < 0.01
                row["canonical64"] = dict(
                    objective=objective,
                    finite=finite,
                    residual_rms_in_noise=rms,
                    kappa_peak=float(kappa[peak]),
                    peak_x=px,
                    peak_y=py,
                    kappa_rms=kappa_rms,
                    physical_sanity_pass=bool(sanity),
                )
                if reference is None:
                    assert (
                        i == 0
                    ), "Missing first full64 reference; cannot silently choose a new one"
                    reference, reference_objective = fields, objective
                comparison = compare_fields(
                    fields, reference, objective, reference_objective
                )
                row["comparison"] = comparison
                row["accepted"] = bool(
                    row["library_success"]
                    and sanity
                    and comparison["reference_consistency_pass"]
                )
                np.savez(output / f"canonical_fields_{i}.npz", **fields)
            write(output / "result.json", report)
    except Exception:
        report["worker_error"] = traceback.format_exc()
        print(report["worker_error"], flush=True)
    write(output / "result.json", report)
    if "worker_error" in report:
        raise RuntimeError("Mixed-precision worker failed; see saved traceback")


def summarize(output):
    summary = {}
    for path in sorted(output.glob("seed*/result.json")):
        report = json.loads(path.read_text())
        rows = report["rows"]
        warm = [r for r in rows if r["phase"] == "warm"]
        completed = [r for r in warm if r["completed"] and r.get("library_success")]
        accepted = [r for r in warm if r.get("accepted")]

        def median(items):
            return (
                statistics.median(r["timings"]["total_seconds"] for r in items)
                if items
                else None
            )

        summary[path.parent.name] = dict(
            attempted=len(rows),
            factorization_passed=sum(r.get("factorization_pass", False) for r in rows),
            completed=sum(r["completed"] for r in rows),
            native_success=sum(r.get("library_success", False) for r in rows),
            accepted=sum(r.get("accepted", False) for r in rows),
            warm_attempted=len(warm),
            warm_completed=len(completed),
            warm_accepted=len(accepted),
            completed_warm_median_seconds=median(completed),
            accepted_warm_median_seconds=median(accepted),
            cold_seconds=rows[0]["timings"]["total_seconds"],
            objective_excess=[
                r["comparison"]["objective_excess"] for r in rows if "comparison" in r
            ],
            worker_error=report.get("worker_error"),
        )
    write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "lbfgs_mixed_precision")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--case", choices=("subhalo", "control"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    if any(args.output.iterdir()):
        raise RuntimeError(
            "Use a fresh output directory; existing records are preserved"
        )
    shutil.copyfile(Path(__file__), args.output / "runner_at_measurement.py")
    write(
        args.output / "protocol.json",
        dict(
            seed=args.seed,
            repeats=args.repeats,
            variants=VARIANTS,
            options=OPTIONS,
            policy=POLICY,
            scope="Same input repeated, not independent noise seeds. All attempts retained; no retries or hidden ridge.",
        ),
    )
    for case in ("subhalo", "control"):
        for variant in VARIANTS:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--variant",
                    variant,
                    "--case",
                    case,
                    "--seed",
                    str(args.seed),
                    "--repeats",
                    str(args.repeats),
                    "--output",
                    str(args.output.resolve()),
                ],
                cwd=ROOT,
                check=True,
            )
    print(json.dumps(summarize(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
