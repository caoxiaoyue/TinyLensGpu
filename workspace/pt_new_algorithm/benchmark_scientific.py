"""Isolated, synchronized timing of original LM and AD + fixed metric L-BFGS.

The parent imports no JAX. Each method/case has its own fresh worker process.
Warm fits retain compiled function identities but recompute numerical metrics,
factorizations and final gradient certificates. No disk metric cache or logs
inside the timed optimizer. Evidence-only work is measured separately for LM.
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
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sync(value):
    import jax

    if isinstance(value, jax.Array):
        value.block_until_ready()
    elif dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            sync(getattr(value, f.name))
    elif isinstance(value, dict):
        for item in value.values():
            sync(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            sync(item)


class PreparedAD:
    """Same numerical method as the prototype, with reusable JIT functions."""

    def __init__(self, problem, strengths, atol, panel=16):
        import jax
        import jax.numpy as jnp
        import jax.scipy.linalg as jsl
        from run_experiment import objective_for

        self.problem, self.strengths, self.atol, self.panel = (
            problem,
            strengths,
            atol,
            panel,
        )
        self.objective = objective_for(problem, strengths)
        self.ftol = 1e-15
        self.gradient = jax.jit(jax.value_and_grad(self.objective))
        weighted = lambda x: problem.forward(x) / problem.noise_active
        prior = lambda x: problem.prior_matvec(x, *strengths)

        @jax.jit
        def metric_block(point, indices):
            _, push = jax.linearize(weighted, point)
            _, pull = jax.vjp(weighted, point)

            def column(unit):
                return pull(push(unit))[0] + prior(unit)

            return jax.vmap(column)(
                jax.nn.one_hot(indices, problem.n_joint, dtype=point.dtype)
            )

        @jax.jit
        def diagonal_block(point, indices):
            _, push = jax.linearize(weighted, point)

            def entry(unit):
                image = push(unit)
                return jnp.vdot(image, image) + jnp.vdot(unit, prior(unit))

            return jax.vmap(entry)(
                jax.nn.one_hot(indices, problem.n_joint, dtype=point.dtype)
            )

        def decode(z, chol):
            return problem.initial_values + jsl.solve_triangular(chol.T, z, lower=False)

        self.metric_block = metric_block
        self.diagonal_block = diagonal_block
        self.decode = decode
        self.value_grad = jax.jit(
            jax.value_and_grad(lambda z, chol: self.objective(decode(z, chol)))
        )

    def matrix(self, point, diagonal=False):
        import jax.numpy as jnp
        import numpy as np

        size = self.problem.n_joint
        result = np.empty((size,) if diagonal else (size, size), dtype=np.float64)
        block = self.diagonal_block if diagonal else self.metric_block
        for start in range(0, size, self.panel):
            rows = np.asarray(
                block(point, jnp.arange(start, start + self.panel, dtype=jnp.int32))
            )
            count = min(self.panel, size - start)
            if diagonal:
                result[start : start + count] = rows[:count]
            else:
                result[:, start : start + count] = rows[:count].T
        return result if diagonal else (result + result.T) / 2

    def fit(self):
        import jax.numpy as jnp
        import numpy as np
        from scipy.optimize import minimize

        started = perf_counter()
        point = self.problem.initial_values
        matrix_start = perf_counter()
        metric = self.matrix(point)
        metric_seconds = perf_counter() - matrix_start
        factor_start = perf_counter()
        chol = jnp.linalg.cholesky(jnp.asarray(metric))
        sync(chol)
        if not np.isfinite(np.asarray(chol)).all():
            raise RuntimeError("AD metric Cholesky failed")
        factor_seconds = perf_counter() - factor_start
        gradient_start = perf_counter()
        _, initial_gradient = self.gradient(point)
        initial_gamma = float(
            jnp.max(jnp.abs(initial_gradient) / jnp.sqrt(jnp.diag(jnp.asarray(metric))))
        )
        threshold = self.atol + 1e-8 * initial_gamma
        z0 = jnp.zeros(self.problem.n_joint, dtype=point.dtype)
        sync(self.value_grad(z0, chol))
        gradient_setup_seconds = perf_counter() - gradient_start

        def fun(z):
            value, grad = self.value_grad(jnp.asarray(z), chol)
            return float(value), np.asarray(grad, dtype=np.float64)

        solve_start = perf_counter()
        result = minimize(
            fun,
            np.zeros(self.problem.n_joint),
            jac=True,
            method="L-BFGS-B",
            options=dict(
                maxiter=3000,
                maxfun=120000,
                maxcor=30,
                maxls=80,
                gtol=1e-8,
                ftol=self.ftol,
            ),
        )
        values = self.decode(jnp.asarray(result.x), chol)
        sync(values)
        solve_seconds = perf_counter() - solve_start
        certificate_start = perf_counter()
        value, grad = self.gradient(values)
        diagonal = self.matrix(values, diagonal=True)
        gamma = float(jnp.max(jnp.abs(grad) / jnp.sqrt(jnp.asarray(diagonal))))
        certificate_seconds = perf_counter() - certificate_start
        products_start = perf_counter()
        products = self.problem._result_products(values, *self.strengths)
        sync(products)
        product_seconds = perf_counter() - products_start
        total = perf_counter() - started
        return values, dict(
            map_products_seconds=total,
            public_fit_seconds=total,
            metric_seconds=metric_seconds,
            factor_seconds=factor_seconds,
            gradient_setup_seconds=gradient_setup_seconds,
            solve_seconds=solve_seconds,
            certificate_seconds=certificate_seconds,
            product_seconds=product_seconds,
            prior_logdet_seconds=0.0,
            final_evidence_seconds=0.0,
            success=bool(result.success and gamma <= threshold),
            library_success=bool(result.success),
            ftol=self.ftol,
            termination=str(result.message),
            iterations=int(result.nit),
            nfev=int(result.nfev),
            objective=float(value),
            scaled_gradient=gamma,
            gradient_threshold=threshold,
        )


class PreparedLM:
    """Run the unmodified public LM fit; observe evidence-only timing boundaries."""

    def __init__(self, problem, strengths, atol):
        from TinyLensGpu.ObservationModel.LensImage.potential_correction_iterative import (
            _DenseIterationBackend,
            _logdet_spd,
        )

        self.problem, self.strengths, self.atol = problem, strengths, atol
        backend = _DenseIterationBackend(problem)
        original_logdets = backend.logdets
        original_products = problem._result_products
        self.events = {}

        def initialize(strengths):
            # Exact original three operations. Synchronize Q before measuring
            # normalization so its construction is NOT excluded from MAP time.
            prior = problem.dense_prior(*strengths)
            sync(prior)
            start = perf_counter()
            logdet, positive = _logdet_spd(prior)
            sync((logdet, positive))
            self.events["prior_logdet_seconds"] = perf_counter() - start
            return prior, logdet, positive

        def logdets(system):
            sync(system.curvature)
            self.events["final_evidence_start"] = perf_counter()
            return original_logdets(system)

        def products(*args, **kwargs):
            entered = perf_counter()
            if "final_evidence_start" in self.events:
                self.events["final_evidence_seconds"] = (
                    entered - self.events["final_evidence_start"]
                )
            value = original_products(*args, **kwargs)
            sync(value)
            self.events["product_seconds"] = perf_counter() - entered
            return value

        backend.initialize = initialize
        backend.logdets = logdets
        problem._dense_backend = backend
        problem._result_products = products

    def fit(self):
        import jax.numpy as jnp

        self.events = {}
        started = perf_counter()
        result = self.problem.fit(
            *self.strengths, backend="dense", max_iter=80, gradient_atol=self.atol
        )
        sync(result)
        elapsed = perf_counter() - started
        evidence = self.events.get("final_evidence_seconds", 0.0)
        prior = self.events.get("prior_logdet_seconds", 0.0)
        initial = result.configuration.get("initial_scaled_gradient")
        threshold = self.atol + 1e-8 * initial if initial is not None else 0.0
        return jnp.r_[result.source_values, result.potential_values], dict(
            map_products_seconds=elapsed - prior - evidence,
            public_fit_seconds=elapsed,
            prior_logdet_seconds=prior,
            final_evidence_seconds=evidence,
            product_seconds=self.events.get("product_seconds", 0.0),
            success=bool(
                result.success
                and result.configuration.get("final_scaled_gradient") is not None
                and result.configuration["final_scaled_gradient"] <= threshold
            ),
            termination=result.termination_reason,
            iterations=result.configuration["accepted_iterations"],
            objective=float(result.objective),
            scaled_gradient=result.configuration.get("final_scaled_gradient"),
            gradient_threshold=threshold,
        )


def worker(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import scipy
    from run_scientific import (
        MockConfig,
        generate_mock_pair,
        SimulatorConfig,
        PotentialCorrectionRenderer,
        IterativePotentialCorrection,
        digest,
        inspect_endpoint,
    )

    cfg = json.loads(
        (
            ROOT
            / "examples/potential_correction/iterative_scientific_configuration.json"
        ).read_text()
    )
    with jax.experimental.enable_x64(), jax.default_matmul_precision("highest"):
        mock = MockConfig.from_dict(cfg["mock"])
        pair = generate_mock_pair(mock)
        observation = getattr(pair, args.case)
        sim = SimulatorConfig(
            mock.dpix,
            mock.npix,
            observation.psf_kernel,
            cfg["inversion_nsub"],
            observation.mask,
        )
        renderer = PotentialCorrectionRenderer(
            observation.macro,
            sim,
            cfg["source_n"],
            tuple(cfg["source_bbox"]),
            potential_spacing=cfg["potential_spacing"],
            dtype=mock.dtype,
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        source = observation.source_truth.brightness(sx, sy).ravel()
        problem = IterativePotentialCorrection(
            renderer,
            observation.image_data,
            observation.noise_map,
            source,
            lambda_amplitude=cfg["lambda_amplitude"],
            source_regularization=cfg["source_regularization"],
            solve_rtol=cfg["solve_rtol"],
            solve_atol=cfg["solve_atol"],
        )
        sync((problem.initial_values, problem.data_active, problem.noise_active))
        p = cfg["fixed_strengths"][args.case]
        strengths = (p["lambda_source"], p["lambda_potential"])
        atol = (
            cfg["case_iteration_options"]
            .get(args.case, {})
            .get("gradient_atol", cfg["iteration_options"]["gradient_atol"])
        )
        prepared = (PreparedAD if args.method == "lbfgs" else PreparedLM)(
            problem, strengths, atol
        )
        if args.method == "lbfgs":
            prepared.ftol = args.ftol
        shutil.copyfile(Path(__file__), args.output / "benchmark_snapshot.py")
        rows = []
        for repetition in range(args.repeats + 1):
            values, row = prepared.fit()
            row["phase"] = "cold" if repetition == 0 else "warm"
            row["repetition"] = repetition
            rows.append(row)
            print(
                f"{args.method}/{args.case} {row['phase']} {repetition}: "
                f"MAP+products={row['map_products_seconds']:.4f}s "
                f"public={row['public_fit_seconds']:.4f}s "
                f"success={row['success']} iter={row['iterations']}",
                flush=True,
            )
            if not row["success"]:
                break
        audit = inspect_endpoint(problem, strengths, values, panel=16)
        report = dict(
            method=args.method,
            case=args.case,
            rows=rows,
            audit=audit,
            input_sha256=digest(
                problem.data_active, problem.noise_active, problem.initial_values
            ),
            config=cfg,
            python=platform.python_version(),
            jax=jax.__version__,
            scipy=scipy.__version__,
            numpy=np.__version__,
            gpu=[d.device_kind for d in jax.devices()],
            persistent_compilation_cache=str(jax.config.jax_compilation_cache_dir),
            ftol=args.ftol,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope="Prepared original truth-init model through certified MAP and raw image/field products. No metric disk cache, plotting, file output or evidence. Warm recomputes numeric metric/factors; stable compiled functions.",
        )
        write(args.output / f"{args.method}_{args.case}.json", report)
        if not all(row["success"] for row in rows):
            raise RuntimeError(
                "A timed reconstruction failed; do not treat it as a valid speed result"
            )


def summarize_results(results):
    """Keep failures visible; timing ratios describe successful samples only."""
    summary = {}
    for case in ("subhalo", "control"):
        lm, lb = results[f"lm/{case}"], results[f"lbfgs/{case}"]
        if lm["input_sha256"] != lb["input_sha256"]:
            raise RuntimeError("Timing methods received different inputs")
        item = {}
        for method, report in (("lm", lm), ("lbfgs", lb)):

            def passed(row):
                g = row.get("scaled_gradient")
                return (
                    row["success"] and g is not None and g <= row["gradient_threshold"]
                )

            warm_rows = [r for r in report["rows"] if r["phase"] == "warm"]
            warm = [r["map_products_seconds"] for r in warm_rows if passed(r)]
            item[method] = dict(
                cold_seconds=report["rows"][0]["map_products_seconds"],
                cold_passed=bool(passed(report["rows"][0])),
                cold_public_seconds=report["rows"][0]["public_fit_seconds"],
                warm_attempted=len(warm_rows),
                warm_passed=len(warm),
                successful_warm_median=statistics.median(warm) if warm else None,
                successful_warm_min=min(warm) if warm else None,
                successful_warm_max=max(warm) if warm else None,
                all_rows_passed=all(passed(r) for r in report["rows"]),
                failed_rows=[r for r in report["rows"] if not passed(r)],
            )
        a, b = (
            item["lm"]["successful_warm_median"],
            item["lbfgs"]["successful_warm_median"],
        )
        item["conditional_successful_warm_ratio_lm_over_lbfgs"] = (
            a / b if a and b else None
        )
        item["all_rows_passed"] = (
            item["lm"]["all_rows_passed"] and item["lbfgs"]["all_rows_passed"]
        )
        summary[case] = item
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "timing_full"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ftol", type=float, default=1e-15)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=("lm", "lbfgs"))
    parser.add_argument("--case", choices=("subhalo", "control"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    results = {}
    for case in ("subhalo", "control"):
        for method in ("lm", "lbfgs"):
            destination = args.output / f"{method}_{case}.json"
            if destination.exists():
                raise RuntimeError(f"Use a fresh output directory: {destination}")
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--method",
                    method,
                    "--case",
                    case,
                    "--output",
                    str(args.output.resolve()),
                    "--repeats",
                    str(args.repeats),
                    "--ftol",
                    str(args.ftol),
                ],
                cwd=ROOT,
                check=False,
            )
            if not destination.exists():
                raise RuntimeError(
                    f"Worker failed without a numerical record: {method}/{case}"
                )
            results[f"{method}/{case}"] = json.loads(destination.read_text())
    summary = summarize_results(results)
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not all(row["all_rows_passed"] for row in summary.values()):
        raise SystemExit(
            "Completed timing records contain failed reconstructions; see failed_rows"
        )


if __name__ == "__main__":
    main()
