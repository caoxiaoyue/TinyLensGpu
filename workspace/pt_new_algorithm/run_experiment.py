"""PROTOTYPE: JAX-autodiff L-BFGS versus the existing fixed-strength LM.

    python workspace/pt_new_algorithm/run_experiment.py

No evidence calculation, macro fitting, hyperparameter tuning, or library changes.
The L-BFGS objective calls only the nonlinear forward and prior actions; it does
not call the analytical gradient, Jacobian, curvature, or LM implementation.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import scipy
from scipy.optimize import minimize

from problems import CASES, make_problem

SCALINGS = ("raw", "diagonal", "whitened", "initial_diagonal", "initial_whitened")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def objective_for(problem, strengths):
    """Exactly .5 * (nonlinear chi2 + s.T Qs s + p.T Qp p)."""

    def objective(values):
        residual = (
            problem.data_active - problem.forward(values)
        ) / problem.noise_active
        prior_values = problem.prior_matvec(values, *strengths)
        return 0.5 * (jnp.vdot(residual, residual) + jnp.vdot(values, prior_values))

    return objective


def coordinate_maps(problem, strengths, requested):
    """Invertible, fixed changes of coordinates; the physical prior is unchanged.

    Dense prior whitening is intentionally limited to this small experiment.
    It is NOT a claim of a scalable matrix-free implementation.
    """
    prior = problem.dense_prior(*strengths)
    chol = jnp.linalg.cholesky(prior)
    chol.block_until_ready()
    if not np.isfinite(np.asarray(chol)).all():
        raise RuntimeError("Prototype prior whitening needs a positive-definite prior")
    origin = problem.initial_values
    diagonal = jnp.sqrt(jnp.diag(prior))

    def white(z):
        return origin + jsl.solve_triangular(chol.T, z, lower=False)

    def white_gradient(gradient):
        return jsl.solve_triangular(chol, gradient, lower=True)

    maps = {
        "raw": lambda z: origin + z,
        "diagonal": lambda z: origin + z / diagonal,
        "whitened": white,
    }
    if any(name.startswith("initial_") for name in requested):
        # One fixed metric from an AUTODIFF Jacobian. There is no handwritten
        # derivative and no updated curvature in the L-BFGS loop. Small-grid
        # diagnostic only: this dense matrix is not a scalable operator method.
        jacobian = jax.jit(jax.jacfwd(problem.forward))(origin)
        weighted = jacobian / problem.noise_active[:, None]
        initial_diagonal = jnp.sqrt(jnp.sum(weighted**2, axis=0) + jnp.diag(prior))
        maps["initial_diagonal"] = lambda z: origin + z / initial_diagonal
        if "initial_whitened" in requested:
            metric_chol = jnp.linalg.cholesky(weighted.T @ weighted + prior)
            metric_chol.block_until_ready()
            if not np.isfinite(np.asarray(metric_chol)).all():
                raise RuntimeError("Initial autodiff metric factorization failed")
            maps["initial_whitened"] = lambda z: origin + jsl.solve_triangular(
                metric_chol.T, z, lower=False
            )
    return maps, white_gradient


def derivative_audit(problem, strengths, objective, maps):
    """Analytical branch gradient is an audit reference, never optimizer input."""
    value_grad = jax.jit(jax.value_and_grad(objective))
    evaluate_analytic = jax.jit(lambda x: problem._evaluate(x, *strengths))
    rng = np.random.default_rng(20261103)
    samples = [problem.initial_values]
    samples.append(
        maps["whitened"](jnp.asarray(rng.normal(size=problem.n_joint) * 1e-3))
    )
    records = []
    for point in samples:
        value, gradient = value_grad(point)
        reference = evaluate_analytic(point)
        analytical = -np.asarray(reference.negative_gradient)
        difference = np.asarray(gradient) - analytical
        records.append(
            dict(
                objective_abs_error=abs(float(value - reference.objective)),
                gradient_relative_l2=float(
                    np.linalg.norm(difference) / max(1.0, np.linalg.norm(analytical))
                ),
                gradient_max_abs_error=float(np.max(np.abs(difference))),
            )
        )
    transformed = jax.jit(lambda z: objective(maps["whitened"](z)))
    point = jnp.asarray(rng.normal(size=problem.n_joint) * 1e-3)
    direction = jnp.asarray(rng.normal(size=problem.n_joint))
    direction = direction / jnp.linalg.norm(direction)
    predicted = float(jnp.vdot(jax.grad(transformed)(point), direction))
    differences = []
    for step in (1e-4, 1e-5, 1e-6):
        finite = float(
            (
                transformed(point + step * direction)
                - transformed(point - step * direction)
            )
            / (2 * step)
        )
        minus = problem.linearize(maps["whitened"](point - step * direction)).mapping
        plus = problem.linearize(maps["whitened"](point + step * direction)).mapping
        same_branches = np.array_equal(
            np.asarray(minus.indices), np.asarray(plus.indices)
        ) and np.array_equal(np.asarray(minus.valid), np.asarray(plus.valid))
        differences.append(
            dict(
                step=step,
                autodiff=predicted,
                finite_difference=finite,
                relative_error=abs(finite - predicted) / max(1.0, abs(predicted)),
                same_interpolation_branches=same_branches,
            )
        )
    return dict(
        analytical_comparison=records, directional_finite_differences=differences
    )


def optimize_lbfgs(objective, decode, size, args):
    """Official SciPy L-BFGS-B, unconstrained, supplied with exact JAX gradients."""
    transformed = lambda z: objective(decode(z))
    value_gradient = jax.jit(jax.value_and_grad(transformed))
    z0 = jnp.zeros(size, dtype=jnp.float64)
    started = perf_counter()
    value_gradient(z0)[0].block_until_ready()
    compile_seconds = perf_counter() - started
    history, evaluations = [], 0
    last = {}

    def fun(z):
        nonlocal evaluations
        value, gradient = value_gradient(jnp.asarray(z))
        f, g = float(value), np.asarray(gradient, dtype=np.float64)
        evaluations += 1
        last.update(z=np.array(z, copy=True), f=f, g=g)
        return f, g

    def callback(z):
        if not np.array_equal(z, last["z"]):
            fun(z)
        history.append(
            dict(
                iteration=len(history) + 1,
                objective=last["f"],
                transformed_gradient_inf=float(np.max(np.abs(last["g"]))),
            )
        )

    started = perf_counter()
    result = minimize(
        fun,
        np.zeros(size),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options=dict(
            maxiter=args.max_iter,
            maxfun=args.max_iter * 30,
            maxcor=args.memory,
            maxls=50,
            ftol=args.ftol,
            gtol=args.gtol,
        ),
    )
    values = decode(jnp.asarray(result.x))
    values.block_until_ready()
    elapsed = perf_counter() - started
    return (
        values,
        dict(
            library_success=bool(result.success),
            status=int(result.status),
            termination=str(result.message),
            iterations=int(result.nit),
            evaluations=evaluations,
            scipy_nfev=int(result.nfev),
            compile_seconds=compile_seconds,
            solve_seconds=elapsed,
            timing_scope="L-BFGS solve excludes first JIT and fixed prior-coordinate preparation; includes CPU/GPU transfers",
        ),
        history,
    )


def summarize(
    problem, strengths, objective, values, common_gradient, initial_objective
):
    value, gradient = jax.jit(jax.value_and_grad(objective))(values)
    image = problem.forward(values)
    residual = (problem.data_active - image) / problem.noise_active
    prior_values = problem.prior_matvec(values, *strengths)
    scaled_norm = float(jnp.max(jnp.abs(common_gradient(gradient))))
    # Independent endpoint diagnostic in the SAME physical diagonal metric
    # used by LM. This AD Jacobian is outside both optimizers and their timers.
    jacobian = jax.jit(jax.jacfwd(problem.forward))(values)
    diagonal = jnp.sum(
        (jacobian / problem.noise_active[:, None]) ** 2, axis=0
    ) + jnp.diag(problem.dense_prior(*strengths))
    gn_scaled_norm = float(jnp.max(jnp.abs(gradient) / jnp.sqrt(diagonal)))
    return dict(
        objective=float(value),
        initial_objective=initial_objective,
        chi2=float(jnp.vdot(residual, residual)),
        penalty=float(2 * value - jnp.vdot(residual, residual)),
        source_penalty=float(
            jnp.vdot(values[: problem.n_source], prior_values[: problem.n_source])
        ),
        potential_penalty=float(
            jnp.vdot(values[problem.n_source :], prior_values[problem.n_source :])
        ),
        physical_gradient_inf=float(jnp.max(jnp.abs(gradient))),
        prior_whitened_gradient_inf=scaled_norm,
        prior_whitened_branch_stationary_at_1e_5=scaled_norm <= 1e-5,
        current_gn_scaled_gradient_inf=gn_scaled_norm,
        gn_branch_stationary_at_1e_6=gn_scaled_norm <= 1e-6,
        source_change_l2=float(
            jnp.linalg.norm(
                values[: problem.n_source] - problem.initial_values[: problem.n_source]
            )
        ),
        potential_l2=float(jnp.linalg.norm(values[problem.n_source :])),
        valid_ray_fraction=float(jnp.mean(problem.linearize(values).mapping.valid)),
    )


def run_case(case, args):
    started = perf_counter()
    problem, strengths, metadata = make_problem(case)
    objective = objective_for(problem, strengths)
    initial_objective = float(objective(problem.initial_values))
    coordinate_started = perf_counter()
    maps, common_gradient = coordinate_maps(problem, strengths, args.scalings)
    coordinate_seconds = perf_counter() - coordinate_started
    audit = derivative_audit(problem, strengths, objective, maps)
    directory = args.output / case
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(
        case=case,
        metadata=metadata,
        configuration=problem._configuration(),
        lambda_source=strengths[0],
        lambda_potential=strengths[1],
        lambda_amplitude=problem.lambda_amplitude,
        initial_objective=initial_objective,
        coordinate_setup_seconds=coordinate_seconds,
    )
    write_json(directory / "configuration.json", config)
    write_json(directory / "derivative_audit.json", audit)
    arrays = dict(
        initial=np.asarray(problem.initial_values),
        data=np.asarray(problem.data_active),
        noise=np.asarray(problem.noise_active),
    )
    results = {}
    for mode in args.scalings:
        values, status, history = optimize_lbfgs(
            objective, maps[mode], problem.n_joint, args
        )
        name = "lbfgs_" + mode
        results[name] = {
            **status,
            **summarize(
                problem,
                strengths,
                objective,
                values,
                common_gradient,
                initial_objective,
            ),
        }
        arrays[name] = np.asarray(values)
        write_json(directory / f"{name}_history.json", history)
        print(
            f"{case:18s} {name:18s} F={results[name]['objective']:.10g} "
            f"g={results[name]['prior_whitened_gradient_inf']:.3g} "
            f"success={status['library_success']} iterations={status['iterations']}",
            flush=True,
        )
    if not args.skip_lm:
        fit_started = perf_counter()
        lm = problem.fit(*strengths, max_iter=args.lm_max_iter, backend="dense")
        lm.source_values.block_until_ready()
        lm.potential_values.block_until_ready()
        elapsed = perf_counter() - fit_started
        values = jnp.concatenate((lm.source_values, lm.potential_values))
        results["lm"] = dict(
            library_success=bool(lm.success),
            termination=lm.termination_reason,
            iterations=lm.configuration["accepted_iterations"],
            total_seconds=elapsed,
            timing_scope="public fit includes lazy JIT, dense preparation, optimization, final curvature/evidence/products; not directly comparable to L-BFGS solve_seconds",
            **summarize(
                problem,
                strengths,
                objective,
                values,
                common_gradient,
                initial_objective,
            ),
        )
        arrays["lm"] = np.asarray(values)
        write_json(directory / "lm_history.json", [dict(h) for h in lm.history])
        for name, result in results.items():
            result["objective_relative_to_lm"] = (
                result["objective"] - results["lm"]["objective"]
            ) / max(1.0, abs(results["lm"]["objective"]))
            result["source_rmse_vs_lm"] = float(
                np.sqrt(
                    np.mean(
                        (
                            arrays[name][: problem.n_source]
                            - arrays["lm"][: problem.n_source]
                        )
                        ** 2
                    )
                )
            )
            result["potential_rmse_vs_lm"] = float(
                np.sqrt(
                    np.mean(
                        (
                            arrays[name][problem.n_source :]
                            - arrays["lm"][problem.n_source :]
                        )
                        ** 2
                    )
                )
            )
            prediction = problem.forward(jnp.asarray(arrays[name]))
            reference = problem.forward(values)
            result["image_rms_difference_in_noise_vs_lm"] = float(
                jnp.sqrt(
                    jnp.mean(((prediction - reference) / problem.noise_active) ** 2)
                )
            )
        print(
            f"{case:18s} {'lm':18s} F={float(lm.objective):.10g} "
            f"success={lm.success} reason={lm.termination_reason}",
            flush=True,
        )
    if case == "ridge":
        for name, result in results.items():
            result["source_max_error_vs_exact"] = float(
                np.max(np.abs(arrays[name][: problem.n_source] - 8 / 7))
            )
    write_json(directory / "results.json", results)
    np.savez_compressed(directory / "fields.npz", **arrays)
    return dict(
        configuration=config,
        audit=audit,
        results=results,
        total_seconds=perf_counter() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument(
        "--scalings",
        nargs="+",
        choices=SCALINGS,
        default=["raw", "whitened", "initial_whitened"],
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "results_final"
    )
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--lm-max-iter", type=int, default=160)
    parser.add_argument("--memory", type=int, default=30)
    parser.add_argument("--gtol", type=float, default=1e-7)
    parser.add_argument("--ftol", type=float, default=1e-15)
    parser.add_argument("--skip-lm", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(
        python=platform.python_version(),
        jax=jax.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
        devices=[str(d) for d in jax.devices()],
        dtype="float64",
        command=sys.argv,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        prototype_sha256={
            name: hashlib.sha256(
                (Path(__file__).resolve().parent / name).read_bytes()
            ).hexdigest()
            for name in ("problems.py", "run_experiment.py")
        },
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        scope="fixed macro, fixed grids/PSF/mask, signed joint source+potential, fixed prior strengths; no detection/evidence validation",
    )
    write_json(args.output / "environment.json", environment)
    results = {}
    with jax.experimental.enable_x64(), jax.default_matmul_precision("highest"):
        for case in args.cases:
            results[case] = run_case(case, args)
            write_json(args.output / "summary.json", results)
    rows = [
        dict(case=case, method=method, **record)
        for case, report in results.items()
        for method, record in report["results"].items()
    ]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
