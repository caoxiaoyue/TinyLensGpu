"""Nonlinear MGE fitting with GPU multi-start projected Optax Adam.

Uses the identical 53-parameter, fixed-width MGE likelihood and flux scaling
as run_model_nonlinear_auto_diff_scipy.py. No linear intensity solve, priors
in the objective, or bijectors are used. Physical boxes, nonnegative fluxes,
and three joint ellipticity disks are enforced after every Adam update.
The extra disk constraint is explicit: q >= --q-min (default 0.01).

Install the optional example dependency in the active environment:
    pip install optax==0.2.6
Run from any directory:
    python /path/to/run_model_nonlinear_auto_diff_optax.py
Defaults: 8 starts, 4 starts per batch, 8000 steps with cosine learning-rate
decay from 0.03 to 3e-5. Use --n-starts to explore more initial conditions.

Each start has independent optimizer state. vmap batches starts and scan
keeps all iterations on the JAX device. The host only loads data, computes
initial flux normalizations, dispatches batches, and exports diagnostics.
Reduce --batch-size if device memory is limited. CPU execution is supported
as a fallback; the actual device is printed and recorded.

Outputs include the best physical fit, all initial/fitted parameters,
decimated histories, gradient checks, MGE fluxes, and an image plot. A finite
fit that exhausts the step budget is saved with converged=false. Multi-start
and small residuals do not establish a global optimum or uncertainties.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

try:
    import optax
except ImportError as exc:
    raise ImportError(
        "Install this example's dependency: pip install optax==0.2.6"
    ) from exc

from run_model_nonlinear_auto_diff_scipy import (
    build_problem,
    check_gradient,
    make_differentiable_loglike,
    scale_flux_coordinates,
    write_csv,
)
from TinyLensGpu.visualizer import plot_model_results


def make_loss(likelihood, scales):
    """Return a pure JAX scalar loss in scaled physical coordinates."""
    n_pixels = int(np.asarray(likelihood.unmask).sum())
    if n_pixels <= len(scales):
        raise ValueError("There must be more unmasked pixels than fitted parameters")
    scales = jnp.asarray(scales, dtype=jnp.float64)
    loglike = make_differentiable_loglike(likelihood)

    def loss(x):
        return (-loglike(x * scales) + likelihood.log_like_const) / n_pixels

    return loss, n_pixels


def make_projection(names, bounds, scales, q_min=0.01):
    """Project boxes and joint ellipticity disks in optimizer coordinates.

    Equal e1/e2 scales make radial projection Euclidean in these coordinates.
    For this example the disks lie inside the [-1, 1] ellipticity boxes, so
    this composition is the exact projection onto the combined feasible set.
    """
    if not 0.0001 < q_min < 1.0:
        raise ValueError(
            "q_min must be between 0.0001 and 1 to avoid ellipticity clipping"
        )
    scales = np.asarray(scales, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if scales.shape != (len(names),) or bounds.shape != (len(names), 2):
        raise ValueError("Names, scales, and bounds must have consistent shapes")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Scales must be finite and positive")
    if np.any(np.isnan(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Bounds must define nonempty intervals")
    radius = (1.0 - q_min) / (1.0 + q_min)
    pairs = []
    for component in ("mass", "src", "lens"):
        indices = np.asarray(
            [names.index(f"e1_{component}"), names.index(f"e2_{component}")]
        )
        if scales[indices[0]] != scales[indices[1]]:
            raise ValueError("Each e1/e2 pair must have equal coordinate scales")
        if np.any(bounds[indices, 0] > -radius) or np.any(bounds[indices, 1] < radius):
            raise ValueError("Ellipticity boxes must contain the entire disk")
        pairs.append((jnp.asarray(indices), radius / scales[indices[0]]))
    lower = jnp.asarray(bounds[:, 0] / scales)
    upper = jnp.asarray(bounds[:, 1] / scales)

    def project(x):
        original = x
        x = optax.projections.projection_box(x, lower, upper)
        for indices, scaled_radius in pairs:
            # Project the original pair: box-clipping it first would change
            # its direction and would not give the nearest point on the disk.
            pair = original[indices]
            norm = jnp.linalg.norm(pair)
            factor = jnp.minimum(1.0, scaled_radius / jnp.maximum(norm, 1e-30))
            x = x.at[indices].set(pair * factor)
        return x

    return project


def make_starts(initial, names, scales, project, n_starts, seed):
    """Reproducible JAX starts in plausible ranges, without mock truth or fits.

    These ranges guide initialization only; optimization uses the full domain.
    Start zero is the same positive guess as the SciPy example. All random
    starts are generated at once, so changing batch size cannot change them.
    """
    if n_starts < 1:
        raise ValueError("n_starts must be positive")
    initial = jnp.asarray(initial, dtype=jnp.float64)
    scales = jnp.asarray(scales, dtype=jnp.float64)
    unit = jax.random.uniform(
        jax.random.PRNGKey(seed), (n_starts, len(names)), dtype=jnp.float64
    )
    theta = jnp.broadcast_to(initial, unit.shape)
    ranges = {
        "theta_E": (1.0, 2.0),
        "e1_mass": (-0.15, 0.15),
        "e2_mass": (-0.15, 0.15),
        "gamma1": (-0.1, 0.1),
        "gamma2": (-0.1, 0.1),
        "e1_src": (-0.2, 0.2),
        "e2_src": (-0.2, 0.2),
        "center_x_src": (-0.4, 0.4),
        "center_y_src": (-0.4, 0.4),
        "e1_lens": (-0.2, 0.2),
        "e2_lens": (-0.2, 0.2),
        "center_x_lens": (-0.02, 0.02),
        "center_y_lens": (-0.02, 0.02),
    }
    for index, name in enumerate(names):
        if name.startswith("flux_"):
            theta = theta.at[:, index].set(
                initial[index] * jnp.exp(1.4 * unit[:, index] - 0.7)
            )
        elif name in ranges:
            lower, upper = ranges[name]
            theta = theta.at[:, index].set(lower + (upper - lower) * unit[:, index])
    theta = theta.at[0].set(initial)
    return jax.jit(jax.vmap(project))(theta / scales)


def make_solver(
    loss,
    project,
    steps=8000,
    learning_rate=0.03,
    final_learning_rate=3e-5,
    gtol=1e-5,
    gradient_clip=10.0,
):
    """Compile independent projected Adam runs with best-point retention.

    The convergence norm is max(abs(x - project(x - grad))), with eta=1,
    evaluated using the original, unclipped gradient. No loss-stagnation stop.
    Nonfinite values stop only their own start and retain its last finite best.
    Projection is outside loss; we do not differentiate through the optimizer.
    """
    if steps < 1 or not 0 < final_learning_rate <= learning_rate:
        raise ValueError(
            "Positive steps and 0 < final learning rate <= learning rate required"
        )
    if (
        not np.isfinite([learning_rate, final_learning_rate, gtol, gradient_clip]).all()
        or gtol <= 0
        or gradient_clip <= 0
    ):
        raise ValueError("Optimizer settings must be finite with positive tolerances")
    schedule = optax.cosine_decay_schedule(
        learning_rate, max(steps - 1, 1), alpha=final_learning_rate / learning_rate
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip), optax.adam(schedule)
    )
    value_and_grad = jax.value_and_grad(loss)

    def evaluate(x):
        value, grad = value_and_grad(x)
        finite = jnp.isfinite(value) & jnp.all(jnp.isfinite(grad))
        pg_norm = jnp.max(jnp.abs(x - project(x - grad)))
        return value, grad, finite, pg_norm

    def solve_one(x0):
        x0 = project(x0)
        carry = (
            x0,
            optimizer.init(x0),
            x0,
            jnp.asarray(jnp.inf),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0),
        )

        def step(carry, unused):
            x, state, best_x, best_value, done, failed, iterations = carry
            value, grad, finite, pg_norm = evaluate(x)
            better = finite & (value < best_value)
            best_x = jnp.where(better, x, best_x)
            best_value = jnp.where(better, value, best_value)
            failed = failed | (~done & ~finite)
            done = done | ~finite | (pg_norm <= gtol)
            active = ~done
            updates, next_state = optimizer.update(
                jnp.where(finite, grad, 0.0), state, x
            )
            candidate = project(optax.apply_updates(x, updates))
            candidate_finite = jnp.all(jnp.isfinite(candidate))
            failed = failed | (active & ~candidate_finite)
            active = active & candidate_finite
            next_x = jnp.where(active, candidate, x)
            next_state = jax.tree.map(
                lambda new, old: jnp.where(active, new, old), next_state, state
            )
            next_carry = (
                next_x,
                next_state,
                best_x,
                best_value,
                done | ~candidate_finite,
                failed,
                iterations + active.astype(jnp.int32),
            )
            return next_carry, (value, best_value, pg_norm)

        carry, history = jax.lax.scan(step, carry, xs=None, length=steps)
        x, _, best_x, best_value, _, failed, iterations = carry
        # Evaluate the last update as well; scan evaluates points before updating.
        value, _, finite, _ = evaluate(x)
        better = finite & (value < best_value)
        best_x = jnp.where(better, x, best_x)
        best_value = jnp.where(better, value, best_value)
        _, _, best_finite, pg_norm = evaluate(best_x)
        return {
            "x": best_x,
            "loss": best_value,
            "projected_gradient_max_abs": pg_norm,
            "converged": best_finite & (pg_norm <= gtol),
            "failed": failed | ~finite | ~best_finite,
            "iterations": iterations,
            "history": history,
        }

    # vmap the entire solver, rather than sharing momentum or stopping criteria.
    return jax.jit(jax.vmap(solve_one))


def _finite_number(value):
    return float(value) if np.isfinite(value) else None


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=script_dir / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "output_auto_diff_optax"
    )
    parser.add_argument(
        "--n-starts", type=int, default=8, help="Number of independent starts"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Starts optimized together on device"
    )
    parser.add_argument(
        "--steps", type=int, default=8000, help="Adam update budget per start"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
        help="Initial cosine-schedule learning rate",
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=3e-5,
        help="Final cosine-schedule learning rate",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=10.0,
        help="Gradient norm limit per start",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-5,
        help="Convergence tolerance for projected gradient mapping",
    )
    parser.add_argument(
        "--q-min",
        type=float,
        default=0.01,
        help="Minimum axis ratio for mass, source, and lens light",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-every", type=int, default=100)
    args = parser.parse_args()
    if min(args.n_starts, args.batch_size, args.steps, args.history_every) < 1:
        parser.error(
            "Start count, batch size, steps, and history interval must be positive"
        )
    if (
        not 0 < args.final_learning_rate <= args.learning_rate
        or not np.isfinite(
            [
                args.learning_rate,
                args.final_learning_rate,
                args.gtol,
                args.gradient_clip,
                args.q_min,
            ]
        ).all()
        or args.gtol <= 0
        or args.gradient_clip <= 0
        or not 0.0001 < args.q_min < 1
    ):
        parser.error("Invalid learning rates, tolerance, clipping norm, or q-min")

    print(
        "Nonlinear MGE maximum likelihood: JAX + multi-start projected Optax Adam",
        flush=True,
    )
    print(
        f"Devices: {jax.devices()}; float64: {jax.config.x64_enabled}; Optax: {optax.__version__}",
        flush=True,
    )
    model, names, initial, bounds, scales = build_problem(args.data_dir)
    setup_start = time.perf_counter()
    scales = scale_flux_coordinates(model, names, initial, scales)
    loss, n_pixels = make_loss(model, scales)
    project = make_projection(names, bounds, scales, args.q_min)
    starts = make_starts(initial, names, scales, project, args.n_starts, args.seed)
    jax.block_until_ready(starts)
    setup_seconds = time.perf_counter() - setup_start
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    gradient_checks = check_gradient(
        value_and_grad, starts[0], names, scales, "initial"
    )
    solver = make_solver(
        loss,
        project,
        args.steps,
        args.learning_rate,
        args.final_learning_rate,
        args.gtol,
        args.gradient_clip,
    )
    batch_size = min(args.batch_size, args.n_starts)
    # Compile without executing a full optimization as a timing warmup.
    compile_start = time.perf_counter()
    executable = solver.lower(starts[:batch_size]).compile()
    compile_seconds = time.perf_counter() - compile_start
    print(
        f"Setup: {setup_seconds:.2f} s; solver compilation: {compile_seconds:.2f} s",
        flush=True,
    )

    results, run_rows, history_rows, batch_seconds = [], [], [], []
    for offset in range(0, args.n_starts, batch_size):
        count = min(batch_size, args.n_starts - offset)
        batch = starts[offset : offset + count]
        # Pad the last batch to reuse the same compiled executable; discard padding.
        if count < batch_size:
            batch = jnp.concatenate(
                [batch, jnp.repeat(batch[-1:], batch_size - count, axis=0)]
            )
        jax.block_until_ready(batch)
        timer = time.perf_counter()
        device_result = executable(batch)
        jax.block_until_ready(device_result)
        seconds = time.perf_counter() - timer
        batch_seconds.append(seconds)
        host_result = jax.device_get(device_result)
        print(
            f"Batch {offset // batch_size + 1}: {count} starts, {seconds:.2f} s",
            flush=True,
        )
        for local_id in range(count):
            start_id = offset + local_id
            result = jax.tree.map(lambda value: value[local_id], host_result)
            results.append(result)
            theta = result["x"] * scales
            run_rows.append(
                {
                    "start": start_id,
                    "converged": bool(result["converged"]),
                    "failed": bool(result["failed"]),
                    "chi2": _finite_number(2 * n_pixels * result["loss"]),
                    "projected_gradient_max_abs": _finite_number(
                        result["projected_gradient_max_abs"]
                    ),
                    "iterations": int(result["iterations"]),
                    "batch_seconds": seconds,
                    **{
                        f"initial_{name}": float(value)
                        for name, value in zip(
                            names, np.asarray(starts[start_id]) * scales
                        )
                    },
                    **dict(zip(names, theta.tolist())),
                }
            )
            current, best, pg = result["history"]
            for iteration in range(0, args.steps, args.history_every):
                history_rows.append(
                    {
                        "start": start_id,
                        "iteration": iteration,
                        "chi2": _finite_number(2 * n_pixels * current[iteration]),
                        "best_chi2": _finite_number(2 * n_pixels * best[iteration]),
                        "projected_gradient_max_abs": _finite_number(pg[iteration]),
                    }
                )
            history_rows.append(
                {
                    "start": start_id,
                    "iteration": args.steps,
                    "chi2": run_rows[-1]["chi2"],
                    "best_chi2": run_rows[-1]["chi2"],
                    "projected_gradient_max_abs": run_rows[-1][
                        "projected_gradient_max_abs"
                    ],
                }
            )
            print(
                f"  Start {start_id}: chi2={run_rows[-1]['chi2']}, "
                f"converged={run_rows[-1]['converged']}, failed={run_rows[-1]['failed']}",
                flush=True,
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "optimization_runs.csv", run_rows)
    write_csv(output_dir / "optimization_history.csv", history_rows)
    valid = [i for i, result in enumerate(results) if np.isfinite(result["loss"])]
    if not valid:
        raise RuntimeError(
            f"No finite fit; inspect {output_dir / 'optimization_runs.csv'}"
        )
    best_id = min(valid, key=lambda i: results[i]["loss"])
    best = results[best_id]
    theta_best = best["x"] * scales
    gradient_checks += check_gradient(
        value_and_grad, best["x"], names, scales, "best_fit"
    )
    chi2 = float(2 * n_pixels * best["loss"])
    prediction = np.asarray(
        model.forward_model(jnp.asarray(theta_best), return_intensity=False)
    )
    residual = (prediction - np.asarray(model.image_data)) / np.asarray(model.noise_map)
    chi2_image = np.sum(residual[np.asarray(model.unmask, dtype=bool)] ** 2)
    if not np.isclose(chi2, chi2_image, rtol=1e-10, atol=1e-8):
        raise RuntimeError("Objective does not agree with the image residual chi2")
    if not np.allclose(
        np.asarray(project(jnp.asarray(best["x"]))), best["x"], rtol=0, atol=1e-12
    ):
        raise RuntimeError("Best fit violates physical constraints")

    fitted = dict(zip(names, theta_best.tolist()))
    mge_rows = []
    for prefix, components in (
        ("src", model.phys_model.source_light),
        ("lens", model.phys_model.lens_light),
    ):
        for index, component in enumerate(components):
            sigma = float(component.sigma.value)
            flux = fitted[f"flux_{prefix}_{index}"]
            mge_rows.append(
                {
                    "plane": prefix,
                    "component": index,
                    "sigma": sigma,
                    "flux": flux,
                    "central_surface_brightness": flux / (2 * np.pi * sigma**2),
                    "at_zero_flux": flux == 0.0,
                }
            )
    summary = {
        "method": "multi-start projected Optax Adam",
        "optax_version": optax.__version__,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "objective": "maximum likelihood; chi2 / (2 * unmasked_pixels)",
        "model": "SIE + Shear; 20 source and 20 lens Gaussian components",
        "use_linear": False,
        "linear_solve_used": False,
        "dpix": 0.074,
        "nsub": 4,
        "data_dir": str(args.data_dir.resolve()),
        "seed": args.seed,
        "options": {
            key: getattr(args, key)
            for key in (
                "n_starts",
                "batch_size",
                "steps",
                "learning_rate",
                "final_learning_rate",
                "gtol",
                "gradient_clip",
                "q_min",
                "history_every",
            )
        },
        "parameter_order": names,
        "scales": dict(zip(names, scales.tolist())),
        "physical_bounds": {
            name: [_finite_number(value) for value in pair]
            for name, pair in zip(names, bounds)
        },
        "ellipticity_max_norm": (1 - args.q_min) / (1 + args.q_min),
        "flux_scaling": "sqrt(unmasked_pixels) / noise-weighted unit-image norm at initial geometry",
        "source_sigmas": [
            float(component.sigma.value) for component in model.phys_model.source_light
        ],
        "lens_sigmas": [
            float(component.sigma.value) for component in model.phys_model.lens_light
        ],
        "best_start": best_id,
        "best_fit": fitted,
        "chi2": chi2,
        "log_likelihood": float(
            make_differentiable_loglike(model)(jnp.asarray(theta_best))
        ),
        "unmasked_pixels": n_pixels,
        "degrees_of_freedom": n_pixels - len(names),
        "reduced_chi2": chi2 / (n_pixels - len(names)),
        "converged": bool(best["converged"]),
        "failed": bool(best["failed"]),
        "projected_gradient_max_abs": _finite_number(
            best["projected_gradient_max_abs"]
        ),
        "projected_gradient_eta": 1.0,
        "converged_starts": sum(row["converged"] for row in run_rows),
        "failed_starts": sum(row["failed"] for row in run_rows),
        "starts_matching_best_chi2": int(
            sum(abs(2 * n_pixels * results[i]["loss"] - chi2) <= 1e-3 for i in valid)
        ),
        "matching_chi2_absolute_tolerance": 1e-3,
        "zero_flux_components": sum(row["at_zero_flux"] for row in mge_rows),
        "total_source_flux": sum(
            row["flux"] for row in mge_rows if row["plane"] == "src"
        ),
        "total_lens_flux": sum(
            row["flux"] for row in mge_rows if row["plane"] == "lens"
        ),
        "setup_seconds": setup_seconds,
        "solver_compile_seconds": compile_seconds,
        "optimization_seconds": sum(batch_seconds),
        "gradient_checks_passed": all(row["passed"] for row in gradient_checks),
    }
    memory = jax.devices()[0].memory_stats()
    if memory is not None:
        summary["device_peak_bytes_in_use"] = int(memory.get("peak_bytes_in_use", 0))
    with (output_dir / "fit_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    write_csv(output_dir / "gradient_check.csv", gradient_checks)
    write_csv(output_dir / "mge_fluxes.csv", mge_rows)
    write_csv(
        output_dir / "best_fit.csv",
        [
            {
                "parameter": name,
                "value": fitted[name],
                "lower_bound": _finite_number(lower),
                "upper_bound": _finite_number(upper),
            }
            for name, (lower, upper) in zip(names, bounds)
        ],
    )
    figure, _ = plot_model_results(
        model,
        theta_best,
        save_path=str(output_dir / "model_visualization.png"),
        title=f"Projected Optax Adam: reduced chi2 = {summary['reduced_chi2']:.4f}",
        show_critical_lines=True,
        show_caustics=True,
    )
    plt.close(figure)
    print(
        f"Best start: {best_id}; reduced chi2: {summary['reduced_chi2']:.6f}; "
        f"projected gradient: {summary['projected_gradient_max_abs']:.3g}; "
        f"converged: {summary['converged']}"
    )
    print(f"Optimization: {sum(batch_seconds):.2f} s; results: {output_dir}")
    # Exhausting Adam's finite step budget is a normal experimental outcome.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
