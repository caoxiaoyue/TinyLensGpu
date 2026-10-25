"""Validate the current 20-MGE example on a frozen source coordinate system."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXAMPLE = (
    ROOT / "examples/pix_src_demo_operator/simple/pix_src_lens_light_mge/fit_joint.py"
)
OUT = Path(__file__).resolve().parent / "output"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--pcg-iterations", type=int, default=400)
    parser.add_argument("--pcg-rtol", type=float, default=1e-5)
    parser.add_argument("--log-lambda", type=float, default=-8.0)
    parser.add_argument("--check-fd", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--label", default="probe")
    parser.add_argument("--point-offset", type=float, default=0.0)
    parser.add_argument("--comp-rtol", type=float, default=1e-4)
    parser.add_argument("--fd-steps", type=float, nargs="+", default=[3e-4, 1e-4, 3e-5])
    parser.add_argument("--refine-iterations", type=int, nargs="*", default=[])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--optimizer-scales", type=float, nargs="+", default=None)
    parser.add_argument("--start-json", type=Path)
    parser.add_argument("--point-json", type=Path)
    parser.add_argument("--check-envelope", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    if args.forward_only and (args.check_fd or args.optimize or args.check_envelope):
        parser.error(
            "--forward-only cannot be combined with derivative/optimization checks"
        )
    for field in ("resume", "start_json", "point_json"):
        path = getattr(args, field)
        if path:
            setattr(args, field, path.resolve())
    OUT.mkdir(exist_ok=True)

    import jax

    jax.config.update("jax_enable_x64", args.x64)
    import jax.numpy as jnp
    import numpy as np
    import TinyLensGpu
    import caskade as ck
    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
        PixelizedImageProbModelOperator,
    )

    setup, marker, _ = EXAMPLE.read_text().partition(
        'print("[5] Running Nautilus sampler ...")'
    )
    if not marker:
        raise RuntimeError("Example setup boundary changed")
    ns = {"__file__": str(EXAMPLE), "__name__": "gradient_validation_setup"}
    exec(compile(setup, str(EXAMPLE), "exec"), ns)
    bbox = ns["prob_model"].infer_source_bbox()
    dtype = jnp.float64 if args.x64 else jnp.float32
    model = PixelizedImageProbModelOperator(
        jnp.asarray(ns["image_data"], dtype=dtype),
        jnp.asarray(ns["noise_map"], dtype=dtype),
        jnp.asarray(ns["psf_kernel"], dtype=dtype),
        ns["DPIX"],
        ns["phys_model"],
        mask=ns["mask"],
        source_seed_mask=ns["source_seed_mask"],
        nsub=4,
        source_bbox_padding=0.2,
        position_likelihood=ns["position_likelihood"],
        fixed_source_bbox=bbox,
        solver_type="pnpg",
        pnpg_max_iter=args.iterations,
        pnpg_rtol=args.rtol,
        pnpg_power_iter=20,
        pnpg_comp_rel_tol=args.comp_rtol,
    )
    model.pcg_max_iter = args.pcg_iterations
    model.pcg_rtol = args.pcg_rtol
    names = ns["param_names"]
    theta = np.asarray(model.get_values(), dtype=np.float64).copy()
    theta[names.index("log_lambda_reg")] = args.log_lambda
    scales = np.array(
        [
            1.0 if name == "log_lambda_reg" else 0.05 if "center_" in name else 0.1
            for name in names
        ]
    )
    theta += (
        args.point_offset * scales * np.array([(-1.0) ** i for i in range(len(theta))])
    )
    if args.point_json:
        point = json.loads(args.point_json.read_text())
        if point["names"] != names:
            raise ValueError("point parameter order differs from model")
        theta = np.asarray(point["parameters"], dtype=np.float64)
    source_weights = jnp.sin(jnp.arange(6400, dtype=dtype) + 1.0) / 6400
    lens_weights = jnp.cos(jnp.arange(20, dtype=dtype) + 1.0) / 20

    def outputs(parameters):
        result = model.evaluate(parameters)
        return jnp.stack(
            [
                result.log_evidence,
                jnp.dot(source_weights, result.source_pixels),
                jnp.dot(lens_weights, result.lens_amplitudes),
            ]
        )

    evaluate = jax.jit(model.evaluate)
    values_fn = jax.jit(outputs)
    jac_fn = jax.jit(jax.jacrev(outputs))
    t0 = time.perf_counter()
    result = jax.block_until_ready(evaluate(jnp.asarray(theta, dtype=dtype)))
    np.savez_compressed(
        OUT / f"{args.label}_initial.npz",
        theta=theta,
        source=np.asarray(result.source_pixels),
        lens=np.asarray(result.lens_amplitudes),
        model_image=np.asarray(result.model_image),
        bbox=np.asarray(model._fixed_source_bbox),
    )
    jac = (
        None
        if args.forward_only
        else np.asarray(jax.block_until_ready(jac_fn(jnp.asarray(theta, dtype=dtype))))
    )
    info = {
        key: np.asarray(value).item()
        for key, value in result.solver_info._asdict().items()
    }
    report = dict(
        package_path=TinyLensGpu.__file__,
        jax_version=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        x64=args.x64,
        source_n=80,
        n_lens_light=20,
        nsub=4,
        bbox=[float(v) for v in model._fixed_source_bbox],
        inferred_bbox=bbox,
        configuration={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        names=names,
        theta=theta.tolist(),
        scales=scales.tolist(),
        log_evidence=float(result.log_evidence),
        solver_info=info,
        jacobian=None if jac is None else jac.tolist(),
        finite_jacobian=None if jac is None else bool(np.isfinite(jac).all()),
        source_min=float(result.source_pixels.min()),
        lens_min=float(result.lens_amplitudes.min()),
        chi2_per_data=float(
            jnp.mean(
                ((model.data_1d - result.model_image[model.unmask]) / model.noise_1d)
                ** 2
            )
        ),
        initial_seconds=time.perf_counter() - t0,
    )

    def active_diagnostics(parameters, source, lens):
        with ck.ActiveContext(model):
            model.fill_params(parameters)
            fixed_bbox = model._fixed_source_bbox
            mapping = model.sim_obj.precompute_operator_data(*fixed_bbox)
            reg_data = model._regularization_data(*fixed_bbox)
            operator, rhs, _ = model.sim_obj.build_joint_system(
                model.noise_1d,
                model.data_1d,
                *fixed_bbox,
                jnp.exp(model.source_model.log_lambda_reg.value),
                reg_data,
                model.lens_light_regularization,
                op_data=mapping,
            )
            x = jnp.concatenate([source, lens])
            Ax = operator.matvec(x)
            residual = Ax - rhs
            free = x > 1e-8
            error = jnp.max(jnp.where(free, jnp.abs(residual), 0.0))
            margin = (
                1e-8 + args.comp_rtol * jnp.maximum(jnp.abs(rhs), jnp.abs(Ax)) + error
            )
            return dict(
                free_stationarity_max=error,
                active_count=jnp.sum(~free),
                active_source_count=jnp.sum(~free[:6400]),
                active_lens_count=jnp.sum(~free[6400:]),
                strict_complementarity_pass=jnp.all(free | (residual > margin)),
                strict_complementarity_min_margin=jnp.min(
                    jnp.where(free, jnp.inf, residual - margin)
                ),
                strict_complementarity_failed_count=jnp.sum(
                    (~free) & (residual <= margin)
                ),
                active_lens=jnp.logical_not(free[6400:]),
                lens_dual=residual[6400:],
                lens_margin=margin[6400:],
                lens_rhs=rhs[6400:],
            )

    diagnostic = jax.jit(active_diagnostics)(
        jnp.asarray(theta, dtype=dtype), result.source_pixels, result.lens_amplitudes
    )
    report["active_diagnostics"] = {
        k: np.asarray(v).tolist() for k, v in diagnostic.items()
    }
    report_path = OUT / f"{args.label}.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print("INITIAL", json.dumps(report), flush=True)
    if args.forward_only:
        return 0 if info["converged"] else 2
    if not report["finite_jacobian"]:
        print(
            "INVALID_GRADIENT: derivative checks cannot pass at this point", flush=True
        )
        if not args.optimize:
            return 2

    if args.check_envelope:
        # At a resolved constrained MAP, the objective's solution derivative
        # vanishes. Freeze that solution to probe explicit objective terms
        # without repeatedly running the expensive inner optimizer.
        original_solve = model._solve_source
        frozen = jnp.concatenate([result.source_pixels, result.lens_amplitudes])
        model._solve_source = lambda *a, **kw: (frozen, result.solver_info)
        try:
            frozen_gradient = np.asarray(
                jax.jit(jax.grad(lambda p: model(p)))(jnp.asarray(theta, dtype=dtype))
            )
            batch_frozen = jax.jit(jax.vmap(lambda p: model(p)))
            envelope = dict(gradient=frozen_gradient.tolist(), differences=[])
            for h in args.fd_steps:
                deltas = np.diag(h * scales)
                points = np.concatenate([theta + deltas, theta - deltas])
                values = np.asarray(batch_frozen(jnp.asarray(points, dtype=dtype)))
                fd = (values[: len(theta)] - values[len(theta) :]) / (2 * h * scales)
                envelope["differences"].append(dict(h=h, gradient=fd.tolist()))
            original_logdet = model._logdet_block_schur
            try:
                model._logdet_block_schur = lambda p: jax.lax.stop_gradient(
                    original_logdet(p)
                )
                without_logdet = np.asarray(
                    jax.jit(jax.grad(lambda p: model(p)))(
                        jnp.asarray(theta, dtype=dtype)
                    )
                )
                envelope["without_logdet_derivative"] = without_logdet.tolist()
                envelope["logdet_contribution"] = (
                    frozen_gradient - without_logdet
                ).tolist()
            finally:
                model._logdet_block_schur = original_logdet
            report["frozen_solution_envelope"] = envelope
            save()
            print("ENVELOPE", json.dumps(envelope), flush=True)
        finally:
            model._solve_source = original_solve

    if args.check_fd and report["finite_jacobian"]:
        comparisons = []
        batch_values = jax.jit(jax.vmap(outputs))
        for h in args.fd_steps:
            finite_difference = np.empty_like(jac)
            points = []
            for index in range(len(theta)):
                delta = np.zeros_like(theta)
                delta[index] = h * scales[index]
                points.extend([theta + delta, theta - delta])
            values = np.asarray(
                jax.block_until_ready(batch_values(jnp.asarray(points, dtype=dtype)))
            )
            for index in range(len(theta)):
                finite_difference[:, index] = (
                    values[2 * index] - values[2 * index + 1]
                ) / (2 * h * scales[index])
            scaled_ad = jac * scales
            scaled_fd = finite_difference * scales
            error = np.abs(scaled_ad - scaled_fd)
            comparison = dict(
                h=h,
                finite_difference=finite_difference.tolist(),
                max_scaled_absolute_error=error.max(axis=1).tolist(),
                relative_l2_error=(
                    np.linalg.norm(scaled_ad - scaled_fd, axis=1)
                    / np.maximum(np.linalg.norm(scaled_fd, axis=1), 1e-12)
                ).tolist(),
                components_pass=np.all(
                    np.isclose(
                        scaled_ad,
                        scaled_fd,
                        rtol=0.02,
                        atol=np.array([1e-2, 1e-6, 1e-5])[:, None],
                    ),
                    axis=1,
                ).tolist(),
            )
            comparisons.append(comparison)
            report["finite_differences"] = comparisons
            save()
            print("FINITE_DIFFERENCE", json.dumps(comparison), flush=True)
        report["finite_difference_pass"] = all(comparisons[-1]["components_pass"])
        save()
        if not report["finite_difference_pass"]:
            print("FINITE_DIFFERENCE_FAILED: optimization was not started", flush=True)
            return 5

    if args.optimize:
        from scipy.optimize import minimize

        n_data = int(model.data_1d.size)
        origin = jnp.asarray(theta, dtype=dtype)
        opt_scales = (
            scales
            if args.optimizer_scales is None
            else np.asarray(args.optimizer_scales)
        )
        if opt_scales.shape != scales.shape or np.any(opt_scales <= 0):
            raise ValueError(
                "optimizer scales must be positive and match all parameters"
            )
        scales_jax = jnp.asarray(opt_scales, dtype=dtype)
        reference = float(result.log_evidence)

        def loss(z):
            return (reference - model(origin + scales_jax * z)) / n_data

        runners = {}
        budgets = sorted(set([args.iterations] + args.refine_iterations))
        current_budget = args.iterations
        history = []
        accepted = []
        retries = []

        def runner(budget):
            if budget not in runners:
                previous = model.pnpg_max_iter
                model.pnpg_max_iter = budget
                try:
                    runners[budget] = (
                        jax.jit(jax.value_and_grad(loss))
                        .lower(jnp.zeros(len(theta), dtype=dtype))
                        .compile()
                    )
                finally:
                    model.pnpg_max_iter = previous
            return runners[budget]

        def objective(z):
            nonlocal current_budget
            for budget in budgets:
                value, gradient = jax.block_until_ready(
                    runner(budget)(jnp.asarray(z, dtype=dtype))
                )
                value, gradient = float(value), np.asarray(gradient, dtype=np.float64)
                if np.isfinite(value) and np.isfinite(gradient).all():
                    current_budget = budget
                    break
                retries.append(
                    dict(budget=budget, z=np.asarray(z).tolist(), value=value)
                )
                print("REFINE", budget, flush=True)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                report["optimizer_failure"] = dict(
                    z=np.asarray(z).tolist(),
                    value=value,
                    gradient=gradient.tolist(),
                    history=history,
                    retries=retries,
                )
                save()
                raise FloatingPointError("Nonfinite objective/gradient during L-BFGS-B")
            history.append(
                dict(
                    value=value,
                    gradient_norm=float(np.linalg.norm(gradient)),
                    z=np.asarray(z).tolist(),
                    budget=current_budget,
                )
            )
            report["optimization_progress"] = dict(
                history=history, accepted_iterations=accepted
            )
            save()
            print(
                "OPT_EVAL",
                len(history),
                value,
                float(np.linalg.norm(gradient)),
                flush=True,
            )
            return value, gradient

        def callback(z):
            for row in reversed(history):
                if np.array_equal(row["z"], z):
                    accepted.append(row.copy())
                    print(
                        "OPT_STEP",
                        len(accepted),
                        row["value"],
                        row["gradient_norm"],
                        flush=True,
                    )
                    break

        start = (
            scales
            / opt_scales
            * np.array(
                [
                    0.1 if name == "log_lambda_reg" else 0.02 * (-1) ** i
                    for i, name in enumerate(names)
                ]
            )
        )
        if args.resume:
            resumed = json.loads(args.resume.resolve().read_text())
            previous_history = resumed.get("optimization_progress", {}).get(
                "history", []
            )
            if not previous_history:
                previous_history = resumed["optimizer_failure"]["history"]
            best = min(previous_history, key=lambda item: item["value"])
            previous_scales = np.asarray(
                resumed.get("optimizer_scales", resumed["scales"])
            )
            physical = np.asarray(resumed["theta"]) + previous_scales * np.asarray(
                best["z"]
            )
            start = (physical - theta) / opt_scales
            report["resumed_from"] = str(args.resume.resolve())
        if args.start_json:
            point = json.loads(args.start_json.read_text())
            if point["names"] != names:
                raise ValueError("start parameter order differs from model")
            start = (np.asarray(point["parameters"]) - theta) / opt_scales
        report["optimizer_scales"] = opt_scales.tolist()
        report["optimizer_initial_parameters"] = (theta + opt_scales * start).tolist()
        save()
        bounds = []
        for index, spec in enumerate(ns["prior_specs"]):
            low, high = spec.limits
            if spec.prior_type == "uniform":
                low, high = max(low, spec.settings[0]), min(high, spec.settings[1])
            bounds.append(
                (
                    (low - theta[index]) / opt_scales[index],
                    (high - theta[index]) / opt_scales[index],
                )
            )
        started = time.perf_counter()
        try:
            optimized = minimize(
                objective,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback,
                options=dict(maxiter=args.maxiter, maxls=30, ftol=1e-10, gtol=1e-6),
            )
        except FloatingPointError as exc:
            print("OPTIMIZATION_FAILED", str(exc), flush=True)
            return 3
        best_theta = theta + opt_scales * optimized.x
        for entry in reversed(history):
            if np.array_equal(entry["z"], optimized.x):
                current_budget = entry["budget"]
                break
        model.pnpg_max_iter = current_budget
        final_evaluate = jax.jit(lambda parameters: model.evaluate(parameters))
        final = jax.block_until_ready(
            final_evaluate(jnp.asarray(best_theta, dtype=dtype))
        )
        bound_array = np.asarray(bounds)
        projected_gradient = optimized.x - np.clip(
            optimized.x - optimized.jac, bound_array[:, 0], bound_array[:, 1]
        )
        projected_norm = float(np.max(np.abs(projected_gradient)))
        gradient_converged = np.isfinite(projected_norm) and projected_norm <= 1e-6
        reconstruction_valid = bool(
            final.solver_info.converged
            and jnp.isfinite(final.log_evidence)
            and jnp.all(jnp.isfinite(final.source_pixels))
            and jnp.all(jnp.isfinite(final.lens_amplitudes))
            and jnp.min(final.source_pixels) >= 0
            and jnp.min(final.lens_amplitudes) >= 0
        )
        report["optimization"] = dict(
            success=bool(optimized.success),
            gradient_converged=bool(gradient_converged),
            reconstruction_valid=reconstruction_valid,
            validation_pass=bool(
                optimized.success and gradient_converged and reconstruction_valid
            ),
            message=str(optimized.message),
            n_iterations=int(optimized.nit),
            n_evaluations=int(optimized.nfev),
            theta=best_theta.tolist(),
            log_evidence=float(final.log_evidence),
            seconds=time.perf_counter() - started,
            history=history,
            accepted_iterations=accepted,
            retries=retries,
            final_iteration_budget=current_budget,
            initial_log_evidence=reference - history[0]["value"] * n_data,
            scaled_gradient_inf_norm=float(np.max(np.abs(optimized.jac))),
            projected_gradient_inf_norm=projected_norm,
            source_min=float(final.source_pixels.min()),
            lens_min=float(final.lens_amplitudes.min()),
            chi2_per_data=float(
                jnp.mean(
                    ((model.data_1d - final.model_image[model.unmask]) / model.noise_1d)
                    ** 2
                )
            ),
            final_solver_info={
                k: np.asarray(v).item() for k, v in final.solver_info._asdict().items()
            },
        )
        np.savez_compressed(
            OUT / f"{args.label}_fit.npz",
            theta=best_theta,
            source=np.asarray(final.source_pixels),
            lens=np.asarray(final.lens_amplitudes),
            model_image=np.asarray(final.model_image),
            bbox=np.asarray(model._fixed_source_bbox),
        )
        save()
        print(
            "OPTIMIZATION",
            json.dumps(
                {k: v for k, v in report["optimization"].items() if k != "history"}
            ),
            flush=True,
        )
        if not optimized.success:
            return 4
        if not gradient_converged or not reconstruction_valid:
            return 6
    if args.check_fd and not report.get("finite_difference_pass", False):
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
