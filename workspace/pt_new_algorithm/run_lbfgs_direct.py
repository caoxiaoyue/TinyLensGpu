"""Research prototype: DIRECT(lambda_s, lambda4, lambda0) around JAX L-BFGS.

Each hyperparameter candidate cold-starts the same nonlinear source/potential
MAP problem. Evidence is the existing local Gauss-Newton approximation, with
complete prior normalization and fresh, undamped endpoint curvature. Float64.
No core-library integration, truth-based selection or inner LM/GN fallback.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

NAMES = ("lambda_source", "lambda_potential", "lambda_amplitude")
BOUNDS = ((-8.0, -1.0), (1.0, 11.0), (0.0, 8.0))
INNER = dict(maxiter=3000, maxfun=120000, maxcor=30, maxls=80, gtol=1e-8, ftol=1e-15)


def safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path, value):
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(args):
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jsl
    import numpy as np
    import scipy
    from scipy.optimize import direct, minimize
    from threadpoolctl import threadpool_limits
    from benchmark_scientific import sync
    from run_scientific import (
        MockConfig,
        generate_mock_pair,
        SimulatorConfig,
        PotentialCorrectionRenderer,
        IterativePotentialCorrection,
        observation_source,
        digest,
        _sample_truth,
        _recovery_metrics,
    )
    from TinyLensGpu.ObservationModel.LensImage.potential_correction_iterative import (
        _DenseIterationBackend,
        _logdet_spd,
    )

    @jax.jit
    def factor(matrix):
        """Equilibration preserves the matrix; no jitter or eigenvalue clipping."""
        scale = jnp.sqrt(jnp.diag(matrix))
        chol = jnp.linalg.cholesky(matrix / scale[:, None] / scale[None, :])
        logdet = 2 * (jnp.log(jnp.diag(chol)).sum() + jnp.log(scale).sum())
        valid = (
            jnp.all(jnp.isfinite(chol))
            & jnp.all(jnp.diag(chol) > 0)
            & jnp.isfinite(logdet)
        )
        return chol, scale, logdet, valid

    class Engine:
        def __init__(self, problem, output):
            self.problem, self.output = problem, output
            self.ns, self.np = problem.n_source, problem.n_potential
            self.q0 = problem.initial_values
            self.panel = args.panel
            preparation = perf_counter()
            problem.dense_prior(1.0, 1.0).block_until_ready()
            self.rs = problem._dense_source_prior
            self.r4 = problem._dense_potential_roughness
            self.eye = jnp.eye(self.np, dtype=jnp.float64)
            _, _, self.logdet_rs, ok = factor(self.rs)
            if not bool(ok):
                raise RuntimeError("Source prior is not SPD; cannot normalize evidence")

            def prior_action(q, strengths):
                source, potential = q[: self.ns], q[self.ns :]
                return jnp.concatenate(
                    (
                        strengths[0]
                        * problem.source_regularizer.matvec_free(
                            source,
                            *problem.renderer.source_bbox,
                            scale=problem.source_scale,
                        ),
                        problem.potential_regularizer.matvec(
                            potential, strengths[1], strengths[2]
                        ),
                    )
                )

            def prior_matrix(strengths):
                potential = strengths[1] * self.r4 + strengths[2] * self.eye
                matrix = jnp.zeros(
                    (problem.n_joint, problem.n_joint), dtype=jnp.float64
                )
                matrix = matrix.at[: self.ns, : self.ns].set(strengths[0] * self.rs)
                return matrix.at[self.ns :, self.ns :].set(potential), potential

            def objective(q, strengths):
                residual = (
                    problem.data_active - problem.forward(q)
                ) / problem.noise_active
                return 0.5 * (
                    jnp.vdot(residual, residual)
                    + jnp.vdot(q, prior_action(q, strengths))
                )

            weighted = lambda q: problem.forward(q) / problem.noise_active

            @jax.jit
            def gram_block(point, indices):
                _, push = jax.linearize(weighted, point)
                _, pull = jax.vjp(weighted, point)
                return jax.vmap(lambda unit: pull(push(unit))[0])(
                    jax.nn.one_hot(indices, problem.n_joint, dtype=jnp.float64)
                )

            def decode(z, chol, scale):
                return self.q0 + jsl.solve_triangular(chol.T, z, lower=False) / scale

            self.prior_action = jax.jit(prior_action)
            self.prior_matrix = jax.jit(prior_matrix)
            self.objective = objective
            self.gradient = jax.jit(jax.value_and_grad(objective))
            self.gram_block = gram_block
            self.decode = jax.jit(decode)
            self.white_gradient = jax.jit(
                jax.value_and_grad(
                    lambda z, strengths, chol, scale: objective(
                        decode(z, chol, scale), strengths
                    )
                )
            )
            self.noise_normalization = float(
                jnp.sum(jnp.log(2 * jnp.pi * problem.noise_active**2))
            )
            gram = self.gram(self.q0)
            self.initial_gram = jnp.asarray(gram)
            sync(self.initial_gram)
            np.save(output / "initial_data_gram.npy", gram)
            np.savez(
                output / "initial_state.npz",
                values=np.asarray(self.q0),
                data=np.asarray(problem.data_active),
                noise=np.asarray(problem.noise_active),
            )
            self.preparation_seconds = perf_counter() - preparation
            self.last_gram = None

        def gram(self, point):
            size = self.problem.n_joint
            result = np.empty((size, size), dtype=np.float64)
            for start in range(0, size, self.panel):
                block = np.asarray(
                    self.gram_block(
                        point, jnp.arange(start, start + self.panel, dtype=jnp.int32)
                    )
                )
                count = min(self.panel, size - start)
                result[:, start : start + count] = block[:count].T
            return 0.5 * (result + result.T)

        def fit(self, coordinates):
            started = perf_counter()
            strengths_np = 10.0 ** np.asarray(coordinates, dtype=np.float64)
            strengths = jnp.asarray(strengths_np)
            row = dict(
                log10_strengths=np.asarray(coordinates).tolist(),
                strengths=dict(zip(NAMES, strengths_np.tolist())),
                accepted=False,
                log_evidence=None,
                evidence_method="local_gauss_newton",
                evidence_is_exact=False,
                determinant_method="full_cholesky",
                initial_state_sha256=digest(self.q0),
                timings={},
            )
            timing = row["timings"]
            q = None
            stage = "prior_and_initial_factor"
            try:
                t = perf_counter()
                prior, potential = self.prior_matrix(strengths)
                _, _, logdet_p, prior_ok = factor(potential)
                logdet_q = self.ns * jnp.log(strengths[0]) + self.logdet_rs + logdet_p
                initial_h = self.initial_gram + prior
                chol0, scale0, _, initial_ok = factor(initial_h)
                sync((chol0, logdet_q))
                timing["prior_initial_factor_seconds"] = perf_counter() - t
                if not bool(prior_ok & initial_ok & jnp.isfinite(logdet_q)):
                    raise ArithmeticError("prior_or_initial_factorization_failed")
                stage = "gradient_setup"
                t = perf_counter()
                _, g0 = self.gradient(self.q0, strengths)
                gamma0 = float(jnp.max(jnp.abs(g0) / jnp.sqrt(jnp.diag(initial_h))))
                zero = jnp.zeros(self.problem.n_joint, dtype=jnp.float64)
                sync(self.white_gradient(zero, strengths, chol0, scale0))
                timing["gradient_setup_seconds"] = perf_counter() - t

                def fun(z):
                    value, gradient = self.white_gradient(
                        jnp.asarray(z), strengths, chol0, scale0
                    )
                    return float(value), np.asarray(gradient, dtype=np.float64)

                stage = "lbfgs"
                t = perf_counter()
                controls = {**INNER, "maxiter": args.max_inner_iter}
                result = minimize(
                    fun,
                    np.zeros(self.problem.n_joint),
                    jac=True,
                    method="L-BFGS-B",
                    options=controls,
                )
                q = self.decode(jnp.asarray(result.x), chol0, scale0)
                sync(q)
                timing["lbfgs_seconds"] = perf_counter() - t
                row["inner"] = dict(
                    library_success=bool(result.success),
                    status=int(result.status),
                    message=str(result.message),
                    iterations=int(result.nit),
                    nfev=int(result.nfev),
                    objective=float(result.fun),
                )
                if not result.success:
                    raise ArithmeticError("inner_native_failure")
                if not bool(jnp.all(jnp.isfinite(q))):
                    raise ArithmeticError("nonfinite_endpoint")
                stage = "endpoint_curvature"
                t = perf_counter()
                f, gradient = self.gradient(q, strengths)
                gram = self.gram(q)
                h = jnp.asarray(gram) + prior
                chol_h, scale_h, logdet_h, h_ok = factor(h)
                sync((chol_h, f, gradient))
                timing["endpoint_curvature_seconds"] = perf_counter() - t
                if not bool(h_ok & jnp.isfinite(f) & jnp.all(jnp.isfinite(gradient))):
                    raise ArithmeticError("endpoint_factorization_or_values_failed")
                stage = "stationarity_and_evidence"
                t = perf_counter()
                whitened_g = jsl.solve_triangular(
                    chol_h, gradient / scale_h, lower=True
                )
                predicted_decrease = float(0.5 * jnp.vdot(whitened_g, whitened_g))
                gamma = float(jnp.max(jnp.abs(gradient) / scale_h))
                threshold = args.legacy_atol + 1e-8 * gamma0
                row.update(
                    objective=float(f),
                    prior_logdet=float(logdet_q),
                    posterior_logdet=float(logdet_h),
                    scaled_gradient=gamma,
                    legacy_gradient_threshold=threshold,
                    legacy_gradient_pass=gamma <= threshold,
                    gn_predicted_decrease=predicted_decrease,
                )
                if (
                    not math.isfinite(predicted_decrease)
                    or predicted_decrease > args.gap_limit
                ):
                    raise ArithmeticError("gn_stationarity_proxy_exceeds_limit")
                loge = float(
                    -f + 0.5 * (logdet_q - logdet_h - self.noise_normalization)
                )
                if not math.isfinite(loge):
                    raise ArithmeticError("nonfinite_evidence")
                row.update(
                    accepted=True,
                    log_evidence=loge,
                    failure_reason=None,
                    physical_gradient_dtype=str(gradient.dtype),
                    endpoint_dtype=str(q.dtype),
                )
                timing["stationarity_evidence_seconds"] = perf_counter() - t
                self.last_gram = gram
            except ArithmeticError as exc:
                row.update(failure_reason=str(exc), failure_stage=stage)
            timing["total_seconds"] = perf_counter() - started
            return q, row

        def audit(self, q, row, name):
            """Same-state check against the original analytical backend, not an optimizer."""
            strengths = row["strengths"]
            original = self.problem
            candidate = IterativePotentialCorrection(
                original.renderer,
                original.image_data,
                original.noise_map,
                self.q0[: self.ns],
                lambda_amplitude=strengths["lambda_amplitude"],
                source_regularization=args.configuration["source_regularization"],
                solve_rtol=args.configuration["solve_rtol"],
                solve_atol=args.configuration["solve_atol"],
            )
            ls, lp = strengths["lambda_source"], strengths["lambda_potential"]
            point = candidate._evaluate_kernel(q, ls, lp)
            backend = _DenseIterationBackend(candidate)
            prior, ldq, qok = backend.initialize((ls, lp))
            system = backend.prepare(point.linearization, prior)
            ldh, hok = backend.logdets(system)
            loge = float(
                -point.objective + 0.5 * (ldq - ldh - self.noise_normalization)
            )
            _, gradient = self.gradient(q, jnp.asarray([strengths[n] for n in NAMES]))
            difference = float(
                jnp.max(
                    jnp.abs(gradient + point.negative_gradient)
                    / jnp.sqrt(system.metric)
                )
            )
            report = dict(
                analytical_log_evidence=loge,
                ad_log_evidence=row["log_evidence"],
                log_evidence_absolute_difference=abs(loge - row["log_evidence"]),
                objective_absolute_difference=abs(
                    float(point.objective) - row["objective"]
                ),
                gradient_difference_in_gn_scale=difference,
                prior_logdet_absolute_difference=abs(float(ldq) - row["prior_logdet"]),
                posterior_logdet_absolute_difference=abs(
                    float(ldh) - row["posterior_logdet"]
                ),
                passed=bool(qok & hok)
                and abs(loge - row["log_evidence"]) <= 1e-3
                and difference <= 1e-8,
            )
            write(self.output / f"{name}_formula_audit.json", report)
            if not report["passed"]:
                raise RuntimeError(
                    "Independent same-state evidence/gradient formula audit failed"
                )
            return report

        def export(self, q, row, name, observation):
            strengths = row["strengths"]
            p = self.problem
            products = p._result_products(
                q, strengths["lambda_source"], strengths["lambda_potential"]
            )
            # The common product helper is bound to baseline lambda0. Replace its
            # penalty/strength metadata using the explicitly dynamic complete Q.
            lam = jnp.asarray([strengths[n] for n in NAMES])
            penalty = float(jnp.vdot(q, self.prior_action(q, lam)))
            products.update(
                strengths=replace(
                    products["strengths"],
                    lambda_amplitude=strengths["lambda_amplitude"],
                ),
                regularization_penalty=penalty,
                objective=0.5 * (products["chi2"] + penalty),
            )
            arrays = {
                key: np.asarray(products[key])
                for key in (
                    "source_values",
                    "source_x_axis",
                    "source_y_axis",
                    "potential_values",
                    "convergence",
                    "deflection_x",
                    "deflection_y",
                    "derivative_valid",
                    "nonlinear_model_image",
                    "nonlinear_image_active",
                    "nonlinear_normalized_residual",
                )
            }
            arrays.update(
                values=np.asarray(q),
                image_data=np.asarray(p.image_data),
                noise_map=np.asarray(p.noise_map),
                potential_x=np.asarray(p.renderer.potential_grid.x),
                potential_y=np.asarray(p.renderer.potential_grid.y),
                core_indices=np.asarray(p.renderer.potential_grid.core_indices),
                node_indices=np.asarray(p.renderer.potential_grid.node_indices),
            )
            np.savez_compressed(self.output / f"{name}_fields.npz", **arrays)
            # Truth enters diagnostics only after the candidate has been selected.
            view = SimpleNamespace(**products)
            truth = _sample_truth(
                view.source_x_axis,
                view.source_y_axis,
                view.potential_grid,
                observation,
                view.dtype,
            )
            recovery, _ = _recovery_metrics(
                view, truth, observation, affine_diagnostics=False
            )
            core = np.asarray(p.renderer.potential_grid.core_indices)
            k = np.asarray(products["convergence"])[core]
            peak = core[int(np.argmax(k))]
            summary = dict(
                recovery=recovery,
                penalty=penalty,
                residual_rms_in_noise=float(
                    np.sqrt(
                        np.mean(
                            np.asarray(
                                (p.data_active - products["nonlinear_image_active"])
                                / p.noise_active
                            )
                            ** 2
                        )
                    )
                ),
                kappa_peak=float(np.max(k)),
                kappa_rms=float(np.sqrt(np.mean(k * k))),
                peak_x=float(p.renderer.potential_grid.x[peak]),
                peak_y=float(p.renderer.potential_grid.y[peak]),
            )
            write(self.output / f"{name}_recovery.json", summary)
            return summary

    output = args.output / args.case
    output.mkdir(parents=True, exist_ok=False)
    (output / "endpoints").mkdir()
    started = perf_counter()
    with threadpool_limits(limits=1), jax.experimental.enable_x64(
        True
    ), jax.default_matmul_precision("highest"):
        cfg = json.loads(
            (
                ROOT
                / "examples/potential_correction/iterative_scientific_configuration.json"
            ).read_text()
        )
        args.configuration = cfg
        args.legacy_atol = (
            cfg["case_iteration_options"]
            .get(args.case, {})
            .get("gradient_atol", cfg["iteration_options"]["gradient_atol"])
        )
        mock = MockConfig.from_dict(cfg["mock"])
        observation = getattr(generate_mock_pair(mock), args.case)
        renderer = PotentialCorrectionRenderer(
            observation.macro,
            SimulatorConfig(
                mock.dpix,
                mock.npix,
                observation.psf_kernel,
                cfg["inversion_nsub"],
                observation.mask,
            ),
            cfg["source_n"],
            tuple(cfg["source_bbox"]),
            potential_spacing=cfg["potential_spacing"],
            dtype="float64",
        )

        def make_problem(source):
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

        baseline_strengths = [
            cfg["fixed_strengths"][args.case]["lambda_source"],
            cfg["fixed_strengths"][args.case]["lambda_potential"],
            cfg["lambda_amplitude"],
        ]
        t = perf_counter()
        if args.initialization == "source_only":
            zero_problem = make_problem(jnp.zeros(renderer.n_source, dtype=jnp.float64))
            initcache = output / "initialization_cache"
            initcache.mkdir()
            source, initinfo = observation_source(
                zero_problem,
                baseline_strengths[0],
                initcache,
                "source_only_baseline",
                args.panel,
            )
            initinfo.update(
                source_initialization="observation_source_only_map",
                lambda_source=baseline_strengths[0],
            )
        else:
            sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
            source = observation.source_truth.brightness(sx, sy).ravel()
            initinfo = dict(
                source_initialization="truth_node_samples", source_truth_used=True
            )
        initinfo["seconds"] = perf_counter() - t
        problem = make_problem(source)
        problem.initialization_provenance.update(initinfo)
        engine = Engine(problem, output)
        bounds = np.asarray(args.bounds).reshape(3, 2)
        protocol = dict(
            case=args.case,
            config=cfg,
            log10_bounds=bounds.tolist(),
            strength_order=NAMES,
            inner_options={**INNER, "maxiter": args.max_inner_iter},
            gap_limit=args.gap_limit,
            gap_interpretation="GN quadratic predicted decrease, not a true objective-gap/evidence-error bound",
            initialization=initinfo,
            initialization_sha256=digest(problem.initial_values),
            input_sha256=digest(problem.data_active, problem.noise_active),
            evidence_method="local_gauss_newton",
            evidence_is_exact=False,
            source_sha256=sha(Path(__file__)),
            dependencies_sha256={
                str(path.relative_to(ROOT)): sha(path)
                for path in (
                    HERE / "run_scientific.py",
                    HERE / "run_experiment.py",
                    HERE / "benchmark_scientific.py",
                    ROOT
                    / "examples/potential_correction/iterative_scientific_configuration.json",
                    ROOT
                    / "TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py",
                    ROOT
                    / "TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py",
                    ROOT / "TinyLensGpu/utils/inversion/potential_regularization.py",
                    ROOT / "TinyLensGpu/utils/inversion/regularization.py",
                    ROOT
                    / "TinyLensGpu/ForwardSimulation/LensImage/potential_correction.py",
                )
            },
            repeat_evidence_tolerance=args.repeat_tolerance,
            software=dict(
                python=platform.python_version(),
                jax=jax.__version__,
                numpy=np.__version__,
                scipy=scipy.__version__,
            ),
            gpu=[d.device_kind for d in jax.devices()],
            cache_scope="Immutable geometry/data, prior bases/logdetRs and fixed initial data Gram only; fresh Q/H0/factor/MAP/endpointGram for each candidate",
            engine_preparation_seconds=engine.preparation_seconds,
        )
        write(output / "protocol.json", protocol)
        trace = []
        cache = {}
        best = {}

        def score(x, phase="direct"):
            key = tuple(float(v) for v in x)
            if key in cache:
                return cache[key]
            q, row = engine.fit(np.asarray(x))
            index = len(trace)
            row.update(index=index, phase=phase)
            trace.append(row)
            value = -row["log_evidence"] if row["accepted"] else np.inf
            cache[key] = value
            if q is not None:
                np.savez(
                    output / "endpoints" / f"{index:05d}.npz", values=np.asarray(q)
                )
            with (output / "evaluations.jsonl").open("a") as f:
                f.write(json.dumps(safe(row), allow_nan=False) + "\n")
            improved = row["accepted"] and (
                not best or row["log_evidence"] > best["row"]["log_evidence"]
            )
            if improved:
                best.update(q=q, row=row)
                write(output / "best.json", row)
            if improved or index % 10 == 0:
                print(
                    f"{args.case} {phase} #{index} accepted={row['accepted']} "
                    f"logE={row['log_evidence']} best={best.get('row',{}).get('log_evidence')} "
                    f"inner={row.get('inner',{}).get('iterations')} "
                    f"seconds={row['timings']['total_seconds']:.3f} {row.get('failure_reason')}",
                    flush=True,
                )
            return value

        baseline_x = np.log10(baseline_strengths)
        score(baseline_x, "baseline")
        baseline_row = trace[0]
        if not baseline_row["accepted"]:
            raise RuntimeError("Baseline did not pass numerical gate; see saved trace")
        baseline_q = jnp.asarray(np.load(output / "endpoints" / "00000.npz")["values"])
        baseline_audit = engine.audit(baseline_q, baseline_row, "baseline")
        baseline_recovery = engine.export(
            baseline_q, baseline_row, "baseline", observation
        )
        repeats = []
        q_repeat, repeat = engine.fit(baseline_x)
        repeat.update(
            phase="baseline_repeat",
            log_evidence_difference=(
                repeat["log_evidence"] - baseline_row["log_evidence"]
                if repeat["accepted"]
                else None
            ),
        )
        repeats.append(repeat)
        write(output / "repeats.json", repeats)
        if (
            not repeat["accepted"]
            or abs(repeat["log_evidence_difference"]) > args.repeat_tolerance
        ):
            raise RuntimeError(
                "Baseline evidence repeat not reproducible within the declared tolerance"
            )
        if args.validate_only:
            write(
                output / "validation.json",
                dict(
                    protocol=protocol,
                    baseline=baseline_row,
                    audit=baseline_audit,
                    repeat=repeat,
                    recovery=baseline_recovery,
                    seconds=perf_counter() - started,
                ),
            )
            return
        search_started = perf_counter()
        result = direct(
            score,
            bounds.tolist(),
            maxfun=args.maxfun,
            maxiter=args.outer_maxiter,
            eps=1e-5,
            len_tol=args.len_tol,
            vol_tol=1e-16,
            locally_biased=True,
        )
        search_seconds = perf_counter() - search_started
        selected = dict(best["row"])
        q = best["q"]
        selected_recovery = engine.export(q, selected, "selected", observation)
        selected_audit = engine.audit(q, selected, "selected")
        for i in range(2):
            rq, rr = engine.fit(np.asarray(selected["log10_strengths"]))
            rr.update(
                phase="selected_repeat",
                repetition=i,
                log_evidence_difference=(
                    rr["log_evidence"] - selected["log_evidence"]
                    if rr["accepted"]
                    else None
                ),
            )
            repeats.append(rr)
            if rq is not None:
                np.savez(output / f"selected_repeat_{i}.npz", values=np.asarray(rq))
            write(output / "repeats.json", repeats)
        distances = np.minimum(
            np.asarray(selected["log10_strengths"]) - bounds[:, 0],
            bounds[:, 1] - np.asarray(selected["log10_strengths"]),
        )
        normalized = distances / (bounds[:, 1] - bounds[:, 0])
        report = dict(
            protocol=protocol,
            baseline=baseline_row,
            selected=selected,
            baseline_recovery=baseline_recovery,
            selected_recovery=selected_recovery,
            baseline_audit=baseline_audit,
            selected_audit=selected_audit,
            log_evidence_gain=selected["log_evidence"] - baseline_row["log_evidence"],
            optimizer=dict(
                success=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                nfev=int(result.nfev),
                nit=int(result.nit),
                x=result.x.tolist(),
                fun=float(result.fun),
                maxfun=args.maxfun,
                len_tol=args.len_tol,
            ),
            selected_boundary_distance_dex=distances.tolist(),
            selected_at_boundary=(normalized <= 0.01).tolist(),
            evaluated_candidates=len(trace),
            accepted_candidates=sum(r["accepted"] for r in trace),
            failed_candidates=[
                dict(
                    index=r["index"],
                    reason=r.get("failure_reason"),
                    strengths=r["strengths"],
                )
                for r in trace
                if not r["accepted"]
            ],
            search_seconds=search_seconds,
            total_process_seconds=perf_counter() - started,
            repeats=repeats,
            selected_repeats_consistent=all(
                r["accepted"]
                and abs(r["log_evidence_difference"]) <= args.repeat_tolerance
                for r in repeats
                if r["phase"] == "selected_repeat"
            ),
            interpretation="Budgeted empirical-Bayes search using local-GN evidence; no global maximum, exact nonlinear marginalization or calibrated detection claim",
        )
        write(output / "summary.json", report)
        print(
            json.dumps(
                safe(
                    {
                        k: report[k]
                        for k in (
                            "log_evidence_gain",
                            "optimizer",
                            "evaluated_candidates",
                            "accepted_candidates",
                            "search_seconds",
                        )
                    }
                ),
                indent=2,
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "lbfgs_direct_evidence")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("subhalo", "control"),
        default=["subhalo", "control"],
    )
    parser.add_argument("--case", choices=("subhalo", "control"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--maxfun", type=int, default=300)
    parser.add_argument("--outer-maxiter", type=int, default=1000)
    parser.add_argument("--len-tol", type=float, default=1e-3)
    parser.add_argument("--max-inner-iter", type=int, default=3000)
    parser.add_argument("--gap-limit", type=float, default=1e-3)
    parser.add_argument("--repeat-tolerance", type=float, default=0.05)
    parser.add_argument("--panel", type=int, default=16)
    parser.add_argument(
        "--initialization", choices=("source_only", "truth"), default="source_only"
    )
    parser.add_argument(
        "--bounds", nargs=6, type=float, default=[v for pair in BOUNDS for v in pair]
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    if any(args.output.iterdir()):
        raise RuntimeError(
            "Use a fresh output directory; existing results are preserved"
        )
    shutil.copyfile(Path(__file__), args.output / "runner_at_measurement.py")
    write(args.output / "protocol.json", vars(args))
    for case in args.cases:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--case",
            case,
            "--output",
            str(args.output),
            "--maxfun",
            str(args.maxfun),
            "--outer-maxiter",
            str(args.outer_maxiter),
            "--len-tol",
            str(args.len_tol),
            "--max-inner-iter",
            str(args.max_inner_iter),
            "--gap-limit",
            str(args.gap_limit),
            "--repeat-tolerance",
            str(args.repeat_tolerance),
            "--panel",
            str(args.panel),
            "--initialization",
            args.initialization,
            "--bounds",
            *[str(v) for v in args.bounds],
        ]
        if args.validate_only:
            command.append("--validate-only")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
