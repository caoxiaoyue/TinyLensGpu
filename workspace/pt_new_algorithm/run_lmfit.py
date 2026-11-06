"""Standalone potential-correction prototype using lmfit's MINPACK LM.

The full objective is preserved as an augmented residual:
    r(x) = [(f(x)-data)/noise, chol(Q).T @ x]
so 0.5*r.T*r is the existing penalized objective, including the accumulated
source and potential (not just increments). Initial whitening and all
derivatives use JAX; lmfit.Minimizer.leastsq performs the actual LM solve.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lmfit
import numpy as np
import scipy
from threadpoolctl import threadpool_limits, threadpool_info

from benchmark_scientific import PreparedAD, sync
from problems import make_problem
from run_scientific import (
    MockConfig,
    generate_mock_pair,
    SimulatorConfig,
    PotentialCorrectionRenderer,
    IterativePotentialCorrection,
    write_json,
    inspect_endpoint,
    record_fields,
    observation_source,
    digest,
)


class LmfitPotentialCorrection(PreparedAD):
    """Reuse the fixed initial AD metric; replace only the optimizer interface."""

    def __init__(self, problem, strengths, atol, panel=16):
        super().__init__(problem, strengths, atol, panel)
        self.names = [f"z{i:05d}" for i in range(problem.n_joint)]

        def residual(z, chol, prior_chol):
            values = self.decode(z, chol)
            image = (
                problem.forward(values) - problem.data_active
            ) / problem.noise_active
            return jnp.concatenate((image, prior_chol.T @ values))

        self.residual = jax.jit(residual)

        @jax.jit
        def jacobian_block(z, chol, prior_chol, indices):
            _, push = jax.linearize(lambda x: residual(x, chol, prior_chol), z)
            return jax.vmap(push)(
                jax.nn.one_hot(indices, problem.n_joint, dtype=problem.dtype)
            )

        self.jacobian_block = jacobian_block

    def fit(self, args, output):
        output.mkdir(parents=True, exist_ok=True)
        total_start = perf_counter()
        p = self.problem
        t = perf_counter()
        metric = self.matrix(p.initial_values)
        metric_seconds = perf_counter() - t
        t = perf_counter()
        chol = jnp.linalg.cholesky(jnp.asarray(metric))
        prior = p.dense_prior(*self.strengths)
        prior_chol = jnp.linalg.cholesky(prior)
        sync((chol, prior_chol))
        if not all(np.isfinite(np.asarray(a)).all() for a in (chol, prior_chol)):
            raise RuntimeError(
                "Physical prior or initial metric not SPD; no hidden jitter"
            )
        factor_seconds = perf_counter() - t
        zero = jnp.zeros(p.n_joint, dtype=p.dtype)
        value0, gradient0 = self.gradient(p.initial_values)
        threshold = self.atol + 1e-8 * float(
            jnp.max(jnp.abs(gradient0) / jnp.sqrt(jnp.diag(jnp.asarray(metric))))
        )
        r0 = self.residual(zero, chol, prior_chol)
        phi_residual = float(0.5 * jnp.vdot(r0, r0))
        np.testing.assert_allclose(phi_residual, float(value0), rtol=1e-10, atol=1e-7)
        # Independent directional chain-rule audit of the whitened residual.
        direction = jnp.asarray(np.random.default_rng(17).normal(size=p.n_joint))
        direction = direction / jnp.linalg.norm(direction)
        _, rdot = jax.jvp(
            lambda z: self.residual(z, chol, prior_chol), (zero,), (direction,)
        )
        derivative = float(jnp.vdot(r0, rdot))
        expected = float(
            jnp.vdot(jsl.solve_triangular(chol, gradient0, lower=True), direction)
        )
        np.testing.assert_allclose(derivative, expected, rtol=1e-8, atol=1e-7)
        identity_audit = dict(
            initial_objective=float(value0),
            augmented_half_sum_squares=phi_residual,
            directional_derivative=derivative,
            expected=expected,
            residual_count=int(len(r0)),
            variables=p.n_joint,
        )
        setup_seconds = perf_counter() - total_start
        params = lmfit.Parameters()
        for name in self.names:
            params.add(name, value=0.0)
        counts = dict(residual=0, jacobian=0)
        timings = dict(
            residual_seconds=0.0,
            jacobian_seconds=0.0,
            covariance_transform_seconds=0.0,
            uncertainty_correlations_seconds=0.0,
        )
        trace = []
        solve_start = perf_counter()
        last = {}
        stop = {"reason": None, "z": None}
        last_callback = {"z": None}
        external_gradient_seconds = 0.0

        def vector(parameters):
            return np.fromiter(
                (parameters[name].value for name in self.names),
                dtype=np.float64,
                count=p.n_joint,
            )

        def residual(parameters):
            t = perf_counter()
            z = vector(parameters)
            values = np.asarray(
                self.residual(jnp.asarray(z), chol, prior_chol), dtype=np.float64
            )
            counts["residual"] += 1
            timings["residual_seconds"] += perf_counter() - t
            last.update(z=z, half_sum_squares=float(0.5 * np.dot(values, values)))
            return values

        def jacobian(parameters):
            t = perf_counter()
            z = jnp.asarray(vector(parameters))
            # col_deriv=True: lmfit/SciPy require (n_variables, n_residuals).
            result = np.empty((p.n_joint, len(r0)), dtype=np.float64)
            for start in range(0, p.n_joint, self.panel):
                part = np.asarray(
                    self.jacobian_block(
                        z,
                        chol,
                        prior_chol,
                        jnp.arange(start, start + self.panel, dtype=jnp.int32),
                    )
                )
                count = min(self.panel, p.n_joint - start)
                result[start : start + count] = part[:count]
            counts["jacobian"] += 1
            np.savez_compressed(
                output / "checkpoint.npz",
                values=np.asarray(self.decode(z, chol)),
                z=np.asarray(z),
                jacobian_calls=counts["jacobian"],
            )
            elapsed = perf_counter() - t
            timings["jacobian_seconds"] += elapsed
            print(
                f"  AD Jacobian {counts['jacobian']} ready ({elapsed:.3f}s, "
                f"elapsed {perf_counter()-solve_start:.1f}s)",
                flush=True,
            )
            return result

        def iteration(parameters, iteration, residual_array, *unused, **kwargs):
            nonlocal external_gradient_seconds
            if stop["reason"] is not None:
                return True
            if getattr(minner.result, "aborted", False):
                return True
            last_callback["z"] = np.array(vector(parameters), copy=True)
            elapsed = perf_counter() - solve_start
            row = dict(
                residual_call=int(iteration),
                seconds=elapsed,
                objective=float(0.5 * np.dot(residual_array, residual_array)),
            )
            trace.append(row)
            if args.stop_on_gradient:
                checked = perf_counter()
                point = self.decode(jnp.asarray(vector(parameters)), chol)
                _, gradient = self.gradient(point)
                diagonal = self.matrix(point, diagonal=True)
                current = float(
                    jnp.max(jnp.abs(gradient) / jnp.sqrt(jnp.asarray(diagonal)))
                )
                external_gradient_seconds += perf_counter() - checked
                row["current_gn_scaled_gradient"] = current
                if current <= threshold and row["objective"] <= float(value0):
                    stop["reason"] = "physical_gradient_converged"
                    stop["z"] = np.array(vector(parameters), copy=True)
                    return True
            if len(trace) == 1 or len(trace) % 5 == 0:
                print(
                    f"  lmfit residual call {iteration}, F={row['objective']:.12g}, elapsed={elapsed:.1f}s",
                    flush=True,
                )
            elapsed = perf_counter() - solve_start
            if args.max_seconds and elapsed > args.max_seconds:
                stop["reason"] = "callback_wall_budget"
                stop["z"] = np.array(vector(parameters), copy=True)
                return True
            return False

        minner = lmfit.Minimizer(
            residual, params, iter_cb=iteration, scale_covar=False, nan_policy="raise"
        )
        # Keep the actual lmfit default covariance/error processing; just measure
        # it separately. calc_covar=False does NOT disable this for leastsq.
        for method, key in (
            ("_int2ext_cov_x", "covariance_transform_seconds"),
            (
                "_calculate_uncertainties_correlations",
                "uncertainty_correlations_seconds",
            ),
        ):
            original = getattr(minner, method)

            def timed(*a, _original=original, _key=key, **kw):
                t = perf_counter()
                try:
                    return _original(*a, **kw)
                finally:
                    timings[_key] += perf_counter() - t

            setattr(minner, method, timed)
        result = minner.leastsq(
            Dfun=jacobian,
            col_deriv=True,
            max_nfev=args.max_nfev,
            ftol=args.ftol,
            xtol=args.xtol,
            gtol=args.gtol,
            factor=100,
        )
        api_seconds = perf_counter() - solve_start
        endpoint_source = "native_returned_parameters"
        if stop["z"] is not None:
            endpoint_source = "owned_copy_of_first_stopping_callback"
        elif result.aborted:
            # max_nfev can abort BEFORE iter_cb. Its final cleanup callback may
            # expose a borrowed/freed MINPACK buffer; never replace our last
            # normally completed callback with that cleanup state.
            stop["reason"] = "native_abort_before_physical_stop"
            stop["z"] = np.array(
                np.asarray(zero) if last_callback["z"] is None else last_callback["z"],
                copy=True,
            )
            endpoint_source = (
                "initial_point_before_any_callback"
                if last_callback["z"] is None
                else "owned_last_completed_callback_before_native_abort"
            )
        # lmfit can re-read a borrowed MINPACK parameter buffer after abort.
        # Keep the FIRST callback stop vector as an owned copy, not that buffer.
        returned_z = vector(result.params)
        z = jnp.asarray(returned_z if stop["z"] is None else stop["z"])
        values = self.decode(z, chol)
        sync(values)
        certificate_start = perf_counter()
        value, grad = self.gradient(values)
        diagonal = self.matrix(values, diagonal=True)
        gamma = float(jnp.max(jnp.abs(grad) / jnp.sqrt(jnp.asarray(diagonal))))
        certificate_seconds = perf_counter() - certificate_start
        product_start = perf_counter()
        products = p._result_products(values, *self.strengths)
        sync(products)
        product_seconds = perf_counter() - product_start
        total_seconds = perf_counter() - total_start
        rfinal = self.residual(z, chol, prior_chol)
        np.testing.assert_allclose(
            float(0.5 * jnp.vdot(rfinal, rfinal)), float(value), rtol=1e-9, atol=1e-6
        )
        stationary = gamma <= threshold
        info = dict(
            accepted_reconstruction=bool(
                stationary
                and (
                    (result.success and not result.aborted)
                    or stop["reason"] == "physical_gradient_converged"
                )
            ),
            library_success=bool(result.success),
            physical_stationarity_passed=stationary,
            aborted=bool(result.aborted),
            stop_reason=stop["reason"],
            external_gradient_stop=stop["reason"] == "physical_gradient_converged",
            endpoint_source=endpoint_source,
            native_parameters_match_captured_stop=(
                None
                if stop["z"] is None
                else bool(np.array_equal(returned_z, stop["z"]))
            ),
            ier=int(result.ier),
            message=str(result.message),
            minpack_message=str(result.lmdif_message),
            nfev=int(result.nfev),
            njev=counts["jacobian"],
            counts=counts,
            objective=float(value),
            scaled_gradient=gamma,
            gradient_threshold=threshold,
            lmfit_chisqr=(
                None
                if getattr(result, "chisqr", None) is None
                else float(result.chisqr)
            ),
            augmented_sum_squares=float(jnp.vdot(rfinal, rfinal)),
            covariance_present=result.covar is not None,
            errorbars=bool(result.errorbars),
            timings=dict(
                total_seconds=total_seconds,
                setup_seconds=setup_seconds,
                metric_seconds=metric_seconds,
                factor_seconds=factor_seconds,
                lmfit_api_seconds=api_seconds,
                certificate_seconds=certificate_seconds,
                product_seconds=product_seconds,
                external_gradient_seconds=external_gradient_seconds,
                **timings,
            ),
            options=dict(
                method="lmfit.Minimizer.leastsq",
                Dfun="JAX AD",
                col_deriv=True,
                max_nfev=args.max_nfev,
                ftol=args.ftol,
                xtol=args.xtol,
                gtol=args.gtol,
                max_seconds=args.max_seconds,
                default_uncertainty_processing=True,
                scale_covar=False,
                stop_on_gradient=args.stop_on_gradient,
                wall_budget_policy="checked at residual callbacks, not a hard process timeout",
            ),
            objective_identity=identity_audit,
            scope="Complete augmented MAP; CPU MINPACK LM with JAX GPU residual/Jacobian; no evidence",
        )
        write_json(output / "optimizer.json", info)
        write_json(output / "history.json", trace)
        np.savez_compressed(
            output / "endpoint.npz", values=np.asarray(values), z=np.asarray(z)
        )
        return values, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", choices=("coupled", "nodal", "scientific"), default="scientific"
    )
    parser.add_argument(
        "--configuration",
        type=Path,
        default=ROOT
        / "examples/potential_correction/iterative_scientific_configuration.json",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT
        / ".scratch/potential-correction/output/iterative-scientific/validated-reference",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("subhalo", "control"),
        default=["subhalo", "control"],
    )
    parser.add_argument(
        "--initialization", choices=("truth", "source_only"), default="truth"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "lmfit_full"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=0,
        help="Additional fits with compiled kernels retained",
    )
    parser.add_argument("--panel", type=int, default=16)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=600.0,
        help="Solver callback wall budget; zero disables",
    )
    parser.add_argument("--ftol", type=float, default=1e-13)
    parser.add_argument("--xtol", type=float, default=1e-13)
    parser.add_argument("--gtol", type=float, default=1e-13)
    parser.add_argument("--blas-threads", type=int, default=16)
    parser.add_argument(
        "--stop-on-gradient",
        action="store_true",
        help="Use lmfit's documented iter_cb to stop when original physical gradient gate passes",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "environment.json").exists():
        raise RuntimeError("Use a fresh output directory to preserve numerical records")
    shutil.copyfile(Path(__file__), args.output / "runner_snapshot.py")
    environment = dict(
        python=platform.python_version(),
        executable=sys.executable,
        jax=jax.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
        lmfit=lmfit.__version__,
        gpu=[d.device_kind for d in jax.devices()],
        blas_threads=args.blas_threads,
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("benchmark_scientific.py"),
                Path(__file__).with_name("run_scientific.py"),
            )
        },
    )
    write_json(args.output / "environment.json", environment)
    records = {}
    with jax.experimental.enable_x64(), jax.default_matmul_precision(
        "highest"
    ), threadpool_limits(limits=args.blas_threads):
        write_json(args.output / "threadpools.json", threadpool_info())
        if args.fixture == "scientific":
            config = json.loads(args.configuration.read_text())
            write_json(args.output / "original_configuration.json", config)
            mock = MockConfig.from_dict(config["mock"])
            pair = generate_mock_pair(mock)
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
            )
            cases = args.cases
        else:
            cases = [args.fixture]
        for case in cases:
            baseline = None
            init_info = dict(source_truth_used=True, macro_fixed_to_truth=True)
            if args.fixture == "scientific":
                observation = getattr(pair, case)
                physical = config["fixed_strengths"][case]
                strengths = (physical["lambda_source"], physical["lambda_potential"])

                def build(source):
                    return IterativePotentialCorrection(
                        renderer,
                        observation.image_data,
                        observation.noise_map,
                        source,
                        lambda_amplitude=config["lambda_amplitude"],
                        source_regularization=config["source_regularization"],
                        solve_rtol=config["solve_rtol"],
                        solve_atol=config["solve_atol"],
                    )

                if args.initialization == "truth":
                    sx, sy = jnp.meshgrid(
                        renderer.source_x_axis, renderer.source_y_axis
                    )
                    source = observation.source_truth.brightness(sx, sy).ravel()
                else:
                    seed = build(jnp.zeros(renderer.n_source, dtype=renderer.dtype))
                    cache = args.output / "source_cache"
                    cache.mkdir(exist_ok=True)
                    source, init_info = observation_source(
                        seed, strengths[0], cache, case, args.panel
                    )
                problem = build(source)
                atol = (
                    config.get("case_iteration_options", {})
                    .get(case, {})
                    .get("gradient_atol", config["iteration_options"]["gradient_atol"])
                )
                reference = args.baseline / case / "dense"
                original_metadata = json.loads((reference / "result.json").read_text())
                if (
                    original_metadata["result_configuration"]["source_n"]
                    != config["source_n"]
                ):
                    raise RuntimeError("Baseline source resolution differs")
                if original_metadata["strengths"] != dict(
                    lambda_source=strengths[0],
                    lambda_potential=strengths[1],
                    lambda_amplitude=config["lambda_amplitude"],
                ):
                    raise RuntimeError("Baseline physical prior differs")
                with np.load(
                    reference / "reconstruction.npz", allow_pickle=False
                ) as saved:
                    for name in ("image_data", "noise_map", "mask"):
                        np.testing.assert_allclose(
                            saved[name],
                            getattr(observation, name),
                            rtol=1e-12,
                            atol=1e-12,
                        )
                    baseline = {
                        name: np.array(saved[name])
                        for name in (
                            "source_values",
                            "potential_values",
                            "convergence",
                            "nonlinear_model_image",
                            "nonlinear_normalized_residual",
                        )
                    }
            else:
                problem, strengths, meta = make_problem(case)
                atol = 1e-6
            prepared = LmfitPotentialCorrection(problem, strengths, atol, args.panel)
            for repetition in range(args.repeats + 1):
                directory = args.output / case / f"repeat_{repetition}"
                write_json(
                    args.output / f"{case}_configuration.json",
                    dict(
                        configuration=problem._configuration(),
                        strengths=list(strengths),
                        initialization=init_info,
                        source_sha256=digest(
                            problem.initial_values,
                            problem.data_active,
                            problem.noise_active,
                        ),
                    ),
                )
                print(
                    f"\nLMFIT {case} repeat {repetition} | {problem.n_joint} variables",
                    flush=True,
                )
                values, info = prepared.fit(args, directory)
                independent = inspect_endpoint(
                    problem, strengths, values, panel=args.panel
                )
                info["independent_audit"] = independent
                info["independent_stationarity_passed"] = (
                    independent["current_gn_scaled_gradient"]
                    <= info["gradient_threshold"]
                    and independent["analytical_gn_scaled_gradient"]
                    <= info["gradient_threshold"]
                )
                info["accepted_reconstruction"] = bool(
                    info["accepted_reconstruction"]
                    and info["independent_stationarity_passed"]
                )
                info["initialization"] = init_info
                if baseline is not None:
                    recovery, products = record_fields(
                        problem, strengths, values, observation, baseline, directory
                    )
                    info["recovery"] = recovery
                    info["comparison_to_original_lm"] = dict(
                        objective_difference=float(
                            info["objective"] - original_metadata["objective"]
                        ),
                        image_rms_difference_in_noise=float(
                            jnp.sqrt(
                                jnp.mean(
                                    (
                                        (
                                            products["nonlinear_model_image"]
                                            - baseline["nonlinear_model_image"]
                                        ).ravel()[problem.renderer.flat_indices]
                                        / problem.noise_active
                                    )
                                    ** 2
                                )
                            )
                        ),
                        kappa_max_abs_difference=float(
                            np.nanmax(
                                np.abs(
                                    np.asarray(products["convergence"])
                                    - baseline["convergence"]
                                )
                            )
                        ),
                    )
                    write_json(
                        directory / "configuration.json",
                        dict(
                            original_configuration=config,
                            initialization=args.initialization,
                            case=case,
                            problem=problem._configuration(),
                        ),
                    )
                else:
                    small = json.loads(
                        (
                            Path(__file__).resolve().parent
                            / "results_final"
                            / case
                            / "results.json"
                        ).read_text()
                    )
                    info["original_lm_objective"] = small["lm"]["objective"]
                write_json(directory / "diagnostics.json", info)
                records[f"{case}/{repetition}"] = info
                write_json(args.output / "summary.json", records)
                print(
                    f"RESULT {case}/{repetition}: accepted={info['accepted_reconstruction']} "
                    f"F={info['objective']:.12g}, g={info['scaled_gradient']:.5g}, "
                    f"threshold={info['gradient_threshold']:.5g}, "
                    f"total={info['timings']['total_seconds']:.3f}s",
                    flush=True,
                )
                if info["aborted"] and not info.get("external_gradient_stop", False):
                    break


if __name__ == "__main__":
    main()
