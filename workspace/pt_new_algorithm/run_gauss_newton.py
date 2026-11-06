"""Standalone JAX potential correction with SciPy trust-region Gauss--Newton.

least_squares(method='trf', tr_solver='lsmr') chooses and safeguards the steps.
JAX supplies augmented residuals and Jacobian products through LinearOperator;
no finite differences, hand-written nonlinear iterations, or dense CPU QR.
The objective and fixed initial whitening match the existing L-BFGS prototype.
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
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator
from threadpoolctl import threadpool_info, threadpool_limits

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


class GaussNewtonPotentialCorrection(PreparedAD):
    """A fixed coordinate transform plus SciPy's protected GN, with JAX AD."""

    def __init__(self, problem, strengths, atol, panel=16, whitening="solve"):
        super().__init__(problem, strengths, atol, panel)
        self.whitening = whitening
        if whitening == "explicit":
            self.decode = lambda z, transform: problem.initial_values + transform @ z
            self.value_grad = jax.jit(
                jax.value_and_grad(
                    lambda z, transform: self.objective(self.decode(z, transform))
                )
            )
        elif whitening != "solve":
            raise ValueError("Unknown whitening mode")

        def residual(z, chol, prior_chol):
            values = self.decode(z, chol)
            image = (
                problem.forward(values) - problem.data_active
            ) / problem.noise_active
            return jnp.concatenate((image, prior_chol.T @ values))

        self.residual = jax.jit(residual)

        @jax.jit
        def push(z, chol, prior_chol, direction):
            return jax.jvp(lambda q: residual(q, chol, prior_chol), (z,), (direction,))[
                1
            ]

        @jax.jit
        def pull(z, chol, prior_chol, weights):
            return jax.vjp(lambda q: residual(q, chol, prior_chol), z)[1](weights)[0]

        self.push = push
        self.pull = pull
        self.push_many = jax.jit(
            lambda z, chol, prior_chol, directions: jax.vmap(
                lambda v: push(z, chol, prior_chol, v)
            )(directions.T).T
        )
        # A jitted factory returns a Partial pytree: its arrays change with the
        # point, while the linear program and application kernels stay cached.
        self.linearization_factory = jax.jit(
            lambda z, chol, prior_chol: jax.linearize(
                lambda q: residual(q, chol, prior_chol), z
            )[1]
        )
        self.cached_push = jax.jit(lambda linear, v: linear(v))
        self.cached_pull = jax.jit(
            lambda linear, u: jax.linear_transpose(
                linear, jnp.zeros(problem.n_joint, dtype=problem.dtype)
            )(u)[0]
        )
        self.cached_push_many = jax.jit(lambda linear, v: jax.vmap(linear)(v.T).T)

        @jax.jit
        def compiled_diagonal(point, prior_diagonal):
            _, linear = jax.linearize(
                lambda values: problem.forward(values) / problem.noise_active, point
            )
            count = ((problem.n_joint + panel - 1) // panel) * panel
            groups = jnp.arange(count, dtype=jnp.int32).reshape(-1, panel)

            def block(indices):
                images = jax.vmap(linear)(
                    jax.nn.one_hot(indices, problem.n_joint, dtype=problem.dtype)
                )
                return jnp.sum(images * images, axis=1)

            diagonal = jax.lax.map(block, groups).reshape(-1)[: problem.n_joint]
            return diagonal + prior_diagonal

        self.compiled_diagonal = compiled_diagonal

    def fit(self, args, output):
        output.mkdir(parents=True, exist_ok=True)
        total_start = perf_counter()
        p = self.problem
        started = perf_counter()
        metric = self.matrix(p.initial_values)
        metric_seconds = perf_counter() - started
        started = perf_counter()
        chol = jnp.linalg.cholesky(jnp.asarray(metric))
        metric_chol = chol
        prior_matrix = p.dense_prior(*self.strengths)
        prior_diagonal = (
            jnp.diag(prior_matrix) if args.diagonal_backend == "compiled" else None
        )
        prior_chol = jnp.linalg.cholesky(prior_matrix)
        sync((chol, prior_chol))
        del prior_matrix
        explicit_whitening_seconds = 0.0
        if self.whitening == "explicit":
            transformed = perf_counter()
            chol = jsl.solve_triangular(
                metric_chol.T, jnp.eye(p.n_joint, dtype=p.dtype), lower=False
            )
            chol.block_until_ready()
            explicit_whitening_seconds = perf_counter() - transformed
        if not all(np.isfinite(np.asarray(a)).all() for a in (chol, prior_chol)):
            raise RuntimeError(
                "Physical prior or initial metric not SPD; no hidden jitter"
            )
        factor_seconds = perf_counter() - started
        zero = jnp.zeros(p.n_joint, dtype=p.dtype)
        value0, gradient0 = self.gradient(p.initial_values)
        initial_gamma = float(
            jnp.max(jnp.abs(gradient0) / jnp.sqrt(jnp.diag(jnp.asarray(metric))))
        )
        threshold = self.atol + 1e-8 * initial_gamma
        r0 = self.residual(zero, chol, prior_chol)
        np.testing.assert_allclose(
            float(0.5 * jnp.vdot(r0, r0)), float(value0), rtol=1e-10, atol=1e-7
        )
        rng = np.random.default_rng(17)
        direction = jnp.asarray(rng.normal(size=p.n_joint))
        direction = direction / jnp.linalg.norm(direction)
        weights = jnp.asarray(rng.normal(size=len(r0)))
        weights = weights / jnp.linalg.norm(weights)
        jv = self.push(zero, chol, prior_chol, direction)
        jtu = self.pull(zero, chol, prior_chol, weights)
        derivative = float(jnp.vdot(r0, jv))
        expected = float(
            jnp.vdot(
                (
                    chol.T @ gradient0
                    if self.whitening == "explicit"
                    else jsl.solve_triangular(chol, gradient0, lower=True)
                ),
                direction,
            )
        )
        adjoint_left, adjoint_right = float(jnp.vdot(weights, jv)), float(
            jnp.vdot(jtu, direction)
        )
        np.testing.assert_allclose(derivative, expected, rtol=1e-8, atol=1e-7)
        np.testing.assert_allclose(adjoint_left, adjoint_right, rtol=1e-8, atol=1e-10)
        cached_audit = {}
        if args.diagonal_backend == "compiled":
            initial_diagonal = self.compiled_diagonal(p.initial_values, prior_diagonal)
            reference_diagonal = jnp.diag(jnp.asarray(metric))
            np.testing.assert_allclose(
                initial_diagonal, reference_diagonal, rtol=1e-9, atol=1e-7
            )
            cached_audit["compiled_diagonal_initial_relative_error"] = float(
                jnp.max(
                    jnp.abs(initial_diagonal - reference_diagonal)
                    / jnp.maximum(jnp.abs(reference_diagonal), 1e-30)
                )
            )
        if self.whitening == "explicit":
            mapped = chol @ direction
            solved = jsl.solve_triangular(metric_chol.T, direction, lower=False)
            np.testing.assert_allclose(mapped, solved, rtol=1e-7, atol=1e-10)
            cached_audit.update(
                transform_vector_relative_error=float(
                    jnp.linalg.norm(mapped - solved) / jnp.linalg.norm(solved)
                ),
                transform_solve_relative_residual=float(
                    jnp.linalg.norm(metric_chol.T @ mapped - direction)
                    / jnp.linalg.norm(direction)
                ),
            )
        if args.linearization == "cached":
            linear = self.linearization_factory(zero, chol, prior_chol)
            cached_jv = self.cached_push(linear, direction)
            cached_jtu = self.cached_pull(linear, weights)
            np.testing.assert_allclose(cached_jv, jv, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(cached_jtu, jtu, rtol=1e-8, atol=1e-9)
            cached_audit.update(
                cached_jvp_max_abs_error=float(jnp.max(jnp.abs(cached_jv - jv))),
                cached_vjp_max_abs_error=float(jnp.max(jnp.abs(cached_jtu - jtu))),
            )
        identity_audit = dict(
            initial_objective=float(value0),
            augmented_half_sum_squares=float(0.5 * jnp.vdot(r0, r0)),
            directional_derivative=derivative,
            expected=expected,
            adjoint_left=adjoint_left,
            adjoint_right=adjoint_right,
            residual_count=int(len(r0)),
            variables=p.n_joint,
            **cached_audit,
        )
        setup_seconds = perf_counter() - total_start
        counts = dict(
            residual=0,
            jacobian=0,
            matvec=0,
            rmatvec=0,
            matmat=0,
            linearization=0,
            full_gate_checks=0,
            candidate_skips=0,
        )
        timings = dict(
            residual_seconds=0.0,
            matvec_seconds=0.0,
            rmatvec_seconds=0.0,
            matmat_seconds=0.0,
            linearization_seconds=0.0,
            host_to_device_seconds=0.0,
            device_call_seconds=0.0,
            device_to_host_seconds=0.0,
            callback_value_gradient_seconds=0.0,
            callback_diagonal_seconds=0.0,
            candidate_screen_seconds=0.0,
        )
        initial_diagonal = jnp.diag(jnp.asarray(metric))

        def current_diagonal(point):
            if args.diagonal_backend == "compiled":
                return self.compiled_diagonal(point, prior_diagonal)
            return self.matrix(point, diagonal=True)

        trace = []
        stop = dict(reason=None, z=None)
        external_gradient_seconds = 0.0
        solve_start = perf_counter()

        def residual(z):
            t = perf_counter()
            result = np.array(
                self.residual(jnp.asarray(z), chol, prior_chol),
                dtype=np.float64,
                copy=True,
            )
            counts["residual"] += 1
            timings["residual_seconds"] += perf_counter() - t
            return result

        def jacobian(z):
            # Each operator owns its linearization point. JIT functions receive
            # it dynamically, retaining compiled kernels across outer steps.
            point = jnp.asarray(np.array(z, dtype=np.float64, copy=True))
            counts["jacobian"] += 1
            if args.linearization == "cached":
                t = perf_counter()
                linear = self.linearization_factory(point, chol, prior_chol)
                jax.block_until_ready(linear)
                timings["linearization_seconds"] += perf_counter() - t
                counts["linearization"] += 1
                functions = (self.cached_push, self.cached_pull, self.cached_push_many)
                arguments = (linear,)
            else:
                functions = (self.push, self.pull, self.push_many)
                arguments = (point, chol, prior_chol)

            def product(kind, function, vector):
                t = perf_counter()
                device_vector = jnp.asarray(vector)
                if args.bridge_profile:
                    device_vector.block_until_ready()
                    transferred = perf_counter()
                    timings["host_to_device_seconds"] += transferred - t
                device_result = function(*arguments, device_vector)
                if args.bridge_profile:
                    device_result.block_until_ready()
                    computed = perf_counter()
                    timings["device_call_seconds"] += computed - transferred
                result = np.array(device_result, dtype=np.float64, copy=True)
                if args.bridge_profile:
                    timings["device_to_host_seconds"] += perf_counter() - computed
                counts[kind] += 1
                timings[kind + "_seconds"] += perf_counter() - t
                return result

            return LinearOperator(
                shape=(len(r0), p.n_joint),
                dtype=np.float64,
                matvec=lambda v: product(
                    "matvec", functions[0], np.asarray(v).reshape(-1)
                ),
                rmatvec=lambda u: product(
                    "rmatvec", functions[1], np.asarray(u).reshape(-1)
                ),
                matmat=lambda v: product("matmat", functions[2], v),
            )

        def callback(intermediate_result):
            nonlocal external_gradient_seconds
            z = np.array(intermediate_result.x, dtype=np.float64, copy=True)
            row = dict(
                iteration=int(intermediate_result.nit),
                nfev=int(intermediate_result.nfev),
                seconds=perf_counter() - solve_start,
                objective=float(intermediate_result.cost),
            )
            if not args.native_stop:
                checked = perf_counter()
                point = self.decode(jnp.asarray(z), chol)
                value, gradient = self.gradient(point)
                jax.block_until_ready((value, gradient))
                timings["callback_value_gradient_seconds"] += perf_counter() - checked
                screen_start = perf_counter()
                screen = (
                    float(jnp.max(jnp.abs(gradient) / jnp.sqrt(initial_diagonal)))
                    if args.gate_policy == "candidate"
                    else None
                )
                row["screen_scaled_gradient"] = screen
                timings["candidate_screen_seconds"] += perf_counter() - screen_start
                if screen is not None and screen > args.candidate_factor * threshold:
                    # Scheduling heuristic only. An unchecked point can NEVER
                    # pass; final acceptance still requires the current metric.
                    gamma = None
                    counts["candidate_skips"] += 1
                else:
                    diagonal_start = perf_counter()
                    diagonal = current_diagonal(point)
                    gamma = float(
                        jnp.max(jnp.abs(gradient) / jnp.sqrt(jnp.asarray(diagonal)))
                    )
                    timings["callback_diagonal_seconds"] += (
                        perf_counter() - diagonal_start
                    )
                    counts["full_gate_checks"] += 1
                external_gradient_seconds += perf_counter() - checked
                row["current_gn_scaled_gradient"] = gamma
                np.testing.assert_allclose(
                    float(value), row["objective"], rtol=1e-9, atol=1e-6
                )
                if (
                    gamma is not None
                    and gamma <= threshold
                    and float(value) <= float(value0)
                ):
                    stop.update(reason="physical_gradient_converged", z=z)
            row["counts"] = counts.copy()
            trace.append(row)
            if len(trace) == 1 or len(trace) % 10 == 0 or stop["reason"] is not None:
                print(
                    f"  GN iteration {row['iteration']}, F={row['objective']:.12g}, "
                    f"gamma={row.get('current_gn_scaled_gradient')}, elapsed={row['seconds']:.2f}s",
                    flush=True,
                )
            if (
                stop["reason"] is None
                and args.max_seconds
                and perf_counter() - solve_start > args.max_seconds
            ):
                stop.update(reason="callback_wall_budget", z=z)
            if stop["reason"] is not None:
                raise StopIteration

        result = least_squares(
            residual,
            np.zeros(p.n_joint, dtype=np.float64),
            jac=jacobian,
            method="trf",
            tr_solver="lsmr",
            loss="linear",
            x_scale=1.0,
            ftol=args.ftol,
            xtol=args.xtol,
            gtol=args.gtol,
            max_nfev=args.max_nfev,
            tr_options=dict(
                atol=args.lsmr_tol,
                btol=args.lsmr_tol,
                maxiter=args.lsmr_maxiter,
                regularize=not args.no_subproblem_regularize,
            ),
            callback=callback,
        )
        api_seconds = perf_counter() - solve_start
        returned_z = np.array(result.x, copy=True)
        z = jnp.asarray(returned_z if stop["z"] is None else stop["z"])
        values = self.decode(z, chol)
        sync(values)
        started = perf_counter()
        value, gradient = self.gradient(values)
        diagonal = current_diagonal(values)
        gamma = float(jnp.max(jnp.abs(gradient) / jnp.sqrt(jnp.asarray(diagonal))))
        certificate_seconds = perf_counter() - started
        started = perf_counter()
        products = p._result_products(values, *self.strengths)
        sync(products)
        product_seconds = perf_counter() - started
        total_seconds = perf_counter() - total_start
        rfinal = self.residual(z, chol, prior_chol)
        np.testing.assert_allclose(
            float(0.5 * jnp.vdot(rfinal, rfinal)), float(value), rtol=1e-9, atol=1e-6
        )
        stationary = gamma <= threshold
        info = dict(
            accepted_reconstruction=bool(
                stationary
                and (result.success or stop["reason"] == "physical_gradient_converged")
            ),
            library_success=bool(result.success),
            physical_stationarity_passed=stationary,
            aborted=bool(result.status == -2),
            stop_reason=stop["reason"],
            external_gradient_stop=stop["reason"] == "physical_gradient_converged",
            endpoint_source=(
                "native_returned_parameters"
                if stop["z"] is None
                else "owned_copy_of_stopping_callback"
            ),
            native_parameters_match_captured_stop=(
                None
                if stop["z"] is None
                else bool(np.array_equal(returned_z, stop["z"]))
            ),
            status=int(result.status),
            message=str(result.message),
            nfev=int(result.nfev),
            njev=int(result.njev),
            iterations=len(trace),
            counts=counts,
            objective=float(value),
            scipy_cost=float(result.cost),
            scaled_gradient=gamma,
            native_optimality=float(result.optimality),
            gradient_threshold=threshold,
            augmented_sum_squares=float(jnp.vdot(rfinal, rfinal)),
            kernel_cache_sizes={
                name: getattr(getattr(self, name), "_cache_size", lambda: None)()
                for name in (
                    "linearization_factory",
                    "cached_push",
                    "cached_pull",
                    "cached_push_many",
                    "push",
                    "pull",
                    "push_many",
                    "compiled_diagonal",
                )
            },
            timings=dict(
                total_seconds=total_seconds,
                setup_seconds=setup_seconds,
                metric_seconds=metric_seconds,
                factor_seconds=factor_seconds,
                explicit_whitening_seconds=explicit_whitening_seconds,
                scipy_api_seconds=api_seconds,
                certificate_seconds=certificate_seconds,
                product_seconds=product_seconds,
                external_gradient_seconds=external_gradient_seconds,
                **timings,
            ),
            options=dict(
                method="scipy.optimize.least_squares",
                algorithm="trf",
                tr_solver="lsmr",
                jac="JAX JVP/VJP LinearOperator",
                loss="linear",
                x_scale=1.0,
                ftol=args.ftol,
                xtol=args.xtol,
                gtol=args.gtol,
                max_nfev=args.max_nfev,
                lsmr_tol=args.lsmr_tol,
                lsmr_maxiter=args.lsmr_maxiter,
                subproblem_regularize=not args.no_subproblem_regularize,
                physical_gradient_stop=not args.native_stop,
                gate_policy=args.gate_policy,
                candidate_factor=args.candidate_factor,
                linearization=args.linearization,
                bridge_profile=args.bridge_profile,
                whitening_mode=self.whitening,
                diagonal_backend=args.diagonal_backend,
                extra_whitening_bytes=(
                    int(chol.size * chol.dtype.itemsize)
                    if self.whitening == "explicit"
                    else 0
                ),
                bridge_profile_scope="synchronized host phases; device call includes dispatch/wait, not pure kernel time",
                max_seconds=args.max_seconds,
                wall_budget_policy="checked after outer iterations, not a hard process timeout",
            ),
            objective_identity=identity_audit,
            scope="Augmented MAP; SciPy TRF/LSMR safeguarded Gauss-Newton; JAX GPU derivatives; no evidence",
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
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "gauss_newton_full",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=0,
        help="Additional fits with compiled kernels retained",
    )
    parser.add_argument("--panel", type=int, default=16)
    parser.add_argument("--max-nfev", type=int, default=1500)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="Solver callback wall budget; zero disables",
    )
    parser.add_argument("--ftol", type=float, default=None)
    parser.add_argument("--xtol", type=float, default=None)
    parser.add_argument("--gtol", type=float, default=1e-12)
    parser.add_argument("--blas-threads", type=int, default=16)
    parser.add_argument(
        "--native-stop",
        action="store_true",
        help="Keep native stopping only; default uses the original physical gradient callback",
    )
    parser.add_argument("--lsmr-tol", type=float, default=1e-10)
    parser.add_argument("--lsmr-maxiter", type=int, default=200)
    parser.add_argument("--no-subproblem-regularize", action="store_true")
    parser.add_argument(
        "--gate-policy", choices=("every", "candidate"), default="every"
    )
    parser.add_argument("--candidate-factor", type=float, default=10.0)
    parser.add_argument(
        "--linearization", choices=("direct", "cached"), default="direct"
    )
    parser.add_argument("--bridge-profile", action="store_true")
    parser.add_argument("--whitening", choices=("solve", "explicit"), default="solve")
    parser.add_argument(
        "--diagonal-backend", choices=("panels", "compiled"), default="panels"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.panel < 1 or args.lsmr_maxiter < 1 or args.lsmr_tol <= 0:
        parser.error("panel/lsmr-maxiter/lsmr-tol must be positive")
    if args.candidate_factor < 1:
        parser.error("candidate-factor must be at least one")
    if (args.output / "environment.json").exists():
        raise RuntimeError("Use a fresh output directory to preserve numerical records")
    shutil.copyfile(Path(__file__), args.output / "runner_snapshot.py")
    environment = dict(
        python=platform.python_version(),
        executable=sys.executable,
        jax=jax.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
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
            prepared = GaussNewtonPotentialCorrection(
                problem, strengths, atol, args.panel, whitening=args.whitening
            )
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
                    f"\nSCIPY_GN {case} repeat {repetition} | {problem.n_joint} variables",
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

    if any(not row["accepted_reconstruction"] for row in records.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
