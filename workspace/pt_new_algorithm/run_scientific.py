"""Expendable full-size scientific mock experiment: AD + fixed metric + L-BFGS.

python workspace/pt_new_algorithm/run_scientific.py

Uses the original 200-pixel configuration unchanged. All optimizer derivatives
and the initial GN metric come from JAX AD; no analytical Jacobian/LM step is
used. Diagnostics may compare the existing analytical gradient and saved LM.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import scipy
from scipy.optimize import minimize

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from examples.potential_correction.mock import MockConfig, generate_mock_pair
from examples.potential_correction._recovery_diagnostics import (
    _recovery_metrics,
    _sample_truth,
    _truth_arrays,
)
from examples.potential_correction.result_products import write_json
from run_experiment import objective_for


def digest(*arrays):
    sha = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        sha.update(str((value.shape, value.dtype)).encode())
        sha.update(value.tobytes())
    return sha.hexdigest()


def ad_metric(forward, prior_action, point, *, panel, diagonal_only=False):
    """Bounded-panel J_AD.T J_AD + Q; forward is already noise weighted."""
    started = perf_counter()
    _, push = jax.linearize(forward, point)
    _, pull = jax.vjp(forward, point)
    size = len(point)

    def column(unit):
        image = push(unit)
        regularized = prior_action(unit)
        if diagonal_only:
            return jnp.vdot(image, image) + jnp.vdot(unit, regularized)
        return pull(image)[0] + regularized

    @jax.jit
    def block(indices):
        return jax.vmap(column)(jax.nn.one_hot(indices, size, dtype=point.dtype))

    result = np.empty((size,) if diagonal_only else (size, size), dtype=np.float64)
    for start in range(0, size, panel):
        indices = jnp.arange(start, start + panel, dtype=jnp.int32)
        values = np.asarray(block(indices))
        count = min(panel, size - start)
        if diagonal_only:
            result[start : start + count] = values[:count]
        else:
            result[:, start : start + count] = values[:count].T
        if start == 0 or start + panel >= size or (start // panel) % 50 == 0:
            print(
                f"  AD {'diagonal' if diagonal_only else 'metric'} {min(start+panel,size)}/{size}",
                flush=True,
            )
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite AD metric; the physical prior was not changed")
    elapsed = perf_counter() - started
    metadata = dict(
        seconds=elapsed, panel=panel, size=size, diagonal_only=diagonal_only
    )
    if not diagonal_only:
        asymmetry = float(np.max(np.abs(result - result.T)))
        metadata["relative_max_asymmetry"] = asymmetry / max(
            1.0, float(np.max(np.abs(result)))
        )
        result = (result + result.T) / 2
    return result, metadata


def optimize(problem, strengths, metric, args, output):
    objective = objective_for(problem, strengths)
    origin = problem.initial_values
    preparation = perf_counter()
    chol = jnp.linalg.cholesky(jnp.asarray(metric))
    chol.block_until_ready()
    if not np.isfinite(np.asarray(chol)).all():
        raise RuntimeError(
            "Initial metric not numerically positive definite; no hidden jitter"
        )

    def decode(z):
        return origin + jsl.solve_triangular(chol.T, z, lower=False)

    value_grad = jax.jit(jax.value_and_grad(lambda z: objective(decode(z))))
    z0 = jnp.zeros(problem.n_joint, dtype=problem.dtype)
    value_grad(z0)[0].block_until_ready()
    setup_seconds = perf_counter() - preparation
    history, last = [], {}
    started = perf_counter()
    history_file = output / "history.jsonl"

    def fun(z):
        value, grad = value_grad(jnp.asarray(z))
        f, g = float(value), np.asarray(grad, dtype=np.float64)
        last.update(z=np.array(z, copy=True), objective=f, gradient=g)
        return f, g

    def callback(z):
        if not np.array_equal(z, last["z"]):
            fun(z)
        row = dict(
            iteration=len(history) + 1,
            objective=last["objective"],
            white_gradient_inf=float(np.max(np.abs(last["gradient"]))),
            seconds=perf_counter() - started,
        )
        history.append(row)
        with history_file.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if len(history) == 1 or len(history) % 100 == 0:
            print(
                f"  L-BFGS {len(history)} F={row['objective']:.12g} g_white={row['white_gradient_inf']:.4g}",
                flush=True,
            )
            np.savez_compressed(
                output / "checkpoint.npz",
                values=np.asarray(decode(jnp.asarray(z))),
                z=z,
            )

    result = minimize(
        fun,
        np.zeros(problem.n_joint),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options=dict(
            maxiter=args.max_iter,
            maxfun=args.max_iter * 40,
            maxcor=args.memory,
            maxls=80,
            ftol=args.ftol,
            gtol=args.gtol,
        ),
    )
    values = decode(jnp.asarray(result.x))
    values.block_until_ready()
    info = dict(
        library_success=bool(result.success),
        status=int(result.status),
        termination=str(result.message),
        iterations=int(result.nit),
        nfev=int(result.nfev),
        njev=int(result.njev),
        coordinate_factorization_and_first_jit_seconds=setup_seconds,
        solve_seconds=perf_counter() - started,
        memory=args.memory,
        ftol=args.ftol,
        gtol=args.gtol,
        timing_scope="solve excludes metric preparation/factorization/first JIT and final diagnostics",
    )
    np.savez_compressed(output / "endpoint.npz", values=np.asarray(values), z=result.x)
    write_json(output / "optimizer.json", info)
    return values, info, history


def inspect_endpoint(problem, strengths, values, *, panel):
    objective = objective_for(problem, strengths)
    value, gradient = jax.jit(jax.value_and_grad(objective))(values)
    diagonal, timing = ad_metric(
        lambda x: problem.forward(x) / problem.noise_active,
        lambda x: problem.prior_matvec(x, *strengths),
        values,
        panel=panel,
        diagonal_only=True,
    )
    g = np.asarray(gradient)
    physical = float(np.max(np.abs(g) / np.sqrt(diagonal)))
    analytical = problem._evaluate_kernel(values, *strengths)
    mismatch = np.linalg.norm(g + np.asarray(analytical.negative_gradient)) / max(
        1.0, np.linalg.norm(g)
    )
    difference = g + np.asarray(analytical.negative_gradient)
    return dict(
        objective=float(value),
        current_gn_scaled_gradient=physical,
        gradient_relative_error_vs_analytical=float(mismatch),
        gradient_difference_in_gn_metric=float(
            np.max(np.abs(difference) / np.sqrt(diagonal))
        ),
        analytical_gn_scaled_gradient=float(
            np.max(np.abs(np.asarray(analytical.negative_gradient)) / np.sqrt(diagonal))
        ),
        analytical_objective_absolute_error=abs(float(value - analytical.objective)),
        endpoint_metric_timing=timing,
    )


def record_fields(problem, strengths, values, observation, baseline, directory):
    # Reuse existing geometry/rendering and recovery domains, but do not pretend
    # a failed optimizer endpoint is an accepted public scientific result.
    products = problem._result_products(values, *strengths)
    view = SimpleNamespace(**products)
    truth = _sample_truth(
        view.source_x_axis,
        view.source_y_axis,
        view.potential_grid,
        observation,
        view.dtype,
    )
    recovery, _ = _recovery_metrics(view, truth, observation, affine_diagnostics=False)
    recovery["residual_rms_noise"] = float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(
                        products["data_active"] - products["nonlinear_image_active"]
                    )
                    / np.asarray(products["noise_active"])
                )
                ** 2
            )
        )
    )
    arrays = {
        name: np.asarray(value)
        for name, value in products.items()
        if isinstance(value, (jax.Array, np.ndarray))
    }
    arrays.update(_truth_arrays(truth, view.potential_grid, view.dtype))
    for name in ("x", "y", "node_indices", "core_indices"):
        arrays["potential_" + name] = np.asarray(getattr(view.potential_grid, name))
    for name in ("image_data", "noise_map", "mask", "noise_realization", "psf_kernel"):
        arrays[name] = np.asarray(getattr(observation, name))
    arrays["truth_model_image"] = np.asarray(observation.model_image)
    arrays["initial_source"] = np.asarray(problem.initial_values[: problem.n_source])
    arrays["source_bbox"] = np.asarray(problem.renderer.source_bbox)
    arrays["joint_values"] = np.asarray(values)
    arrays["initial_joint_values"] = np.asarray(problem.initial_values)
    for name in (
        "source_values",
        "potential_values",
        "convergence",
        "nonlinear_model_image",
        "nonlinear_normalized_residual",
    ):
        arrays["lm_" + name] = np.asarray(baseline[name])
    np.savez_compressed(directory / "diagnostic_fields.npz", **arrays)
    return recovery, products


def cached_metric(forward, prior_action, point, cache, key, panel):
    path = cache / (key + ".npy")
    metadata_path = path.with_suffix(".json")
    if path.exists():
        info = json.loads(metadata_path.read_text())
        if info["key"] != key:
            raise RuntimeError("Metric cache mismatch")
        return np.load(path, allow_pickle=False), {**info, "cache_hit": True}
    matrix, info = ad_metric(forward, prior_action, point, panel=panel)
    np.save(path, matrix)
    info.update(
        key=key,
        cache_hit=False,
        ad_source="jax.linearize + jax.vjp; bounded one-hot panels",
    )
    write_json(metadata_path, info)
    return matrix, info


def observation_source(problem, strength, cache, key, panel):
    """Exact signed source-only ridge solve at zero perturbation; no truth."""
    mapping = problem.renderer.source_mapping(macro_mapping=problem.macro_mapping)
    forward = (
        lambda s: problem.renderer.source_forward(s, mapping) / problem.noise_active
    )
    prior = lambda s: strength * problem.source_regularizer.matvec_free(
        s, *problem.renderer.source_bbox
    )
    zero = jnp.zeros(problem.n_source, dtype=problem.dtype)
    matrix, info = cached_metric(forward, prior, zero, cache, key, panel)
    _, pull = jax.vjp(forward, zero)
    rhs = pull(problem.data_active / problem.noise_active)[0]
    chol = jnp.linalg.cholesky(jnp.asarray(matrix))
    source = jsl.solve_triangular(
        chol.T, jsl.solve_triangular(chol, rhs, lower=True), lower=False
    )
    source.block_until_ready()
    if not np.isfinite(np.asarray(source)).all():
        raise RuntimeError("Observation-only source solve failed")
    residual = np.asarray(jnp.asarray(matrix) @ source - rhs)
    info = {
        **info,
        "source_equation_relative_residual": float(
            np.linalg.norm(residual) / max(1.0, np.linalg.norm(np.asarray(rhs)))
        ),
        "source_truth_used": False,
        "macro_fixed_to_truth": True,
    }
    return source, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "scientific_full",
    )
    parser.add_argument(
        "--metric-cache",
        type=Path,
        default=Path(__file__).resolve().parent / "scientific_metric_cache",
    )
    parser.add_argument(
        "--initializations",
        nargs="+",
        choices=("truth", "source_only"),
        default=["truth", "source_only"],
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("subhalo", "control"),
        default=["subhalo", "control"],
    )
    parser.add_argument("--panel", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--memory", type=int, default=30)
    parser.add_argument("--gtol", type=float, default=1e-8)
    parser.add_argument("--ftol", type=float, default=1e-15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.metric_cache.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.configuration.read_text())
    write_json(args.output / "original_configuration.json", config)
    environment = dict(
        python=platform.python_version(),
        jax=jax.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
        devices=[str(d) for d in jax.devices()],
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("run_experiment.py"),
                ROOT / "examples/potential_correction/mock.py",
            )
        },
        scope="original full scientific mock, fixed true macro and physical prior; no evidence/detection calibration",
    )
    write_json(args.output / "environment.json", environment)
    shutil.copyfile(Path(__file__), args.output / "runner_snapshot.py")
    report = {}
    with jax.experimental.enable_x64(), jax.default_matmul_precision("highest"):
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
        config_key = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        for initialization in args.initializations:
            for case in args.cases:
                directory = args.output / initialization / case
                if (directory / "optimizer.json").exists() or (
                    directory / "history.jsonl"
                ).exists():
                    raise RuntimeError(
                        f"Use a fresh output to preserve the previous run: {directory}"
                    )
                directory.mkdir(parents=True, exist_ok=True)
                print(
                    f"\n{initialization}/{case}: original {mock.npix}x{mock.npix} mock",
                    flush=True,
                )
                observation = getattr(pair, case)
                strengths_record = config["fixed_strengths"][case]
                strengths = (
                    strengths_record["lambda_source"],
                    strengths_record["lambda_potential"],
                )

                def problem_with(source):
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

                init_info = dict(
                    source_truth_used=initialization == "truth",
                    macro_fixed_to_truth=True,
                )
                if initialization == "truth":
                    sx, sy = jnp.meshgrid(
                        renderer.source_x_axis, renderer.source_y_axis
                    )
                    source = observation.source_truth.brightness(sx, sy).ravel()
                else:
                    seed_problem = problem_with(
                        jnp.zeros(renderer.n_source, dtype=renderer.dtype)
                    )
                    source_key = (
                        "source-" + config_key + "-" + digest(seed_problem.noise_active)
                    )
                    source, init_info = observation_source(
                        seed_problem,
                        strengths[0],
                        args.metric_cache,
                        source_key,
                        args.panel,
                    )
                problem = problem_with(source)
                baseline_directory = args.baseline / case / "dense"
                baseline_metadata = json.loads(
                    (baseline_directory / "result.json").read_text()
                )
                if baseline_metadata["mock_configuration"] != asdict(mock):
                    raise RuntimeError("Saved LM baseline uses a different mock")
                if baseline_metadata["strengths"] != dict(
                    lambda_source=strengths[0],
                    lambda_potential=strengths[1],
                    lambda_amplitude=config["lambda_amplitude"],
                ):
                    raise RuntimeError(
                        "Saved LM baseline uses a different physical prior"
                    )
                for name, expected in (
                    ("source_regularization", config["source_regularization"]),
                    ("nsub", config["inversion_nsub"]),
                ):
                    if baseline_metadata["result_configuration"][name] != expected:
                        raise RuntimeError(f"Saved LM baseline has a different {name}")
                with np.load(
                    baseline_directory / "reconstruction.npz", allow_pickle=False
                ) as saved:
                    baseline = {
                        name: np.array(saved[name])
                        for name in (
                            "source_values",
                            "potential_values",
                            "convergence",
                            "nonlinear_model_image",
                            "nonlinear_normalized_residual",
                            "image_data",
                            "noise_map",
                            "mask",
                            "potential_node_indices",
                            "source_x_axis",
                            "source_y_axis",
                        )
                    }
                for name in ("image_data", "noise_map", "mask"):
                    np.testing.assert_allclose(
                        np.asarray(getattr(observation, name)),
                        baseline[name],
                        rtol=1e-12,
                        atol=1e-12,
                    )
                np.testing.assert_array_equal(
                    np.asarray(renderer.potential_grid.node_indices),
                    baseline["potential_node_indices"],
                )
                np.testing.assert_allclose(
                    renderer.source_x_axis, baseline["source_x_axis"], atol=1e-15
                )
                np.testing.assert_allclose(
                    renderer.source_y_axis, baseline["source_y_axis"], atol=1e-15
                )
                objective = objective_for(problem, strengths)
                value0, gradient0 = jax.jit(jax.value_and_grad(objective))(
                    problem.initial_values
                )
                key = (
                    "joint-"
                    + config_key
                    + "-"
                    + digest(
                        problem.initial_values,
                        problem.noise_active,
                        np.asarray(strengths),
                    )
                )
                metric, metric_info = cached_metric(
                    lambda x: problem.forward(x) / problem.noise_active,
                    lambda x: problem.prior_matvec(x, *strengths),
                    problem.initial_values,
                    args.metric_cache,
                    key,
                    args.panel,
                )
                initial_gradient = float(
                    np.max(np.abs(np.asarray(gradient0)) / np.sqrt(np.diag(metric)))
                )
                stopping = {
                    **config["iteration_options"],
                    **config.get("case_iteration_options", {}).get(case, {}),
                }
                threshold = (
                    stopping["gradient_atol"]
                    + stopping.get("gradient_rtol", 1e-8) * initial_gradient
                )
                run_config = dict(
                    original_configuration=config,
                    case=case,
                    initialization=initialization,
                    initialization_details=init_info,
                    problem=problem._configuration(),
                    initial_objective=float(value0),
                    initial_gn_scaled_gradient=initial_gradient,
                    physical_gradient_threshold=threshold,
                    metric=metric_info,
                    saved_lm_baseline=str(baseline_directory),
                    same_initialization_as_saved_lm=initialization == "truth",
                )
                write_json(directory / "configuration.json", run_config)
                values, optimizer, history = optimize(
                    problem, strengths, metric, args, directory
                )
                endpoint = inspect_endpoint(
                    problem, strengths, values, panel=args.panel
                )
                stationary = endpoint["current_gn_scaled_gradient"] <= threshold
                accepted = optimizer["library_success"] and stationary
                recovery, products = record_fields(
                    problem, strengths, values, observation, baseline, directory
                )
                baseline_values = jnp.asarray(
                    np.r_[baseline["source_values"], baseline["potential_values"]]
                )
                baseline_objective = float(objective(baseline_values))
                comparison = dict(
                    same_initialization=initialization == "truth",
                    saved_objective=baseline_metadata["objective"],
                    saved_fields_rerendered_objective=baseline_objective,
                    objective_difference=endpoint["objective"] - baseline_objective,
                    image_rms_difference_in_noise=float(
                        np.sqrt(
                            np.mean(
                                (
                                    (
                                        np.asarray(products["nonlinear_model_image"])
                                        - baseline["nonlinear_model_image"]
                                    )[~baseline["mask"]]
                                    / mock.noise_sigma
                                )
                                ** 2
                            )
                        )
                    ),
                    source_relative_difference=float(
                        np.linalg.norm(
                            np.asarray(products["source_values"])
                            - baseline["source_values"]
                        )
                        / np.linalg.norm(baseline["source_values"])
                    ),
                    kappa_max_abs_difference=float(
                        np.nanmax(
                            np.abs(
                                np.asarray(products["convergence"])
                                - baseline["convergence"]
                            )
                        )
                    ),
                    saved_lm_recovery=baseline_metadata["summary"],
                )
                result = dict(
                    accepted_reconstruction=accepted,
                    endpoint_role=(
                        "accepted reconstruction"
                        if accepted
                        else "unconverged endpoint diagnostics"
                    ),
                    library_success_is_not_acceptance=True,
                    physical_stationarity_passed=stationary,
                    physical_gradient_threshold=threshold,
                    optimizer=optimizer,
                    endpoint=endpoint,
                    recovery=recovery,
                    comparison_to_saved_lm=comparison,
                    no_evidence_computed=True,
                    initialization=init_info,
                )
                write_json(directory / "diagnostics.json", result)
                report[f"{initialization}/{case}"] = result
                write_json(args.output / "summary.json", report)
                print(
                    f"RESULT {initialization}/{case}: accepted={accepted} F={endpoint['objective']:.12g} "
                    f"g={endpoint['current_gn_scaled_gradient']:.5g} threshold={threshold:.5g} "
                    f"peak={recovery['convergence']['all_core']['recovered_max']:.6g} "
                    f"at {recovery['convergence_peak_xy_arcsec']}",
                    flush=True,
                )
                del metric
                jax.clear_caches()
    print(f"Results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
