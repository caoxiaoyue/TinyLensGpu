"""Ten independent noisy mocks: original JAX/whitened L-BFGS-B stability.

Seeds change simulated Gaussian noise, not the deterministic optimizer.
Each seed runs in a fresh process, with a matched zero-perturber control.
The original stopping settings are retained; old gradient-gate results and
physical/reference checks are reported separately. No selective retries.
"""

import argparse
import hashlib
import json
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

POLICY = dict(
    objective_excess_max=1e-3,
    model_difference_rms_in_noise_max=1e-3,
    source_relative_difference_max=0.005,
    kappa_difference_rms_atol=1e-5,
    kappa_difference_rms_rtol=0.01,
    residual_rms_in_noise_min=0.9,
    residual_rms_in_noise_max=1.1,
    peak_distance_grid_diagonals_max=1.0,
    control_kappa_rms_gross_max=0.01,
    matched_peak_contrast_gross_min=5.0,
    ad_analytic_gradient_gn_difference_max=1e-8,
    objective_identity_absolute_error_max=1e-6,
)
LBFGS_OPTIONS = dict(
    maxiter=3000, maxfun=120000, maxcor=30, maxls=80, gtol=1e-8, ftol=1e-15
)


def write(path, value):
    def scalar(item):
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"Unsupported JSON value {type(item).__name__}")

    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=scalar) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(args):
    import traceback
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jsl
    import numpy as np
    import scipy
    import TinyLensGpu
    from scipy.optimize import minimize
    from threadpoolctl import threadpool_info, threadpool_limits
    from benchmark_scientific import PreparedAD, sync
    from run_gauss_newton import GaussNewtonPotentialCorrection
    from run_scientific import (
        MockConfig,
        generate_mock_pair,
        SimulatorConfig,
        PotentialCorrectionRenderer,
        IterativePotentialCorrection,
        observation_source,
        inspect_endpoint,
        digest,
        _sample_truth,
        _recovery_metrics,
        _truth_arrays,
    )

    class RecordedLBFGS(PreparedAD):
        """Same numerical algorithm as PreparedAD.fit; save native state/history."""

        def fit_recorded(self, directory):
            started = perf_counter()
            t = perf_counter()
            metric = self.matrix(self.problem.initial_values)
            metric_seconds = perf_counter() - t
            t = perf_counter()
            chol = jnp.linalg.cholesky(jnp.asarray(metric))
            sync(chol)
            if not np.isfinite(np.asarray(chol)).all():
                raise RuntimeError("Initial metric Cholesky failed; no hidden ridge")
            factor_seconds = perf_counter() - t
            _, g0 = self.gradient(self.problem.initial_values)
            initial_gamma = float(
                jnp.max(jnp.abs(g0) / jnp.sqrt(jnp.diag(jnp.asarray(metric))))
            )
            threshold = self.atol + 1e-8 * initial_gamma
            zero = jnp.zeros(self.problem.n_joint, dtype=self.problem.dtype)
            sync(self.value_grad(zero, chol))
            history = []

            def fun(z):
                value, gradient = self.value_grad(jnp.asarray(z), chol)
                return float(value), np.asarray(gradient, dtype=np.float64)

            def callback(intermediate_result):
                history.append(
                    dict(
                        iteration=len(history) + 1,
                        objective=float(intermediate_result.fun),
                        seconds=perf_counter() - solve_start,
                    )
                )

            solve_start = perf_counter()
            result = minimize(
                fun,
                np.zeros(self.problem.n_joint),
                jac=True,
                method="L-BFGS-B",
                options=LBFGS_OPTIONS.copy(),
                callback=callback,
            )
            values = self.decode(jnp.asarray(result.x), chol)
            sync(values)
            solve_seconds = perf_counter() - solve_start
            t = perf_counter()
            value, gradient = self.gradient(values)
            diagonal = self.matrix(values, diagonal=True)
            gamma = float(jnp.max(jnp.abs(gradient) / jnp.sqrt(jnp.asarray(diagonal))))
            certificate_seconds = perf_counter() - t
            t = perf_counter()
            products = self.problem._result_products(values, *self.strengths)
            sync(products)
            product_seconds = perf_counter() - t
            info = dict(
                library_success=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                objective=float(value),
                iterations=int(result.nit),
                nfev=int(result.nfev),
                njev=int(result.njev),
                white_gradient_inf=float(np.max(np.abs(result.jac))),
                scaled_gradient=gamma,
                gradient_threshold=threshold,
                legacy_gradient_pass=gamma <= threshold,
                timings=dict(
                    total_seconds=perf_counter() - started,
                    metric_seconds=metric_seconds,
                    factor_seconds=factor_seconds,
                    solve_seconds=solve_seconds,
                    certificate_seconds=certificate_seconds,
                    product_seconds=product_seconds,
                ),
                options=LBFGS_OPTIONS,
            )
            np.savez_compressed(
                directory / "endpoint.npz",
                values=np.asarray(values),
                z=np.asarray(result.x),
            )
            write(directory / "optimizer.json", info)
            write(directory / "history.json", history)
            return values, products, info

    def fields(problem, strengths, values, products, observation, directory):
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
        core = np.asarray(view.potential_grid.core_indices)
        kappa = np.asarray(products["convergence"])[core]
        peak_index = core[int(np.argmax(kappa))]
        peak_xy = [
            float(np.asarray(view.potential_grid.x)[peak_index]),
            float(np.asarray(view.potential_grid.y)[peak_index]),
        ]
        rms = float(
            np.sqrt(
                np.mean(
                    np.asarray(
                        (problem.data_active - products["nonlinear_image_active"])
                        / problem.noise_active
                    )
                    ** 2
                )
            )
        )
        finite = bool(
            np.isfinite(np.asarray(values)).all()
            and np.isfinite(np.asarray(products["nonlinear_image_active"])).all()
            and np.isfinite(kappa).all()
            and np.isfinite(rms)
        )
        distance = float(
            np.hypot(
                peak_xy[0] - observation.subhalo_truth.center_x,
                peak_xy[1] - observation.subhalo_truth.center_y,
            )
        )
        sanity = dict(
            finite=finite,
            residual_rms_ok=POLICY["residual_rms_in_noise_min"]
            <= rms
            <= POLICY["residual_rms_in_noise_max"],
        )
        if observation.has_subhalo:
            sanity.update(
                positive_peak=float(np.max(kappa)) > 0,
                peak_localized=distance
                <= np.sqrt(2)
                * config["potential_spacing"]
                * POLICY["peak_distance_grid_diagonals_max"],
            )
        else:
            sanity["control_kappa_rms_gross_ok"] = (
                float(np.sqrt(np.mean(kappa * kappa)))
                <= POLICY["control_kappa_rms_gross_max"]
            )
        summary = dict(
            sanity=sanity,
            sanity_pass=all(sanity.values()),
            residual_rms_noise=rms,
            peak=float(np.max(kappa)),
            peak_xy=peak_xy,
            peak_distance=distance,
            kappa_rms=float(np.sqrt(np.mean(kappa * kappa))),
            kappa_max_abs=float(np.max(np.abs(kappa))),
            recovery=recovery,
        )
        arrays = {
            k: np.asarray(v)
            for k, v in products.items()
            if isinstance(v, (jax.Array, np.ndarray))
        }
        arrays.update(_truth_arrays(truth, view.potential_grid, view.dtype))
        for key in ("x", "y", "node_indices", "core_indices"):
            arrays["potential_" + key] = np.asarray(getattr(view.potential_grid, key))
        for key in (
            "image_data",
            "noise_map",
            "mask",
            "noise_realization",
            "psf_kernel",
        ):
            arrays[key] = np.asarray(getattr(observation, key))
        arrays.update(
            initial_joint_values=np.asarray(problem.initial_values),
            joint_values=np.asarray(values),
            source_bbox=np.asarray(problem.renderer.source_bbox),
        )
        np.savez_compressed(directory / "fields.npz", **arrays)
        write(directory / "physical.json", summary)
        return summary

    def compare(problem, products, info, reference_products, reference_info):
        core = np.asarray(products["potential_grid"].core_indices)
        kappa = np.asarray(products["convergence"])[core]
        reference_kappa = np.asarray(reference_products["convergence"])[core]
        image_delta = float(
            np.sqrt(
                np.mean(
                    np.asarray(
                        (
                            products["nonlinear_image_active"]
                            - reference_products["nonlinear_image_active"]
                        )
                        / problem.noise_active
                    )
                    ** 2
                )
            )
        )
        source_delta = float(
            np.linalg.norm(
                np.asarray(
                    products["source_values"] - reference_products["source_values"]
                )
            )
            / max(
                1e-12, np.linalg.norm(np.asarray(reference_products["source_values"]))
            )
        )
        kappa_delta = float(np.sqrt(np.mean((kappa - reference_kappa) ** 2)))
        kappa_limit = POLICY["kappa_difference_rms_atol"] + POLICY[
            "kappa_difference_rms_rtol"
        ] * float(np.sqrt(np.mean(reference_kappa**2)))
        objective_excess = info["objective"] - reference_info["objective"]
        checks = dict(
            objective_excess_ok=objective_excess <= POLICY["objective_excess_max"],
            image_difference_ok=image_delta
            <= POLICY["model_difference_rms_in_noise_max"],
            source_difference_ok=source_delta
            <= POLICY["source_relative_difference_max"],
            kappa_difference_ok=kappa_delta <= kappa_limit,
        )
        return dict(
            checks=checks,
            pass_all=all(checks.values()),
            objective_excess=objective_excess,
            image_difference_rms_noise=image_delta,
            source_relative_difference=source_delta,
            kappa_difference_rms=kappa_delta,
            kappa_difference_limit=kappa_limit,
        )

    def audit_pass(audit):
        return bool(
            audit["gradient_difference_in_gn_metric"]
            <= POLICY["ad_analytic_gradient_gn_difference_max"]
            and audit["analytical_objective_absolute_error"]
            <= POLICY["objective_identity_absolute_error_max"]
        )

    args.output.mkdir(parents=True, exist_ok=True)
    assert (
        ROOT in Path(TinyLensGpu.__file__).resolve().parents
    ), "Unexpected installed checkout"
    config = json.loads(
        (
            ROOT
            / "examples/potential_correction/iterative_scientific_configuration.json"
        ).read_text()
    )
    config["mock"]["seed"] = args.seed
    write(args.output / "configuration.json", config)
    write(
        args.output / "environment.json",
        dict(
            python=platform.python_version(),
            executable=sys.executable,
            jax=jax.__version__,
            scipy=scipy.__version__,
            numpy=np.__version__,
            gpu=[d.device_kind for d in jax.devices()],
            runner_sha256=sha(Path(__file__)),
            package_path=TinyLensGpu.__file__,
            base_configuration_sha256=sha(
                ROOT
                / "examples/potential_correction/iterative_scientific_configuration.json"
            ),
        ),
    )
    results = {}
    with jax.experimental.enable_x64(), jax.default_matmul_precision(
        "highest"
    ), threadpool_limits(limits=16):
        write(args.output / "threadpools.json", threadpool_info())
        mock = MockConfig.from_dict(config["mock"])
        pair = generate_mock_pair(mock)
        np.testing.assert_array_equal(
            pair.subhalo.noise_realization, pair.control.noise_realization
        )
        write(
            args.output / "noise.json",
            dict(
                seed=args.seed,
                sha256=digest(pair.subhalo.noise_realization),
                paired_identical=True,
            ),
        )
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
        for case in ("subhalo", "control"):
            observation = getattr(pair, case)
            directory = args.output / case
            directory.mkdir()
            strengths = (
                config["fixed_strengths"][case]["lambda_source"],
                config["fixed_strengths"][case]["lambda_potential"],
            )
            atol = (
                config.get("case_iteration_options", {})
                .get(case, {})
                .get("gradient_atol", config["iteration_options"]["gradient_atol"])
            )

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

            try:
                seed_problem = build(jnp.zeros(renderer.n_source, dtype=renderer.dtype))
                cache = directory / "source_initialization"
                cache.mkdir()
                source, initialization = observation_source(
                    seed_problem, strengths[0], cache, case, args.panel
                )
                problem = build(source)
                write(
                    directory / "inputs.json",
                    dict(
                        configuration=problem._configuration(),
                        strengths=list(strengths),
                        initialization=initialization,
                        input_sha256=digest(
                            problem.initial_values,
                            problem.data_active,
                            problem.noise_active,
                        ),
                    ),
                )
                reference_directory = directory / "reference_gn"
                reference_options = SimpleNamespace(
                    linearization="cached",
                    gate_policy="candidate",
                    candidate_factor=10.0,
                    bridge_profile=False,
                    diagonal_backend="panels",
                    native_stop=False,
                    max_seconds=args.reference_seconds,
                    ftol=1e-13,
                    xtol=1e-13,
                    gtol=1e-12,
                    max_nfev=1500,
                    lsmr_tol=1e-8,
                    lsmr_maxiter=200,
                    no_subproblem_regularize=False,
                )
                print("REFERENCE", args.seed, case, flush=True)
                reference_products = reference_info = None
                reference_usable = False
                try:
                    reference_values, reference_info = GaussNewtonPotentialCorrection(
                        problem, strengths, atol, args.panel, whitening="explicit"
                    ).fit(reference_options, reference_directory)
                    reference_audit = inspect_endpoint(
                        problem, strengths, reference_values, panel=args.panel
                    )
                    reference_products = problem._result_products(
                        reference_values, *strengths
                    )
                    sync(reference_products)
                    reference_physical = fields(
                        problem,
                        strengths,
                        reference_values,
                        reference_products,
                        observation,
                        reference_directory,
                    )
                    reference_usable = bool(
                        (
                            reference_info["library_success"]
                            or reference_info["external_gradient_stop"]
                        )
                        and reference_info["stop_reason"] != "callback_wall_budget"
                        and reference_physical["sanity_pass"]
                        and audit_pass(reference_audit)
                    )
                    write(
                        reference_directory / "diagnostics.json",
                        dict(
                            optimizer=reference_info,
                            audit=reference_audit,
                            physical=reference_physical,
                            usable_reference=reference_usable,
                        ),
                    )
                except Exception as exc:
                    reference_products = None
                    reference_directory.mkdir(exist_ok=True)
                    write(
                        reference_directory / "error.json",
                        dict(error=str(exc), traceback=traceback.format_exc()),
                    )
                    print(
                        "REFERENCE ERROR; LBFGS still runs",
                        args.seed,
                        case,
                        str(exc),
                        flush=True,
                    )
                prepared = RecordedLBFGS(problem, strengths, atol, args.panel)
                for repeat in range(args.repeats):
                    path = directory / f"repeat_{repeat}"
                    path.mkdir()
                    print("LBFGS", args.seed, case, repeat, flush=True)
                    values, products, info = prepared.fit_recorded(path)
                    audit = inspect_endpoint(
                        problem, strengths, values, panel=args.panel
                    )
                    physical = fields(
                        problem, strengths, values, products, observation, path
                    )
                    comparison = (
                        compare(
                            problem, products, info, reference_products, reference_info
                        )
                        if reference_products is not None
                        else dict(pass_all=False, reason="reference unavailable")
                    )
                    record = dict(
                        seed=args.seed,
                        case=case,
                        repeat=repeat,
                        optimizer=info,
                        independent_audit=audit,
                        physical=physical,
                        reference_usable=reference_usable,
                        comparison=comparison,
                        initialization=initialization,
                        independent_formulas_pass=audit_pass(audit),
                    )
                    results[f"{case}/{repeat}"] = record
                    write(path / "diagnostics.json", record)
                    write(args.output / "results.json", results)
                    print(
                        "ENDPOINT",
                        args.seed,
                        case,
                        repeat,
                        "native",
                        info["library_success"],
                        "sanity",
                        physical["sanity_pass"],
                        "agreement",
                        comparison["pass_all"],
                        "legacy_gradient",
                        info["legacy_gradient_pass"],
                        flush=True,
                    )
            except Exception as exc:
                error = dict(
                    seed=args.seed,
                    case=case,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
                write(directory / "error.json", error)
                results[case + "/error"] = error
                write(args.output / "results.json", results)
                print("ERROR", args.seed, case, str(exc), flush=True)
        for repeat in range(args.repeats):
            sub = results.get(f"subhalo/{repeat}")
            control = results.get(f"control/{repeat}")
            if not sub or not control:
                continue
            contrast = sub["physical"]["peak"] / max(
                1e-30, control["physical"]["kappa_max_abs"]
            )
            pair_ok = contrast >= POLICY["matched_peak_contrast_gross_min"]
            for record in (sub, control):
                record["paired_contrast"] = contrast
                record["paired_contrast_gross_ok"] = pair_ok
                record["physical_pass"] = bool(
                    record["physical"]["sanity_pass"] and pair_ok
                )
                record["overall_pass"] = bool(
                    record["optimizer"]["library_success"]
                    and record["physical_pass"]
                    and record["reference_usable"]
                    and record["comparison"]["pass_all"]
                    and record["independent_formulas_pass"]
                )
                record["inconclusive_reference"] = bool(
                    not record["reference_usable"]
                    or (
                        not record["comparison"]["pass_all"]
                        and record["comparison"].get("objective_excess", 0.0)
                        < -POLICY["objective_excess_max"]
                    )
                )
                write(
                    args.output
                    / record["case"]
                    / f"repeat_{repeat}"
                    / "diagnostics.json",
                    record,
                )
        write(args.output / "results.json", results)
    if any(not r.get("overall_pass", False) for r in results.values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE / "lbfgs_stability_10seeds")
    parser.add_argument("--reference-seconds", type=float, default=60.0)
    parser.add_argument("--panel", type=int, default=16)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.repeats < 1 or args.panel < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive repeats/panel and unique seeds required")
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        raise RuntimeError(
            "Use a fresh output directory; no previous outcomes are overwritten"
        )
    source_paths = [
        Path(__file__),
        HERE / "benchmark_scientific.py",
        HERE / "run_gauss_newton.py",
        HERE / "run_scientific.py",
        HERE / "run_experiment.py",
        ROOT / "examples/potential_correction/mock.py",
        ROOT / "examples/potential_correction/iterative_scientific_configuration.json",
        ROOT / "examples/potential_correction/_recovery_diagnostics.py",
        ROOT / "examples/potential_correction/result_products.py",
    ]
    source_paths.extend(sorted((ROOT / "TinyLensGpu").rglob("*.py")))
    snapshot = args.output / "source_snapshots"
    snapshot.mkdir()
    for path in source_paths:
        destination = snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    manifest = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    write(
        args.output / "protocol.json",
        dict(
            seeds=args.seeds,
            repeats=args.repeats,
            policy=POLICY,
            lbfgs_options=LBFGS_OPTIONS,
            source_sha256=manifest,
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            initialization="Observation-only signed source MAP; zero potential; fixed true macro",
            interpretation="Native stopping, legacy gradient, numerical reference agreement, and gross physical sanity are separate. No noise-seed selection or retries.",
            reference="Same-seed SciPy TRF/LSMR GN: cached, explicit whitening, candidate gate, lsmr1e-8, ftol/xtol1e-13, gtol1e-12",
            scope="Fixed physical system, priors and true macro; only simulated noise seed changes. Matched cases share noise and have original different potential priors; contrast is not a detection significance.",
            speed_scope="Fresh processes/data and compilation per seed; this study tests stability, not the previous warm speed.",
        ),
    )
    all_results = {}
    noise_hashes = {}
    for seed in args.seeds:
        directory = args.output / f"seed_{seed:03d}"
        directory.mkdir()
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--seed",
            str(seed),
            "--repeats",
            str(args.repeats),
            "--panel",
            str(args.panel),
            "--reference-seconds",
            str(args.reference_seconds),
            "--output",
            str(directory),
        ]
        write(directory / "command.json", command)
        print("START SEED", seed, flush=True)
        with (directory / "console.log").open("w") as log:
            process = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        if (directory / "results.json").exists():
            records = json.loads((directory / "results.json").read_text())
            all_results[str(seed)] = dict(
                returncode=process.returncode, records=records
            )
            if (directory / "noise.json").exists():
                noise_hashes[str(seed)] = json.loads(
                    (directory / "noise.json").read_text()
                )["sha256"]
            passed = sum(r.get("overall_pass", False) for r in records.values())
            print(
                "COMPLETE SEED",
                seed,
                "passed",
                passed,
                "/",
                2 * args.repeats,
                flush=True,
            )
        else:
            all_results[str(seed)] = dict(
                returncode=process.returncode, error="No result file; see console.log"
            )
            print("FAILED SEED", seed, "no numerical records", flush=True)
        assert all(
            sha(path) == manifest[str(path.relative_to(ROOT))] for path in source_paths
        ), "Source changed during experiment"
        write(
            args.output / "summary.json",
            dict(
                results=all_results,
                noise_hashes=noise_hashes,
                unique_noise_hashes=len(set(noise_hashes.values())),
            ),
        )
    rows = [
        r for seed in all_results.values() for r in seed.get("records", {}).values()
    ]
    endpoints = [r for r in rows if "optimizer" in r]
    errors = [r for r in rows if "error" in r]
    success = bool(
        len(endpoints) == len(args.seeds) * 2 * args.repeats
        and not errors
        and all(r.get("overall_pass", False) for r in rows)
        and len(set(noise_hashes.values())) == len(args.seeds)
    )
    write(
        args.output / "final.json",
        dict(
            overall_pass=success,
            expected_fits=len(args.seeds) * 2 * args.repeats,
            recorded_fits=len(endpoints),
            error_records=len(errors),
            passed=sum(r.get("overall_pass", False) for r in rows),
            native_success=sum(
                r.get("optimizer", {}).get("library_success", False) for r in rows
            ),
            legacy_gradient_pass=sum(
                r.get("optimizer", {}).get("legacy_gradient_pass", False) for r in rows
            ),
            unique_noise_hashes=len(set(noise_hashes.values())),
        ),
    )
    print("FINAL", json.loads((args.output / "final.json").read_text()), flush=True)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
