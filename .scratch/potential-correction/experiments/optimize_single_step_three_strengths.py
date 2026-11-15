"""Explore three regularization strengths on the unchanged scientific mock.

This is an experiment, not a change to the fixed-amplitude public API. Each
candidate is a separate shallow copy with a declared positive amplitude prior.
Only strength-independent dense arrays and geometry are shared. All evidence
evaluations and final reconstructions use the public dense single-step fitter.
Truth is inspected only after selection, never by the search objective.
"""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np
import scipy
from scipy.optimize import direct, minimize

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from examples.potential_correction.mock import MockConfig, generate_mock_pair
from examples.potential_correction.shared.result_products import (
    export_result,
    json_safe,
    write_json,
)


NAMES = ("lambda_source", "lambda_potential", "lambda_amplitude")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Objective:
    def __init__(self, problem, path, *, resume=False):
        self.problem = problem
        self.path = path
        self.cache = {}
        self.rows = []
        self.best_x = None
        self.best_result = None
        self.phase = "initialization"
        # These arrays do not depend on any of the searched strengths.
        problem.dense_mapping().block_until_ready()
        problem.dense_prior(1.0, 1.0).block_until_ready()
        if resume and path.exists():
            self.rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.cache = {
                tuple(row["log10_strengths"]): -row["log_evidence"] if row["success"] else np.inf
                for row in self.rows
            }
            valid = [row for row in self.rows if row["success"]]
            if valid:
                best = max(valid, key=lambda row: row["log_evidence"])
                self.best_x = np.asarray(best["log10_strengths"])
                self.best_result = self.fit(self.best_x)

    def fit(self, coordinates):
        strengths = np.power(10.0, np.asarray(coordinates, dtype=float))
        candidate = copy.copy(self.problem)
        candidate.lambda_amplitude = float(strengths[2])
        candidate._operator_backends = {}
        result = candidate.fit(float(strengths[0]), float(strengths[1]), backend="dense")
        jax.block_until_ready(result.potential_values)
        assert result.strengths.lambda_amplitude == float(strengths[2])
        return result

    def __call__(self, coordinates):
        x = np.asarray(coordinates, dtype=float)
        key = tuple(x)
        if key in self.cache:
            return self.cache[key]
        start = time.perf_counter()
        result = self.fit(x)
        value = -float(result.log_evidence) if result.success else np.inf
        row = dict(
            index=len(self.rows), phase=self.phase, log10_strengths=x.tolist(),
            strengths=asdict(result.strengths), success=result.success,
            failure_reason=result.failure_reason,
            log_evidence=-value if result.success else None,
            chi2=float(result.chi2),
            regularization_penalty=float(result.regularization_penalty),
            prior_logdet=float(result.prior_logdet),
            posterior_logdet=float(result.posterior_logdet),
            relative_residual=result.solver_info.relative_residual,
            seconds=time.perf_counter() - start,
        )
        self.rows.append(row)
        self.cache[key] = value
        with self.path.open("a") as stream:
            stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
        improved = result.success and (
            self.best_result is None
            or float(result.log_evidence) > float(self.best_result.log_evidence)
        )
        if improved:
            self.best_x, self.best_result = x.copy(), result
        if improved or len(self.rows) % 25 == 0:
            print(json.dumps(json_safe(dict(
                phase=self.phase, evaluations=len(self.rows), improved=improved,
                best_log_evidence=None if self.best_result is None else float(self.best_result.log_evidence),
                best_strengths=None if self.best_result is None else asdict(self.best_result.strengths),
                last_success=result.success,
            ))), flush=True)
        return value


def optimizer_record(result):
    return dict(
        success=bool(result.success), message=str(result.message),
        status=int(result.status), nfev=int(result.nfev),
        nit=int(result.nit), log10_strengths=result.x.tolist(), objective=float(result.fun),
    )


def local_search(objective, start, bounds, budget):
    return minimize(
        objective, np.asarray(start), method="L-BFGS-B", jac="3-point", bounds=bounds,
        options=dict(maxfun=budget, maxiter=50, ftol=1e-12, gtol=1e-3,
                     finite_diff_rel_step=1e-3, maxls=12),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["subhalo", "control"], default="subhalo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-bounds", nargs=2, type=float, default=[-8, -1])
    parser.add_argument("--potential-bounds", nargs=2, type=float, default=[1, 11])
    parser.add_argument("--amplitude-bounds", nargs=2, type=float, default=[0, 8])
    parser.add_argument("--direct-maxfun", type=int, default=300)
    parser.add_argument("--fixed-maxfun", type=int, default=150)
    parser.add_argument("--local-maxfun", type=int, default=160)
    parser.add_argument("--slices", type=int, default=11)
    parser.add_argument("--resume", action="store_true", help="Reuse saved candidate scores after an interrupted export")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    previous = (json.loads((args.output / "summary.json").read_text())
                if args.resume and (args.output / "summary.json").exists() else None)
    bounds = np.asarray([args.source_bounds, args.potential_bounds, args.amplitude_bounds])
    if not np.all(np.isfinite(bounds)) or not np.all(bounds[:, 1] > bounds[:, 0]):
        parser.error("Bounds must be finite increasing log10 intervals")
    config_path = ROOT / "examples/potential_correction/mock/single_step/configurations/scientific.json"
    config = json.loads(config_path.read_text())
    config["three_strength_experiment"] = vars(args)
    mock = MockConfig.from_dict(config["mock"])
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              check=True, capture_output=True, text=True).stdout.strip()
    record = dict(
        scope="Single-step dense exact conditional linear Gaussian evidence; positive strengths; fixed true macro and frozen analytic source gradient",
        selection_rule="Maximize log evidence only; no truth-based filter or recovery selection",
        caveat="An empirical-Bayes point search, not hyperparameter posterior sampling or a proof of global optimality",
        configuration=config, log10_bounds=bounds.tolist(),
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in [
            Path(__file__), config_path,
            ROOT / "TinyLensGpu/ObservationModel/LensImage/potential_correction.py",
            ROOT / "TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py",
            ROOT / "TinyLensGpu/utils/inversion/potential_regularization.py",
        ]},
        git_commit=revision,
        software=dict(python=platform.python_version(), jax=jax.__version__,
                      numpy=np.__version__, scipy=scipy.__version__, device=str(jax.devices()[0])),
        searches={}, exports={},
    )
    if previous is not None:
        record["resumed_from"] = dict(
            source_sha256=previous["source_sha256"],
            evaluation_count=previous["evaluation_count"],
            elapsed_seconds_at_checkpoint=previous["elapsed_seconds"],
        )
    assert jax.devices()[0].platform == "gpu", "This scientific run requires the GPU environment"
    start = time.perf_counter()
    with jax.experimental.enable_x64():
        pair = generate_mock_pair(mock)
        observation = getattr(pair, args.case)
        renderer = PotentialCorrectionRenderer(
            pair.control.macro,
            SimulatorConfig(mock.dpix, mock.npix, pair.control.psf_kernel,
                            config["inversion_nsub"], pair.control.mask),
            config["source_n"], tuple(config["source_bbox"]),
            potential_spacing=config["potential_spacing"], dtype="float64",
        )
        problem = SingleStepPotentialCorrection(
            renderer, observation.image_data, observation.noise_map,
            observation.source_truth.gradient, lambda_amplitude=config["lambda_amplitude"],
            solve_rtol=config["solve_rtol"],
        )
        record["counts"] = dict(data=int(problem.data_active.size), source=problem.n_source,
                                potential=problem.n_potential, core=renderer.potential_grid.n_core)
        objective = Objective(problem, args.output / "evaluations.jsonl", resume=args.resume)
        original = config["fixed_strengths"][args.case]
        original_x = np.log10([original["lambda_source"], original["lambda_potential"],
                              config["lambda_amplitude"]])
        objective.phase = "original_baseline"
        objective(original_x)
        baseline = objective.fit(original_x)
        assert baseline.success
        reference = (ROOT / ".scratch/potential-correction/output/single-step-scientific/reference"
                     / args.case / "dense")
        metadata = json.loads((reference / "result.json").read_text())
        with np.load(reference / "reconstruction.npz", allow_pickle=False) as saved:
            for name in ("image_data", "noise_map", "mask", "psf_kernel"):
                np.testing.assert_array_equal(np.asarray(getattr(observation, name)), saved[name])
            errors = {}
            for name in ("source_values", "potential_values", "convergence"):
                actual = np.asarray(getattr(baseline, name))
                np.testing.assert_allclose(actual, saved[name], rtol=1e-7, atol=1e-8)
                errors[name] = float(np.nanmax(np.abs(actual - saved[name])))
        evidence_error = abs(float(baseline.log_evidence) - metadata["log_evidence"])
        assert evidence_error < 1e-5
        record["baseline_reproduction"] = dict(
            reference=str(reference.relative_to(ROOT)), log_evidence_error=evidence_error,
            field_max_errors=errors, inputs_identical=True,
        )
        # Independently check that the experimental sharing does not freeze lambda0.
        check_x = original_x + [0, 0, np.log10(3)]
        shared = objective.fit(check_x)
        fresh = SingleStepPotentialCorrection(
            renderer, observation.image_data, observation.noise_map,
            observation.source_truth.gradient, lambda_amplitude=float(10**check_x[2]),
            solve_rtol=config["solve_rtol"],
        )
        # Reusing the response is safe: it depends on geometry/gradient, not priors.
        fresh._dense_design = problem._dense_design
        independent = fresh.fit(float(10**check_x[0]), float(10**check_x[1]), backend="dense")
        assert shared.success and independent.success
        np.testing.assert_allclose(shared.potential_values, independent.potential_values,
                                   rtol=1e-9, atol=1e-10)
        assert abs(float(shared.log_evidence - independent.log_evidence)) < 1e-6
        record["amplitude_adapter_check"] = dict(
            fresh_constructor_matches=True, lambda_amplitude=shared.strengths.lambda_amplitude,
            log_evidence_error=float(abs(shared.log_evidence - independent.log_evidence)),
        )
        del fresh, independent, shared

        def checkpoint():
            record["evaluation_count"] = len(objective.rows)
            record["failed_evaluation_count"] = sum(not row["success"] for row in objective.rows)
            record["elapsed_seconds"] = time.perf_counter() - start
            record["candidate_fit_seconds"] = sum(row["seconds"] for row in objective.rows)
            write_json(args.output / "summary.json", record)

        checkpoint()
        objective.phase = "joint_global_direct"
        joint = direct(objective, bounds.tolist(), maxfun=args.direct_maxfun, maxiter=1000,
                       eps=1e-5, len_tol=1e-4, vol_tol=1e-16, locally_biased=True)
        record["searches"][objective.phase] = optimizer_record(joint)
        checkpoint()
        starts = [objective.best_x.copy(), original_x]
        for index, initial in enumerate(starts):
            objective.phase = f"joint_local_{index}"
            fitted = local_search(objective, initial, bounds.tolist(), args.local_maxfun)
            record["searches"][objective.phase] = optimizer_record(fitted)
            checkpoint()

        # A fair incremental comparison also reoptimizes source/D4 at fixed lambda0.
        amplitude = original_x[2]
        fixed_rows = []

        def fixed_objective(x):
            full = np.r_[x, amplitude]
            value = objective(full)
            fixed_rows.append((value, full))
            return value

        objective.phase = "fixed_amplitude_global_direct"
        fixed = direct(fixed_objective, bounds[:2].tolist(), maxfun=args.fixed_maxfun,
                       maxiter=1000, eps=1e-5, len_tol=1e-4, vol_tol=1e-16)
        record["searches"][objective.phase] = optimizer_record(fixed)
        objective.phase = "fixed_amplitude_local"
        fixed_local = local_search(fixed_objective, fixed.x, bounds[:2].tolist(), args.local_maxfun)
        record["searches"][objective.phase] = optimizer_record(fixed_local)
        fixed_x = min(fixed_rows, key=lambda item: item[0])[1]
        fixed_result = objective.fit(fixed_x)
        objective.phase = "joint_local_from_fixed_optimum"
        fitted = local_search(objective, fixed_x, bounds.tolist(), args.local_maxfun)
        record["searches"][objective.phase] = optimizer_record(fitted)
        checkpoint()

        # Conditional slices expose trends and boundaries; these are not profiles.
        objective.phase = "conditional_slices"
        center = objective.best_x.copy()
        slices = {}
        for dimension, name in enumerate(NAMES):
            values = np.unique(np.r_[np.linspace(*bounds[dimension], args.slices), center[dimension]])
            slices[name] = []
            for value in values:
                coordinates = center.copy()
                coordinates[dimension] = value
                score = objective(coordinates)
                slices[name].append(dict(log10_strength=float(value),
                                         log_evidence=-score if np.isfinite(score) else None))
        record["conditional_slices"] = dict(
            center_log10_strengths=center.tolist(), other_strengths="fixed, not profiled", curves=slices,
        )
        if np.linalg.norm(objective.best_x - center) > 0.01:
            objective.phase = "joint_local_after_slices"
            fitted = local_search(objective, objective.best_x.copy(), bounds.tolist(), args.local_maxfun)
            record["searches"][objective.phase] = optimizer_record(fitted)

        best_x = objective.best_x.copy()
        best = objective.fit(best_x)
        repeated = objective.fit(best_x)
        record["repeat_best_log_evidence_difference"] = float(repeated.log_evidence - best.log_evidence)
        distance = np.minimum(best_x - bounds[:, 0], bounds[:, 1] - best_x)
        record["selected"] = dict(
            log10_strengths=best_x.tolist(), strengths=asdict(best.strengths),
            log_evidence=float(best.log_evidence), distance_to_nearest_bound_dex=distance.tolist(),
            within_0_1_dex_of_bound=(distance < 0.1).tolist(),
            delta_log_evidence_vs_original=float(best.log_evidence - baseline.log_evidence),
            delta_log_evidence_vs_reoptimized_fixed_amplitude=float(best.log_evidence - fixed_result.log_evidence),
        )
        checkpoint()
        # Truth-dependent diagnostics begin here, after all selections are frozen.
        for name, result in (("original", baseline), ("fixed_amplitude_optimized", fixed_result),
                             ("joint_optimized", best)):
            export_config = copy.deepcopy(config)
            export_config.update(lambda_amplitude=result.strengths.lambda_amplitude,
                                 mode="three_strength_experiment", backends=["dense"], make_plots=False)
            export_config["fixed_strengths"][args.case] = {
                key: getattr(result.strengths, key) for key in NAMES[:2]
            }
            export_config["calibration_provenance"] = (
                "Original documented hand-selected strengths, reproduced on unchanged data."
                if name == "original" else
                "Selected by this experiment's full dense conditional Gaussian evidence search; "
                + ("lambda0 fixed at 10000." if name == "fixed_amplitude_optimized"
                   else "source, D4, and amplitude strengths jointly searched.")
            )
            metadata = export_result(result, observation, args.output / name,
                                     configuration=export_config, make_plots=False, affine_diagnostics=True)
            valid = np.asarray(result.derivative_valid)
            with np.load(args.output / name / "reconstruction.npz") as arrays:
                truth = arrays["truth_discrete_convergence"][valid]
                peak = int(np.argmax(truth))
                recovered = arrays["convergence"][valid]
                local = arrays["local_core"][valid]
                background = float(np.sqrt(np.mean(recovered[~local]**2)))
            record["exports"][name] = dict(
                path=str((args.output / name).relative_to(ROOT)),
                strengths=asdict(result.strengths), log_evidence=float(result.log_evidence),
                chi2=float(result.chi2), regularization_penalty=float(result.regularization_penalty),
                prior_logdet=float(result.prior_logdet), posterior_logdet=float(result.posterior_logdet),
                solver_info=asdict(result.solver_info), summary=metadata["summary"],
                kappa_background_rms=background,
                recovery_at_truth_peak=None if not truth[peak] else float(recovered[peak] / truth[peak]),
                truth_discrete_peak=float(truth[peak]), recovered_at_truth_peak=float(recovered[peak]),
            )
        checkpoint()
        print(json.dumps(json_safe(dict(completed=True, case=args.case,
                                       selected=record["selected"], evaluations=len(objective.rows),
                                       elapsed_seconds=record["elapsed_seconds"]))), flush=True)


if __name__ == "__main__":
    main()
