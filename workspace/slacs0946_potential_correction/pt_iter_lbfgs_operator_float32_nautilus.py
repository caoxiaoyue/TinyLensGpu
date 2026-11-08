"""Sample J0946 strengths with Nautilus and the Float32 operator L-BFGS-B fit.

Use a uniform prior in log10 strengths and score each qualified fit by its
local Gauss–Newton evidence. Retain the highest-evidence qualified fit visited.
Use --plot-only to redraw its diagnostics from the saved output.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parents[1]))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import jax
import numpy as np
import scipy
from nautilus import Sampler

from TinyLensGpu.ObservationModel import PotentialCorrectionStrengths
from examples.potential_correction.result_products import json_safe, write_json
from workspace.slacs0946_potential_correction.pt_iter_lbfgs import (
    STRENGTH_ORDER,
    add_gn_stationarity_argument,
    log10_bounds,
    plot_saved_diagnostics,
    save_reconstruction,
)
from workspace.slacs0946_potential_correction.pt_iter_lbfgs_operator_float32 import (
    prepare_problem,
)


OUTPUT_SUFFIX = "_nautilus"
FAILED_LOG_LIKELIHOOD = -1e8


def search(problem, bounds, output, *, sampler_options, run_options, fit_options):
    """Sample the strengths, logging each fit and retaining the best qualified result."""
    lower, upper = bounds.T
    width = upper - lower
    best = best_result = None
    evaluations = 0
    failures = Counter()
    qualified_samples = set()

    def prior(unit_point):
        return lower + width * unit_point

    with (output / f"evaluations{OUTPUT_SUFFIX}.jsonl").open(
        "x", buffering=1
    ) as stream:

        def likelihood(point):
            nonlocal best, best_result, evaluations
            strengths = PotentialCorrectionStrengths(*map(float, 10.0**point))
            result = problem.fit(strengths, **fit_options)
            evidence = (
                None if result.log_evidence is None else float(result.log_evidence)
            )
            qualified = bool(
                result.success and evidence is not None and np.isfinite(evidence)
            )
            score = evidence if qualified else FAILED_LOG_LIKELIHOOD
            failure_reason = None
            if not qualified:
                failure_reason = (
                    result.failure_reason or "missing_or_nonfinite_log_evidence"
                )
                failures[failure_reason] += 1
            row = {
                "index": evaluations,
                "log10_strengths": point,
                "strengths": asdict(strengths),
                "qualified": qualified,
                "failure_reason": failure_reason,
                "objective": result.objective,
                "log_evidence": evidence,
                "log_likelihood": score,
                "optimizer_info": asdict(result.optimizer_info),
                "timings": result.configuration["timings"],
            }
            stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
            evaluations += 1
            if qualified:
                qualified_samples.add((*point, score))
                if best is None or score > best["log_evidence"]:
                    best, best_result = row, result
                    print(
                        f"[nautilus] fit {row['index']} best log evidence {score:.8f}",
                        flush=True,
                    )
            if evaluations % 10 == 0:
                qualified_count = evaluations - sum(failures.values())
                print(
                    f"[nautilus] {evaluations} fits, {qualified_count} qualified",
                    flush=True,
                )
            return score

        sampler = Sampler(
            prior, likelihood, n_dim=len(STRENGTH_ORDER), **sampler_options
        )
        complete = bool(sampler.run(**run_options))

    summary = {
        "success": best is not None,
        "sampler_converged": complete,
        "n_like": sampler.n_like,
        "sampler_n_eff": sampler.n_eff,
        "evaluations": evaluations,
        "qualified": evaluations - sum(failures.values()),
        "failure_counts": dict(failures),
    }
    if best is None:
        summary["failure_reason"] = "no_qualified_fit"
        return best, best_result, summary

    if not complete:
        # Retain the exploratory draws in a budget-limited run.
        sampler.discard_exploration = False
    points, log_weights, scores = sampler.posterior()
    # Match the fit qualification even if a valid evidence equals the penalty.
    valid = np.asarray(
        [(*point, score) in qualified_samples for point, score in zip(points, scores)],
        dtype=bool,
    ) & np.isfinite(log_weights)
    log_weights = log_weights[valid]
    weights = (
        np.exp(log_weights - log_weights.max()) if log_weights.size else log_weights
    )
    if weights.size:
        weights /= weights.sum()
    posterior = f"posterior{OUTPUT_SUFFIX}.npz"
    np.savez_compressed(
        output / posterior,
        log10_strengths=points[valid],
        weights=weights,
        log_evidence=scores[valid],
        parameter_names=STRENGTH_ORDER,
    )
    summary.update(
        posterior=posterior,
        posterior_sample_count=weights.size,
        posterior_includes_exploration=not sampler.discard_exploration,
    )
    return best, best_result, summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--macro-dir", type=Path, default=HERE / "macro")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / f"potential_lbfgs_operator_float32{OUTPUT_SUFFIX}",
        help="Directory for the Nautilus results",
    )
    parser.add_argument("--potential-spacing", type=float, default=0.1)
    parser.add_argument("--log10-lambda-s", type=float, nargs=2)
    parser.add_argument("--log10-lambda4", type=float, nargs=2)
    parser.add_argument("--log10-lambda0", type=float, nargs=2)
    parser.add_argument("--n-live", type=int, default=100, help="Nautilus live points")
    parser.add_argument(
        "--n-eff", type=int, default=500, help="Target effective sample size"
    )
    parser.add_argument(
        "--max-evals", type=int, default=3000, help="Maximum candidate fits"
    )
    parser.add_argument("--seed", type=int, default=947, help="Nautilus random seed")
    parser.add_argument(
        "--max-iter", type=int, default=3000, help="L-BFGS-B iterations per candidate"
    )
    add_gn_stationarity_argument(parser)
    parser.add_argument(
        "--plot-only", action="store_true", help="Plot a saved reconstruction"
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.plot_only:
        plot_saved_diagnostics(output, artifact_suffix=OUTPUT_SUFFIX)
        return
    if args.n_live <= len(STRENGTH_ORDER):
        parser.error("Use at least 4 live points for the three strengths")
    if min(args.n_eff, args.max_evals, args.max_iter) < 1 or args.seed < 0:
        parser.error("Use positive sample and iteration budgets and a nonnegative seed")
    macro_dir = args.macro_dir.resolve()
    if output.exists() or output.is_relative_to(macro_dir):
        parser.error("Use a new output directory outside the M3 handoff")

    sampler_options = {"n_live": args.n_live, "seed": args.seed, "n_batch": 1}
    run_options = {
        "n_eff": args.n_eff,
        "n_like_max": args.max_evals,
        "f_live": 0.01,
        "discard_exploration": True,
    }
    fit_options = {
        "backend": "operator",
        "max_iter": args.max_iter,
        "gn_stationarity_tol": args.gn_stationarity_tol,
        "pcg_max_iter": 2000,
    }
    with jax.experimental.enable_x64(False), jax.default_matmul_precision("highest"):
        problem, metadata, observation = prepare_problem(
            macro_dir, args.potential_spacing
        )
        try:
            bounds = log10_bounds(args, metadata)
        except ValueError as exc:
            parser.error(str(exc))
        with np.errstate(over="ignore", under="ignore"):
            strengths = np.asarray(10.0**bounds, dtype=np.float32)
        if not np.all(np.isfinite(strengths) & (strengths > 0)):
            parser.error("Log10 bounds must give finite positive Float32 strengths")
        output.mkdir(parents=True, exist_ok=False)
        write_json(
            output / f"manifest{OUTPUT_SUFFIX}.json",
            {
                "macro_dir": macro_dir,
                "m3_handoff_json_sha256": hashlib.sha256(
                    (macro_dir / "m3_handoff.json").read_bytes()
                ).hexdigest(),
                "fixed_macro": metadata,
                "initialization_provenance": problem.initialization_provenance,
                "potential_spacing_arcsec": args.potential_spacing,
                "native_sampling_dtype": "float32",
                "fit_dtype": problem.dtype.name,
                "optimizer_state_dtype": "float64",
                "backend": fit_options["backend"],
                "determinant_method": "principal_block_cholesky",
                "strength_order": STRENGTH_ORDER,
                "log10_bounds": bounds,
                "prior": "independent uniform in log10 strengths",
                "failed_log_likelihood": FAILED_LOG_LIKELIHOOD,
                "sampler_options": sampler_options,
                "sampler_run_options": run_options,
                "fit_options": fit_options,
                "selection": "Highest qualified local GN log evidence among evaluated samples",
                "versions": {
                    "jax": jax.__version__,
                    "scipy": scipy.__version__,
                    "numpy": np.__version__,
                    "nautilus": version("nautilus-sampler"),
                },
                "device": jax.devices()[0].device_kind,
            },
        )
        best, result, summary = search(
            problem,
            bounds,
            output,
            sampler_options=sampler_options,
            run_options=run_options,
            fit_options=fit_options,
        )

    if result is not None:
        destination = output / f"maximum_evidence{OUTPUT_SUFFIX}"
        save_reconstruction(
            destination, result, best, observation, artifact_suffix=OUTPUT_SUFFIX
        )
        point = best["log10_strengths"]
        summary["selected"] = best
        summary["distance_to_global_bounds_dex"] = dict(
            zip(STRENGTH_ORDER, np.minimum(point - bounds[:, 0], bounds[:, 1] - point))
        )
        summary["result"] = str(
            destination.relative_to(output) / f"result{OUTPUT_SUFFIX}.json"
        )
    write_json(output / f"completion{OUTPUT_SUFFIX}.json", summary)
    if result is None:
        raise SystemExit(
            f"Nautilus found no qualified fit; see evaluations{OUTPUT_SUFFIX}.jsonl"
        )
    plot_saved_diagnostics(output, artifact_suffix=OUTPUT_SUFFIX)
    print(
        json.dumps(
            {
                "strengths": best["strengths"],
                "log_evidence": best["log_evidence"],
                "sampler_converged": summary["sampler_converged"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
