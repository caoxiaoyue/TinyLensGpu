"""Re-score a frozen J0946 campaign after the joint/fallback QR correction.

The original sampler, likelihood trace and maps are read-only. New numerical
records live in a separate directory. Float64 checks use the same frozen
Float32 geometry/design and an independent SciPy Gaussian marginalization;
they are arithmetic controls, not a new Float64 lens fit or sampler run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_driver():
    path = HERE / "pt.py"
    spec = importlib.util.spec_from_file_location("j0946_qr_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key(point):
    return np.asarray(point, dtype=np.float32).tobytes()


def weighted_changes(driver, points, weights, old, current):
    from scipy.special import logsumexp

    finite = np.isfinite(current) & np.isfinite(old)
    positive = finite & (weights > 0)
    delta = current[positive].astype(float) - old[positive]
    w = weights[positive] / weights[positive].sum()
    logw = np.full(weights.shape, -np.inf)
    logw[positive] = np.log(weights[positive]) + delta
    adjusted = np.exp(logw - logsumexp(logw))

    def quantiles(w):
        return np.array(
            [
                driver.weighted_quantile(points[:, i], w, [0.16, 0.5, 0.84])
                for i in range(3)
            ]
        )

    report = dict(
        evaluated_points=len(points),
        finite_points=int(finite.sum()),
        invalid_original_posterior_weight=float(weights[~finite].sum()),
        weighted_mean_delta_log_evidence=float(w @ delta),
        weighted_rms_delta_log_evidence=float(np.sqrt(w @ delta**2)),
        maximum_absolute_delta_on_positive_weight=float(abs(delta).max()),
        maximum_absolute_delta_all_finite=float(
            abs(current[finite].astype(float) - old[finite]).max()
        ),
        importance_total_variation=float(0.5 * abs(adjusted - weights).sum()),
        reweighted_kish_size=float(1 / (adjusted @ adjusted)),
        original_log10_quantiles_16_50_84=quantiles(weights),
        reweighted_log10_quantiles_16_50_84=quantiles(adjusted),
        original_strength_quantiles_16_50_84=10.0 ** quantiles(weights),
        reweighted_strength_quantiles_16_50_84=10.0 ** quantiles(adjusted),
    )
    return report, adjusted


def evaluate(args):
    output = args.output.resolve()
    run, macro = args.run.resolve(), args.macro.resolve()
    if output == run:
        raise ValueError("Audit output directory must differ from the original run")

    import jax
    import jax.numpy as jnp
    from scipy import sparse

    driver = load_driver()
    jax.config.update("jax_enable_x64", False)
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((run / "manifest.json").read_text())
    identity = dict(
        driver_sha256=digest(HERE / "pt.py"),
        run=str(run),
        run_manifest_sha256=digest(run / "manifest.json"),
        posterior_sha256=digest(run / "posterior.npz"),
        trace_sha256=digest(run / "evaluations.jsonl"),
        handoff_npz_sha256=digest(macro / "m3_handoff.npz"),
        handoff_json_sha256=digest(macro / "m3_handoff.json"),
    )
    manifest_path = output / "audit_manifest.json"
    if manifest_path.exists():
        saved_inputs = json.loads(manifest_path.read_text())["inputs"]
        # Recognize the previous workspace path while still requiring every hash.
        if run.is_relative_to(HERE):
            previous_run = HERE.with_name("single_step") / run.relative_to(HERE)
            if saved_inputs["run"] == str(previous_run):
                saved_inputs["run"] = str(run)
        if not args.resume or saved_inputs != identity:
            raise ValueError("Use --resume with unchanged code and campaign inputs")
    else:
        driver.write_json(
            manifest_path,
            dict(
                created_utc=datetime.now(timezone.utc).isoformat(),
                inputs=identity,
                numerical_policy=driver.NUMERICAL_POLICY,
                original_numerical_policy=original["settings"]["numerical_policy"],
                scope="Re-evaluation of existing sampled support and evaluated candidates; no new sampler or proof of unsampled-volume coverage",
                devices=[str(device) for device in jax.devices()],
                jax_version=jax.__version__,
            ),
        )
    settings = original["settings"]
    problem, metadata, context = driver.load_problem(
        macro, settings["potential_spacing"], settings["solve_rtol"]
    )
    if driver.json_safe(metadata) != original["fixed_macro"]:
        raise ValueError("The frozen macro does not match the original campaign")
    start = time.perf_counter()
    evidence = driver.DenseEvidence(problem, formulation="data")
    evidence.source_root.block_until_ready()
    print(json.dumps(dict(preparation_seconds=time.perf_counter() - start)), flush=True)
    qr_cache = {}
    compiled_joint = evidence.joint_solve

    def joint(point):
        result = compiled_joint(point)
        qr_cache[key(point)] = np.asarray(result[0])
        return result

    evidence.joint_solve = joint
    with np.load(run / "posterior.npz") as archive:
        points = archive["log10_strengths"]
        weights = archive["weights"] / archive["weights"].sum()
        old_scores = archive["log_evidence"]
    selected = dict(
        maximum_evidence=points[int(np.argmax(old_scores))],
        posterior_median=np.array(
            [
                driver.weighted_quantile(points[:, i], weights, [0.5])[0]
                for i in range(3)
            ]
        ),
    )
    preflight = {}
    for name, point in selected.items():
        x = jnp.asarray(point, dtype=jnp.float32)
        times, rows = {}, {}
        for method, action in (
            ("default", evidence.solve),
            ("qr", evidence.joint_solve),
        ):
            t = time.perf_counter()
            rows[method] = np.asarray(action(x)[0])
            times[method + "_first_seconds"] = time.perf_counter() - t
            t = time.perf_counter()
            np.asarray(action(x)[0])
            times[method + "_warm_seconds"] = time.perf_counter() - t
        preflight[name] = dict(log10_strengths=point, metrics=rows, timing=times)
        print(
            json.dumps(driver.json_safe(dict(name=name, **preflight[name]))), flush=True
        )
    driver.write_json(output / "preflight.json", preflight)

    def rescore(label, coordinates, *, qr):
        path = output / f"{label}_rescore.npz"
        width = 6 if qr else 7
        rows = np.empty((len(coordinates), width), dtype=np.float32)
        completed = 0
        if args.resume and path.exists():
            with np.load(path) as archive:
                completed = len(archive["metrics"])
                np.testing.assert_array_equal(
                    archive["log10_strengths"], coordinates[:completed]
                )
                rows[:completed] = archive["metrics"]
        started, last = time.perf_counter(), time.perf_counter()
        for i in range(completed, len(coordinates)):
            point = coordinates[i]
            if qr and key(point) in qr_cache:
                rows[i] = qr_cache[key(point)]
            else:
                action = evidence.joint_solve if qr else evidence.solve
                rows[i] = np.asarray(action(jnp.asarray(point, dtype=jnp.float32))[0])
            if i + 1 == len(coordinates) or time.perf_counter() - last >= 20:
                np.savez_compressed(
                    path, log10_strengths=coordinates[: i + 1], metrics=rows[: i + 1]
                )
                status = dict(
                    phase=label,
                    completed=i + 1,
                    total=len(coordinates),
                    invalid=int((~np.isfinite(rows[: i + 1, 0])).sum()),
                    elapsed_seconds=time.perf_counter() - started,
                )
                driver.write_json(output / "progress.json", status)
                print(json.dumps(status), flush=True)
                last = time.perf_counter()
        return rows

    posterior_qr = rescore("posterior_qr", points, qr=True)
    posterior_default = rescore("posterior_default", points, qr=False)
    qr_report, qr_weights = weighted_changes(
        driver, points, weights, old_scores, posterior_qr[:, 0]
    )
    default_report, _ = weighted_changes(
        driver, points, weights, old_scores, posterior_default[:, 0]
    )
    driver.write_json(
        output / "posterior_impact.json", dict(qr=qr_report, default=default_report)
    )
    trace = [
        json.loads(line)
        for line in (run / "evaluations.jsonl").read_text().splitlines()
    ]
    evaluated = np.asarray([row["log10_strengths"] for row in trace])
    previous = np.asarray(
        [
            row["log_evidence"] if row["log_evidence"] is not None else -np.inf
            for row in trace
        ]
    )
    original_fallback = np.asarray([row["fallback_float32"] for row in trace])
    candidate_rows = rescore("evaluated_candidates", evaluated, qr=False)
    old_valid, new_valid = np.isfinite(previous), np.isfinite(candidate_rows[:, 0])
    rescued, lost = ~old_valid & new_valid, old_valid & ~new_valid
    common = old_valid & new_valid
    differences = candidate_rows[common, 0].astype(float) - previous[common]
    candidate_report = dict(
        evaluated_candidates=len(trace),
        original_fallback_candidates=int(original_fallback.sum()),
        corrected_fallback_candidates=int(candidate_rows[:, 6].sum()),
        original_invalid_candidates=int((~old_valid).sum()),
        corrected_invalid_candidates=int((~new_valid).sum()),
        rescued_candidates=int(rescued.sum()),
        newly_rejected_candidates=int(lost.sum()),
        maximum_retained_original_log_evidence=float(old_scores.max()),
        maximum_corrected_candidate_log_evidence=float(
            candidate_rows[new_valid, 0].max()
        ),
        maximum_rescued_log_evidence=(
            float(candidate_rows[rescued, 0].max()) if rescued.any() else None
        ),
        maximum_previously_accepted_newly_rejected_log_evidence=(
            float(previous[lost].max()) if lost.any() else None
        ),
        maximum_absolute_delta_on_common_valid_candidates=float(abs(differences).max()),
    )
    driver.write_json(output / "candidate_impact.json", candidate_report)
    selected["qr_reweighted_median"] = np.asarray(
        qr_report["reweighted_log10_quantiles_16_50_84"]
    )[:, 1]
    for label, mask, score in (
        ("highest_original_fallback", original_fallback, candidate_rows[:, 0]),
        ("highest_rescued", rescued, candidate_rows[:, 0]),
        ("highest_newly_rejected", lost, previous),
    ):
        if mask.any():
            indices = np.flatnonzero(mask)
            selected[label] = evaluated[indices[np.argmax(score[indices])]]
    if common.any():
        indices = np.flatnonzero(common)
        selected["largest_changed_candidate"] = evaluated[
            indices[np.argmax(abs(differences))]
        ]

    def qr_export(point):
        metrics, values = evidence.joint_solve(point)
        return jnp.concatenate((metrics, jnp.zeros(1, dtype=problem.dtype))), values

    qr_adapter = SimpleNamespace(solve=qr_export, formulation="joint", rhs=evidence.rhs)
    map_comparisons = {}
    for name in ("maximum_evidence", "posterior_median", "qr_reweighted_median"):
        result = driver.export_fit(
            problem, qr_adapter, selected[name], output / name, context
        )
        if name == "qr_reweighted_median":
            continue
        with np.load(run / name / "reconstruction.npz") as archive:
            old = {
                k: archive[k]
                for k in ("source_values", "potential_values", "convergence")
            }
        valid = np.asarray(result.derivative_valid)
        summary = dict(log_evidence=result.log_evidence, success=result.success)
        for field, reference in old.items():
            current = np.asarray(getattr(result, field))
            if field == "convergence":
                reference, current = reference[valid], current[valid]
            summary[field + "_relative_l2_change"] = float(
                np.linalg.norm(current.astype(float) - reference)
                / np.linalg.norm(reference)
            )
        peak = np.flatnonzero(valid)[np.argmax(np.asarray(result.convergence)[valid])]
        summary["peak_kappa"] = float(result.convergence[peak])
        summary["peak_position_arcsec"] = [
            float(result.potential_grid.x[peak]),
            float(result.potential_grid.y[peak]),
        ]
        map_comparisons[name] = summary
    driver.write_json(output / "map_impact.json", map_comparisons)
    reference_points = []
    for name, point in selected.items():
        x = jnp.asarray(point, dtype=jnp.float32)
        metrics, _ = evidence.joint_solve(x)
        reference_points.append(
            dict(
                name=name,
                log10_strengths=point,
                actual_strengths=jnp.power(jnp.float32(10), x),
                qr_metrics=metrics,
            )
        )
    driver.write_json(output / "reference_points.json", reference_points)
    source = sparse.csr_matrix(np.asarray(evidence.source))
    difference = sparse.csr_matrix(np.asarray(evidence.difference))
    np.savez_compressed(
        output / "reference_matrices.npz",
        design=np.asarray(evidence.design),
        data=np.asarray(problem.data_active),
        noise=np.asarray(problem.noise_active),
        n_source=problem.n_source,
        source_data=source.data,
        source_indices=source.indices,
        source_indptr=source.indptr,
        difference_shape=difference.shape,
        difference_data=difference.data,
        difference_indices=difference.indices,
        difference_indptr=difference.indptr,
    )
    driver.write_json(
        output / "evaluation_complete.json",
        dict(
            complete=True,
            candidate_impact=candidate_report,
            allocator_stats=jax.devices()[0].memory_stats(),
        ),
    )


def reference(args):
    import scipy.linalg as sla
    from scipy import sparse
    from scipy.sparse.linalg import splu

    driver = load_driver()
    output = args.output.resolve()
    if not (output / "evaluation_complete.json").exists():
        raise ValueError("Complete the GPU evaluation before the Float64 reference")
    with np.load(output / "reference_matrices.npz") as archive:
        ns = int(archive["n_source"])
        design = archive["design"].astype(float)
        data, noise = archive["data"].astype(float), archive["noise"].astype(float)
        source = sparse.csr_matrix(
            (
                archive["source_data"].astype(float),
                archive["source_indices"],
                archive["source_indptr"],
            ),
            shape=(ns, ns),
        )
        difference = sparse.csr_matrix(
            (
                archive["difference_data"].astype(float),
                archive["difference_indices"],
                archive["difference_indptr"],
            ),
            shape=tuple(archive["difference_shape"]),
        )
    weighted, y = design / noise[:, None], data / noise
    factor = splu(source.tocsc())
    print("Preparing independent Float64 source covariance", flush=True)
    source_covariance = weighted[:, :ns] @ factor.solve(weighted[:, :ns].T)
    roughness = (difference.T @ difference).toarray()
    noise_term = 2 * np.log(noise).sum() + len(y) * np.log(2 * np.pi)
    reports = []
    for row in json.loads((output / "reference_points.json").read_text()):
        # Reuse the actual GPU strength transform, not a fresh CPU exp10
        # whose Float32 rounding can differ across backends.
        strengths = np.asarray(row["actual_strengths"], dtype=float)
        ls, l4, l0 = strengths
        prior = l4 * roughness + l0 * np.eye(roughness.shape[0])
        chol_prior = sla.cho_factor(prior, lower=True)
        potential_covariance = weighted[:, ns:] @ sla.cho_solve(
            chol_prior, weighted[:, ns:].T
        )
        covariance = np.eye(len(y)) + source_covariance / ls + potential_covariance
        covariance = 0.5 * (covariance + covariance.T)
        chol_covariance = sla.cho_factor(covariance, lower=True)
        dual = sla.cho_solve(chol_covariance, y)
        source_values = factor.solve(weighted[:, :ns].T @ dual) / ls
        potential_values = sla.cho_solve(chol_prior, weighted[:, ns:].T @ dual)
        logdet = 2 * np.log(np.diag(chol_covariance[0])).sum()
        score = -0.5 * (y @ dual + logdet + noise_term)
        residual = y - weighted @ np.concatenate((source_values, potential_values))
        penalty = (
            ls * source_values @ (source @ source_values)
            + l4 * np.linalg.norm(difference @ potential_values) ** 2
            + l0 * (potential_values @ potential_values)
        )
        equivalent = -0.5 * (residual @ residual + penalty + logdet + noise_term)
        item = dict(
            name=row["name"],
            log10_strengths=row["log10_strengths"],
            actual_strengths=strengths,
            float64_log_evidence=score,
            float64_quadratic_identity_error=equivalent - score,
            qr_accepted=row["qr_metrics"][0] is not None,
            qr_physical_relative_residual=row["qr_metrics"][1],
            qr_minus_float64_log_evidence=(
                row["qr_metrics"][0] - score
                if row["qr_metrics"][0] is not None
                else None
            ),
            qr_ungated_log_evidence_reconstructed_from_terms=-0.5
            * (
                row["qr_metrics"][2]
                + row["qr_metrics"][3]
                + row["qr_metrics"][5]
                - row["qr_metrics"][4]
                + noise_term
            ),
        )
        path = output / row["name"] / "reconstruction.npz"
        if path.exists():
            with np.load(path) as archive:
                for name, current in (
                    ("source_values", source_values),
                    ("potential_values", potential_values),
                ):
                    item[name + "_qr_vs_float64_relative_l2"] = float(
                        np.linalg.norm(archive[name].astype(float) - current)
                        / np.linalg.norm(current)
                    )
            np.savez_compressed(
                output / row["name"] / "float64_reference.npz",
                source_values=source_values,
                potential_values=potential_values,
            )
        reports.append(item)
        driver.write_json(output / "float64_reference.json", reports)
        print(json.dumps(driver.json_safe(item)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "potential")
    parser.add_argument("--macro", type=Path, default=HERE / "macro")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("evaluate", "reference"), default="evaluate"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    (evaluate if args.phase == "evaluate" else reference)(args)


if __name__ == "__main__":
    main()
