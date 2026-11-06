"""Separate stationarity, field stability and interpolation-model differences.

Saved audit: python workspace/iteration/check_practical_convergence.py
Actual fits: add --refine --output <fresh directory>. Original artifacts are
read-only. This is a J0946 experiment, not a new optimizer acceptance policy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent


def read_trial(run, row):
    with np.load(run / row["directory"] / "reconstruction.npz") as a:
        core, active = a["derivative_valid"], ~a["mask"]
        assert int(active.sum()) == 2293
        return dict(q=np.array(a["joint_values"]), source=np.array(a["source_values"]),
                    potential=np.array(a["potential_values"]), kappa=np.array(a["convergence"][core]),
                    image=np.array(a["nonlinear_source_model_image"][active]),
                    noise=np.array(a["noise_map"][active]))


def difference(a, b):
    """Norms relative to b; image RMS is per-pixel noise-normalized."""
    k = a["kappa"] - b["kappa"]
    kappa_norm = np.linalg.norm(b["kappa"])
    return dict(source_relative_L2=float(np.linalg.norm(a["source"]-b["source"])/np.linalg.norm(b["source"])),
                potential_max_abs=float(np.max(abs(a["potential"]-b["potential"]))),
                kappa_max_abs=float(np.max(abs(k))), kappa_RMS=float(np.sqrt(np.mean(k*k))),
                kappa_relative_L2=float(np.linalg.norm(k)/kappa_norm) if kappa_norm > 0 else None,
                image_noise_RMS=float(np.sqrt(np.mean(((a["image"]-b["image"])/b["noise"])**2))),
                peak_difference=float(np.max(a["kappa"])-np.max(b["kappa"])))


def saved_audit(run):
    rows = json.loads((run / "summary.json").read_text())["rows"]
    fields = {r["index"]: read_trial(run, r) for r in rows}
    groups = {}
    for profile in dict.fromkeys(r["profile"] for r in rows):
        group = [r for r in rows if r["profile"] == profile]
        pairs = []
        for repeat in sorted({r["repeat"] for r in group}):
            b, c = [next(r for r in group if r["method"] == m and r["repeat"] == repeat)
                    for m in ("bilinear", "cubic")]
            pairs.append(dict(bilinear_index=b["index"], cubic_index=c["index"],
                              **difference(fields[b["index"]], fields[c["index"]])))
        methods = {}
        for method in ("bilinear", "cubic"):
            selected = [r for r in group if r["method"] == method]
            repeat_pairs = [dict(indices=[a["index"], b["index"]],
                                 **difference(fields[a["index"]], fields[b["index"]]))
                            for a, b in itertools.combinations(selected, 2)]
            methods[method] = dict(repeat_pairs=repeat_pairs, endpoints=[dict(
                **{k: r[k] for k in ("index", "native_success", "native_message", "accepted", "iterations",
                                     "objective", "gn_stationarity_proxy", "delta_kappa_peak",
                                     "diagnostic_log_evidence", "chi2_per_pixel")},
                objective_tail_ranges={str(n): float(np.ptp(r["objective_history"][-n:]))
                                       for n in (10, 100, 1000) if len(r["objective_history"]) >= n})
                for r in selected])
        groups[profile] = dict(cross_interpolation_pairs=pairs, methods=methods)
    return dict(groups=groups, actual_new_optimization=False), rows, fields


def refine(run, output, rows, fields, maxiter):
    import pt  # Sets allocation policy and checkout import paths before JAX.
    import jax
    import jax.numpy as jnp
    import lbfgs
    from lbfgs import EvidenceEngine
    from run_fixed_boundary import zero_ring_problem
    from run_cubic_source import cubic_problem, audit_endpoint
    from examples.potential_correction.result_products import write_json

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    protocol = json.loads((run / "protocol.json").read_text())
    for path, digest in protocol["source_hashes"].items():
        assert pt.sha256(pt.ROOT / path) == digest, path
    for path, digest in protocol["input_hashes"].items():
        assert pt.sha256(Path(path)) == digest, path
    inputs = [run / "summary.json", run / "protocol.json", run / "initial_values.npz"]
    inputs += [run / r["directory"] / name for r in rows
               for name in ("result.json", "reconstruction.npz")]
    input_hashes = {str(p.resolve()): pt.sha256(p) for p in inputs}
    # Fixed in advance: the budget-limited first/worst endpoints, a cold replay,
    # and both cross-start directions. No endpoint is picked by new results.
    cases = [("bilinear_cold", "bilinear", None),
             ("bilinear_resume_18", "bilinear", 18),
             ("bilinear_resume_22", "bilinear", 22),
             ("bilinear_from_cubic_19", "bilinear", 19),
             ("cubic_from_bilinear_18", "cubic", 18),
             ("cubic_resume_19", "cubic", 19)]
    write_json(output / "protocol.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        cases=cases, maxiter=maxiter, source_run=str(run.resolve()), input_hashes=input_hashes,
        diagnostic_source_sha256=pt.sha256(Path(__file__)),
        original_optimizer=protocol["optimizer"], changed_optimizer_setting="only per-case maxiter if requested",
        interpretation="Measure stability; preserve original native/accepted flags; no new physical acceptance threshold"))
    hard, metadata, arrays = pt.load_problem()
    reference = EvidenceEngine(hard, maxiter=maxiter)
    engines = dict(bilinear=EvidenceEngine(zero_ring_problem(hard, arrays, 1.),
                                          maxiter=maxiter, reference_engine=reference),
                   cubic=EvidenceEngine(cubic_problem(hard, arrays, 1.),
                                        maxiter=maxiter, reference_engine=reference))
    row18 = next(r for r in rows if r["index"] == 18)
    logs = np.asarray(row18["log10_strengths"])
    strengths = jnp.asarray(10.**logs)
    coords = reference.prepare_coordinates(strengths)
    with np.load(run / "initial_values.npz") as a:
        cold = np.array(a["joint_values"])

    def render(engine, q):
        q = np.asarray(q)
        rendered = engine.problem.renderer.render(q[:hard.n_source], q[hard.n_source:],
                                                   macro_mapping=engine.problem.macro_mapping)
        core = np.asarray(rendered.derivative_valid)
        return dict(q=q, source=q[:hard.n_source], potential=q[hard.n_source:],
                    kappa=np.asarray(rendered.convergence)[core], image=np.asarray(rendered.image_active),
                    noise=np.asarray(engine.problem.noise_active))

    # Reproduce the roundoff-scale gradient flip without optimizing either state.
    with np.load(HERE / "m3_source_fixed_v1/trials/0037/reconstruction.npz") as a:
        q37 = np.array(a["joint_values"])
    with np.load(HERE / "m3_source_fixed_v1/trials/0038/reconstruction.npz") as a:
        q38 = np.array(a["joint_values"])
    previous = json.loads((HERE / "m3_source_fixed_v1/summary.json").read_text())
    replay_strengths = jnp.asarray([previous["best"]["strengths"][k]
                                   for k in ("lambda_s", "lambda4", "lambda0")])
    branch_replay = dict(max_joint_difference=float(np.max(abs(q38-q37))), methods={})
    for method, engine in engines.items():
        f37, g37 = engine.physical_value_grad(jnp.asarray(q37), replay_strengths)
        f38, g38 = engine.physical_value_grad(jnp.asarray(q38), replay_strengths)
        branch_replay["methods"][method] = dict(objective_abs_difference=abs(float(f38-f37)),
            gradient_L2_difference=float(jnp.linalg.norm(g38-g37)),
            fields=difference(render(engine, q38), render(engine, q37)))
    write_json(output / "branch_replay.json", branch_replay)

    # Evaluate BOTH endpoint arrays on EACH objective; cross-model F values
    # cannot be interpreted as optimization gaps.
    cross_objectives = {m: {str(i): float(e.objective(jnp.asarray(fields[i]["q"]), strengths))
                            for i in (18, 19, 20, 22)} for m, e in engines.items()}
    write_json(output / "cross_objectives.json", cross_objectives)
    results = []
    for name, method, index in cases:
        engine = engines[method]
        initial = cold if index is None else fields[index]["q"]
        initial_fields = render(engine, initial)
        snapshots = {0: np.zeros(hard.n_joint)}
        iteration = 0
        original_minimize = lbfgs.minimize

        def recording_minimize(*args, **kwargs):
            original_callback = kwargs["callback"]

            def callback(z):
                nonlocal iteration
                iteration += 1
                if iteration in (1, 10) or iteration % 50 == 0:
                    snapshots[iteration] = np.array(z, copy=True)
                original_callback(z)

            kwargs["callback"] = callback
            result = original_minimize(*args, **kwargs)
            snapshots[result.nit] = np.array(result.x, copy=True)
            return result

        print(f"[fit] {name}", flush=True)
        with patch.object(lbfgs, "minimize", recording_minimize):
            row, q = engine.fit(logs, initial_values=initial, initialization_label=name,
                                prepared_coordinates=(strengths, coords))
        assert iteration == row["iterations"]
        row["analytical_audit"] = audit_endpoint(q, np.asarray(strengths), HERE / "macro", method, engine)
        final_fields = render(engine, q)
        tail_step = max(i for i in snapshots if i <= max(0, iteration-100))
        qtail = engine.decode(snapshots[tail_step], strengths, coords, base=jnp.asarray(initial))
        tail_fields = render(engine, qtail)
        record = dict(name=name, method=method, original_index=index, optimizer=row,
                      initial_to_final=difference(final_fields, initial_fields),
                      final_vs_saved_bilinear_18=difference(final_fields, fields[18]),
                      final_vs_saved_cubic_19=difference(final_fields, fields[19]),
                      tail_start_iteration=tail_step, tail_to_final=difference(final_fields, tail_fields),
                      kappa_peak=float(np.max(final_fields["kappa"])))
        trace = []
        checkpoint_arrays = {}
        for i, z in snapshots.items():
            qi = np.asarray(engine.decode(z, strengths, coords, base=jnp.asarray(initial)))
            checkpoint_arrays[f"q_{i:04d}"] = qi
            # Use model evaluation, not a cached line-search trial value.
            trace.append(dict(iteration=i, objective=float(engine.objective(jnp.asarray(qi), strengths)),
                              final_difference=difference(render(engine, qi), final_fields)))
        trial = output / name
        trial.mkdir()
        np.savez_compressed(trial / "states.npz", initial=initial, final=q, **checkpoint_arrays)
        write_json(trial / "trace.json", trace)
        write_json(trial / "result.json", record)
        results.append(record)
        print(json.dumps(dict(name=name, iterations=row["iterations"], objective=row["objective"],
                              native_success=row["native_success"], proxy=row["gn_stationarity_proxy"],
                              peak=record["kappa_peak"], change=record["initial_to_final"])), flush=True)
    unchanged = all(pt.sha256(Path(p)) == h for p, h in input_hashes.items())
    assert unchanged
    return dict(actual_new_optimization=True, cases=results, branch_replay=branch_replay,
                cross_objectives=cross_objectives, original_artifacts_unchanged=unchanged,
                versions=dict(jax=jax.__version__, scipy=pt.scipy.__version__), devices=str(jax.devices()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "cubic_source_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument("--require-stationary", action="store_true",
                        help="Expected failure on historical bilinear endpoints; replay the original gate")
    args = parser.parse_args()
    if args.maxiter < 1:
        parser.error("--maxiter must be positive")
    if args.refine and args.output is None:
        parser.error("--refine requires --output")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise FileExistsError("Use a fresh output directory")
    audit, rows, fields = saved_audit(args.run)
    if args.refine:
        audit["refinement"] = refine(args.run, args.output, rows, fields, args.maxiter)
        audit["actual_new_optimization"] = True
    if args.output is not None:
        (args.output / "analysis.json").write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")
    for profile, group in audit["groups"].items():
        print(json.dumps(dict(profile=profile, first_pair=group["cross_interpolation_pairs"][0],
                              bilinear_endpoints=group["methods"]["bilinear"]["endpoints"])))
    if args.require_stationary:
        assert all(r["accepted"] for r in rows), "Historical bilinear stationary gate fails despite stable fields"


if __name__ == "__main__":
    main()
