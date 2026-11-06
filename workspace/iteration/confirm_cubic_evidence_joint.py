"""Additional near-optimum Powell confirmation; never modify the parent run.

The parent second start reached its outer maxfev despite matching the first
solution. This diagnostic starts a fresh Powell search at the retained point.
The original candidate/repeat certification is preserved if improvement<=.001,
the same material-change threshold declared for the parent stencil test.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pt
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from examples.potential_correction.result_products import write_json
from lbfgs import EvidenceEngine
from run_cubic_source import cubic_problem, audit_endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxfun", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    started = time.perf_counter()
    protocol = json.loads((args.parent/"protocol.json").read_text())
    summary = json.loads((args.parent/"summary.json").read_text())
    original = summary["best"]
    parent_files = [args.parent/name for name in ("protocol.json", "summary.json", "initial_values.npz")]
    parent_files += [args.parent/"maximum_evidence"/name for name in ("result.json", "reconstruction.npz", "validation.json")]
    parent_hashes = {str(p.resolve()):pt.sha256(p) for p in parent_files}
    assert all(pt.sha256(pt.ROOT/p) == h for p,h in protocol["source_hashes"].items())
    source_files = [Path(__file__)]+[pt.ROOT/p for p in protocol["source_hashes"]]
    macro = Path(protocol["arguments"]["macro_dir"])
    hard, metadata, arrays = pt.load_problem(macro, .1)
    with np.load(args.parent/"initial_values.npz") as a:
        initial = np.array(a["joint_values"])
        np.testing.assert_array_equal(a["source_scale"], arrays["source_scale"])
    bounds = np.asarray(protocol["hyper_log10_bounds"])
    start = np.asarray(original["log10_strengths"])
    before = dict(parent_hashes=parent_hashes, source_hashes={str(p):pt.sha256(p) for p in source_files},
        created_utc=datetime.now(timezone.utc).isoformat(), parent=str(args.parent.resolve()),
        purpose="separate near-optimum native-stop confirmation; no parent state replacement",
        start_log10_hypers=start, fixed_rho=metadata["adaptive_reg_rho"], source_scale_fixed=True,
        source_lambda_optimized=True, initial_values="same literal parent array and zero potential",
        options=dict(maxfev=args.maxfun, maxiter=12, xtol=.002, ftol=1e-8), material_improvement_threshold=.001)
    write_json(args.output/"protocol.json", before)
    reference = EvidenceEngine(hard, maxiter=protocol["optimizer"]["maxiter"])
    engine = EvidenceEngine(cubic_problem(hard, arrays, 1.), maxiter=protocol["optimizer"]["maxiter"], reference_engine=reference)
    with np.load(args.parent/original["directory"]/"reconstruction.npz") as a:
        audit = audit_endpoint(np.array(a["joint_values"]), 10.**start, macro, "cubic", engine)
    rows = []; cache = {tuple(np.round(start,12)):-original["log_evidence"]}
    cache_hits = 0

    def evaluate(x):
        nonlocal cache_hits
        x = np.asarray(x, dtype=np.float64); key = tuple(np.round(x,12))
        if key in cache:
            cache_hits += 1
            return cache[key]
        assert x.shape == (3,) and np.all(x >= bounds[:,0]-1e-10) and np.all(x <= bounds[:,1]+1e-10)
        directory = f"trials/{len(rows):04d}"; target = args.output/directory
        row,q = engine.fit(x, initial_values=initial, initialization_label="same_parent_frozen_initial_array")
        row.update(index=len(rows), directory=directory)
        row = pt.export_solution(target,engine,arrays,metadata,row,q)
        rows.append(row)
        with (args.output/"evaluations.jsonl").open("a") as f:
            f.write(json.dumps(pt.json_safe(row),allow_nan=False)+"\n")
        score = -row["log_evidence"] if row["accepted"] else 1e30
        cache[key] = score
        print(f"[confirm {row['index']:03d}] E={row['log_evidence']} peak={row['delta_kappa_peak']} accepted={row['accepted']}",flush=True)
        return score

    result = minimize(evaluate,start,method="Powell",bounds=bounds,
        options=dict(maxfev=args.maxfun,maxiter=12,xtol=.002,ftol=1e-8))
    accepted = [r for r in rows if r["accepted"]]
    best = max(accepted,key=lambda r:r["log_evidence"]) if accepted else None
    gain = max(0.,best["log_evidence"]-original["log_evidence"]) if best else 0.
    final_score = cache[tuple(np.round(result.x,12))]
    confirmed = bool(result.success and final_score < 1e29 and gain <= .001)
    assert all(pt.sha256(Path(p)) == h for p,h in parent_hashes.items())
    assert all(pt.sha256(Path(p)) == h for p,h in before["source_hashes"].items())
    write_json(args.output/"summary.json",dict(native_result=pt.optimizer_record(result),rows=rows,best=best,
        cache_hits=cache_hits,material_evidence_improvement=gain,confirmed=confirmed,
        parent_original_preserved=True,parent_best_index=original["index"],parent_analytical_audit=audit,
        elapsed_seconds=time.perf_counter()-started))
    print(f"[confirmation complete] success={result.success} confirmed={confirmed} gain={gain}",flush=True)


if __name__ == "__main__":
    main()
