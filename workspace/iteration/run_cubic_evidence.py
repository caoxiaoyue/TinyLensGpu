"""C1 J0946: fixed M3 source prior, DIRECT + two-start Powell on lens hypers.

Every evidence callback independently solves the same joint source/potential
MAP with AD L-BFGS, from the literal initial array used in the paired C1 test.
Only log10 lambda4 and log10 lambda0 change. The full coupled endpoint GN
evidence, original hard-M3 metric, original acceptance and repeat gates remain.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import pt
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import direct, minimize

from examples.potential_correction.result_products import write_json
from lbfgs import EvidenceEngine
from run_cubic_source import cubic_problem, audit_endpoint


HERE = Path(__file__).resolve().parent
PROTOCOL = "J0946_C1_fixed_M3_source_prior_DIRECT_two_start_Powell_v1"


class FitBudgetExceeded(RuntimeError):
    """A hard new-fit cap; report separately from native optimizer status."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"cubic_evidence_v1")
    parser.add_argument("--macro-dir", type=Path, default=HERE/"macro")
    parser.add_argument("--previous-run", type=Path, default=HERE/"cubic_source_v1")
    parser.add_argument("--direct-maxfun", type=int, default=120)
    parser.add_argument("--direct-fit-cap", type=int, default=160)
    parser.add_argument("--local-maxfun", type=int, default=160)
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.direct_maxfun, args.direct_fit_cap, args.local_maxfun, args.maxiter) < 1 or args.repeats < 3:
        raise ValueError("Positive budgets and at least three endpoint repeats required")
    if args.direct_fit_cap < args.direct_maxfun:
        raise ValueError("DIRECT explicit cap must allow its nominal maxfun")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory; original results are immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    started = time.perf_counter()
    hard, metadata, arrays = pt.load_problem(args.macro_dir, .1)
    old_protocol = json.loads((args.previous_run/"protocol.json").read_text())
    initial_path = args.previous_run/"initial_values.npz"
    assert pt.sha256(initial_path) == old_protocol["initial_values_archive_sha256"]
    with np.load(initial_path) as archive:
        initial = np.array(archive["joint_values"], dtype=np.float64)
        np.testing.assert_array_equal(archive["source_scale"], arrays["source_scale"])
        np.testing.assert_array_equal(archive["saved_M3_source"], arrays["source_values"])
    assert initial.shape == (hard.n_joint,) and np.all(initial[hard.n_source:] == 0)
    assert metadata["lambda_source"] == old_protocol["fixed_source_lambda_M3"]
    assert metadata["adaptive_reg_rho"] == old_protocol["fixed_rho_M3"]
    fixed_log_source = np.log10(metadata["lambda_source"])
    profiles = {k:np.asarray(v, dtype=np.float64) for k,v in old_protocol["profiles_lens_log10"].items()}
    source_names = ("run_cubic_evidence.py", "analyze_cubic_evidence.py", "pt.py", "lbfgs.py",
        "run_cubic_source.py", "source_cubic.py", "source_boundary.py", "run_fixed_boundary.py", "check_source_knots.py")
    sources = [HERE/name for name in source_names]
    sources += [pt.ROOT/"TinyLensGpu"/name for name in (
        "ForwardSimulation/LensImage/potential_correction.py", "ForwardSimulation/LensImage/pixelized_operator.py",
        "ObservationModel/LensImage/potential_correction_iterative.py",
        "ObservationModel/LensImage/_potential_correction_common.py", "utils/inversion/regularization.py",
        "utils/inversion/potential_regularization.py", "utils/lensing/potential.py")]
    inputs = [args.macro_dir/name for name in ("m3_handoff.json", "m3_handoff.npz")]
    inputs += [args.previous_run/name for name in ("protocol.json", "summary.json", "initial_values.npz")]
    protocol = dict(protocol=PROTOCOL, created_utc=datetime.now(timezone.utc).isoformat(), arguments=vars(args),
        source_hashes={str(p.relative_to(pt.ROOT)):pt.sha256(p) for p in sources},
        input_hashes={str(p.resolve()):pt.sha256(p) for p in inputs},
        source_lambda_M3=metadata["lambda_source"], source_lambda_effective=float(10.**fixed_log_source),
        rho_M3=metadata["adaptive_reg_rho"], source_scale_fixed=True, source_template_fixed=True,
        source_regularization=metadata["source_regularization"], source_pixels_frozen=False,
        interpolation="C1 cardinal bicubic with fixed zero exterior nodes, original nodal intensities",
        shared_whitening_reference="unchanged hard-bilinear M3 source / zero potential",
        initial_values="identical frozen source-only array from prior paired experiment, zero potential; no warm starts",
        outer_variables=["log10_lambda4", "log10_lambda0"], baseline_lens_log10=profiles,
        search_budget=dict(DIRECT_nominal_maxfun=args.direct_maxfun, DIRECT_explicit_unique_fit_cap=args.direct_fit_cap,
            DIRECT_maxiter=50, DIRECT_locally_biased=False, DIRECT_len_tol=.002,
            Powell_maxfev_each=args.local_maxfun, Powell_maxiter_each=12, Powell_xtol=.002, Powell_ftol=1e-8,
            Powell_starts=["best_after_DIRECT", "single_step_lens_weights"],
            local_stencil_step_log10=.02, local_stencil_points=8,
            stencil_refine_if_evidence_improvement_exceeds=.001, optional_stencil_refine_maxfev=120),
        qualification=dict(native_success=True, gn_proxy_max=.001),
        repeat_gate=dict(all_attempts_accepted=True, diagnostic_log_evidence_range_max=.01,
            maximum_pairwise_kappa_difference_max=1e-4, additional_cold_repeats=args.repeats),
        optimizer=dict(method="L-BFGS-B", maxiter=args.maxiter, maxfun=120000, maxcor=30, maxls=80,
            ftol=1e-15, gtol=1e-8),
        evidence="full coupled endpoint local-GN Gaussian approximation with unchanged normalized nodal priors",
        real_data_no_injected_truth=True, smooth_refit_performed=False,
        selection="largest qualified evidence, never selected by kappa amplitude",
        versions=dict(jax=jax.__version__, scipy=pt.scipy.__version__, numpy=np.__version__),
        devices=str(jax.devices()), configuration=hard._configuration())
    for path in sources:
        target = args.output/"source_snapshot"/path.relative_to(pt.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    np.savez_compressed(args.output/"initial_values.npz", joint_values=initial,
        source_scale=arrays["source_scale"], saved_M3_source=arrays["source_values"])
    protocol["initial_values_archive_sha256"] = pt.sha256(args.output/"initial_values.npz")
    write_json(args.output/"protocol.json", protocol)
    reference = EvidenceEngine(hard, maxiter=args.maxiter)
    engine = EvidenceEngine(cubic_problem(hard, arrays, 1.), maxiter=args.maxiter, reference_engine=reference)
    bounds = reference.bounds(metadata["lambda_source"])[1:]
    protocol["lens_log10_bounds"] = bounds
    write_json(args.output/"protocol.json", protocol)
    strengths0 = jnp.asarray(10.**np.r_[fixed_log_source, profiles["strong_anchor"]])
    warm = reference.prepare_coordinates(strengths0)
    pd, pc, vp, vectors, coefficients, _ = warm
    begin = time.perf_counter()
    engine.white_value_grad(jnp.zeros(hard.n_joint), strengths0, jnp.asarray(initial), engine.sd, engine.sc,
        engine.s0, pd, pc, vp, vectors, coefficients)[1].block_until_ready()
    engine.physical_value_grad(jnp.asarray(initial), strengths0)[1].block_until_ready()
    engine.decode(np.zeros(hard.n_joint), strengths0, warm, base=jnp.asarray(initial)).block_until_ready()
    write_json(args.output/"preparation.json", dict(reference_seconds=reference.preparation_seconds,
        cubic_seconds=engine.preparation_seconds, explicit_JIT_warmup_seconds=time.perf_counter()-begin))
    rows = []; cache = {}; best = None; outer = {}; counters = {}; repeats = []

    def checkpoint():
        write_json(args.output/"checkpoint.json", dict(protocol=PROTOCOL, rows=rows, best=best,
            outer=outer, counters=counters, elapsed_seconds=time.perf_counter()-started))

    def evaluate(lens, *, phase, repeat=None, audit=False, update_best=True, cache_allowed=True):
        nonlocal best
        lens = np.asarray(lens, dtype=np.float64)
        if lens.shape != (2,) or not np.isfinite(lens).all() or np.any(lens < bounds[:,0]-1e-10) or np.any(lens > bounds[:,1]+1e-10):
            raise ValueError("Lens coordinates must be finite and inside declared bounds")
        key = tuple(np.round(lens, 12))
        if phase in counters:
            counters[phase]["callbacks"] += 1
        if cache_allowed and key in cache:
            if phase in counters:
                counters[phase]["cache_hits"] += 1
            old = rows[cache[key]]
            return -old["log_evidence"] if old["accepted"] else 1e30
        if phase in counters:
            cap = args.direct_fit_cap if phase == "DIRECT" else (120 if phase == "Powell_stencil" else args.local_maxfun)
            if counters[phase]["new_fits"] >= cap:
                raise FitBudgetExceeded(f"{phase} reached {cap} new fits")
            counters[phase]["new_fits"] += 1
        x = np.r_[fixed_log_source, lens]
        index = len(rows); directory = f"trials/{index:04d}"
        trial = args.output/directory; trial.mkdir(parents=True)
        print(f"[fit {index:03d}] {phase} lens={lens}", flush=True)
        try:
            row, q = engine.fit(x, initial_values=initial, initialization_label="shared_frozen_previous_source_MAP", audit=audit)
            write_json(trial/"optimizer_result.json", row)
            np.savez_compressed(trial/"optimizer_state.npz", joint_values=q)
            assert np.isclose(row["strengths"]["lambda_s"], protocol["source_lambda_effective"], rtol=1e-14, atol=0)
            np.testing.assert_array_equal(engine.last_initial_values, initial)
            if audit:
                row["analytical_audit"] = audit_endpoint(q, 10.**x, args.macro_dir, "cubic", engine)
            row.update(index=index, directory=directory, phase=phase, repeat=repeat,
                source_hyperparameters_fixed=True, source_pixels_frozen=False,
                source_relative_change_from_M3=float(np.linalg.norm(q[:hard.n_source]-arrays["source_values"])/np.linalg.norm(arrays["source_values"])))
            row = pt.export_solution(trial, engine, arrays, metadata, row, q)
            exported = json.loads((trial/"result.json").read_text())
            exported.update(protocol=PROTOCOL, selection_scope=protocol["selection"],
                source_interpolation="C1_cardinal_cubic", source_lambda_fixed_M3=metadata["lambda_source"],
                rho_fixed_M3=metadata["adaptive_reg_rho"], source_pixels_frozen=False)
            write_json(trial/"result.json", exported)
        except FloatingPointError as exc:
            row = dict(index=index, directory=directory, phase=phase, repeat=repeat,
                log10_strengths=x.tolist(), accepted=False, log_evidence=None, failure=str(exc),
                source_hyperparameters_fixed=True, source_pixels_frozen=False)
        rows.append(row)
        # Repeats cannot change the score cached for an original search point.
        if cache_allowed:
            cache[key] = index
        with (args.output/"evaluations.jsonl").open("a") as handle:
            handle.write(json.dumps(pt.json_safe(row), allow_nan=False)+"\n")
        if update_best and row["accepted"] and (best is None or row["log_evidence"] > best["log_evidence"]):
            best = dict(row)
            selected = args.output/"maximum_evidence"; selected.mkdir(exist_ok=True)
            for name in ("result.json", "reconstruction.npz"):
                shutil.copy2(trial/name, selected/name)
            write_json(args.output/"best.json", best)
        checkpoint()
        print(f"[result {index:03d}] {phase} accepted={row['accepted']} E={row.get('log_evidence')} "
            f"nit={row.get('iterations')} proxy={row.get('gn_stationarity_proxy')} "
            f"peak={row.get('delta_kappa_peak')} seconds={row.get('total_seconds')}", flush=True)
        return -row["log_evidence"] if row["accepted"] else 1e30

    def search(phase, start=None, maxfun=None):
        counters[phase] = dict(callbacks=0, new_fits=0, cache_hits=0)
        begin = time.perf_counter()
        status = dict(start_log10_lens=np.asarray(start).tolist() if start is not None else None)
        try:
            if phase == "DIRECT":
                result = direct(lambda x:evaluate(x, phase=phase), bounds=bounds.tolist(),
                    maxfun=args.direct_maxfun, maxiter=50, locally_biased=False, len_tol=.002)
            else:
                result = minimize(lambda x:evaluate(x, phase=phase), np.asarray(start),
                    method="Powell", bounds=bounds,
                    options=dict(maxfev=maxfun or args.local_maxfun, maxiter=12, xtol=.002, ftol=1e-8))
            status.update(native_result=pt.optimizer_record(result), explicit_cap_interrupted=False)
        except FitBudgetExceeded as exc:
            status.update(native_result=None, explicit_cap_interrupted=True, termination=str(exc))
        status.update(counters=dict(counters[phase]), wall_seconds=time.perf_counter()-begin,
            best_visited_index=best["index"] if best else None)
        outer[phase] = status
        write_json(args.output/(phase.lower()+"_status.json"), status)
        checkpoint()

    # Re-measure all four known settings under the exact current physical model.
    for name, lens in profiles.items():
        evaluate(lens, phase="baseline_"+name, audit=True)
    search("DIRECT")
    if best is None:
        raise RuntimeError("No qualified candidate; inspect raw trace")
    search("Powell_direct_best", np.asarray(best["log10_strengths"])[1:])
    search("Powell_single_step", profiles["single_step_lens"])
    before_stencil = dict(best)
    center = np.asarray(before_stencil["log10_strengths"])[1:]
    for dx, dy in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
        lens = np.clip(center+.02*np.array([dx,dy]), bounds[:,0], bounds[:,1])
        evaluate(lens, phase="local_stencil")
    if best["log_evidence"] > before_stencil["log_evidence"]+.001:
        search("Powell_stencil", np.asarray(best["log10_strengths"])[1:], maxfun=120)
    selected = dict(best)
    center = np.asarray(selected["log10_strengths"])[1:].copy()
    # Do not promote a lucky repetition or overwrite the selected original state.
    for repeat in range(args.repeats):
        repeats.append(len(rows))
        evaluate(center, phase="selected_repeat", repeat=repeat, audit=True, update_best=False, cache_allowed=False)
    assert best["index"] == selected["index"]
    assert all(pt.sha256(Path(p)) == h for p,h in protocol["input_hashes"].items())
    assert all(pt.sha256(pt.ROOT/p) == h for p,h in protocol["source_hashes"].items())
    write_json(args.output/"summary.json", dict(protocol=PROTOCOL, rows=rows, best=selected,
        selected_repeat_indices=repeats, outer=outer, counters=counters,
        before_stencil_best_index=before_stencil["index"], input_hashes_unchanged=True,
        elapsed_seconds=time.perf_counter()-started, source_pixels_frozen=False, outer_dimensions=2))
    print(f"[complete] {len(rows)} fits in {time.perf_counter()-started:.2f}s; "
        f"best={selected['index']} E={selected['log_evidence']} peak={selected['delta_kappa_peak']}", flush=True)


if __name__ == "__main__":
    main()
