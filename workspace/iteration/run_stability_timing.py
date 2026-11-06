"""Paired J0946 timing: original L-BFGS stop versus objective/field stability.

Uses immutable previous inputs, identical shared coordinates and cold initial
arrays. Does not change historical engines, interpolation, priors or results.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import pt
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from examples.potential_correction.result_products import write_json
from lbfgs import EvidenceEngine, low_rank_whiten, prior_response, scaled_cholesky, stationarity_proxy
from run_cubic_source import cubic_problem, audit_endpoint
from run_fixed_boundary import zero_ring_problem
from check_practical_convergence import difference
from stability_stop import StabilityPolicy, minimize_with_stability


HERE = Path(__file__).resolve().parent
POLICIES = dict(loose=StabilityPolicy(objective_rtol=1e-6, solution_rtol=1e-2),
                standard=StabilityPolicy(),
                tight=StabilityPolicy(objective_rtol=1e-8, solution_rtol=1e-4))
OPTIONS = dict(maxiter=3000, maxfun=120000, maxcor=30, maxls=80, ftol=1e-15, gtol=1e-8)


def field_sampler(engine):
    grid = engine.problem.renderer.potential_grid

    @jax.jit
    def sample(z, strengths, base, sd, sc, s0, pd, pc, vp, vectors, coefficients):
        q = base + low_rank_whiten(z, sd, sc, s0, pd, pc, vp, strengths[0], vectors, coefficients)
        s, p = q[:engine.ns], q[engine.ns:]
        return s, p, grid.derivatives(p).convergence

    return sample


def endpoint(engine, q, strengths, coordinates):
    """Same coupled local-GN diagnostic and evidence as the historical engine."""
    start = time.perf_counter()
    _, pc, _, _, _, logdet_p = coordinates
    pd = coordinates[0]
    value, gradient = engine.physical_value_grad(q, strengths)
    jacobian = engine.jacobian(q)
    vs = prior_response(jacobian[:, :engine.ns].T, engine.sd, engine.sc) / jnp.sqrt(strengths[0])
    vp = prior_response(jacobian[:, engine.ns:].T, pd, pc)
    cd, cc, logdet = scaled_cholesky(engine.eye + vs.T @ vs + vp.T @ vp)
    engine._finite_factor(cc, logdet, "endpoint full data covariance")
    gs = jsp.linalg.solve_triangular(engine.sc, gradient[:engine.ns] / engine.sd,
                                     lower=True) / jnp.sqrt(strengths[0])
    gp = jsp.linalg.solve_triangular(pc, gradient[engine.ns:] / pd, lower=True)
    proxy = float(stationarity_proxy(vs, vp, jnp.concatenate((gs, gp)), cd, cc))
    rendered = engine.problem.renderer.render(q[:engine.ns], q[engine.ns:],
                                               macro_mapping=engine.problem.macro_mapping)
    image = np.asarray(rendered.image_active)
    noise = np.asarray(engine.problem.noise_active)
    chi2 = float(np.sum(((image-np.asarray(engine.problem.data_active))/noise)**2))
    kappa = np.asarray(rendered.convergence)[np.asarray(rendered.derivative_valid)]
    loge = -float(value)-.5*(float(logdet)+engine.noise_normalization)
    row = dict(objective=float(value), gn_stationarity_proxy=proxy, gradient_inf=float(jnp.max(abs(gradient))),
               gradient_gate_passed=proxy <= engine.proxy_tolerance,
               chi2=chi2, chi2_per_pixel=chi2/engine.nd, penalty=2*float(value)-chi2,
               diagnostic_log_evidence=loge,
               logdet_prior=float(engine.ns*jnp.log(strengths[0])+engine.source_logdet+logdet_p),
               logdet_posterior_minus_prior=float(logdet), kappa_peak=float(np.max(kappa)),
               kappa_min=float(np.min(kappa)), kappa_RMS=float(np.sqrt(np.mean(kappa*kappa))))
    row["finite"] = bool(np.all(np.isfinite(q)) and all(np.isfinite(v) for v in row.values()))
    row["endpoint_seconds"] = time.perf_counter()-start
    fields = dict(q=np.asarray(q), source=np.asarray(q[:engine.ns]), potential=np.asarray(q[engine.ns:]),
                  kappa=kappa, image=image, noise=noise)
    return row, fields


def fit(engine, sample, initial, strengths, coordinates, mode, options):
    pd, pc, vp, vectors, coefficients, _ = coordinates
    base = jnp.asarray(initial)
    kernel_args = (strengths, base, engine.sd, engine.sc, engine.s0, pd, pc, vp, vectors, coefficients)
    initial_objective = float(engine.objective(base, strengths))

    def fun(z):
        value, gradient = engine.white_value_grad(jnp.asarray(z), *kernel_args)
        return float(value), np.asarray(gradient, dtype=np.float64)

    def physical_fields(z):
        arrays = sample(jnp.asarray(z), *kernel_args)
        return dict(zip(("source", "potential", "kappa"), (np.asarray(a) for a in arrays)))

    policy = None if mode == "native_plain" else POLICIES["standard" if mode == "native_monitored" else mode]
    result, diagnostic = minimize_with_stability(fun, np.zeros(engine.problem.n_joint),
        initial_objective=initial_objective, sample_fields=physical_fields, policy=policy,
        stop_when_stable=mode != "native_monitored", options=options)
    begin = time.perf_counter()
    q = engine.decode(result.x, strengths, coordinates, base=base)
    engine.physical_value_grad(q, strengths)[1].block_until_ready()
    diagnostic["inner_seconds"] = diagnostic["optimizer_seconds"] + time.perf_counter()-begin
    row, fields = endpoint(engine, q, strengths, coordinates)
    row.update(diagnostic, native_success=bool(result.success), native_status=int(result.status),
               native_message=str(result.message), iterations=int(result.nit), nfev=int(result.nfev),
               white_gradient_inf=float(np.max(abs(result.jac))), initial_objective=initial_objective)
    row["legacy_accepted"] = bool(result.success and row["finite"] and row["gradient_gate_passed"])
    row["fit_and_endpoint_seconds"] = row["inner_seconds"]+row["endpoint_seconds"]
    return row, fields


def read_fields(path):
    with np.load(path) as a:
        return {k: np.array(a[k]) for k in ("q", "source", "potential", "kappa", "image", "noise")}


def analyze(output, rows):
    groups = {}
    for r in rows:
        key = f"{r['profile']}/{r['method']}/{r['mode']}"
        groups.setdefault(key, []).append(r)
    references = {(r["profile"], r["method"], r["repeat"]): r for r in rows if r["mode"] == "native_plain"}
    summary = {}
    for key, group in groups.items():
        comparisons = []
        for r in group:
            ref = references[r["profile"], r["method"], r["repeat"]]
            fields = read_fields(output/r["directory"]/"fields.npz")
            reference_fields = read_fields(output/ref["directory"]/"fields.npz")
            comparisons.append(dict(index=r["index"], reference_index=ref["index"],
                objective_excess=r["objective"]-ref["objective"],
                **difference(fields, reference_fields)))
        summary[key] = dict(attempts=len(group), indices=[r["index"] for r in group],
            stability_stops=sum(r["stability_stop"] for r in group),
            native_successes=sum(r["native_success"] for r in group),
            gradient_passes=sum(r["gradient_gate_passed"] for r in group),
            iterations=[r["iterations"] for r in group],
            median_inner_seconds=float(np.median([r["inner_seconds"] for r in group])),
            median_monitor_seconds=float(np.median([r["monitor_seconds"] for r in group])),
            median_fit_and_endpoint_seconds=float(np.median([r["fit_and_endpoint_seconds"] for r in group])),
            objective_range=[min(r["objective"] for r in group), max(r["objective"] for r in group)],
            kappa_peak_range=[min(r["kappa_peak"] for r in group), max(r["kappa_peak"] for r in group)],
            comparisons_to_same_model_native=comparisons)
    timing = {}
    profiles = dict.fromkeys(r["profile"] for r in rows)
    for profile in profiles:
        modes = ["native_plain", "standard"]
        if profile == "single_step_lens":
            modes += ["native_monitored", "loose", "tight"]
        ratios = {}
        for mode in modes:
            b = summary[f"{profile}/bilinear/{mode}"]
            c = summary[f"{profile}/cubic/{mode}"]
            ratios[mode] = dict(inner_bilinear_over_cubic=b["median_inner_seconds"]/c["median_inner_seconds"],
                fit_and_endpoint_bilinear_over_cubic=b["median_fit_and_endpoint_seconds"]/c["median_fit_and_endpoint_seconds"])
        bn = summary[f"{profile}/bilinear/native_plain"]["median_inner_seconds"]
        cn = summary[f"{profile}/cubic/native_plain"]["median_inner_seconds"]
        bs = summary[f"{profile}/bilinear/standard"]["median_inner_seconds"]
        cs = summary[f"{profile}/cubic/standard"]["median_inner_seconds"]
        timing[profile] = dict(ratios=ratios, bilinear_fraction_time_saved=1-bs/bn,
            absolute_inner_gap_native_seconds=bn-cn, absolute_inner_gap_standard_seconds=bs-cs,
            fraction_absolute_inner_gap_removed=1-(bs-cs)/(bn-cn) if bn != cn else None)
    pairs = []
    for profile in profiles:
        for repeat in sorted({r["repeat"] for r in rows}):
            b, c = [next(r for r in rows if r["profile"] == profile and r["repeat"] == repeat
                         and r["method"] == method and r["mode"] == "standard")
                    for method in ("bilinear", "cubic")]
            pairs.append(dict(profile=profile, repeat=repeat, indices=[b["index"], c["index"]],
                **difference(read_fields(output/b["directory"]/"fields.npz"),
                             read_fields(output/c["directory"]/"fields.npz"))))
    result = dict(groups=summary, timing=timing, cross_interpolation_standard=pairs,
                  reference_scope="Same model, same literal cold start and coordinates; native endpoint is not proven global MAP")
    write_json(output/"analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"stability_timing_v1")
    parser.add_argument("--previous", type=Path, default=HERE/"cubic_source_v1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hardest-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 3:
        raise ValueError("Use at least three paired repetitions")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    prior_protocol = json.loads((args.previous/"protocol.json").read_text())
    old_rows = json.loads((args.previous/"summary.json").read_text())["rows"]
    for path, digest in prior_protocol["source_hashes"].items():
        assert pt.sha256(pt.ROOT/path) == digest, path
    input_hashes = dict(prior_protocol["input_hashes"])
    input_hashes.update({str((args.previous/name).resolve()): pt.sha256(args.previous/name)
                         for name in ("initial_values.npz", "protocol.json", "summary.json")})
    assert all(pt.sha256(Path(p)) == h for p, h in input_hashes.items())
    with np.load(args.previous/"initial_values.npz") as a:
        initial = np.array(a["joint_values"])
    profiles = ["single_step_lens"]
    if not args.hardest_only:
        profiles += [p for p in prior_protocol["profiles_lens_log10"] if p != "single_step_lens"]
    plan = []
    for profile in profiles:
        modes = ["native_plain", "standard"]
        if profile == "single_step_lens":
            modes = ["native_plain", "native_monitored", "loose", "standard", "tight"]
        for repeat in range(args.repeats):
            for mode in modes if repeat % 2 == 0 else modes[::-1]:
                for method in ("bilinear", "cubic") if repeat % 2 == 0 else ("cubic", "bilinear"):
                    plan.append(dict(profile=profile, repeat=repeat, mode=mode, method=method))
    sources = [Path(__file__), HERE/"stability_stop.py", HERE/"check_practical_convergence.py",
               pt.ROOT/"examples/tests/test_j0946_stability_stop.py"]
    sources += [pt.ROOT/p for p in prior_protocol["source_hashes"]]
    for path in sources:
        target = args.output/"source_snapshot"/path.relative_to(pt.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), arguments=vars(args),
        experiment="J0946_same_initial_objective_field_stability_stop_timing_v1", plan=plan,
        policies={name: asdict(policy) for name, policy in POLICIES.items()}, optimizer=OPTIONS,
        acceptance="Additional stability callback OR unchanged native exit; SciPy status 99 is an intentional callback stop",
        window_rule="All accepted F values; pairwise diameters of fields sampled each 10 steps; 50-step window, 3 consecutive checks",
        input_hashes=input_hashes, source_hashes={str(p.relative_to(pt.ROOT)): pt.sha256(p) for p in sources},
        initial_array_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        shared_coordinates="One hard-M3 factorization per profile, reused literally across modes/methods/repeats",
        timing="Warm inner wall time includes all monitoring, final decode/gradient; endpoint reported separately; preparation, audits and IO excluded",
        warmup="One untimed 1-step fit plus full endpoint and field sampler for each method before measurements",
        physical_model_changes=False, outer_evidence_search=False,
        versions=dict(jax=jax.__version__, scipy=pt.scipy.__version__), devices=str(jax.devices()))
    write_json(args.output/"protocol.json", protocol)
    start = time.perf_counter()
    hard, _, arrays = pt.load_problem()
    reference = EvidenceEngine(hard)
    engines = dict(bilinear=EvidenceEngine(zero_ring_problem(hard, arrays, 1.), reference_engine=reference),
                   cubic=EvidenceEngine(cubic_problem(hard, arrays, 1.), reference_engine=reference))
    samplers = {method: field_sampler(engine) for method, engine in engines.items()}
    strengths = {}
    coordinates = {}
    prep = {}
    for profile in profiles:
        row = next(r for r in old_rows if r["profile"] == profile)
        strengths[profile] = jnp.asarray(10.**np.asarray(row["log10_strengths"]))
        begin = time.perf_counter()
        coordinates[profile] = reference.prepare_coordinates(strengths[profile])
        prep[profile] = time.perf_counter()-begin
    warmup = {}
    for method, engine in engines.items():
        begin = time.perf_counter()
        fit(engine, samplers[method], initial, strengths[profiles[0]], coordinates[profiles[0]],
            "standard", {**OPTIONS, "maxiter": 1})
        warmup[method] = time.perf_counter()-begin
    write_json(args.output/"preparation.json", dict(coordinate_seconds=prep, warmup_seconds=warmup,
        engine_preparation_seconds=reference.preparation_seconds+sum(e.preparation_seconds for e in engines.values())))
    rows = []
    for index, case in enumerate(plan):
        profile, method, mode = case["profile"], case["method"], case["mode"]
        print(f"[stability-timing {index+1}/{len(plan)}] {profile} {method} {mode} repeat={case['repeat']}", flush=True)
        row, fields = fit(engines[method], samplers[method], initial, strengths[profile], coordinates[profile], mode, OPTIONS)
        if case["repeat"] == 0 and (mode == "standard" or (profile == "single_step_lens" and mode == "native_plain")):
            row["independent_gradient_audit"] = audit_endpoint(fields["q"], np.asarray(strengths[profile]),
                                                                HERE/"macro", method, engines[method])
        row.update(case, index=index, directory=f"trials/{index:04d}",
                   strengths=dict(zip(("lambda_s", "lambda4", "lambda0"), np.asarray(strengths[profile]).tolist())))
        trial = args.output/row["directory"]
        trial.mkdir(parents=True)
        np.savez_compressed(trial/"fields.npz", **fields)
        write_json(trial/"result.json", row)
        rows.append(row)
        with (args.output/"evaluations.jsonl").open("a") as handle:
            handle.write(json.dumps(pt.json_safe(row), allow_nan=False)+"\n")
        print(f"[result] nit={row['iterations']} inner={row['inner_seconds']:.4f}s F={row['objective']:.9f} "
              f"stable_stop={row['stability_stop']} native={row['native_status']} proxy={row['gn_stationarity_proxy']:.3g}", flush=True)
    unchanged = all(pt.sha256(Path(p)) == h for p, h in input_hashes.items())
    assert unchanged and all(r["finite"] for r in rows)
    write_json(args.output/"summary.json", dict(rows=rows, original_inputs_unchanged=unchanged,
        elapsed_seconds=time.perf_counter()-start, protocol_path=str(args.output/"protocol.json")))
    analysis = analyze(args.output, rows)
    print(json.dumps(analysis["timing"], indent=2), flush=True)


if __name__ == "__main__":
    main()
