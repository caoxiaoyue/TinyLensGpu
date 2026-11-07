"""Paired J0946 bilinear/C1 source test, fixed M3 hypers and fixed lens pairs.

No outer search or integration changes. Both methods start from the exact same
saved source-only array of the preceding M3-hyperparameter experiment, with
zero potential. Source coefficients remain free. All fits share the original
hard-M3 whitening reference and the original strict acceptance thresholds.
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

from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from examples.potential_correction.result_products import write_json
from lbfgs import EvidenceEngine
from run_fixed_boundary import zero_ring_problem
from source_boundary import zero_ring_mapping, zero_ring_gradient
from source_cubic import CubicSourceRenderer, cubic_mapping, cubic_gradient
from check_source_knots import knot_probe


HERE = Path(__file__).resolve().parent
PROTOCOL = "J0946_fixed_M3_prior_paired_bilinear_C1_cardinal_cubic_v1"


def cubic_problem(hard, arrays, lambda0):
    r = hard.renderer
    renderer = CubicSourceRenderer(
        r.macro_model, r.sim_config, r.source_n, r.source_bbox,
        potential_grid=r.potential_grid, dtype="float64",
        source_interpolation="cubic_hermite",
    )
    problem = IterativePotentialCorrection(renderer, arrays["lens_subtracted_image"],
        arrays["noise_map"], arrays["source_values"], lambda_amplitude=lambda0,
        source_regularization=hard.source_regularizer.regularization_type,
        initialization_provenance=hard.initialization_provenance)
    problem.source_scale = jnp.asarray(arrays["source_scale"], dtype=jnp.float64)
    problem.initialization_provenance.update(source_initialization="shared_frozen_previous_source_MAP",
        source_truth_used=False, source_boundary=renderer.source_boundary,
        gradient_policy="current_C1_cardinal_cubic_analytic_matches_AD", extra_fitted_parameters=0)
    return problem


def audit_endpoint(q, strengths, macro_dir, method, engine):
    # lambda0 is captured by the library JIT: construct a fresh actual-strength
    # public problem, rather than mutating a traced problem's lambda_amplitude.
    hard, _, arrays = pt.load_problem(macro_dir, .1, float(strengths[2]))
    factory = zero_ring_problem if method == "bilinear" else cubic_problem
    problem = factory(hard, arrays, float(strengths[2]))
    point = problem._evaluate_kernel(jnp.asarray(q), float(strengths[0]), float(strengths[1]))
    value, gradient = engine.physical_value_grad(jnp.asarray(q), jnp.asarray(strengths))
    vf = abs(float(value-point.objective))
    gf = float(jnp.linalg.norm(gradient+point.negative_gradient)/jnp.maximum(1., jnp.linalg.norm(gradient)))
    if not bool(point.finite) or vf > 1e-6 or gf > 1e-7:
        raise FloatingPointError(f"Independent AD/analytic discrepancy: F={vf}, g={gf}")
    return dict(passed=True, objective_absolute_error=vf, gradient_relative_error=gf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"cubic_source_v1")
    parser.add_argument("--macro-dir", type=Path, default=HERE/"macro")
    parser.add_argument("--previous-run", type=Path, default=HERE/"m3_source_fixed_v1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=3000)
    args = parser.parse_args()
    if args.repeats < 3 or args.maxiter < 1:
        raise ValueError("Use at least three identical-start repeats and a positive iteration budget")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    start = time.perf_counter()
    hard, metadata, arrays = pt.load_problem(args.macro_dir, .1)
    previous = json.loads((args.previous_run/"summary.json").read_text())
    old_protocol = json.loads((args.previous_run/"protocol.json").read_text())
    initial_path = args.previous_run/"initial_values.npz"
    if pt.sha256(initial_path) != old_protocol["source_scale_sha256_array"]:
        raise ValueError("Previous frozen initialization archive has changed")
    with np.load(initial_path) as archive:
        initial = np.array(archive["zero_ring"], dtype=np.float64)
        np.testing.assert_array_equal(archive["source_scale"], arrays["source_scale"])
        np.testing.assert_array_equal(archive["saved_M3_source"], arrays["source_values"])
    assert initial.shape == (hard.n_joint,) and np.all(initial[hard.n_source:] == 0)
    assert metadata["lambda_source"] == old_protocol["source_lambda_requested_M3"]
    assert metadata["adaptive_reg_rho"] == old_protocol["source_rho_fixed"]
    for name, digest in old_protocol["handoff_hashes"].items():
        assert pt.sha256(args.macro_dir/name) == digest
    profiles = {name: np.asarray(old_protocol["probe_lens_log10"][name])
                for name in ("strong_anchor", "previous_iterative_lens")}
    best = previous["best"]["strengths"]
    profiles["previous_fixed_M3_selected_lens"] = np.log10([best["lambda4"], best["lambda0"]])
    profiles["single_step_lens"] = np.asarray(old_protocol["probe_lens_log10"]["single_step_lens"])
    inputs = [args.macro_dir/name for name in ("m3_handoff.json", "m3_handoff.npz")]
    inputs += [args.previous_run/name for name in ("protocol.json", "summary.json", "initial_values.npz")]
    inputs += [args.previous_run/f"trials/{i:04d}/reconstruction.npz" for i in (37, 38)]
    sources = [HERE/name for name in ("run_cubic_source.py", "source_cubic.py", "check_source_knots.py",
        "analyze_cubic_source.py", "pt.py", "lbfgs.py", "source_boundary.py", "run_fixed_boundary.py")]
    sources += [pt.ROOT/"examples/tests/test_j0946_cubic_source.py"]
    sources += [pt.ROOT/"TinyLensGpu"/name for name in (
        "ForwardSimulation/LensImage/potential_correction.py", "ForwardSimulation/LensImage/pixelized_operator.py",
        "ObservationModel/LensImage/potential_correction_iterative.py",
        "ObservationModel/LensImage/_potential_correction_common.py", "utils/inversion/regularization.py",
        "utils/inversion/potential_regularization.py", "utils/lensing/potential.py")]
    protocol = dict(protocol=PROTOCOL, created_utc=datetime.now(timezone.utc).isoformat(), arguments=vars(args),
        source_hashes={str(p.relative_to(pt.ROOT)): pt.sha256(p) for p in sources},
        input_hashes={str(p.resolve()): pt.sha256(p) for p in inputs},
        fixed_source_lambda_M3=metadata["lambda_source"], fixed_rho_M3=metadata["adaptive_reg_rho"],
        source_scale_fixed=True, source_regularization=metadata["source_regularization"], source_pixels_frozen=False,
        source_node_semantics="original brightness intensities, no B-spline coefficient conversion",
        initialization="identical literal previous zero-ring source-only MAP for BOTH methods; zero potential",
        reference_metric="unchanged hard-bilinear M3 source and zero potential for BOTH methods",
        extra_fitted_parameters=0, cubic="tensor cardinal cubic Hermite, centered slopes, Catmull-Rom a=-0.5",
        exterior="all exterior nodes fixed zero; bilinear compact support 1 cell, cubic 2 cells",
        source_positivity="original signed coefficients; cubic negative weights/overshoot possible, no clipping",
        profiles_lens_log10=profiles, qualification=dict(native_success=True, gn_proxy_max=.001),
        repeatability=dict(all_attempts_accepted=True, diagnostic_log_evidence_range_max=.01,
            maximum_pairwise_kappa_difference_max=1e-4),
        optimizer=dict(method="L-BFGS-B", maxiter=args.maxiter, maxfun=120000, maxcor=30,
            maxls=80, ftol=1e-15, gtol=1e-8),
        evidence="same full coupled endpoint local-GN approximation and original normalized nodal priors",
        outer_search_performed=False, smooth_refit_performed=False, real_data_no_injected_truth=True,
        configuration=hard._configuration(), identical_start_repeats=args.repeats,
        gpu=str(jax.devices()), versions=dict(jax=jax.__version__, scipy=pt.scipy.__version__, numpy=np.__version__))
    for path in sources:
        target = args.output/"source_snapshot"/path.relative_to(pt.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    np.savez_compressed(args.output/"initial_values.npz", joint_values=initial,
        saved_M3_source=arrays["source_values"], source_scale=arrays["source_scale"])
    protocol["initial_values_archive_sha256"] = pt.sha256(args.output/"initial_values.npz")
    write_json(args.output/"protocol.json", protocol)
    # Actual old problematic state: first validate what this interpolation changes.
    with np.load(args.previous_run/"trials/0038/reconstruction.npz") as archive:
        witness_source, witness_potential = archive["source_values"], archive["potential_values"]
    mapping = hard.renderer.source_mapping(jnp.asarray(witness_potential))
    beta = np.column_stack((mapping.beta_x, mapping.beta_y))
    witness = {}
    for name, functions in dict(bilinear=(zero_ring_mapping, zero_ring_gradient),
                                cubic=(cubic_mapping, cubic_gradient)).items():
        witness[name] = knot_probe(witness_source, beta, hard.renderer.source_bbox,
            hard.renderer.source_n, *functions)
    write_json(args.output/"knot_replay.json", witness)
    assert not witness["bilinear"]["derivative_continuity_pass"]
    assert witness["cubic"]["derivative_continuity_pass"]
    print("[mechanism] old derivative jump reproduced; cubic derivative continuous", flush=True)
    reference = EvidenceEngine(hard, maxiter=args.maxiter)
    engines = dict(bilinear=EvidenceEngine(zero_ring_problem(hard, arrays, 1.), maxiter=args.maxiter, reference_engine=reference),
                   cubic=EvidenceEngine(cubic_problem(hard, arrays, 1.), maxiter=args.maxiter, reference_engine=reference))
    with np.load(args.previous_run/"trials/0037/reconstruction.npz") as a:
        q37 = np.array(a["joint_values"])
    with np.load(args.previous_run/"trials/0038/reconstruction.npz") as a:
        q38 = np.array(a["joint_values"])
    replay_strengths = jnp.asarray(10.**np.r_[np.log10(metadata["lambda_source"]), profiles["previous_fixed_M3_selected_lens"]])
    gradient_replay = {}
    for name, engine in engines.items():
        f37, g37 = engine.physical_value_grad(jnp.asarray(q37), replay_strengths)
        f38, g38 = engine.physical_value_grad(jnp.asarray(q38), replay_strengths)
        gradient_replay[name] = dict(objective_absolute_difference=abs(float(f38-f37)),
            physical_gradient_difference_L2=float(jnp.linalg.norm(g38-g37)),
            physical_gradient_difference_inf=float(jnp.max(jnp.abs(g38-g37))),
            physical_gradient_relative_difference=float(jnp.linalg.norm(g38-g37)/jnp.maximum(1., jnp.linalg.norm(g37))))
    write_json(args.output/"saved_endpoint_gradient_replay.json", dict(optimization_rerun=False,
        maximum_joint_input_difference=float(np.max(abs(q38-q37))),
        maximum_potential_input_difference=float(np.max(abs(q38[hard.n_source:]-q37[hard.n_source:]))),
        methods=gradient_replay))
    strengths0 = jnp.asarray(10.**np.r_[np.log10(metadata["lambda_source"]), profiles["strong_anchor"]])
    warm_coordinates = reference.prepare_coordinates(strengths0)
    warm_times = {}
    for name, engine in engines.items():
        begin = time.perf_counter()
        pd, pc, vp, vectors, coefficients, _ = warm_coordinates
        _, gradient = engine.white_value_grad(jnp.zeros(hard.n_joint), strengths0, jnp.asarray(initial),
            engine.sd, engine.sc, engine.s0, pd, pc, vp, vectors, coefficients)
        gradient.block_until_ready()
        engine.physical_value_grad(jnp.asarray(initial), strengths0)[1].block_until_ready()
        engine.decode(np.zeros(hard.n_joint), strengths0, warm_coordinates, base=jnp.asarray(initial)).block_until_ready()
        warm_times[name] = time.perf_counter()-begin
    write_json(args.output/"preparation.json", dict(reference_seconds=reference.preparation_seconds,
        model_preparation_seconds={k:e.preparation_seconds for k,e in engines.items()},
        forward_gradient_JIT_warmup_seconds=warm_times, timing="per-fit timings exclude this one-time preparation and explicit warmup"))
    rows = []
    for profile, lens in profiles.items():
        x = np.r_[np.log10(metadata["lambda_source"]), lens]
        for repeat in range(args.repeats):
            # Paired order; no cached optimization, reused endpoint, or cherry picking.
            for method, engine in engines.items():
                index = len(rows); directory = f"trials/{index:04d}"
                trial = args.output/directory; trial.mkdir(parents=True)
                print(f"[fit {index:02d}] {profile} {method} repeat={repeat}", flush=True)
                try:
                    row, q = engine.fit(x, initial_values=initial, initialization_label="shared_frozen_previous_source_MAP",
                        audit=(profile == "strong_anchor" and repeat == 0))
                    write_json(trial/"optimizer_result.json", row)
                    np.savez_compressed(trial/"optimizer_state.npz", joint_values=q)
                    assert np.isclose(row["strengths"]["lambda_s"], metadata["lambda_source"], rtol=1e-14, atol=0)
                    assert np.array_equal(engine.last_initial_values, initial)
                    if repeat == 0:
                        row["analytical_audit"] = audit_endpoint(q, 10.**x, args.macro_dir, method, engine)
                    row.update(index=index, directory=directory, profile=profile, method=method, repeat=repeat,
                        source_hyperparameters_fixed=True, source_pixels_frozen=False,
                        source_relative_change_from_M3=float(np.linalg.norm(q[:hard.n_source]-arrays["source_values"])/np.linalg.norm(arrays["source_values"])))
                    row = pt.export_solution(trial, engine, arrays, metadata, row, q)
                    exported = json.loads((trial/"result.json").read_text())
                    exported.update(protocol=PROTOCOL, selection_scope="predeclared fixed lens pair; no hyperparameter selection",
                        source_interpolation=method, source_support_bbox=engine.problem.renderer.support_bbox)
                    write_json(trial/"result.json", exported)
                except FloatingPointError as exc:
                    row = dict(index=index, directory=directory, profile=profile, method=method, repeat=repeat,
                        accepted=False, log_evidence=None, failure=str(exc), log10_strengths=x.tolist(),
                        source_hyperparameters_fixed=True, source_pixels_frozen=False)
                rows.append(row)
                with (args.output/"evaluations.jsonl").open("a") as handle:
                    handle.write(json.dumps(pt.json_safe(row), allow_nan=False)+"\n")
                print(f"[result {index:02d}] {method} accepted={row['accepted']} nit={row.get('iterations')} "
                    f"proxy={row.get('gn_stationarity_proxy')} inner={row.get('inner_seconds')} peak={row.get('delta_kappa_peak')}", flush=True)
    unchanged = all(pt.sha256(Path(path)) == digest for path,digest in protocol["input_hashes"].items())
    assert unchanged, "An input/reference was modified"
    write_json(args.output/"summary.json", dict(protocol=PROTOCOL, rows=rows,
        input_hashes_unchanged=unchanged, elapsed_seconds=time.perf_counter()-start,
        preparation_seconds=reference.preparation_seconds+sum(e.preparation_seconds for e in engines.values())))
    print(f"[complete] {len(rows)} fits in {time.perf_counter()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
