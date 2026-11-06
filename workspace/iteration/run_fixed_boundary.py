"""Fixed-lambda J0946 boundary/initialization experiment, with paired repeats.

Four conditions: hard/zero-ring source boundary x source-only/single-step
joint initialization. Every condition uses the same physical priors, original
free nodes, and hard-M3 whitening metric. No outer hyperparameter search.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import pt  # Configures checkout imports and GPU allocation before JAX work.
import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from examples.potential_correction.result_products import write_json
from lbfgs import EvidenceEngine
from source_boundary import ZeroRingRenderer


HERE = Path(__file__).resolve().parent
PROTOCOL = "J0946_fixed_single_step_lambdas_boundary_initialization_v1"


def zero_ring_problem(hard, arrays, lambda0):
    r = hard.renderer
    renderer = ZeroRingRenderer(r.macro_model, r.sim_config, r.source_n, r.source_bbox,
                                potential_grid=r.potential_grid, dtype="float64")
    problem = IterativePotentialCorrection(renderer, arrays["lens_subtracted_image"],
        arrays["noise_map"], arrays["source_values"], lambda_amplitude=lambda0,
        source_regularization=hard.source_regularizer.regularization_type,
        initialization_provenance=hard.initialization_provenance)
    problem.source_scale = jnp.asarray(arrays["source_scale"], dtype=jnp.float64)
    problem.initialization_provenance.update(source_initialization="explicit_comparison_start",
        source_truth_used=False, gradient_policy="current_zero_extended_bilinear_derivative",
        source_boundary="one_fixed_zero_ghost_ring", extra_fitted_parameters=0)
    return problem


def check_single_step(path, hard, arrays):
    result = json.loads((path / "result.json").read_text())
    with np.load(path / "reconstruction.npz") as archive:
        saved = {key: archive[key] for key in archive.files}
    checks = {
        "source_reference": np.array_equal(saved["macro_source_values"], arrays["source_values"]),
        "source_scale": np.array_equal(saved["source_scale"], arrays["source_scale"]),
        "mask": np.array_equal(saved["mask"], arrays["mask"]),
        "data": np.array_equal(saved["data_active"], np.asarray(hard.data_active)),
        "noise": np.array_equal(saved["noise_active"], np.asarray(hard.noise_active)),
        "potential_topology": np.array_equal(saved["potential_node_indices"], np.asarray(hard.renderer.potential_grid.node_indices)),
        "derivative_valid": np.array_equal(saved["derivative_valid"], np.asarray(hard.renderer.potential_grid.derivative_valid)),
        "source_axes": all(np.allclose(saved[key], np.asarray(getattr(hard.renderer, key)), rtol=0, atol=1e-7)
                           for key in ("source_x_axis", "source_y_axis")),
    }
    if not all(checks.values()):
        raise ValueError(f"Single-step handoff does not match fixed experiment: {checks}")
    warm = np.r_[saved["source_values"], saved["potential_values"]].astype(np.float64)
    return result, warm, checks


def public_audit(problem, engine, q, strengths):
    point = problem._evaluate_kernel(jnp.asarray(q), strengths[0], strengths[1])
    value, gradient = engine.physical_value_grad(jnp.asarray(q), strengths)
    vf = abs(float(value-point.objective))
    gf = float(jnp.linalg.norm(gradient+point.negative_gradient)/jnp.maximum(jnp.linalg.norm(gradient), 1.0))
    passed = bool(point.finite) and vf < 1e-6 and gf < 1e-7
    if not passed:
        raise FloatingPointError(f"Current-model AD/analytic disagreement: F={vf}, g={gf}")
    return dict(passed=passed, objective_absolute_error=vf, gradient_relative_error=gf)


def real_boundary_profile(engines, log_strengths, output):
    """Replay the saved troublesome state near an outer edge, without fitting."""
    path = HERE / "preflight" / "candidate_0000.npz"
    if not path.exists():
        write_json(output, dict(skipped="Historical endpoint unavailable"))
        return
    with np.load(path) as archive:
        q = jnp.asarray(archive["joint_values"])
    problem = engines["hard"].problem
    mapping = problem.linearize(q).mapping
    bx, by = np.asarray(mapping.beta_x), np.asarray(mapping.beta_y)
    xmin, xmax, ymin, ymax = np.asarray(problem.renderer.source_bbox)
    distances = np.stack((bx-xmin, xmax-bx, by-ymin, ymax-by), axis=1)
    # Distance to a finite rectangle side, not to its infinite supporting line.
    eligible = np.stack(((by>=ymin)&(by<=ymax), (by>=ymin)&(by<=ymax),
                         (bx>=xmin)&(bx<=xmax), (bx>=xmin)&(bx<=xmax)), axis=1)
    selected = np.where(eligible, np.abs(distances), np.inf)
    ray, side = np.unravel_index(np.argmin(selected), selected.shape)
    axis = "x" if side < 2 else "y"
    bound = (xmin, xmax, ymin, ymax)[side]
    current = (bx if axis == "x" else by)[ray]
    grid = problem.renderer.potential_grid
    direction = -getattr(grid, axis)  # beta moves by +t along the chosen axis.
    center = q.at[problem.n_source:].add((bound-current)*direction)
    strengths = jnp.asarray(10.**np.asarray(log_strengths))
    rows = []
    for name, engine in engines.items():
        for h in (1e-6, 1e-8, 1e-10):
            values, images, objectives = [], [], []
            for sign in (-1, 1):
                point = center.at[problem.n_source:].add(sign*h*direction)
                mapped = engine.problem.linearize(point).mapping
                values.append(float(jnp.sum(mapped.weights[ray]*point[:problem.n_source][mapped.indices[ray]])))
                images.append(np.asarray(engine.problem.forward(point)))
                objectives.append(float(engine.objective(point, strengths)))
            rows.append(dict(boundary=name, displacement_arcsec=h,
                query_brightness_difference=abs(values[1]-values[0]),
                maximum_image_difference=float(np.max(abs(images[1]-images[0]))),
                objective_difference=objectives[1]-objectives[0]))
    write_json(output, dict(saved_endpoint=str(path), ray=int(ray), side=int(side),
        original_distance_arcsec=float(selected[ray, side]), profiles=rows))


def summarize_case(rows, output, evidence_tolerance=.01, kappa_tolerance=1e-4):
    valid = [r for r in rows if "objective" in r]
    if len(valid) != len(rows):
        return dict(runs=len(rows), failures=[r.get("failure") for r in rows if "failure" in r],
                    all_repeats_accepted=False)
    first = np.load(output / rows[0]["directory"] / "reconstruction.npz")
    q0, k0 = first["joint_values"].copy(), first["convergence"].copy()
    first.close()
    differences = []
    for row in rows:
        with np.load(output / row["directory"] / "reconstruction.npz") as archive:
            differences.append(float(np.nanmax(abs(archive["convergence"]-k0))))
            row["relative_joint_difference_to_first"] = float(np.linalg.norm(archive["joint_values"]-q0)/np.linalg.norm(q0))
    evidence_range = float(np.ptp([r["diagnostic_log_evidence"] for r in rows]))
    field_range = max(differences)
    return dict(runs=len(rows), accepted=sum(r["accepted"] for r in rows),
        native_success=sum(r["native_success"] for r in rows),
        all_repeats_accepted=all(r["accepted"] for r in rows),
        diagnostic_evidence_range=evidence_range, maximum_kappa_difference_to_first=field_range,
        evidence_repeatability_pass=evidence_range <= evidence_tolerance,
        field_repeatability_pass=field_range <= kappa_tolerance,
        proxy_range=[min(r["gn_stationarity_proxy"] for r in rows), max(r["gn_stationarity_proxy"] for r in rows)],
        objective_range=[min(r["objective"] for r in rows), max(r["objective"] for r in rows)],
        kappa_peak_range=[min(r["delta_kappa_peak"] for r in rows), max(r["delta_kappa_peak"] for r in rows)],
        median_iterations=float(np.median([r["iterations"] for r in rows])),
        median_inner_seconds=float(np.median([r["inner_seconds"] for r in rows])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "fixed_boundary_v1")
    parser.add_argument("--macro-dir", type=Path, default=HERE / "macro")
    parser.add_argument("--single-step-dir", type=Path, default=HERE.parent / "single_step/potential_qr_v2/maximum_evidence")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=3000)
    args = parser.parse_args()
    if args.repeats < 2 or args.maxiter < 1:
        raise ValueError("Use at least two repeats and a positive iteration budget")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    start = time.perf_counter()
    single = json.loads((args.single_step_dir / "result.json").read_text())
    wanted = single["strengths"]
    log_strengths = np.log10([wanted["lambda_source"], wanted["lambda_potential"], wanted["lambda_amplitude"]])
    strengths = jnp.asarray(10.**log_strengths)
    hard, metadata, arrays = pt.load_problem(args.macro_dir, .1, float(strengths[2]))
    _, warm, input_checks = check_single_step(args.single_step_dir, hard, arrays)
    zero = zero_ring_problem(hard, arrays, float(strengths[2]))
    before = {name: pt.sha256(args.macro_dir / name) for name in ("m3_handoff.json", "m3_handoff.npz")}
    protocol = dict(protocol=PROTOCOL, created_utc=datetime.now(timezone.utc).isoformat(),
        arguments=vars(args), requested_single_step_strengths=wanted,
        effective_strengths=dict(zip(("lambda_s", "lambda4", "lambda0"), np.asarray(strengths))),
        source_grid_bbox=np.asarray(hard.renderer.source_bbox),
        zero_ring_support_bbox=zero.renderer.support_bbox,
        fitted_source_nodes=hard.n_source, fitted_potential_nodes=hard.n_potential,
        extra_fitted_parameters=0, configuration=hard._configuration(),
        input_checks=input_checks, handoff_hashes=before,
        preconditioner="Same fixed hard-boundary M3 metric and same coordinate factors within each repeat",
        initializations="Hard-model source-only MAP at zero potential, or saved single-step joint solution; literal arrays shared across boundary modes",
        prior="Original adaptive first-order source prior and full D4+all-node amplitude potential prior; no ghost variables in normalization",
        optimizer=dict(method="L-BFGS-B", maxiter=args.maxiter, maxcor=30, maxls=80, ftol=1e-15, gtol=1e-8),
        qualification=dict(native_success=True, gn_proxy_max=.001),
        repeatability=dict(diagnostic_log_evidence_range_max=.01, kappa_max_absolute_difference_max=1e-4),
        evidence_method="full coupled endpoint local-GN Gaussian approximation",
        fixed_real_data_no_injected_truth=True, hyperparameter_search=False, smooth_refit_performed=False)
    source_files = [HERE / name for name in ("run_fixed_boundary.py", "source_boundary.py", "lbfgs.py", "pt.py")]
    source_files += [pt.ROOT / "TinyLensGpu" / name for name in (
        "ForwardSimulation/LensImage/potential_correction.py", "utils/lensing/mapping.py",
        "ObservationModel/LensImage/potential_correction_iterative.py",
        "ObservationModel/LensImage/_potential_correction_common.py",
        "utils/inversion/regularization.py", "utils/inversion/potential_regularization.py",
        "utils/lensing/potential.py")]
    protocol["source_hashes"] = {str(path.relative_to(pt.ROOT)): pt.sha256(path) for path in source_files}
    for path in source_files:
        target = args.output / "source_snapshot" / path.relative_to(pt.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    protocol["single_step_hashes"] = {name: pt.sha256(args.single_step_dir / name)
                                     for name in ("result.json", "reconstruction.npz")}
    write_json(args.output / "protocol.json", protocol)
    engines = {"hard": EvidenceEngine(hard, maxiter=args.maxiter)}
    engines["zero_ring"] = EvidenceEngine(zero, maxiter=args.maxiter, reference_engine=engines["hard"])
    real_boundary_profile(engines, log_strengths, args.output / "real_boundary_profile.json")
    rows = []
    cases = [(boundary, initialization) for initialization in ("source_only", "single_step")
             for boundary in ("hard", "zero_ring")]
    for repeat in range(args.repeats):
        prepare_start = time.perf_counter()
        cold = np.asarray(engines["hard"].conditional_source(strengths[0]))
        coordinates = engines["hard"].prepare_coordinates(strengths)
        prepare_seconds = time.perf_counter()-prepare_start
        np.savez_compressed(args.output / f"initial_values_{repeat:02d}.npz", source_only=cold, single_step=warm)
        order = cases if repeat % 2 == 0 else list(reversed(cases))
        for boundary, initialization in order:
            engine = engines[boundary]
            initial = cold if initialization == "source_only" else warm
            directory = f"{boundary}/{initialization}/repeat_{repeat:02d}"
            print(f"[fit] {directory}", flush=True)
            try:
                row, q = engine.fit(log_strengths, initial_values=initial,
                    initialization_label=initialization, prepared_coordinates=(strengths, coordinates),
                    audit=repeat == 0)
                trial = args.output / directory
                trial.mkdir(parents=True, exist_ok=True)
                write_json(trial / "optimizer_result.json", row)
                np.savez_compressed(trial / "optimizer_state.npz", joint_values=q,
                                    initial_joint_values=engine.last_initial_values)
                row["analytical_audit"] = public_audit(engine.problem, engine, q, strengths)
                row.update(boundary=boundary, initialization_case=initialization, repeat=repeat,
                           directory=directory, paired_preparation_seconds=prepare_seconds,
                           objective_decrease=row["initial_objective"]-row["objective"])
                if row["objective_decrease"] < -1e-7:
                    raise FloatingPointError("L-BFGS returned an increased objective")
                engine.problem.initialization_provenance.update(source_initialization=initialization,
                    potential_initialization="zero" if initialization == "source_only" else "saved_single_step_potential",
                    source_truth_used=False)
                row = pt.export_solution(args.output / directory, engine, arrays, metadata, row, q)
                result_path = args.output / directory / "result.json"
                result = json.loads(result_path.read_text())
                result.update(protocol=PROTOCOL, selection_scope="fixed_lambda_repeat_not_hyperparameter_optimization",
                    source_boundary=boundary, source_support_bbox=zero.renderer.support_bbox
                        if boundary == "zero_ring" else hard.renderer.source_bbox,
                    extra_fitted_parameters=0)
                write_json(result_path, result)
            except FloatingPointError as exc:
                row = dict(boundary=boundary, initialization_case=initialization, repeat=repeat,
                           directory=directory, accepted=False, failure=str(exc))
            rows.append(row)
            with (args.output / "evaluations.jsonl").open("a") as handle:
                handle.write(json.dumps(pt.json_safe(row), allow_nan=False)+"\n")
            print(f"[result] {directory} accepted={row['accepted']} nit={row.get('iterations')} "
                  f"proxy={row.get('gn_stationarity_proxy')} F={row.get('objective')} "
                  f"diagnostic_logE={row.get('diagnostic_log_evidence')}", flush=True)
    summaries = {}
    for boundary, initialization in cases:
        group = [r for r in rows if r["boundary"] == boundary and r["initialization_case"] == initialization]
        summaries[f"{boundary}/{initialization}"] = summarize_case(group, args.output)
    after = {name: pt.sha256(args.macro_dir / name) for name in before}
    single_unchanged = all(pt.sha256(args.single_step_dir / name) == value
                           for name, value in protocol["single_step_hashes"].items())
    if before != after or not single_unchanged:
        raise RuntimeError("Frozen M3 or single-step inputs changed")
    write_json(args.output / "summary.json", dict(protocol=PROTOCOL, cases=summaries,
        runs=rows, elapsed_seconds=time.perf_counter()-start,
        handoff_unchanged=True, single_step_inputs_unchanged=True,
        smooth_refit_performed=False, hyperparameter_search=False,
        all_cases_fully_passed=all(v["all_repeats_accepted"] and v.get("evidence_repeatability_pass", False)
                                 and v.get("field_repeatability_pass", False) for v in summaries.values())))
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
