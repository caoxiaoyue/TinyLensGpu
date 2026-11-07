"""J0946 nonlinear L-BFGS potential correction, DIRECT + bounded Powell.

Run from any directory in the tinylens_gpu environment. This entry point only
reads the copied M3 handoff; it never runs or imports macro_model.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.optimize import direct, minimize

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from TinyLensGpu.PhysicalModel import EPL, PhysicalModel, Shear
from TinyLensGpu.example_support import resolve_source_interpolation
from examples.potential_correction.result_products import json_safe, write_json
from lbfgs import EvidenceEngine

HERE = Path(__file__).resolve().parent
PROTOCOL = "J0946_M3_joint_AD_LBFGS_full_data_GN_evidence_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def optimizer_record(result):
    if result is None:
        return None
    return {key: getattr(result, key, None)
            for key in ("success", "status", "message", "nfev", "nit", "x", "fun")}


def load_handoff(macro_dir):
    macro_dir = Path(macro_dir)
    for name in ("m3_handoff.json", "m3_handoff.npz"):
        if not (macro_dir / name).is_file():
            raise FileNotFoundError(f"Missing copied M3 handoff: {macro_dir / name}; smooth fit is never run automatically")
    metadata = json.loads((macro_dir / "m3_handoff.json").read_text())
    if not metadata.get("complete") or sha256(macro_dir / "m3_handoff.npz") != metadata["handoff_sha256"]:
        raise ValueError("Incomplete or checksum-mismatched M3 handoff")
    with np.load(macro_dir / "m3_handoff.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    scale = arrays["source_scale"]
    if scale.shape != (metadata["source_n"]**2,) or not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("M3 adaptive source_scale must be finite, positive and match source nodes")
    return metadata, arrays


def load_problem(macro_dir=HERE / "macro", potential_spacing=0.1, lambda_amplitude=1.0):
    metadata, arrays = load_handoff(macro_dir)
    interpolation = resolve_source_interpolation(metadata=metadata, context="M3 handoff")
    p = metadata["macro_parameters"]
    # Preserve the saved native Float32 image sampling, then promote it. The
    # source bbox/node count and macro parameters are the frozen M3 definition.
    with jax.experimental.enable_x64(False):
        config = SimulatorConfig(
            metadata["dpix"], arrays["mask"].shape[0],
            jnp.asarray(arrays["psf_kernel"], dtype=jnp.float32),
            nsub=metadata["nsub"], mask=arrays["mask"],
        )
    macro = PhysicalModel(lens_mass=[
        EPL(theta_E=p["theta_E"], gamma=p["gamma"], e1=p["e1_mass"],
            e2=p["e2_mass"], center_x=p["center_x_mass"], center_y=p["center_y_mass"]),
        Shear(gamma1=p["gamma1"], gamma2=p["gamma2"]),
    ], source_light=[], lens_light=[])
    renderer = PotentialCorrectionRenderer(
        macro, config, metadata["source_n"], tuple(arrays["source_bbox"]),
        potential_spacing=potential_spacing, dtype="float64",
        source_interpolation=interpolation,
    )
    problem = IterativePotentialCorrection(renderer, arrays["lens_subtracted_image"],
        arrays["noise_map"], arrays["source_values"], lambda_amplitude=lambda_amplitude,
        source_regularization=metadata["source_regularization"],
        initialization_provenance=dict(handoff_sha256=metadata["handoff_sha256"],
            representative=metadata["representative"], source_truth_used=False))
    # Workspace adapter for the constructor's absent source_scale keyword.
    # Set before any JIT tracing or dense prior construction, also in audits.
    with jax.experimental.enable_x64():
        problem.source_scale = jnp.asarray(arrays["source_scale"], dtype=jnp.float64)
    problem.initialization_provenance.update(
        source_initialization="saved_M3_source",
        gradient_policy=f"JAX_autodiff_current_{interpolation}_source",
        adaptive_scale="fixed_M1_template_and_M2_rho_as_used_in_M3",
    )
    return problem, metadata, arrays


def analytical_audit(q, strengths, macro_dir, potential_spacing, engine):
    # Fresh instance because lambda_amplitude is captured by the public JIT.
    reference, _, _ = load_problem(macro_dir, potential_spacing, strengths[2])
    point = reference._evaluate_kernel(jnp.asarray(q), strengths[0], strengths[1])
    value, gradient = engine.physical_value_grad(jnp.asarray(q), jnp.asarray(strengths))
    value_error = abs(float(value - point.objective))
    gradient_error = float(jnp.linalg.norm(gradient + point.negative_gradient) /
                           jnp.maximum(jnp.linalg.norm(gradient), 1.0))
    if not bool(point.finite) or value_error > 1e-6 or gradient_error > 1e-7:
        raise FloatingPointError(f"AD/public analytic audit failed: F={value_error}, g={gradient_error}")
    return dict(objective_absolute_error=value_error, gradient_relative_error=gradient_error,
                adaptive_scale_preserved=True, passed=True)


def export_solution(output, engine, arrays, metadata, row, q):
    problem = engine.problem
    source, potential = q[:problem.n_source], q[problem.n_source:]
    initial_source = engine.last_initial_values[:problem.n_source]
    rendered = problem.renderer.render(source, potential, macro_mapping=problem.macro_mapping)
    grid = problem.renderer.potential_grid
    source_model = np.asarray(rendered.model_image)
    residual = np.asarray(problem.data_active - rendered.image_active)
    full_residual = np.full(problem.renderer.image_shape, np.nan)
    full_residual.ravel()[np.asarray(problem.renderer.flat_indices)] = residual / np.asarray(problem.noise_active)
    kappa = np.asarray(rendered.convergence)
    finite = np.isfinite(kappa)
    peak = int(np.nanargmax(kappa))
    row = dict(row, delta_kappa_peak=float(kappa[peak]),
        delta_kappa_peak_xy_arcsec=[float(grid.x[peak]), float(grid.y[peak])],
        delta_kappa_min=float(np.nanmin(kappa)), delta_kappa_core_rms=float(np.sqrt(np.mean(kappa[finite]**2))))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "reconstruction.npz",
        joint_values=q, source_values=source, potential_values=potential,
        source_x_axis=np.asarray(problem.renderer.source_x_axis),
        source_y_axis=np.asarray(problem.renderer.source_y_axis),
        source_scale=arrays["source_scale"], source_bbox=arrays["source_bbox"],
        potential_x=np.asarray(grid.x), potential_y=np.asarray(grid.y),
        potential_node_indices=np.asarray(grid.node_indices),
        derivative_valid=np.asarray(rendered.derivative_valid), convergence=kappa,
        deflection_x=np.asarray(rendered.deflection_x), deflection_y=np.asarray(rendered.deflection_y),
        data_image=arrays["image_data"], lens_subtracted_image=arrays["lens_subtracted_image"],
        lens_light_model=arrays["lens_light_model"], noise_map=arrays["noise_map"], mask=arrays["mask"],
        nonlinear_source_model_image=source_model,
        model_image=source_model + arrays["lens_light_model"].astype(np.float64),
        normalized_residual=full_residual, macro_source_model_image=arrays["macro_model_image"],
        source_initial=initial_source, source_reference_M3=arrays["source_values"],
    )
    write_json(output / "result.json", dict(
        protocol=PROTOCOL, lens="SDSS J0946+1006 (Jackpot)", **row,
        configuration=problem._configuration(), initialization=problem.initialization_provenance,
        evidence_method="local_Gauss_Newton_Gaussian_approximation",
        evidence_is_exact=False, determinant_method="full_data_covariance_cholesky",
        selection_scope="highest_evidence_among_numerically_accepted_evaluated_points",
        fixed_macro_parameters=metadata["macro_parameters"],
        real_data_no_injected_truth=True, smooth_refit_performed=False,
        note="Conditional on fixed macro/lens light/grid/adaptive map; peak is not detection significance or a mass.",
    ))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-dir", type=Path, default=HERE / "macro")
    parser.add_argument("--output", type=Path, default=HERE / "potential")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-log10", nargs=3, type=float,
                        metavar=("SOURCE", "FOURTH", "ZERO"))
    parser.add_argument("--direct-maxfun", type=int, default=120)
    parser.add_argument("--local-maxfun", type=int, default=80)
    parser.add_argument("--inner-maxiter", type=int, default=3000)
    parser.add_argument("--proxy-tolerance", type=float, default=1e-3)
    parser.add_argument("--potential-spacing", type=float, default=0.1)
    parser.add_argument("--source-initialization", choices=("conditional_source", "saved_M3"),
                        default="conditional_source")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a fresh --output directory; refusing to replace {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    start = time.perf_counter()
    problem, metadata, arrays = load_problem(args.macro_dir, args.potential_spacing)
    hashes = {name: sha256(args.macro_dir / name) for name in ("m3_handoff.json", "m3_handoff.npz")}
    initial_model = np.asarray(jax.jit(problem.forward)(problem.initial_values))
    saved_model = arrays["macro_model_image"].ravel()[np.asarray(problem.renderer.flat_indices)]
    initial_residual = (np.asarray(problem.data_active) - initial_model) / np.asarray(problem.noise_active)
    engine = EvidenceEngine(problem, maxiter=args.inner_maxiter, proxy_tolerance=args.proxy_tolerance,
                            source_initialization=args.source_initialization)
    problem.initialization_provenance.update(
        source_initialization=args.source_initialization,
        source_reference="saved_M3_source_for_fixed_whitening_metric",
        conditional_source_rule="exact source MAP with candidate lambda_s and zero potential"
            if args.source_initialization == "conditional_source" else None)
    bounds = engine.bounds(metadata["lambda_source"])
    protocol = dict(protocol=PROTOCOL, created_utc=datetime.now(timezone.utc).isoformat(),
        arguments=vars(args), configuration=problem._configuration(),
        jax_version=jax.__version__, scipy_version=scipy.__version__, numpy_version=np.__version__,
        devices=[str(d) for d in jax.devices()], handoff_hashes=hashes,
        initialization=problem.initialization_provenance, log10_bounds=bounds,
        bounds_rule="lambda_s M3 +/-2 dex; lambda4 dataGram/R4 -3,+6 dex; lambda0 dataGram +/-3 dex",
        initial_saved_M3_forward_max_abs_difference=float(np.max(np.abs(initial_model - saved_model))),
        initial_chi2=float(initial_residual @ initial_residual),
        preparation_seconds=engine.preparation_seconds,
        inner_options=dict(maxcor=30, maxls=80, ftol=1e-15, gtol=1e-8,
                           maxiter=args.inner_maxiter, maxfun=120000),
        local_options=dict(method="Powell", xtol=0.01, ftol=1e-6, maxfev=args.local_maxfun),
        acceptance="native success + finite SPD evidence + GN stationarity proxy <= tolerance",
        evidence="full coupled local-GN Gaussian approximation, no damping/jitter",
        coordinate_policy="once per hyperparameter candidate, exact GN low-rank whitening at fixed M3 reference",
        cold_start=args.source_initialization + "; zero potential, independent of other candidates",
        smooth_refit_performed=False,
    )
    source_files = [HERE / "pt.py", HERE / "lbfgs.py", HERE / "macro_model.py",
        ROOT / "TinyLensGpu/ForwardSimulation/LensImage/potential_correction.py",
        ROOT / "TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py",
        ROOT / "TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py",
        ROOT / "TinyLensGpu/utils/inversion/regularization.py",
        ROOT / "TinyLensGpu/utils/inversion/potential_regularization.py",
        ROOT / "TinyLensGpu/utils/lensing/mapping.py",
        ROOT / "TinyLensGpu/utils/lensing/potential.py"]
    protocol["source_hashes"] = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
    for path in source_files:
        target = args.output / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(args.output / "protocol.json", protocol)
    print(f"[J0946] {problem._configuration()}\n[search] log10 bounds={bounds.tolist()}", flush=True)
    rows, best = [], None
    cache = {}
    phase = "baseline"

    def evaluate(x, *, audit=False, update_best=True):
        nonlocal best
        key = tuple(np.round(x, 12))
        if key in cache and not audit:
            old = rows[cache[key]]
            return -old["log_evidence"] if old.get("accepted") else 1e30
        index = len(rows)
        try:
            row, q = engine.fit(x, audit=audit)
            if audit:
                row["analytical_audit"] = analytical_audit(q, 10.0**np.asarray(x),
                                                          args.macro_dir, args.potential_spacing, engine)
            row.update(index=index, phase=phase)
            np.savez_compressed(args.output / f"candidate_{index:04d}.npz", joint_values=q,
                                initial_joint_values=engine.last_initial_values)
            if update_best and row["accepted"] and (best is None or row["log_evidence"] > best["log_evidence"]):
                best = export_solution(args.output / "maximum_evidence", engine, arrays, metadata, row, q)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            row = dict(index=index, phase=phase, log10_strengths=np.asarray(x).tolist(),
                       accepted=False, log_evidence=None, failure=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        cache[key] = index
        with (args.output / "evaluations.jsonl").open("a") as handle:
            handle.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
        score = row.get("log_evidence")
        print(f"[{phase} {index:03d}] x={np.round(x, 4)} accepted={row['accepted']} "
              f"logE={score} nit={row.get('iterations')} proxy={row.get('gn_stationarity_proxy')} "
              f"seconds={row.get('total_seconds')}", flush=True)
        return -score if row["accepted"] else 1e30

    baseline_x = (np.asarray([bounds[0].mean(), bounds[1, 1], bounds[2, 1]])
                  if args.validation_log10 is None else np.asarray(args.validation_log10))
    if np.any(baseline_x < bounds[:, 0]) or np.any(baseline_x > bounds[:, 1]):
        raise ValueError("Validation point must lie inside the declared search bounds")
    evaluate(baseline_x, audit=True)
    if not rows[-1].get("algebra_audit") or not rows[-1].get("analytical_audit"):
        write_json(args.output / "failure.json", rows[-1])
        raise RuntimeError("Baseline algebra/AD validation failed; inspect before searching")
    outer_direct, outer_local = None, None
    selected_repeat_passed = None
    if not args.validate_only:
        phase = "DIRECT"
        outer_direct = direct(evaluate, bounds=bounds.tolist(), maxfun=args.direct_maxfun,
                              maxiter=1000, locally_biased=False, len_tol=1e-4)
        write_json(args.output / "direct_status.json", optimizer_record(outer_direct))
        if best is None:
            write_json(args.output / "failure.json", dict(
                reason="No numerically accepted candidate after DIRECT", evaluations=rows,
                direct=dict(success=bool(outer_direct.success), status=int(outer_direct.status),
                            message=str(outer_direct.message), nfev=int(outer_direct.nfev),
                            x=outer_direct.x, fun=float(outer_direct.fun))))
            write_json(args.output / "summary.json", dict(
                status="no_certified_candidate", best=None,
                unique_evaluations=len(rows), accepted_evaluations=0,
                rejected_evaluations=len(rows), total_wall_seconds=time.perf_counter()-start,
                local=None, smooth_refit_performed=False,
                handoff_unchanged=all(sha256(args.macro_dir / name) == value for name, value in hashes.items())))
            raise RuntimeError("No accepted inner solutions; inspect evaluations.jsonl")
        phase = "Powell"
        outer_local = minimize(evaluate, np.asarray(best["log10_strengths"]), method="Powell",
            bounds=bounds, options=dict(xtol=0.01, ftol=1e-6,
                                       maxfev=args.local_maxfun, maxiter=12))
        write_json(args.output / "local_status.json", optimizer_record(outer_local))
        # Re-evaluate selected coordinates from the same cold start, with QR and
        # public analytical audits, independently of the retained best state.
        phase = "selected_audit"
        selected_x = np.asarray(best["log10_strengths"])
        selected_log_evidence = best["log_evidence"]
        evaluate(selected_x, audit=True, update_best=False)
        final = rows[-1]
        selected_repeat_passed = bool(final.get("accepted") and
            abs(final["log_evidence"] - selected_log_evidence) <= 0.01)
        if not selected_repeat_passed:
            write_json(args.output / "failure.json", dict(
                reason="Selected cold-start repeat/audit did not reproduce retained evidence",
                original_selected_log_evidence=selected_log_evidence, repeat=final))
    hashes_after = {name: sha256(args.macro_dir / name) for name in hashes}
    if hashes_after != hashes:
        raise RuntimeError("M3 handoff changed during correction")

    write_json(args.output / "summary.json", dict(
        protocol=PROTOCOL, validate_only=args.validate_only, best=best,
        recorded_fit_evaluations=len(rows), unique_evaluations=len(cache),
        accepted_evaluations=sum(r["accepted"] for r in rows),
        rejected_evaluations=sum(not r["accepted"] for r in rows),
        direct=optimizer_record(outer_direct), local=optimizer_record(outer_local),
        total_wall_seconds=time.perf_counter() - start,
        selected_audit=None if args.validate_only else {
            key: rows[-1].get(key) for key in (
                "accepted", "log_evidence", "algebra_audit", "analytical_audit")},
        selected_repeat_passed=selected_repeat_passed,
        handoff_unchanged=hashes_after == hashes, smooth_refit_performed=False,
        selected_distance_to_log10_bounds=None if best is None else
            np.minimum(np.asarray(best["log10_strengths"]) - bounds[:, 0],
                       bounds[:, 1] - np.asarray(best["log10_strengths"])),
    ))
    print(f"[done] best logE={None if best is None else best['log_evidence']}; output={args.output}", flush=True)
    if selected_repeat_passed is False:
        raise RuntimeError("Selected repeat failed; complete status saved in summary.json and failure.json")


if __name__ == "__main__":
    main()
