"""J0946: freeze M3 source hyperparameters, search only lens lambda4/lambda0.

Source intensities remain free in the original joint nonlinear MAP. The
primary model retains the previously tested fixed zero ring. Original hard
boundary fits are controls, never competing evidence scores for that model.
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
from run_fixed_boundary import zero_ring_problem


HERE=Path(__file__).resolve().parent
PROTOCOL="J0946_fixed_M3_source_hyperparameters_lens_only_outer_v1"


def joint_coordinates(lens_coordinates, source_lambda):
    lens=np.asarray(lens_coordinates,dtype=np.float64)
    if lens.shape!=(2,) or not np.isfinite(lens).all() or not np.isfinite(source_lambda) or source_lambda<=0:
        raise ValueError("Use two finite log10 lens coordinates and positive fixed source lambda")
    return np.r_[np.log10(source_lambda),lens]


class SearchBudgetExceeded(RuntimeError):
    pass


def analytical_audit(q, strengths, macro_dir, boundary, engine):
    reference,_,arrays=pt.load_problem(macro_dir,.1,float(strengths[2]))
    if boundary=="zero_ring":
        reference=zero_ring_problem(reference,arrays,float(strengths[2]))
    point=reference._evaluate_kernel(jnp.asarray(q),float(strengths[0]),float(strengths[1]))
    value,gradient=engine.physical_value_grad(jnp.asarray(q),jnp.asarray(strengths))
    error=abs(float(value-point.objective))
    gerror=float(jnp.linalg.norm(gradient+point.negative_gradient)/jnp.maximum(1.,jnp.linalg.norm(gradient)))
    if not bool(point.finite) or error>1e-6 or gerror>1e-7:
        raise FloatingPointError(f"AD/analytical mismatch: F={error}, g={gerror}")
    return dict(passed=True,objective_absolute_error=error,gradient_relative_error=gerror)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"m3_source_fixed_v1")
    parser.add_argument("--macro-dir",type=Path,default=HERE/"macro")
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--maxiter",type=int,default=3000)
    parser.add_argument("--direct-fits",type=int,default=16)
    parser.add_argument("--local-fits",type=int,default=16)
    parser.add_argument("--probe-only",action="store_true")
    args=parser.parse_args()
    if args.repeats<2 or args.maxiter<1 or min(args.direct_fits,args.local_fits)<1:
        raise ValueError("Use positive budgets and at least two repeats")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    jax.config.update("jax_enable_x64",True)
    jax.config.update("jax_default_matmul_precision","highest")
    start=time.perf_counter()
    hard,metadata,arrays=pt.load_problem(args.macro_dir,.1)
    fixed_source=float(metadata["lambda_source"])
    effective_source=float(10.**np.log10(fixed_source))
    zero=zero_ring_problem(hard,arrays,hard.lambda_amplitude)
    for problem in (hard,zero):
        problem.initialization_provenance.update(source_hyperparameters_fixed=True,
            lambda_source_M3=fixed_source,adaptive_reg_rho_fixed=metadata["adaptive_reg_rho"],
            source_pixels_frozen=False,source_truth_used=False,
            source_initialization="fixed_M3_lambda_conditional_source_MAP",
            source_reference="saved_M3_for_whitening_only")
    before={name:pt.sha256(args.macro_dir/name) for name in ("m3_handoff.json","m3_handoff.npz")}
    single_path=HERE.parent/"single_step/potential_qr_v2/maximum_evidence/result.json"
    previous_path=HERE/"direct_powell/maximum_evidence/result.json"
    single=json.loads(single_path.read_text())["strengths"]
    previous=json.loads(previous_path.read_text())["strengths"]
    probes=dict(strong_anchor=np.array([7.5,6.5]),
        single_step_lens=np.log10([single["lambda_potential"],single["lambda_amplitude"]]),
        previous_iterative_lens=np.log10([previous["lambda4"],previous["lambda0"]]))
    sources=[HERE/name for name in ("run_m3_source_fixed.py","pt.py","lbfgs.py","source_boundary.py","run_fixed_boundary.py")]
    sources += [pt.ROOT/"TinyLensGpu"/name for name in (
        "ForwardSimulation/LensImage/potential_correction.py","utils/lensing/mapping.py",
        "ObservationModel/LensImage/potential_correction_iterative.py",
        "ObservationModel/LensImage/_potential_correction_common.py","utils/inversion/regularization.py",
        "utils/inversion/potential_regularization.py","utils/lensing/potential.py")]
    protocol=dict(protocol=PROTOCOL,created_utc=datetime.now(timezone.utc).isoformat(),arguments=vars(args),
        source_lambda_requested_M3=fixed_source,source_lambda_effective_Float64=effective_source,
        source_log_lambda_natural_M3=metadata["log_lambda_source"],
        source_rho_fixed=metadata["adaptive_reg_rho"],source_scale_fixed=True,
        source_regularization=metadata["source_regularization"],source_template_fixed=True,
        M3_source_hypers_provenance="M2 posterior-median hyperparameters held fixed during M3",
        source_pixels_frozen=False,inner_variables="10000 source + 1134 potential coefficients",
        outer_variables=["log10_lambda4","log10_lambda0"],outer_dimensions=2,
        primary_boundary="one_fixed_zero_ghost_ring",hard_boundary_control="single_step_lens probes only",
        preconditioner="same saved hard-M3 reference, once per candidate, unchanged from prior experiment",
        cold_start="one fixed conditional source MAP per boundary at M3 source lambda; zero potential; no warm starts",
        handoff_hashes=before,probe_lens_log10=probes,
        repeated_probes=args.repeats,qualification=dict(native_success=True,gn_proxy_max=.001),
        repeatability=dict(diagnostic_log_evidence_range_max=.01,kappa_max_absolute_difference_max=1e-4),
        optimizer=dict(method="L-BFGS-B",maxiter=args.maxiter,maxfun=120000,maxcor=30,maxls=80,ftol=1e-15,gtol=1e-8),
        outer_budgets=dict(DIRECT_unique_fits=args.direct_fits,DIRECT_maxiter=3,
            Powell_unique_fits=args.local_fits,Powell_maxfev=args.local_fits,Powell_maxiter=3),
        budget_policy="explicit unique-fit cap; intercepted native calls are reported as interrupted, not converged",
        evidence_method="full coupled endpoint local-GN Gaussian approximation, original source normalization retained",
        smooth_refit_performed=False,real_data_no_injected_truth=True,
        previous_strength_files_sha256={str(p):pt.sha256(p) for p in (single_path,previous_path)},
        source_hashes={str(p.relative_to(pt.ROOT)):pt.sha256(p) for p in sources})
    for path in sources:
        target=args.output/"source_snapshot"/path.relative_to(pt.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write_json(args.output/"protocol.json",protocol)
    engines={"hard":EvidenceEngine(hard,maxiter=args.maxiter)}
    engines["zero_ring"]=EvidenceEngine(zero,maxiter=args.maxiter,reference_engine=engines["hard"])
    initial={name:np.asarray(e.conditional_source(effective_source)) for name,e in engines.items()}
    np.savez_compressed(args.output/"initial_values.npz",hard=initial["hard"],zero_ring=initial["zero_ring"],
        saved_M3_source=arrays["source_values"],source_scale=arrays["source_scale"])
    bounds=engines["hard"].bounds(fixed_source)[1:]
    protocol["lens_log10_bounds"]=bounds
    protocol["configuration"]=hard._configuration()
    protocol["source_scale_sha256_array"]=pt.sha256(args.output/"initial_values.npz")
    write_json(args.output/"protocol.json",protocol)
    rows=[];cache={};best=None
    counters={p:dict(callbacks=0,new_fits=0,cache_hits=0) for p in ("DIRECT","Powell")}

    def evaluate(lens,*,boundary="zero_ring",phase,repeat=None,audit=False,update_best=True):
        nonlocal best
        lens=np.asarray(lens,dtype=np.float64)
        x=joint_coordinates(lens,fixed_source)
        key=(boundary,*np.round(lens,12))
        if phase in counters:
            counters[phase]["callbacks"]+=1
            if key in cache:
                counters[phase]["cache_hits"]+=1
                old=rows[cache[key]]
                return -old["log_evidence"] if old.get("accepted") else 1e30
            limit=args.direct_fits if phase=="DIRECT" else args.local_fits
            if counters[phase]["new_fits"]>=limit:
                raise SearchBudgetExceeded(f"{phase} used {limit} new fits")
            counters[phase]["new_fits"]+=1
        index=len(rows);directory=f"trials/{index:04d}"
        trial=args.output/directory;trial.mkdir(parents=True)
        print(f"[fit {index:03d}] {phase} {boundary} lens={lens}",flush=True)
        engine=engines[boundary]
        try:
            row,q=engine.fit(x,initial_values=initial[boundary],
                initialization_label="fixed_M3_lambda_conditional_source_MAP",audit=audit)
            write_json(trial/"optimizer_result.json",row)
            np.savez_compressed(trial/"optimizer_state.npz",joint_values=q,initial_values=initial[boundary])
            if not np.isclose(row["strengths"]["lambda_s"],effective_source,rtol=1e-14,atol=0):
                raise AssertionError("Source hyperparameter changed during lens-only search")
            if audit:
                row["analytical_audit"]=analytical_audit(q,10.**x,args.macro_dir,boundary,engine)
            row.update(index=index,directory=directory,phase=phase,boundary=boundary,repeat=repeat,
                source_hyperparameters_fixed=True,source_pixels_frozen=False,
                objective_decrease=row["initial_objective"]-row["objective"],
                source_relative_change_from_M3=float(np.linalg.norm(q[:hard.n_source]-arrays["source_values"])/np.linalg.norm(arrays["source_values"])))
            row=pt.export_solution(trial,engine,arrays,metadata,row,q)
            exported=json.loads((trial/"result.json").read_text())
            exported.update(protocol=PROTOCOL,selection_scope="fixed_M3_source_hypers_lens_only_search",
                source_lambda_M3=fixed_source,source_rho_M3=metadata["adaptive_reg_rho"],
                source_boundary=boundary,source_pixels_frozen=False)
            write_json(trial/"result.json",exported)
        except FloatingPointError as exc:
            row=dict(index=index,directory=directory,phase=phase,boundary=boundary,repeat=repeat,
                source_hyperparameters_fixed=True,source_pixels_frozen=False,accepted=False,
                log_evidence=None,failure=str(exc),log10_strengths=x.tolist())
        rows.append(row);cache[key]=index
        with (args.output/"evaluations.jsonl").open("a") as handle:
            handle.write(json.dumps(pt.json_safe(row),allow_nan=False)+"\n")
        if boundary=="zero_ring" and update_best and row.get("accepted") and (best is None or row["log_evidence"]>best["log_evidence"]):
            best=dict(row)
            selected=args.output/"maximum_evidence";selected.mkdir(exist_ok=True)
            for name in ("result.json","reconstruction.npz"):
                shutil.copy2(trial/name,selected/name)
            write_json(args.output/"best.json",best)
        print(f"[result {index:03d}] accepted={row['accepted']} nit={row.get('iterations')} "
            f"proxy={row.get('gn_stationarity_proxy')} logE={row.get('log_evidence')} "
            f"peak={row.get('delta_kappa_peak')}",flush=True)
        return -row["log_evidence"] if row.get("accepted") else 1e30

    # Audit an easy, previously certified-strength anchor before expensive probes.
    for name,lens in probes.items():
        for repeat in range(args.repeats):
            evaluate(lens,phase="probe_"+name,repeat=repeat,audit=repeat==0)
    for repeat in range(args.repeats):
        evaluate(probes["single_step_lens"],boundary="hard",phase="hard_control_single_step_lens",
                 repeat=repeat,audit=repeat==0,update_best=False)
    outer={}
    if not args.probe_only:
        for phase in ("DIRECT","Powell"):
            if phase=="Powell" and best is None:
                outer[phase]=dict(skipped="No accepted primary candidate");continue
            begin=time.perf_counter()
            try:
                fun=lambda lens:evaluate(lens,phase=phase)
                if phase=="DIRECT":
                    result=direct(fun,bounds=bounds.tolist(),maxfun=args.direct_fits,maxiter=3,
                                  locally_biased=False,len_tol=1e-4)
                else:
                    result=minimize(fun,np.asarray(best["log10_strengths"])[1:],method="Powell",bounds=bounds,
                        options=dict(xtol=.01,ftol=1e-6,maxfev=args.local_fits,maxiter=3))
                status=dict(native_result=pt.optimizer_record(result),explicit_cap_interrupted=False)
            except SearchBudgetExceeded as exc:
                status=dict(native_result=None,explicit_cap_interrupted=True,termination=str(exc))
            status.update(counters=counters[phase],wall_seconds=time.perf_counter()-begin)
            outer[phase]=status
            write_json(args.output/(phase.lower()+"_status.json"),status)
    selected_indices=[]
    selected_snapshot=dict(best) if best is not None else None
    if best is not None and not args.probe_only:
        lens=np.asarray(best["log10_strengths"])[1:].copy()
        for repeat in range(args.repeats):
            selected_indices.append(len(rows))
            evaluate(lens,phase="selected_repeat",repeat=repeat,audit=True,update_best=False)
    after={name:pt.sha256(args.macro_dir/name) for name in before}
    if before!=after or any(pt.sha256(path)!=value for path,value in protocol["previous_strength_files_sha256"].items()):
        raise RuntimeError("Frozen source/smooth/reference inputs changed")
    write_json(args.output/"summary.json",dict(protocol=PROTOCOL,rows=rows,best=selected_snapshot,
        selected_repeat_indices=selected_indices,outer=outer,elapsed_seconds=time.perf_counter()-start,
        handoff_unchanged=True,source_hyperparameters_fixed=True,source_pixels_frozen=False,
        smooth_refit_performed=False,search_dimensions=2,probe_only=args.probe_only))
    print(f"[done] {args.output}",flush=True)


if __name__=="__main__":
    main()
