"""Audit fixed-M3-source trials and write an explicit selected validation sidecar."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    protocol=json.loads((args.run/"protocol.json").read_text())
    summary=json.loads((args.run/"summary.json").read_text())
    rows=summary["rows"]
    repo=Path(__file__).resolve().parents[2]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(sha(repo/p)==h for p,h in protocol["source_hashes"].items())
    macro=Path(protocol["arguments"]["macro_dir"])
    assert all(sha(macro/p)==h for p,h in protocol["handoff_hashes"].items())
    assert all(sha(Path(p))==h for p,h in protocol["previous_strength_files_sha256"].items())
    # The runner's historical key hashes the entire initial-values NPZ, not
    # just its scale array; audit the latter explicitly below as well.
    assert sha(args.run/"initial_values.npz")==protocol["source_scale_sha256_array"]
    metadata=json.loads((macro/"m3_handoff.json").read_text())
    with np.load(macro/"m3_handoff.npz") as archive:
        m3={k:archive[k] for k in archive.files}
    with np.load(args.run/"initial_values.npz") as archive:
        initial={k:archive[k] for k in archive.files}
    np.testing.assert_array_equal(initial["source_scale"],m3["source_scale"])
    for boundary in ("hard","zero_ring"):
        assert initial[boundary].shape==(11134,) and np.all(initial[boundary][10000:]==0)
    assert np.isclose(protocol["source_lambda_requested_M3"],np.exp(metadata["log_lambda_source"]),rtol=1e-14)
    assert protocol["source_rho_fixed"]==metadata["adaptive_reg_rho"]
    groups={};source_changes=[];verified=0
    for row in rows:
        assert row["source_hyperparameters_fixed"] and not row["source_pixels_frozen"]
        if "strengths" not in row:
            continue
        ls=row["strengths"]["lambda_s"]
        assert np.isclose(ls,metadata["lambda_source"],rtol=1e-14,atol=0)
        with np.load(args.run/row["directory"]/"reconstruction.npz") as archive:
            np.testing.assert_array_equal(archive["source_scale"],m3["source_scale"])
            np.testing.assert_array_equal(archive["source_bbox"],m3["source_bbox"])
            np.testing.assert_array_equal(archive["mask"],m3["mask"])
            np.testing.assert_array_equal(archive["noise_map"],m3["noise_map"])
            np.testing.assert_array_equal(archive["lens_subtracted_image"],m3["lens_subtracted_image"])
            np.testing.assert_array_equal(archive["source_initial"],initial[row["boundary"]][:10000])
            change=float(np.linalg.norm(archive["source_values"]-m3["source_values"])/np.linalg.norm(m3["source_values"]))
            assert abs(change-row["source_relative_change_from_M3"])<1e-12
            source_changes.append(change);verified+=1
    names=sorted(set((r["boundary"],r["phase"]) for r in rows))
    for boundary,phase in names:
        group=[r for r in rows if r["boundary"]==boundary and r["phase"]==phase]
        finite=[r for r in group if "objective" in r]
        item=dict(attempts=len(group),accepted=sum(r["accepted"] for r in group),
            native_success=sum(r.get("native_success",False) for r in group),failures=[r.get("failure") for r in group if "failure" in r])
        if finite:
            item.update(median_iterations=statistics.median(r["iterations"] for r in finite),
                median_inner_seconds=statistics.median(r["inner_seconds"] for r in finite),
                proxy_range=[min(r["gn_stationarity_proxy"] for r in finite),max(r["gn_stationarity_proxy"] for r in finite)],
                objective_range=[min(r["objective"] for r in finite),max(r["objective"] for r in finite)],
                diagnostic_evidence_range=[min(r["diagnostic_log_evidence"] for r in finite),max(r["diagnostic_log_evidence"] for r in finite)],
                kappa_peak_range=[min(r["delta_kappa_peak"] for r in finite),max(r["delta_kappa_peak"] for r in finite)],
                source_relative_change_from_M3_range=[min(r["source_relative_change_from_M3"] for r in finite),max(r["source_relative_change_from_M3"] for r in finite)])
        groups[f"{boundary}/{phase}"]=item
    best=summary["best"]
    selected_validation=None
    if best is not None and summary["selected_repeat_indices"]:
        selected=[best]+[rows[i] for i in summary["selected_repeat_indices"]]
        with np.load(args.run/best["directory"]/"reconstruction.npz") as archive:
            k0=archive["convergence"].copy()
        differences=[]
        for row in selected:
            path=args.run/row["directory"]/"reconstruction.npz"
            if path.exists():
                with np.load(path) as archive:
                    differences.append(float(np.nanmax(abs(archive["convergence"]-k0))))
        complete=all("diagnostic_log_evidence" in r for r in selected) and len(differences)==len(selected)
        erange=float(np.ptp([r["diagnostic_log_evidence"] for r in selected])) if complete else None
        kdiff=max(differences) if complete else None
        accepted=all(r["accepted"] for r in selected)
        passed=bool(complete and accepted and erange<=.01 and kdiff<=1e-4)
        selected_validation=dict(status="passed" if passed else "repeat_gate_failed",passed=passed,
            all_attempts_accepted=accepted,attempts=len(selected),diagnostic_evidence_range=erange,
            maximum_kappa_difference_from_original=kdiff,records=[{k:r.get(k) for k in
                ("index","phase","repeat","accepted","native_success","iterations","gn_stationarity_proxy","diagnostic_log_evidence","directory")} for r in selected])
        (args.run/"maximum_evidence/validation.json").write_text(json.dumps(selected_validation,indent=2)+"\n")
    finite=[r for r in rows if "objective" in r]
    audits=[r["analytical_audit"] for r in finite if "analytical_audit" in r]
    result=dict(source_lambda_fixed=metadata["lambda_source"],rho_fixed=metadata["adaptive_reg_rho"],
        source_pixels_frozen=False,source_normalized_prior_retained=True,outer_dimensions=2,
        source_and_reference_hashes_unchanged=True,verified_export_count=verified,
        source_relative_change_from_M3_range=[min(source_changes),max(source_changes)] if source_changes else None,
        total_fits=len(rows),accepted=sum(r["accepted"] for r in rows),
        rejected=sum(not r["accepted"] for r in rows),groups=groups,
        best={k:best.get(k) for k in ("index","strengths","log_evidence","gn_stationarity_proxy","delta_kappa_peak",
            "delta_kappa_peak_xy_arcsec","objective","chi2","source_relative_change_from_M3","directory")} if best else None,
        selected_validation=selected_validation,outer=summary["outer"],elapsed_seconds=summary["elapsed_seconds"],
        max_audited_objective_error=max((a["objective_absolute_error"] for a in audits),default=None),
        max_audited_gradient_relative_error=max((a["gradient_relative_error"] for a in audits),default=None))
    (args.run/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
