"""Summarize every recorded resolution fit, without rerunning an optimizer."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    summary=json.loads((args.run/"summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("At least one worker failed; inspect all logs before summarizing")
    protocol=json.loads((args.run/"protocol.json").read_text())
    frozen_meta=json.loads((args.run/"frozen_inputs.json").read_text())
    with np.load(args.run/"frozen_inputs.npz") as archive:
        noise,active=archive["noise_map"],~archive["mask"]
    groups={}
    all_rows=[]
    for case,group in summary["groups"].items():
        rows=group["rows"]
        warm=[r for r in rows if r["phase"].startswith("warm_")]
        assert len(warm)==protocol["warm_repeats"]
        env=json.loads((args.run/case/"environment.json").read_text())
        assert env["input_hashes"]==frozen_meta["hashes"] and env["checked_geometry_identical"]
        first=np.load(args.run/rows[0]["directory"]/"fields.npz")
        k0=first["convergence"][first["potential_core_indices"]]
        image0=first["model_image"]
        kdiff,idiff=[],[]
        for row in rows:
            with np.load(args.run/row["directory"]/"fields.npz") as archive:
                kdiff.append(float(np.max(abs(archive["convergence"][archive["potential_core_indices"]]-k0))))
                idiff.append(float(np.sqrt(np.mean(((archive["model_image"]-image0)[active]/noise[active])**2))))
            all_rows.append(row)
        first.close()
        groups[case]=dict(all_fits=len(rows),all_accepted=sum(r["accepted"] for r in rows),
            warm_fits=len(warm),warm_accepted=sum(r["accepted"] for r in warm),
            warm_iterations=[r["iterations"] for r in warm],
            median_warm_iterations=statistics.median(r["iterations"] for r in warm),
            median_warm_timings={key:statistics.median(r["timings"][key] for r in warm)
                                 for key in warm[0]["timings"]},
            warm_reconstruction_range=[min(r["timings"]["reconstruction_seconds"] for r in warm),
                                       max(r["timings"]["reconstruction_seconds"] for r in warm)],
            cold_reconstruction_seconds=rows[0]["timings"]["reconstruction_seconds"],
            proxy_range=[min(r["gn_stationarity_proxy"] for r in rows),max(r["gn_stationarity_proxy"] for r in rows)],
            legacy_gradient_passes=sum(r["legacy_gradient_pass"] for r in rows),
            kappa_peak_range=[min(r["kappa_peak"] for r in rows),max(r["kappa_peak"] for r in rows)],
            residual_rms_noise_range=[min(r["residual_rms_noise"] for r in rows),max(r["residual_rms_noise"] for r in rows)],
            warm_objective_range=[min(r["objective"] for r in warm),max(r["objective"] for r in warm)],
            maximum_repeat_kappa_difference_from_cold=max(kdiff),maximum_repeat_model_rms_in_noise_from_cold=max(idiff),
            dimensions=rows[0]["dimensions"])
    comparisons={}
    for profile in protocol["profiles"]:
        small,large=groups[f"{profile}/n49"],groups[f"{profile}/n100"]
        with np.load(args.run/profile/"n49/cold/fields.npz") as a, np.load(args.run/profile/"n100/cold/fields.npz") as b:
            assert np.array_equal(a["potential_x"],b["potential_x"])
            assert np.array_equal(a["potential_y"],b["potential_y"])
            kdiff=b["convergence"][b["potential_core_indices"]]-a["convergence"][a["potential_core_indices"]]
            comparisons[profile]=dict(
                reconstruction_time_ratio=large["median_warm_timings"]["reconstruction_seconds"]/small["median_warm_timings"]["reconstruction_seconds"],
                lbfgs_time_ratio=large["median_warm_timings"]["lbfgs_seconds"]/small["median_warm_timings"]["lbfgs_seconds"],
                median_iteration_ratio=large["median_warm_iterations"]/small["median_warm_iterations"],
                source_parameters_ratio=10000/2401,joint_parameters_ratio=10956/3357,
                model_difference_rms_in_noise=float(np.sqrt(np.mean(((b["model_image"]-a["model_image"])[active]/noise[active])**2))),
                core_kappa_difference_rms=float(np.sqrt(np.mean(kdiff**2))),
                core_kappa_difference_max_abs=float(np.max(abs(kdiff))))
    result=dict(groups=groups,comparisons=comparisons,all_fits=len(all_rows),
        accepted_fits=sum(r["accepted"] for r in all_rows),same_frozen_inputs_and_potential=True,
        maximum_analytical_gradient_relative_error=max(r["analytical_gradient_relative_error"] for r in all_rows),
        maximum_analytical_objective_error=max(r["analytical_objective_absolute_error"] for r in all_rows),
        maximum_whitening_quadratic_error=max(r["whitening_quadratic_error"] for r in all_rows),
        maximum_source_initial_equation_error=max(r["source_initial_equation_error"] for r in all_rows))
    # A fixed physical test field audits the discretized prior independently
    # of any fitted source or reconstruction-quality selection.
    audit=[]
    for n in (49,100):
        x,y=np.meshgrid(np.linspace(-.4,.4,n),np.linspace(-.4,.4,n))
        s=np.exp(-.5*((x/.16)**2+(y/.12)**2))
        h=.8/(n-1);r=48/(n-1)
        raw=dict(curvature=float((np.sum(np.diff(s,n=2,axis=1)**2)+np.sum(np.diff(s,n=2,axis=0)**2))/h**4),
                 boundary_gradient=float((np.sum((s[:,-1]-s[:,-2])**2)+np.sum((s[-1,:]-s[-2,:])**2))/h**4),
                 boundary_amplitude=float((np.sum(s[:,-1]**2)+np.sum(s[-1,:]**2))/h**4))
        matched={k:raw[k]*r**power for k,power in [("curvature",2),("boundary_gradient",3),("boundary_amplitude",5)]}
        audit.append(dict(source_n=n,raw_components=raw,matched_components=matched,
                          raw_total=sum(raw.values()),matched_total=sum(matched.values())))
    result["fixed_physical_gaussian_prior_audit"]=audit
    repo=Path(__file__).resolve().parents[2]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    unchanged=all(sha(repo/path)==value for path,value in protocol["sources_sha256"].items())
    assert unchanged,"A measured source file changed; preserve the snapshot before interpreting results"
    assert sha(args.run/"frozen_inputs.npz")==summary["frozen_input_sha256"]==frozen_meta["input_file_sha256"]
    selected=Path(protocol["profiles_provenance"]["evidence_selected"])
    assert sha(selected)==protocol["profiles_provenance"]["selected_file_sha256"]
    result["measured_sources_and_frozen_inputs_unchanged"]=unchanged
    with np.load(args.run/"frozen_inputs.npz") as a, np.load(selected.parent/"baseline_fields.npz") as b:
        result["matches_previous_mock_saved_arrays"]={k:bool(np.array_equal(a[k],b[k]))
            for k in ("image_data","noise_map","potential_x","potential_y")}
        assert all(result["matches_previous_mock_saved_arrays"].values())
    (args.run/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    flat=[dict(profile=r["profile"],source_n=r["source_n"],phase=r["phase"],accepted=r["accepted"],
               native_success=r["native_success"],iterations=r["iterations"],nfev=r["nfev"],
               objective=r["objective"],proxy=r["gn_stationarity_proxy"],kappa_peak=r["kappa_peak"],**r["timings"])
          for r in all_rows]
    with (args.run/"all_fits.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
