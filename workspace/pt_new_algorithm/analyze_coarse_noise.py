"""Audit every high-noise fit against its frozen data and injected fields."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from coarse_noise import add_matched_noise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    protocol=json.loads((args.run/"protocol.json").read_text())
    summary=json.loads((args.run/"summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Some worker failed: preserve and inspect all raw endpoints/logs")
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    repo=Path(__file__).resolve().parents[2]
    assert all(sha(repo/p)==h for p,h in protocol["sources_sha256"].items())
    reference=Path(protocol["reference"])
    assert all(sha(reference/p)==h for p,h in protocol["reference_files_sha256"].items())
    assert sha(args.run/"frozen_inputs.npz")==protocol["frozen_input_sha256"]==sha(reference/"frozen_inputs.npz")
    assert sha(args.run/"coarse_base.npz")==protocol["coarse_base_sha256"]
    assert sha(args.run/"postfit_truth.npz")==protocol["truth_sha256"]
    with np.load(args.run/"coarse_base.npz") as a:
        base={k:a[k] for k in a.files}
    with np.load(args.run/"frozen_inputs.npz") as a:
        fine_active=~a["mask"]
        fine_data=a["image_data"][fine_active];fine_sigma=a["noise_map"][fine_active]
    np.testing.assert_allclose(base["data"],fine_data[base["indices"]].mean(axis=1),atol=1e-14)
    np.testing.assert_allclose(base["sigma"],np.sqrt((fine_sigma[base["indices"]]**2).sum(axis=1))/9,atol=1e-16)
    with np.load(args.run/"postfit_truth.npz") as a:
        truth={k:a[k] for k in a.files}
    noise_inputs={}
    for seed in protocol["extra_noise_seeds"]:
        path=args.run/f"noise_seed_{seed:02d}.npz"
        assert sha(path)==protocol["noise_inputs"][str(seed)]["sha256"]
        with np.load(path) as a:
            values={k:a[k] for k in a.files}
        regenerated=add_matched_noise(base["data"],base["sigma"],protocol["target_sigma"],seed)
        for key in regenerated:
            np.testing.assert_array_equal(values[key],regenerated[key])
        np.testing.assert_allclose(values["base_sigma"]**2+values["extra_sigma"]**2,values["sigma"]**2,atol=1e-20)
        noise_inputs[seed]=values
    groups={};all_rows=[];errors=[];quality=[]
    center=np.array([protocol["configuration"]["mock"]["subhalo_x"],protocol["configuration"]["mock"]["subhalo_y"]])
    for name,group in summary["groups"].items():
        rows=group["rows"];warm=[r for r in rows if r["phase"].startswith("warm_")]
        assert len(warm)==protocol["warm_repeats"]
        seed=rows[0]["noise_seed"];noisy=noise_inputs[seed]
        env=json.loads((args.run/name/"environment.json").read_text())
        assert env["noise_input_sha256"]==protocol["noise_inputs"][str(seed)]["sha256"]
        assert env["n_observations"]==2312 and env["checked_geometry_identical"]
        with np.load(args.run/rows[0]["directory"]/"fields.npz") as a:
            k0=a["convergence"][a["potential_core_indices"]]
        kd=[];source_errors=[];core_errors=[];local_errors=[];distances=[]
        for row in rows:
            with np.load(args.run/row["directory"]/"fields.npz") as a:
                np.testing.assert_array_equal(a["observed_data"],noisy["data"])
                np.testing.assert_array_equal(a["observed_noise"],noisy["sigma"])
                np.testing.assert_array_equal(a["observation_indices"],base["indices"])
                model=a["fine_model_image"][fine_active][base["indices"]].mean(axis=1)
                actual=a["model_image"][~a["observation_mask"]]
                error=float(np.max(abs(model-actual)));assert error<1e-12;errors.append(error)
                expected_rms=float(np.sqrt(np.mean(((noisy["data"]-model)/noisy["sigma"])**2)))
                assert abs(expected_rms-row["residual_rms_noise"])<1e-12
                core=a["potential_core_indices"]
                kappa=a["convergence"][core]
                dk=kappa-truth["kappa"]
                se=float(np.linalg.norm(a["source_values"]-truth["source"])/np.linalg.norm(truth["source"]))
                ke=float(np.sqrt(np.mean(dk**2)))
                le=float(np.sqrt(np.mean(dk[truth["local_core"]]**2)))
                peak_distance=float(np.linalg.norm(np.asarray(row["kappa_peak_xy"])-center))
                source_errors.append(se);core_errors.append(ke);local_errors.append(le);distances.append(peak_distance)
                kd.append(float(np.max(abs(kappa-k0))))
                quality.append(dict(profile=row["profile"],noise_seed=seed,phase=row["phase"],
                    source_relative_truth_error=se,core_kappa_truth_rmse=ke,local_kappa_truth_rmse=le,
                    kappa_peak=row["kappa_peak"],peak_distance_arcsec=peak_distance))
                assert len(a["joint_values"])==10956 and len(actual)==2312
            all_rows.append(row)
        groups[name]=dict(runs=len(rows),accepted=sum(r["accepted"] for r in rows),
            native_successes=sum(r["native_success"] for r in rows),warm_runs=len(warm),
            warm_accepted=sum(r["accepted"] for r in warm),
            warm_iterations=[r["iterations"] for r in warm],median_warm_iterations=statistics.median(r["iterations"] for r in warm),
            median_warm_timings={k:statistics.median(r["timings"][k] for r in warm) for k in warm[0]["timings"]},
            median_warm_verified_seconds=statistics.median(r["timings"]["reconstruction_seconds"]+r["timings"]["endpoint_qualification_seconds"] for r in warm),
            proxy_range=[min(r["gn_stationarity_proxy"] for r in rows),max(r["gn_stationarity_proxy"] for r in rows)],
            legacy_gradient_passes=sum(r["legacy_gradient_pass"] for r in rows),
            peak_range=[min(r["kappa_peak"] for r in rows),max(r["kappa_peak"] for r in rows)],
            peak_positions=sorted(set(tuple(r["kappa_peak_xy"]) for r in rows)),
            peak_distance_range=[min(distances),max(distances)],
            residual_rms_range=[min(r["residual_rms_noise"] for r in rows),max(r["residual_rms_noise"] for r in rows)],
            source_truth_error_range=[min(source_errors),max(source_errors)],
            core_kappa_truth_rmse_range=[min(core_errors),max(core_errors)],
            local_kappa_truth_rmse_range=[min(local_errors),max(local_errors)],
            maximum_technical_repeat_kappa_difference_from_cold=max(kd))
    profiles={}
    previous=json.loads((reference/"analysis.json").read_text())
    for profile in protocol["profiles"]:
        rows=[r for r in all_rows if r["profile"]==profile]
        warm=[r for r in rows if r["phase"].startswith("warm_")]
        cold_quality=[q for q in quality if q["profile"]==profile and q["phase"]=="cold"]
        profiles[profile]=dict(runs=len(rows),accepted=sum(r["accepted"] for r in rows),
            warm_runs=len(warm),warm_accepted=sum(r["accepted"] for r in warm),
            median_warm_iterations=statistics.median(r["iterations"] for r in warm),
            median_warm_lbfgs_seconds=statistics.median(r["timings"]["lbfgs_seconds"] for r in warm),
            median_warm_reconstruction_seconds=statistics.median(r["timings"]["reconstruction_seconds"] for r in warm),
            median_warm_verified_seconds=statistics.median(r["timings"]["reconstruction_seconds"]+r["timings"]["endpoint_qualification_seconds"] for r in warm),
            technical_proxy_range=[min(r["gn_stationarity_proxy"] for r in rows),max(r["gn_stationarity_proxy"] for r in rows)],
            previous_low_sigma_control=previous["groups"][f"{profile}/binned"],cold_quality_by_added_seed=cold_quality)
    result=dict(groups=groups,profiles=profiles,total_fits=len(all_rows),accepted_fits=sum(r["accepted"] for r in all_rows),
        native_successes=sum(r["native_success"] for r in all_rows),
        native_messages=sorted(set(r["native_message"] for r in all_rows)),
        maximum_export_model_error=max(errors),source_and_reference_hashes_unchanged=True,
        all_data_noise_exactly_match_frozen_inputs=True,
        max_analytical_objective_error=max(r["analytical_objective_absolute_error"] for r in all_rows),
        max_analytical_gradient_relative_error=max(r["analytical_gradient_relative_error"] for r in all_rows),
        max_whitening_error=max(r["whitening_quadratic_error"] for r in all_rows),
        max_initial_normal_equation_error=max(r["source_initial_equation_error"] for r in all_rows),
        injected_discrete_kappa_peak=float(truth["kappa"].max()),elapsed_seconds=summary["elapsed_seconds"])
    (args.run/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    flat=[dict(profile=r["profile"],noise_seed=r["noise_seed"],phase=r["phase"],accepted=r["accepted"],
        native_success=r["native_success"],iterations=r["iterations"],nfev=r["nfev"],objective=r["objective"],
        proxy=r["gn_stationarity_proxy"],kappa_peak=r["kappa_peak"],residual_rms=r["residual_rms_noise"],**r["timings"]) for r in all_rows]
    for filename,items in [("all_fits.csv",flat),("recovery_scores.csv",quality)]:
        with (args.run/filename).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
