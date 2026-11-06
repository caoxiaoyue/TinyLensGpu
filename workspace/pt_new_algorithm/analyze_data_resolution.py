"""Read-only audit of all image-resolution fits and the exact noise/binning."""

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
    protocol=json.loads((args.run/"protocol.json").read_text())
    if not summary["complete"]:
        raise ValueError("Some conditions failed; inspect raw logs and retain their failures")
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    repo=Path(__file__).resolve().parents[2]
    assert all(sha(repo/p)==h for p,h in protocol["sources_sha256"].items())
    assert sha(args.run/"frozen_inputs.npz")==protocol["frozen_input_sha256"]==summary["frozen_input_sha256"]
    reference=Path(protocol["reference"])
    assert sha(reference/"protocol.json")==protocol["reference_protocol_sha256"]
    assert sha(reference/"frozen_inputs.npz")==sha(args.run/"frozen_inputs.npz")
    with np.load(args.run/"frozen_inputs.npz") as a:
        active=~a["mask"]
        fine_data=a["image_data"][active]
        fine_noise=a["noise_map"][active]
    # Truth is sampled only after every optimizer endpoint has been frozen.
    # These scores describe recovery, never select/restart a fitted endpoint.
    truth_path=args.run/"postfit_truth.npz"
    if not truth_path.exists():
        from run_source_resolution import runtime
        from examples.potential_correction._recovery_diagnostics import _sample_truth
        jax,jnp,np_runtime,cfg,obs,SimulatorConfig,Renderer=runtime()
        sim=SimulatorConfig(cfg["mock"]["dpix"],cfg["mock"]["npix"],obs.psf_kernel,cfg["inversion_nsub"],obs.mask)
        renderer=Renderer(obs.macro,sim,100,cfg["source_bbox"],potential_spacing=cfg["potential_spacing"],dtype="float64")
        with np.load(args.run/"frozen_inputs.npz") as a:
            np.testing.assert_array_equal(renderer.potential_grid.x,a["potential_x"])
            np.testing.assert_array_equal(renderer.potential_grid.y,a["potential_y"])
        truth=_sample_truth(renderer.source_x_axis,renderer.source_y_axis,renderer.potential_grid,obs,"float64")
        np.savez_compressed(truth_path,**truth)
    with np.load(truth_path) as a:
        truth={k:a[k] for k in a.files}
    true_peak=float(truth["kappa"].max())
    groups={};rows=[];export_errors=[];saved={}
    frozen_meta=json.loads((args.run/"frozen_inputs.json").read_text())
    for name,group in summary["groups"].items():
        items=group["rows"];warm=[r for r in items if r["phase"].startswith("warm_")]
        assert len(warm)==protocol["warm_repeats"]
        env=json.loads((args.run/name/"environment.json").read_text())
        assert env["input_hashes"]==frozen_meta["hashes"] and env["checked_geometry_identical"]
        with np.load(args.run/items[0]["directory"]/"fields.npz") as a:
            cold={k:a[k].copy() for k in a.files}
        saved[name]=cold
        k0=cold["convergence"][cold["potential_core_indices"]]
        kd=[];kerrors=[];local_errors=[];source_errors=[]
        for row in items:
            with np.load(args.run/row["directory"]/"fields.npz") as a:
                indices=a["observation_indices"]
                assert np.unique(indices).size==indices.size
                count=indices.shape[1]
                expected_data=fine_data[indices].mean(axis=1)
                expected_noise=np.sqrt((fine_noise[indices]**2).sum(axis=1))/count
                expected_model=a["fine_model_image"][active][indices].mean(axis=1)
                actual_model=a["model_image"][~a["observation_mask"]]
                errors=[float(np.max(abs(a["observed_data"]-expected_data))),
                        float(np.max(abs(a["observed_noise"]-expected_noise))),
                        float(np.max(abs(actual_model-expected_model)))]
                assert max(errors)<1e-12
                export_errors.append(errors)
                assert len(a["joint_values"])==10956 and len(expected_data)==row["dimensions"]["n_data"]
                kd.append(float(np.max(abs(a["convergence"][a["potential_core_indices"]]-k0))))
                error=a["convergence"][a["potential_core_indices"]]-truth["kappa"]
                kerrors.append(float(np.sqrt(np.mean(error**2))))
                local_errors.append(float(np.sqrt(np.mean(error[truth["local_core"]]**2))))
                source_errors.append(float(np.linalg.norm(a["source_values"]-truth["source"])/np.linalg.norm(truth["source"])))
            rows.append(row)
        groups[name]=dict(all_runs=len(items),accepted=sum(r["accepted"] for r in items),
            warm_runs=len(warm),warm_accepted=sum(r["accepted"] for r in warm),
            data_points=items[0]["dimensions"]["n_data"],
            parameter_to_data_ratio=10956/items[0]["dimensions"]["n_data"],
            cold_reconstruction_seconds=items[0]["timings"]["reconstruction_seconds"],
            median_warm_iterations=statistics.median(r["iterations"] for r in warm),
            warm_iterations=[r["iterations"] for r in warm],
            median_warm_timings={k:statistics.median(r["timings"][k] for r in warm) for k in warm[0]["timings"]},
            median_warm_reconstruction_plus_certificate_seconds=statistics.median(
                r["timings"]["reconstruction_seconds"]+r["timings"]["endpoint_qualification_seconds"] for r in warm),
            proxy_range=[min(r["gn_stationarity_proxy"] for r in items),max(r["gn_stationarity_proxy"] for r in items)],
            legacy_gradient_passes=sum(r["legacy_gradient_pass"] for r in items),
            kappa_peak_range=[min(r["kappa_peak"] for r in items),max(r["kappa_peak"] for r in items)],
            residual_rms_noise_range=[min(r["residual_rms_noise"] for r in items),max(r["residual_rms_noise"] for r in items)],
            maximum_repeat_core_kappa_difference_from_cold=max(kd),noise_min=env["noise_min"],noise_max=env["noise_max"],
            postfit_source_relative_error_range=[min(source_errors),max(source_errors)],
            postfit_core_kappa_truth_rmse_range=[min(kerrors),max(kerrors)],
            postfit_local_kappa_truth_rmse_range=[min(local_errors),max(local_errors)])
    comparisons={}
    for profile in protocol["profiles"]:
        full=saved[f"{profile}/full"];matched=saved[f"{profile}/matched_fine"];coarse=saved[f"{profile}/binned"]
        indices=coarse["observation_indices"]
        support=np.unique(indices)
        assert np.array_equal(support,matched["observation_indices"].ravel())
        for left,right in [("full","matched_fine"),("matched_fine","binned")]:
            a,b=saved[f"{profile}/{left}"],saved[f"{profile}/{right}"]
            k=a["potential_core_indices"]
            dk=b["convergence"][k]-a["convergence"][k]
            di=(b["fine_model_image"][active]-a["fine_model_image"][active])[support]/fine_noise[support]
            comparisons[f"{profile}/{left}_to_{right}"]=dict(core_kappa_difference_rms=float(np.sqrt(np.mean(dk**2))),
                core_kappa_difference_max_abs=float(np.max(abs(dk))),
                matched_fine_model_difference_rms_in_fine_noise=float(np.sqrt(np.mean(di**2))))
        with np.load(reference/profile/"n100/cold/fields.npz") as old:
            dk=full["convergence"][full["potential_core_indices"]]-old["convergence"][old["potential_core_indices"]]
            comparisons[f"{profile}/full_to_previous_n100"]=dict(maximum_core_kappa_difference=float(np.max(abs(dk))),
                model_rms_difference_in_fine_noise=float(np.sqrt(np.mean(((full["fine_model_image"][active]-old["model_image"][active])/fine_noise)**2))))
    result=dict(groups=groups,comparisons=comparisons,total_runs=len(rows),accepted_runs=sum(r["accepted"] for r in rows),
        native_successes=sum(r["native_success"] for r in rows),
        native_messages=sorted(set(r["native_message"] for r in rows)),
        sources_and_frozen_inputs_unchanged=True,
        maximum_export_data_noise_model_errors=np.max(export_errors,axis=0).tolist(),
        maximum_analytical_gradient_relative_error=max(r["analytical_gradient_relative_error"] for r in rows),
        maximum_analytical_objective_error=max(r["analytical_objective_absolute_error"] for r in rows),
        maximum_whitening_quadratic_error=max(r["whitening_quadratic_error"] for r in rows),
        maximum_source_initial_equation_error=max(r["source_initial_equation_error"] for r in rows),
        postfit_truth_sha256=sha(truth_path),truth_used_in_optimization=False,injected_discrete_kappa_peak=true_peak,
        elapsed_seconds=summary["elapsed_seconds"])
    (args.run/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    flat=[dict(profile=r["profile"],observation=r["observation"],phase=r["phase"],n_data=r["dimensions"]["n_data"],
        accepted=r["accepted"],native_success=r["native_success"],iterations=r["iterations"],nfev=r["nfev"],
        objective=r["objective"],proxy=r["gn_stationarity_proxy"],kappa_peak=r["kappa_peak"],
        residual_rms_noise=r["residual_rms_noise"],**r["timings"]) for r in rows]
    with (args.run/"all_fits.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
