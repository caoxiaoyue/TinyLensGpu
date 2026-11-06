"""Audit C1 lens-hyperparameter search, full exports, native exits and repeats."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def analyze(run):
    protocol = json.loads((run/"protocol.json").read_text())
    summary = json.loads((run/"summary.json").read_text())
    root = Path(__file__).resolve().parents[2]
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(sha(root/p) == h for p,h in protocol["source_hashes"].items())
    assert all(sha(Path(p)) == h for p,h in protocol["input_hashes"].items())
    assert sha(run/"initial_values.npz") == protocol["initial_values_archive_sha256"]
    with np.load(Path(protocol["arguments"]["macro_dir"])/"m3_handoff.npz") as archive:
        m3 = {k:archive[k] for k in archive.files}
    with np.load(run/"initial_values.npz") as archive:
        initial = np.array(archive["joint_values"])
        np.testing.assert_array_equal(archive["source_scale"], m3["source_scale"])
        np.testing.assert_array_equal(archive["saved_M3_source"], m3["source_values"])
    assert initial.shape == (11134,) and np.all(initial[10000:] == 0)
    assert not summary["source_pixels_frozen"] and summary["outer_dimensions"] == 2
    rows = summary["rows"]; best = summary["best"]
    assert best is not None and best["accepted"]
    qualified = [r for r in rows if r["accepted"] and r["phase"] != "selected_repeat"]
    assert best["index"] == max(qualified, key=lambda r:r["log_evidence"])["index"]
    fields = {}; topology = None; verified = 0
    selected_indices = [best["index"]]+summary["selected_repeat_indices"]
    for row in rows:
        assert row["source_hyperparameters_fixed"] and not row["source_pixels_frozen"]
        if "strengths" not in row:
            assert not row["accepted"] and row["log_evidence"] is None
            continue
        assert np.isclose(row["strengths"]["lambda_s"], protocol["source_lambda_M3"], rtol=1e-14, atol=0)
        expected = bool(row["native_success"] and row["gn_stationarity_proxy"] <= .001
                        and np.isfinite(row["diagnostic_log_evidence"]))
        assert row["accepted"] == expected
        assert (row["log_evidence"] is not None) == row["accepted"]
        with np.load(run/row["directory"]/"reconstruction.npz") as archive:
            for key in ("source_scale", "source_bbox", "mask", "noise_map", "lens_subtracted_image"):
                np.testing.assert_array_equal(archive[key], m3[key])
            np.testing.assert_array_equal(archive["source_initial"], initial[:10000])
            np.testing.assert_array_equal(archive["source_reference_M3"], m3["source_values"])
            if topology is None:
                topology = {k:np.array(archive[k]) for k in ("potential_node_indices", "potential_x", "potential_y", "derivative_valid", "source_x_axis", "source_y_axis")}
            for key, value in topology.items():
                np.testing.assert_array_equal(archive[key], value)
            if row["index"] in selected_indices:
                fields[row["index"]] = np.array(archive["convergence"])
            change = np.linalg.norm(archive["source_values"]-m3["source_values"])/np.linalg.norm(m3["source_values"])
            assert abs(change-row["source_relative_change_from_M3"]) < 1e-12
            assert np.isfinite(archive["joint_values"]).all()
            verified += 1
    selected = [rows[i] for i in selected_indices]
    complete = all(i in fields for i in selected_indices)
    erange = kdiff = None
    if complete:
        stack = np.stack([fields[i] for i in selected_indices])
        finite = np.all(np.isfinite(stack), axis=0)
        kdiff = float(np.max(np.ptp(stack[:,finite], axis=0)))
        erange = float(np.ptp([r["diagnostic_log_evidence"] for r in selected]))
    passed = bool(complete and all(r["accepted"] for r in selected) and erange <= .01 and kdiff <= 1e-4)
    validation = dict(passed=passed, status="passed" if passed else "repeat_gate_failed",
        attempts=len(selected), original_index=best["index"], all_attempts_accepted=all(r["accepted"] for r in selected),
        diagnostic_log_evidence_range=erange, maximum_pairwise_kappa_difference=kdiff,
        records=[{k:r.get(k) for k in ("index", "phase", "repeat", "accepted", "native_success", "iterations",
            "gn_stationarity_proxy", "diagnostic_log_evidence", "delta_kappa_peak", "inner_seconds", "directory")} for r in selected])
    (run/"maximum_evidence/validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    groups = {}
    for phase in dict.fromkeys(r["phase"] for r in rows):
        group = [r for r in rows if r["phase"] == phase]
        finite = [r for r in group if "objective" in r]
        item = dict(attempts=len(group), accepted=sum(r["accepted"] for r in group),
            native_success=sum(r.get("native_success", False) for r in group),
            native_maxiter=sum(r.get("iterations") == protocol["optimizer"]["maxiter"] and not r.get("native_success", False) for r in group),
            failures=[r["failure"] for r in group if "failure" in r])
        if finite:
            for key in ("iterations", "gn_stationarity_proxy", "delta_kappa_peak", "diagnostic_log_evidence"):
                values = [r[key] for r in finite]
                item[key+"_range"] = [min(values), max(values)]
            for key in ("iterations", "inner_seconds", "total_seconds"):
                item["median_"+key] = statistics.median(r[key] for r in finite)
        groups[phase] = item
    baseline = next(r for r in rows if r["phase"] == "baseline_single_step_lens")
    bounds = np.asarray(protocol["lens_log10_bounds"])
    point = np.asarray(best["log10_strengths"])[1:]
    lower, upper = point-bounds[:,0], bounds[:,1]-point
    audits = [r["analytical_audit"] for r in rows if "analytical_audit" in r]
    result = dict(protocol=protocol["protocol"], source_lambda_fixed=protocol["source_lambda_M3"],
        rho_fixed=protocol["rho_M3"], source_pixels_frozen=False, inputs_and_source_hashes_unchanged=True,
        total_fits=len(rows), accepted=sum(r["accepted"] for r in rows), verified_exports=verified,
        formal_log_evidence_only_when_qualified=True, groups=groups, outer=summary["outer"],
        best={k:best.get(k) for k in ("index", "phase", "strengths", "log10_strengths", "log_evidence", "objective",
            "chi2", "iterations", "gn_stationarity_proxy", "delta_kappa_peak", "delta_kappa_peak_xy_arcsec",
            "coordinate_seconds", "inner_seconds", "endpoint_seconds", "total_seconds", "directory")},
        baseline_single_step_lens={k:baseline.get(k) for k in ("index", "strengths", "log_evidence", "objective",
            "chi2", "delta_kappa_peak", "delta_kappa_peak_xy_arcsec", "directory")},
        evidence_gain_over_single_step_lens=float(best["log_evidence"]-baseline["log_evidence"]),
        peak_ratio_to_single_step_lens=float(best["delta_kappa_peak"]/baseline["delta_kappa_peak"]),
        bounds_distance_log10=dict(lower=lower.tolist(), upper=upper.tolist()),
        near_search_boundary=bool(min(np.min(lower),np.min(upper)) <= .02),
        selected_validation=validation, elapsed_seconds=summary["elapsed_seconds"],
        max_independent_objective_error=max((a["objective_absolute_error"] for a in audits), default=None),
        max_independent_gradient_relative_error=max((a["gradient_relative_error"] for a in audits), default=None),
        global_optimum_certified=False, physical_subhalo_recovery_validated=False)
    (run/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2))


if __name__ == "__main__":
    main()
