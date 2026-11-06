"""Audit immutable inputs and paired fixed-lambda C1-source repeat results."""

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
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
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
    assert not protocol["source_pixels_frozen"] and not protocol["outer_search_performed"]
    fields = {}; verified = 0
    for row in summary["rows"]:
        assert row["source_hyperparameters_fixed"] and not row["source_pixels_frozen"]
        if "strengths" not in row:
            continue
        assert np.isclose(row["strengths"]["lambda_s"], protocol["fixed_source_lambda_M3"], rtol=1e-14, atol=0)
        expected = 10.**np.asarray(protocol["profiles_lens_log10"][row["profile"]])
        np.testing.assert_allclose([row["strengths"][k] for k in ("lambda4", "lambda0")], expected, rtol=1e-14)
        with np.load(run/row["directory"]/"reconstruction.npz") as archive:
            for key in ("source_scale", "source_bbox", "mask", "noise_map", "lens_subtracted_image"):
                np.testing.assert_array_equal(archive[key], m3[key])
            np.testing.assert_array_equal(archive["source_initial"], initial[:10000])
            np.testing.assert_array_equal(archive["source_reference_M3"], m3["source_values"])
            change = np.linalg.norm(archive["source_values"]-m3["source_values"])/np.linalg.norm(m3["source_values"])
            assert abs(change-row["source_relative_change_from_M3"]) < 1e-12
            fields[row["index"]] = np.array(archive["convergence"])
            verified += 1
    groups = {}; comparisons = {}
    for profile in protocol["profiles_lens_log10"]:
        paired = {}
        for method in ("bilinear", "cubic"):
            group = [r for r in summary["rows"] if r["profile"] == profile and r["method"] == method]
            assert len(group) == protocol["identical_start_repeats"]
            finite = [r for r in group if "diagnostic_log_evidence" in r]
            complete = len(finite) == len(group) and all(r["index"] in fields for r in group)
            result = dict(attempts=len(group), accepted=sum(r["accepted"] for r in group),
                native_success=sum(r.get("native_success", False) for r in group),
                failures=[r["failure"] for r in group if "failure" in r],
                trial_indices=[r["index"] for r in group])
            if finite:
                for key in ("iterations", "gn_stationarity_proxy", "objective", "chi2", "diagnostic_log_evidence",
                            "delta_kappa_peak", "source_relative_change_from_M3"):
                    values = [r[key] for r in finite]
                    result[key+"_range"] = [min(values), max(values)]
                result["peak_positions_arcsec"] = [r["delta_kappa_peak_xy_arcsec"] for r in finite]
                for key in ("iterations", "inner_seconds", "coordinate_seconds", "endpoint_seconds", "total_seconds"):
                    result["median_"+key] = statistics.median(r[key] for r in finite)
            if complete:
                stack = np.stack([fields[r["index"]] for r in group])
                valid = np.all(np.isfinite(stack), axis=0)
                kdiff = float(np.ptp(stack[:, valid], axis=0).max())
                erange = float(np.ptp([r["diagnostic_log_evidence"] for r in group]))
                result.update(maximum_pairwise_kappa_difference=kdiff, diagnostic_log_evidence_range_width=erange)
                result["repeat_gate_passed"] = bool(all(r["accepted"] for r in group) and kdiff <= 1e-4 and erange <= .01)
                paired[method] = stack.mean(axis=0)
            else:
                result["repeat_gate_passed"] = False
            groups[f"{profile}/{method}"] = result
        if len(paired) == 2:
            valid = np.isfinite(paired["bilinear"]) & np.isfinite(paired["cubic"])
            delta = paired["cubic"][valid]-paired["bilinear"][valid]
            comparisons[profile] = dict(mean_field_difference_core_RMS=float(np.sqrt(np.mean(delta**2))),
                mean_field_maximum_absolute_difference=float(abs(delta).max()),
                fixed_lambda_cubic_to_bilinear_inner_time_ratio=groups[f"{profile}/cubic"]["median_inner_seconds"]/groups[f"{profile}/bilinear"]["median_inner_seconds"],
                same_forward_objective=False, reason="Interpolation changes the forward representation; no injected real-data truth")
    audits = [r["analytical_audit"] for r in summary["rows"] if "analytical_audit" in r]
    result = dict(protocol=protocol["protocol"], fixed_source_lambda=protocol["fixed_source_lambda_M3"],
        fixed_rho=protocol["fixed_rho_M3"], source_pixels_frozen=False, normalized_source_prior_preserved=True,
        inputs_and_source_hashes_unchanged=True, verified_exports=verified, total_fits=len(summary["rows"]),
        accepted=sum(r["accepted"] for r in summary["rows"]), groups=groups, comparisons=comparisons,
        max_independent_objective_error=max((a["objective_absolute_error"] for a in audits), default=None),
        max_independent_gradient_relative_error=max((a["gradient_relative_error"] for a in audits), default=None),
        knot_replay=json.loads((run/"knot_replay.json").read_text()),
        elapsed_seconds=summary["elapsed_seconds"], outer_search_performed=False,
        physical_subhalo_recovery_validated=False)
    (run/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2))


if __name__ == "__main__":
    main()
