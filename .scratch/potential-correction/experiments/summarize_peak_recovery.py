"""Retain compact, provenance-linked measurements of the recovery study."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnose_peak_recovery import measure


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / ".scratch/potential-correction/output"
STUDY = OUTPUT / "peak-recovery"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product(directory):
    row = measure(directory)
    meta = json.loads((directory / "result.json").read_text())
    row.update(
        product=str(directory.relative_to(ROOT)),
        metadata_sha256=digest(directory / "result.json"),
        arrays_sha256=digest(directory / "reconstruction.npz"),
        algorithm=meta["algorithm"],
        backend=meta["backend"],
        log_evidence=meta["log_evidence"],
        source_relative_l2=meta["summary"]["source_relative_l2"],
        final_scaled_gradient=meta["result_configuration"].get("final_scaled_gradient"),
        gradient_atol=meta["result_configuration"].get("gradient_atol"),
    )
    with np.load(directory / "reconstruction.npz", allow_pickle=False) as a:
        valid = a["derivative_valid"]
        error = a["convergence"][valid] - a["truth_discrete_convergence"][valid]
        local = a["local_core"][valid]
        row["all_core_kappa_rmse"] = float(np.sqrt(np.mean(error**2)))
        row["background_kappa_rmse"] = float(np.sqrt(np.mean(error[~local]**2)))
        row["image_data_sha256"] = hashlib.sha256(a["image_data"].tobytes()).hexdigest()
        if "truth_kappa_peak" in row:
            r = np.hypot(a["potential_x"] - 0.75, a["potential_y"] - 0.1)
            row["integrated_kappa_apertures"] = {}
            for radius in (0.075, 0.15, 0.225, 0.3, 0.35777087639996635):
                selected = valid & (r <= radius)
                t = float(a["truth_discrete_convergence"][selected].sum() * 0.075**2)
                f = float(a["convergence"][selected].sum() * 0.075**2)
                row["integrated_kappa_apertures"][str(radius)] = {
                    "nodes": int(selected.sum()), "truth_arcsec2": t,
                    "recovered_arcsec2": f, "ratio": f / t,
                }
    return row


def selection(directory):
    manifest = json.loads((directory / "selection.json").read_text())
    result = {case: product(directory / case / "dense") for case in ("subhalo", "control")}
    result["gate"] = {
        "peak_recovery_above_60_percent": result["subhalo"]["recovery_at_truth_peak"] >= 0.6,
        "control_rms_below_0.0011": result["control"]["kappa_rms"] <= 0.0011,
        "control_max_abs_below_0.0031": result["control"]["kappa_max_abs"] <= 0.0031,
    }
    result["selection"] = {
        "rule": manifest["rule"], "source_sha256": manifest.get("selection_source_sha256"),
        "attempted_per_case": {case: len(value["attempted"]) for case, value in manifest["cases"].items()},
        "failed_per_case": {case: sum(not row["success"] for row in value["attempted"]) for case, value in manifest["cases"].items()},
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/potential-correction/research/peak-recovery/summary.json")
    args = parser.parse_args()
    policy_path = ROOT / ".scratch/potential-correction/research/peak-recovery/validation-policy.json"
    policy = json.loads(policy_path.read_text())
    report = {"scope": "Configuration-specific peak-bias investigation; no detection significance or calibrated mass precision",
              "validation_policy": policy, "validation_policy_sha256": digest(policy_path),
              "baseline": {}, "selected": {}, "heldout": {}, "ablations": {}}
    for algorithm, relative in {
        "single_step": "single-step-scientific/reference",
        "iterative": "iterative-scientific/validated-reference",
    }.items():
        report["baseline"][algorithm] = {case: product(OUTPUT / relative / case / "dense") for case in ("subhalo", "control")}
        report["selected"][algorithm] = selection(STUDY / f"selected-{algorithm.replace('_', '-')}-final")
        for case in ("subhalo", "control"):
            assert report["baseline"][algorithm][case]["image_data_sha256"] == report["selected"][algorithm][case]["image_data_sha256"]
    for seed in policy["held_out_noise_seeds"]:
        directory = STUDY / f"heldout-selected-{seed}"
        if (directory / "selection.json").exists():
            result = selection(directory)
            candidates = json.loads((STUDY / f"heldout-seed-{seed}/summary.json").read_text())
            a = candidates["arguments"]
            assert a["seed"] == seed and a["sources"] == [policy["lambda_source"]]
            assert a["potentials"] == policy["lambda_potential_candidates"]
            assert a["gradient_atol"] == policy["numerical_controls"]["gradient_atol"]
            assert a["max_iter"] == policy["numerical_controls"]["max_iter"]
            report["heldout"][str(seed)] = result
        else:
            report["heldout"][str(seed)] = {"status": "no_selection_export; inspect retained candidate statuses"}
    for path in sorted(STUDY.glob("*/summary.json")):
        value = json.loads(path.read_text())
        report["ablations"][path.parent.name] = {
            "arguments": {key: v for key, v in value["arguments"].items() if key != "output"},
            "source_sha256": value["source_sha256"],
            "truth_penalties": value["truth_penalties"],
            "fits": [{key: v for key, v in row.items() if key != "elapsed_seconds"} for row in value["fits"]],
        }
    report["frozen_linear_bias_decomposition"] = json.loads((STUDY / "linear-bias-decomposition.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for group in ("baseline", "selected", "heldout"):
        print(group)
        for key, value in report[group].items():
            if "subhalo" in value:
                print(key, "peak recovery", value["subhalo"]["recovery_at_truth_peak"],
                      "control RMS", value["control"]["kappa_rms"],
                      "control max", value["control"]["kappa_max_abs"],
                      "gate", value.get("gate"))
            else:
                print(key, value)


if __name__ == "__main__":
    main()
