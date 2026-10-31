"""Fast deterministic replay of scientific fit exports and recovery criteria.

This diagnostic reads products of the real public fitter, never fabricates a
recovery result. Use --assert-target for an explicit red/green improvement gate.
"""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / ".scratch/potential-correction/output/iterative-scientific/validated-reference"


def measure(directory):
    metadata = json.loads((directory / "result.json").read_text())
    with np.load(directory / "reconstruction.npz", allow_pickle=False) as arrays:
        valid = arrays["derivative_valid"]
        kappa = arrays["convergence"][valid]
        truth = arrays["truth_discrete_convergence"][valid]
        local = arrays["local_core"][valid]
        peak = int(np.argmax(truth))
        output = {
            "success": metadata["success"],
            "strengths": metadata["strengths"],
            "kappa_peak": float(np.max(kappa)),
            "kappa_max_abs": float(np.max(np.abs(kappa))),
            "kappa_rms": float(np.sqrt(np.mean(kappa**2))),
            "local_kappa_rmse": float(np.sqrt(np.mean((kappa[local] - truth[local])**2))),
            "background_kappa_rms": float(np.sqrt(np.mean(kappa[~local]**2))),
            "nonlinear_residual_rms_noise": metadata["summary"]["nonlinear_residual_rms_noise"],
        }
        if truth[peak] > 0:
            output.update({
                "truth_kappa_peak": float(truth[peak]),
                "recovery_at_truth_peak": float(kappa[peak] / truth[peak]),
                "global_peak_ratio": float(np.max(kappa) / truth[peak]),
                "suppression_at_truth_peak_percent": float(100 * (1 - kappa[peak] / truth[peak])),
            })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subhalo", type=Path, default=BASE / "subhalo/dense")
    parser.add_argument("--control", type=Path, default=BASE / "control/dense")
    parser.add_argument("--min-recovery", type=float, default=0.60)
    parser.add_argument("--max-control-rms", type=float, default=0.0011)
    parser.add_argument("--max-control-peak", type=float, default=0.0031)
    parser.add_argument("--assert-target", action="store_true")
    parser.add_argument("--require-matched-strengths", action="store_true")
    args = parser.parse_args()
    subhalo, control = measure(args.subhalo), measure(args.control)
    criteria = {
        "both_fits_converged": bool(subhalo["success"] and control["success"]),
        "peak_recovery_improved": subhalo["recovery_at_truth_peak"] >= args.min_recovery,
        "control_rms_bounded": control["kappa_rms"] <= args.max_control_rms,
        "control_peak_bounded": control["kappa_max_abs"] <= args.max_control_peak,
    }
    matched = subhalo["strengths"] == control["strengths"]
    if args.require_matched_strengths:
        criteria["matched_strengths"] = matched
    print(json.dumps({"subhalo": subhalo, "control": control,
                      "matched_strengths": matched, "criteria": criteria}, indent=2))
    if args.assert_target and not all(criteria.values()):
        raise SystemExit("FAIL: " + ", ".join(key for key, passed in criteria.items() if not passed))


if __name__ == "__main__":
    main()
