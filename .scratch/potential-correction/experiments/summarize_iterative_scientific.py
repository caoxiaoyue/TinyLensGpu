"""Summarize retained iterative-scientific exports using host float64 only.

No fitting, Jacobian reconstruction, alignment, or timing occurs here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS = (
    "source_values",
    "potential_values",
    "deflection_x",
    "deflection_y",
    "convergence",
)
DERIVATIVES = FIELDS[2:]


def read_state(directory, failed_key=None):
    directory = Path(directory)
    if failed_key is None:
        if (directory / "result.json").exists():
            metadata = json.loads((directory / "result.json").read_text())
            arrays_path = directory / "reconstruction.npz"
            prefix = ""
        else:
            metadata = json.loads((directory / "failure.json").read_text())
            arrays_path = directory.parents[1] / "diagnostic-failed-terminals.npz"
            prefix = directory.parent.name + "_" + directory.name + "_"
    else:
        metadata = json.loads(
            (directory / "diagnostic-failed-terminals.json").read_text()
        )["runs"][failed_key]
        arrays_path = directory / "diagnostic-failed-terminals.npz"
        prefix = failed_key.replace("/", "_") + "_"
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {
            key[len(prefix) :]: np.array(saved[key])
            for key in saved.files
            if key.startswith(prefix)
        }
    return metadata, arrays


def compare(left, right):
    lm, la = left
    rm, ra = right
    assert np.array_equal(la["derivative_valid"], ra["derivative_valid"])
    assert np.array_equal(la["data_active"], ra["data_active"])
    assert np.array_equal(la["noise_active"], ra["noise_active"])
    valid = la["derivative_valid"]
    differences = {}
    for field in FIELDS:
        delta = np.asarray(ra[field], dtype=np.float64) - np.asarray(
            la[field], dtype=np.float64
        )
        if field in DERIVATIVES:
            delta = delta[valid]
        differences[field] = {
            "max_absolute": float(np.max(np.abs(delta))),
            "rms": float(np.sqrt(np.mean(delta**2))),
        }
    image = (
        np.asarray(ra["nonlinear_image_active"], dtype=np.float64)
        - np.asarray(la["nonlinear_image_active"], dtype=np.float64)
    ) / la["noise_active"]
    output = {
        "reference_success": lm["success"],
        "candidate_success": rm["success"],
        "candidate_minus_reference_objective": float(rm["objective"])
        - float(lm["objective"]),
        "raw_field_differences": differences,
        "nonlinear_image_rms_difference_noise": float(np.sqrt(np.mean(image**2))),
        "nonlinear_image_max_absolute_difference_noise": float(np.max(np.abs(image))),
        "affine_alignment_applied": False,
        "reference_strengths": lm.get("strengths"),
        "candidate_strengths": rm.get("strengths"),
    }
    if lm.get("log_evidence") is not None and rm.get("log_evidence") is not None:
        output.update(
            candidate_minus_reference_log_evidence=float(rm["log_evidence"])
            - float(lm["log_evidence"]),
            evidence_methods=[lm["evidence_method"], rm["evidence_method"]],
            determinant_methods=[lm["determinant_method"], rm["determinant_method"]],
            posterior_logdet_difference=float(rm["posterior_logdet"])
            - float(lm["posterior_logdet"]),
            prior_logdet_difference=float(rm["prior_logdet"])
            - float(lm["prior_logdet"]),
        )
    else:
        output["candidate_minus_reference_log_evidence"] = None
    return output


def fit_row(metadata):
    config = metadata["result_configuration"]
    history = metadata["history"]
    return {
        **{
            key: metadata.get(key)
            for key in [
                "success",
                "termination_reason",
                "objective",
                "chi2",
                "regularization_penalty",
                "log_evidence",
                "evidence_method",
                "determinant_method",
                "evidence_is_exact",
                "solver_info",
                "strengths",
                "summary",
            ]
        },
        "controls": {
            key: config.get(key)
            for key in [
                "solve_rtol",
                "solve_atol",
                "gradient_atol",
                "gradient_rtol",
                "initial_scaled_gradient",
                "final_scaled_gradient",
                "final_damping",
            ]
        },
        "accepted_trials": sum(row.get("accepted", False) for row in history),
        "rejected_trials": sum(not row.get("accepted", False) for row in history[1:]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--summary-directory", type=Path, required=True)
    args = p.parse_args()
    root = args.output_root
    out = args.summary_directory
    out.mkdir(parents=True, exist_ok=True)

    def state(name, case="subhalo", backend="dense"):
        return read_state(root / name / case / backend)

    strict = read_state(root / "strict-subhalo-replay", "subhalo/dense")
    calibration = {
        "strict_failed_dense_to_gradient5e-4": compare(
            strict, state("subhalo-gradient5e-4")
        ),
        "dense_gradient5e-4_to_gradient1e-3": compare(
            state("subhalo-gradient5e-4"), state("subhalo-gradient1e-3")
        ),
        "operator_gradient5e-4_to_gradient1e-3": compare(
            state("validated-reference", backend="operator"),
            state("operator-gradient1e-3", backend="operator"),
        ),
        "strict_converged_operator_to_looser_residual_dense_control": compare(
            state("reference", "control", "operator"),
            state("control-residual5e-5", "control", "dense"),
        ),
        "strict_converged_operator_to_looser_residual_operator_control": compare(
            state("reference", "control", "operator"),
            state("control-residual5e-5", "control", "operator"),
        ),
        "calibration_to_adopted_dense_subhalo": compare(
            state("subhalo-gradient5e-4"), state("validated-reference")
        ),
        "strict_failed_terminal_evidence": None,
        "convention": "Raw signed fields; derivatives restricted to C; host float64; failed-state comparison has no evidence.",
    }
    sensitivity = {}
    final = {}
    for case in ["subhalo", "control"]:
        baseline = state("validated-reference", case)
        final[case] = {
            backend: fit_row(state("validated-reference", case, backend)[0])
            for backend in ["dense", "operator"]
        }
        sensitivity[case] = {
            "baseline_lambda0": 10000.0,
            "fixed_smoothing": baseline[0]["strengths"],
            "profiles": {},
        }
        for strength in [3000, 30000]:
            name = f"amplitude{strength}"
            current = state(name, case)
            sensitivity[case]["profiles"][str(strength)] = {
                "fit": fit_row(current[0]),
                "difference_from_lambda0_10000": compare(baseline, current),
            }

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    write("stopping200-comparison.json", calibration)
    write("amplitude200-sensitivity.json", sensitivity)
    write("scientific200-table.json", final)
    directories = [
        "calibration64",
        "reference",
        "strict-subhalo-replay",
        "subhalo-gradient5e-4",
        "subhalo-gradient1e-3",
        "control-residual5e-5",
        "validated-reference",
        "operator-gradient1e-3",
        "amplitude3000",
        "amplitude30000",
    ]
    entries = []
    for name in directories:
        for path in sorted((root / name).rglob("*")):
            if path.is_file():
                entries.append(
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    write(
        "artifact-manifest.json",
        {
            "root": str(root),
            "scope": "Actual retained fits, diagnostic failures, configurations, histories, arrays and figures; no timing measurements. Historical diagnosis has its separately retained hash manifest.",
            "files": entries,
        },
    )
    print(
        json.dumps(
            {
                "stopping_comparisons": list(calibration),
                "sensitivity_cases": list(sensitivity),
                "manifest_files": len(entries),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
