"""Apply the same finite evidence-selection rule to both mock observations.

Only convergence status and log evidence determine selection. Truth fields,
peak location, peak height and control-map amplitudes never enter the rule.
All attempts, including failures, are retained in the selection manifest.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


def select_candidates(directories, output):
    reference = None
    candidates = {"subhalo": {}, "control": {}}
    for directory in directories:
        record = json.loads((directory / "summary.json").read_text())
        arguments = record["arguments"]
        signature = {
            "arguments": {key: value for key, value in arguments.items()
                          if key not in ("output", "sources", "potentials", "cases")},
            "mock": record["configuration"]["mock"],
            "source_bbox": record["configuration"]["source_bbox"],
            "inversion_nsub": record["configuration"]["inversion_nsub"],
        }
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("Candidate runs differ in data, geometry, algorithm, precision or numerical controls")
        for row in record["fits"]:
            point = (row["lambda_source"], row["lambda_potential"], row["lambda_amplitude"])
            case = row["case"]
            product = directory / row["key"]
            path = product / ("result.json" if row["success"] else "failure.json")
            metadata = json.loads(path.read_text())
            candidate = {
                "strengths": list(point), "success": bool(metadata["success"]),
                "termination": metadata["termination_reason"],
                "log_evidence": metadata["log_evidence"] if metadata["success"] else None,
                "directory": str(product.resolve()),
                "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if point in candidates[case]:
                old = candidates[case][point]
                if old["success"] != candidate["success"] or old["log_evidence"] != candidate["log_evidence"]:
                    raise ValueError("Conflicting repeated candidate")
            candidates[case][point] = candidate
    if set(candidates["subhalo"]) != set(candidates["control"]) or not candidates["subhalo"]:
        raise ValueError("Both observations must have attempted the identical nonempty candidate set")
    selected = {}
    for case, points in candidates.items():
        valid = [row for row in points.values() if row["success"] and row["log_evidence"] is not None
                 and math.isfinite(row["log_evidence"])]
        if not valid:
            raise ValueError(f"No converged candidate for {case}; no selected result")
        selected[case] = max(valid, key=lambda row: row["log_evidence"])
    if output.exists():
        raise FileExistsError("Use a fresh selection output to preserve previous products")
    output.mkdir(parents=True)
    manifest = {
        "rule": "Choose maximum finite log evidence among converged fits independently for each observation; identical candidate set and fitting controls; no truth metrics used",
        "configuration": reference, "cases": {},
        "selection_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for case, points in candidates.items():
        rows = list(points.values())
        best = selected[case]
        target = output / case / "dense"
        shutil.copytree(best["directory"], target)
        manifest["cases"][case] = {"selected": best, "selected_export": str(target.resolve()), "attempted": rows}
    (output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({case: row["selected"] for case, row in manifest["cases"].items()}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    select_candidates(args.runs, args.output)
