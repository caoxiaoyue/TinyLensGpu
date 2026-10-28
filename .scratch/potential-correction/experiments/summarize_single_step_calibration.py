"""Retain independently complete calibration runs with strict input checks.

No reconstruction is rerun or warm started. The regular one-process driver
regenerates the same six configured runs without requiring this consolidation.
"""

import argparse
import gzip
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

FIELDS = (
    "source_values",
    "source_x_axis",
    "source_y_axis",
    "potential_values",
    "deflection_x",
    "deflection_y",
    "convergence",
    "derivative_valid",
    "potential_window_coverage",
    "linear_model_image",
    "nonlinear_model_image",
    "linear_residual",
    "nonlinear_residual",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(config):
    return {
        key: value
        for key, value in config.items()
        if key not in ("lambda_amplitudes", "cases")
    }


def _scan_fit(fit):
    result = {
        key: fit[key]
        for key in ("success", "failure_reason", "log_evidence")
        if key in fit
    }
    if fit["success"]:
        result.update(
            source_relative_l2=fit["source_relative_l2"],
            linear_residual_rms_noise=fit["linear_residual_rms_noise"],
            nonlinear_residual_rms_noise=fit["nonlinear_residual_rms_noise"],
            actual_relative_residual=fit["solver_info"]["relative_residual"],
            solver_iterations=fit["solver_info"]["iterations"],
            kappa_rmse={
                domain: metrics["rmse_discrete_truth"]
                for domain, metrics in fit["convergence"].items()
            },
        )
    return result


def consolidate(directories, output, implementation_commit, completion_date=None):
    records, inputs, selected_arrays, manifest, summaries = [], {}, {}, [], []
    template = None
    seen = set()
    initialization = dict(
        gradient_coordinates="fixed_macro_image_subpixel_rays",
        gradient_provider="SourceTruth.gradient",
        potential_linearization="zero_potential",
        source="analytic mock truth gradient; fixed macro rays",
    )
    for directory in directories:
        path = directory / "calibration.json"
        record = json.loads(path.read_text())
        assert len(record["runs"]) == 2 and {row["case"] for row in record["runs"]} == {
            "subhalo",
            "control",
        }, "Incomplete component run set"
        assert record["configuration"].get("cases", ["subhalo", "control"]) == [
            "subhalo",
            "control",
        ]
        if template is None:
            template = record
        assert normalized(record["configuration"]) == normalized(
            template["configuration"]
        )
        assert record["block_partition"] == template["block_partition"]
        assert record["sizes"] == template["sizes"]
        with np.load(directory / "inputs.npz", allow_pickle=False) as arrays:
            assert (
                "potential_core_indices" in arrays.files
            ), "Component has not finished writing geometry"
            for key in arrays.files:
                value = arrays[key]
                if key in inputs:
                    assert np.array_equal(
                        inputs[key], value, equal_nan=True
                    ), f"Changed common input {key}"
                else:
                    inputs[key] = value.copy()
        with np.load(directory / "reconstructions.npz", allow_pickle=False) as arrays:
            for row in record["runs"]:
                identity = (row["case"], row["lambda_amplitude"])
                assert identity not in seen, "Duplicate independently completed run"
                seen.add(identity)
                assert row["order_check"]["passed"], "Reordered candidate check failed"
                assert (
                    not row["failure_check"]["success"]
                    and row["failure_check"]["objective"] is None
                )
                assert row["failure_check"]["failure_reason"] == "pcg_iteration_limit"
                summary = dict(
                    key=row["key"],
                    case=row["case"],
                    lambda_amplitude=row["lambda_amplitude"],
                    supplied_truth=row["supplied_truth"],
                    order_check={
                        key: value
                        for key, value in row["order_check"].items()
                        if key != "evaluations"
                    },
                    dense_selected_minus_complete_at_operator_log_evidence=row[
                        "dense_selected_minus_complete_at_operator_log_evidence"
                    ],
                    at_dense_selected_strengths=row["at_dense_selected_strengths"],
                    at_operator_selected_strengths=row[
                        "at_operator_selected_strengths"
                    ],
                    selected_field_comparison=row["selected_field_comparison"],
                    selected={},
                )
                for backend in ("dense", "operator"):
                    fit = row["selected"][backend]
                    assert fit["has_valid_fit"] and fit["success"]
                    assert all(
                        e["strengths"]["lambda_amplitude"] == row["lambda_amplitude"]
                        for e in fit["evaluations"]
                    )
                    if "initialization_provenance" not in fit:
                        fit["initialization_provenance"] = initialization
                        fit["initialization_metadata_source"] = (
                            "Explicit runner construction in the recorded numerical implementation; formatter omission completed after the run"
                        )
                    selected = {
                        key: value
                        for key, value in fit.items()
                        if key not in ("evaluations", "configuration")
                    }
                    core = inputs["potential_core_indices"]
                    kappa = arrays[f'{row["key"]}_{backend}_convergence'][core]
                    selected["mean_convergence"] = {
                        domain: dict(
                            recovered=float(
                                kappa[inputs[f'{row["case"]}_{domain}']].mean()
                            ),
                            truth=float(
                                inputs[f'{row["case"]}_discrete_kappa_truth'][
                                    inputs[f'{row["case"]}_{domain}']
                                ].mean()
                            ),
                        )
                        for domain in ("local_core", "illuminated_core")
                    }
                    selected["source_negative_fraction"] = float(
                        np.mean(arrays[f'{row["key"]}_{backend}_source_values'] < 0)
                    )
                    summary["selected"][backend] = selected
                    for field in FIELDS:
                        key = f'{row["key"]}_{backend}_{field}'
                        selected_arrays[key] = arrays[key].copy()
                successful = [
                    entry["fixed_strength_comparison"]
                    for entry in row["scan"]
                    if entry["fixed_strength_comparison"]["success"]
                ]
                assert successful, "No valid fixed-strength comparison"
                summary["scan_failed_comparisons"] = len(row["scan"]) - len(successful)
                summary["scan_max_abs_numerical_difference"] = {
                    key: max(item[key] for item in successful)
                    for key in successful[0]
                    if key.endswith("_max_abs_difference")
                }
                summary["scan"] = [
                    {
                        key: value
                        for key, value in entry.items()
                        if key in ("lambda_source", "lambda_potential")
                    }
                    | {
                        backend: _scan_fit(entry[backend])
                        for backend in ("dense", "operator")
                    }
                    for entry in row["scan"]
                ]
                records.append(row)
                summaries.append(summary)
        manifest.append(
            dict(
                directory=str(directory),
                configuration=record["configuration"],
                record_sha256=digest(path),
                inputs_sha256=digest(directory / "inputs.npz"),
                reconstructions_sha256=digest(directory / "reconstructions.npz"),
            )
        )
    amplitudes = sorted({identity[1] for identity in seen})
    assert seen == {(case, a) for a in amplitudes for case in ("subhalo", "control")}
    assert np.array_equal(inputs["subhalo_noise"], inputs["control_noise"])
    assert not np.count_nonzero(inputs["control_potential_truth"])
    output.mkdir(parents=True, exist_ok=True)
    configuration = {**template["configuration"], "lambda_amplitudes": amplitudes}
    complete = {
        **template,
        "configuration": configuration,
        "runs": records,
        "numeric_implementation_commit": implementation_commit,
        "component_manifest": manifest,
        "initialization_provenance": initialization,
        "completion_date": completion_date,
        "collection_environment": {
            "python": platform.python_version(),
            "scipy": version("scipy"),
        },
    }
    payload = json.dumps(complete, separators=(",", ":"), allow_nan=False).encode()
    with (output / "complete-record.json.gz").open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
            compressed.write(payload)
    np.savez_compressed(output / "inputs.npz", **inputs)
    np.savez_compressed(output / "selected-fields.npz", **selected_arrays)
    summary = dict(
        scope="Small fixed-macro conditional single-step calibration; no detection or mass-precision claim",
        numeric_implementation_commit=implementation_commit,
        configuration=configuration,
        sizes=template["sizes"],
        block_partition={
            key: value
            for key, value in template["block_partition"].items()
            if key != "indices"
        }
        | {
            "partition_sha256": template["runs"][0]["selected"]["operator"][
                "configuration"
            ]["block_partition_sha256"]
        },
        truth_conventions=template["truth_conventions"],
        software=template["software"] | complete["collection_environment"],
        completion_date=completion_date,
        component_manifest=[
            {key: value for key, value in item.items() if key != "configuration"}
            | {
                "lambda_amplitudes": item["configuration"]["lambda_amplitudes"],
                "cases": item["configuration"]["cases"],
            }
            for item in manifest
        ],
        runs=summaries,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--completion-date")
    args = parser.parse_args()
    consolidate(
        args.inputs, args.output, args.implementation_commit, args.completion_date
    )


if __name__ == "__main__":
    main()
