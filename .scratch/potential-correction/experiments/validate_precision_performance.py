"""Reproduce the bounded stage 13 campaign with serial isolated GPU workers.

Launch from the checkout root in tinylens_gpu. The coordinator imports no JAX.
Use a fresh --output to retain prior measurements and diagnostic failures.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "benchmarks/potential_correction_workflow.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invoke(command, output):
    if output.exists():
        print(f"Retaining existing {output}", flush=True)
        return
    subprocess.run(
        [sys.executable, str(WORKER), *command, "--output", str(output)], check=True
    )


def references(canonical, output):
    rows = []
    for algorithm, relative in (
        ("single_step", "single-step-scientific/reference"),
        ("iterative", "iterative-scientific/validated-reference"),
    ):
        for case in ("subhalo", "control"):
            for backend in ("dense", "operator"):
                p = canonical / relative / case / backend
                r = json.loads((p / "result.json").read_text())
                a = np.load(p / "reconstruction.npz")
                assert r["success"] and a["source_values"].dtype == np.float64
                rows.append(
                    dict(
                        algorithm=algorithm,
                        case=case,
                        backend=backend,
                        success=True,
                        dtype="float64",
                        metadata_sha256=sha(p / "result.json"),
                        arrays_sha256=sha(p / "reconstruction.npz"),
                        path=str(p),
                    )
                )
    (output / "reference-gate.json").write_text(json.dumps(rows, indent=2) + "\n")


def fit(
    output,
    case,
    algorithm,
    backend,
    dtype,
    profile,
    *,
    observation="subhalo",
    repeats=1,
    search=0,
):
    tail = f"-search{search}" if search else ""
    name = f"{case}-{observation}-{algorithm}-{backend}-{dtype}-{profile}{tail}"
    # Preserve the shorter names used by the first four numerical references.
    if case == "coupled37":
        name = f"{case}-{algorithm}-{backend}-{dtype}-{profile}{tail}"
    command = [
        "--inputs",
        str(
            output
            / "fixtures"
            / f"{case}{'' if case=='coupled37' else '-'+observation}.npz"
        ),
        "--algorithm",
        algorithm,
        "--backend",
        backend,
        "--dtype",
        dtype,
        "--profile",
        profile,
        "--repeats",
        str(repeats),
    ]
    if search:
        command += ["--search", str(search)]
    invoke(command, output / "workers" / (name + ".json"))


def difference(reference_path, candidate_path):
    r, c = (json.loads(p.read_text()) for p in (reference_path, candidate_path))
    a, b = (np.load(p.with_suffix(".npz")) for p in (reference_path, candidate_path))
    assert (
        r["input_sha256"] == c["input_sha256"]
    ), "Comparison uses different observations/initialization"
    assert np.array_equal(a["potential_node_indices"], b["potential_node_indices"])
    metrics = {}
    for name in (
        "source_values",
        "potential_values",
        "linear_image_active",
        "nonlinear_image_active",
        "deflection_x",
        "deflection_y",
        "convergence",
        "linear_normalized_residual",
        "nonlinear_normalized_residual",
    ):
        if name not in a or name not in b:
            continue
        x, y = np.asarray(a[name], dtype=np.float64), np.asarray(
            b[name], dtype=np.float64
        )
        finite = np.isfinite(x) & np.isfinite(y)
        delta = y[finite] - x[finite]
        denom = np.linalg.norm(x[finite])
        metrics[name] = dict(
            max_absolute=float(np.max(abs(delta))) if delta.size else None,
            rms=float(np.sqrt(np.mean(delta**2))) if delta.size else None,
            relative_l2=None if denom == 0 else float(np.linalg.norm(delta) / denom),
            valid_count=int(finite.sum()),
        )
    for name in (
        "objective",
        "chi2",
        "regularization_penalty",
        "prior_logdet",
        "posterior_logdet",
        "log_evidence",
    ):
        x, y = r["result"].get(name), c["result"].get(name)
        metrics[name] = None if x is None or y is None else y - x
    guard = None
    if c["case"] == "coupled37" and c["result"]["success"]:
        guard = True
        samples = (
            [c["timing"]["cold_status"], *c["timing"]["warm_status"]]
            if not c["search"]
            else []
        )
        for row in samples:
            guard &= row["success"]
            for field in (
                "source_values",
                "potential_values",
                "nonlinear_image_active",
            ):
                guard &= bool(np.allclose(row[field], a[field], atol=2e-4, rtol=2e-4))
            guard &= bool(
                np.isclose(
                    row["objective"], float(a["objective"]), atol=2e-4, rtol=2e-4
                )
            )
        if c["search"]:
            for field in (
                "source_values",
                "potential_values",
                "nonlinear_image_active",
                "objective",
            ):
                guard &= bool(np.allclose(b[field], a[field], atol=2e-4, rtol=2e-4))
    evidence_error = None
    if c["case"] == "coupled37" and not c["search"] and c["result"]["success"]:
        samples = [c["timing"]["cold_status"], *c["timing"]["warm_status"]]
        evidence_error = max(
            abs(row["log_evidence"] - r["result"]["log_evidence"]) for row in samples
        )
    return dict(
        reference=str(reference_path.name),
        candidate=str(candidate_path.name),
        reference_success=r["result"]["success"],
        candidate_success=c["result"]["success"],
        candidate_termination=c["result"]["termination_reason"],
        comparison_scope=(
            "selected bounded searches" if c["search"] else "fixed strengths"
        ),
        reference_strengths=r["result"]["strengths"],
        candidate_strengths=c["result"]["strengths"],
        field_guard_atol=2e-4,
        field_guard_rtol=2e-4,
        all_cold_warm_field_guards=guard,
        max_cold_warm_evidence_difference=evidence_error,
        small_evidence_budget_passes=(
            None if evidence_error is None else evidence_error < 2e-3
        ),
        metrics=metrics,
        geometry=c["geometry"],
        evidence_methods=[
            r["result"]["evidence_method"],
            c["result"]["evidence_method"],
        ],
        determinant_methods=[
            r["result"]["determinant_method"],
            c["result"]["determinant_method"],
        ],
    )


def summarize(output, destination):
    records = {
        p.name: json.loads(p.read_text())
        for p in sorted((output / "workers").glob("*.json"))
    }
    compact = {}
    for name, r in records.items():
        if r.get("worker_failure"):
            compact[name] = r
            continue
        peaks = [x["stats"].get("peak_bytes_in_use") for x in r["allocator_snapshots"]]
        search = r["search"]
        if search:
            search = {
                **search,
                "evaluations": [
                    {
                        k: e[k]
                        for k in (
                            "log_strengths",
                            "log_evidence",
                            "success",
                            "failure_reason",
                            "termination_reason",
                        )
                    }
                    for e in search["evaluations"]
                ],
            }
        compact[name] = dict(
            case=r["case"],
            observation=r["observation"],
            algorithm=r["algorithm"],
            backend=r["backend"],
            dtype=r["dtype"],
            profile=r["profile"],
            counts=r["counts"],
            success=r["result"]["success"],
            termination=r["result"]["termination_reason"],
            solver=r["result"]["solver_info"],
            objective=r["result"]["objective"],
            log_evidence=r["result"]["log_evidence"],
            final_scaled_gradient=r["result"]["configuration"].get(
                "final_scaled_gradient"
            ),
            evidence_method=r["result"]["evidence_method"],
            determinant_method=r["result"]["determinant_method"],
            timing={
                k: v
                for k, v in r["timing"].items()
                if k not in ("cold_status", "warm_status")
            },
            jax_allocator_peak_bytes=max(
                (p for p in peaks if p is not None and p >= 0), default=None
            ),
            sampled_process_gpu_peak_bytes=r["process_gpu_memory"]["peak_bytes"],
            gpu_sample_count=r["process_gpu_memory"]["sample_count"],
            host_peak_rss_bytes=r["host_peak_rss_bytes"],
            live_arrays={k: v for k, v in r["live_arrays"].items() if k != "shapes"},
            geometry=r["geometry"],
            dtype_audit=r["dtype_audit"],
            precision_defaults=r["precision_defaults"],
            input_sha256=r["input_sha256"],
            record_sha256=sha(output / "workers" / name),
            search=search,
        )
        expected = "float64" if r["dtype"] == "float64" else "float32"
        for field, value in r["dtype_audit"].items():
            if value is not None:
                wanted = (
                    ("complex128" if expected == "float64" else "complex64")
                    if field == "psf_fft"
                    else expected
                )
                assert value == wanted, (name, field, value)
        assert r["precision_defaults"]["before"] == r["precision_defaults"]["after"]
    comparisons = []
    for algorithm in ("single_step", "iterative"):
        for backend in ("dense", "operator"):
            stem = f"coupled37-{algorithm}-{backend}"
            for ref, candidate in (
                (f"{stem}-float64-strict.json", f"{stem}-float64-small32.json"),
                (f"{stem}-float64-small32.json", f"{stem}-float32-small32.json"),
                (f"{stem}-float64-strict.json", f"{stem}-float32-small32.json"),
            ):
                if ref in records and candidate in records:
                    comparisons.append(
                        difference(
                            output / "workers" / ref, output / "workers" / candidate
                        )
                    )
            for case in ("subhalo", "control"):
                a = f"scientific200-{case}-{algorithm}-{backend}-float64-scientific.json"
                b = f"scientific200-{case}-{algorithm}-{backend}-float32-scientific.json"
                if a in records and b in records:
                    comparisons.append(
                        difference(output / "workers" / a, output / "workers" / b)
                    )
            a = f"{stem}-float64-small32-search25.json"
            b = f"{stem}-float32-small32-search25.json"
            if a in records and b in records:
                comparisons.append(
                    difference(output / "workers" / a, output / "workers" / b)
                )
        a = f"coupled37-{algorithm}-dense-float64-small32-search25.json"
        b = f"coupled37-{algorithm}-operator-float64-small32-search25.json"
        if a in records and b in records:
            comparisons.append(
                difference(output / "workers" / a, output / "workers" / b)
            )
    rejected = []
    for p in sorted((output / "rejected-calibration-pcg1e-5").glob("*.json")):
        row = json.loads(p.read_text())
        reference = (
            output
            / "workers"
            / f"coupled37-{row['algorithm']}-{row['backend']}-float64-strict.json"
        )
        rejected.append(
            {
                "record_sha256": sha(p),
                "reason": "Shared profile rejected: single-step operator field guard fails even in float64",
                **difference(reference, p),
            }
        )
    for row in comparisons:
        if (
            row["candidate"].startswith("coupled37")
            and row["comparison_scope"] == "fixed strengths"
        ):
            assert row["all_cold_warm_field_guards"], row["candidate"]
            assert row["small_evidence_budget_passes"], row["candidate"]
    report = dict(
        scope="fixed-input source-potential precision and isolated resource measurements; failed-fit times are diagnostics",
        workers=compact,
        comparisons=comparisons,
        rejected_calibration=rejected,
        software=next(
            (r["software"] for r in records.values() if "software" in r), None
        ),
        reference_gate=json.loads((output / "reference-gate.json").read_text()),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")


def reference_reproduction(canonical, output, destination):
    rows = []
    for algorithm, folder in (
        ("single_step", "single-step-scientific/reference"),
        ("iterative", "iterative-scientific/validated-reference"),
    ):
        for case in ("subhalo", "control"):
            for backend in ("dense", "operator"):
                p = (
                    output
                    / "workers"
                    / f"scientific200-{case}-{algorithm}-{backend}-float64-scientific.json"
                )
                if not p.exists():
                    continue
                row = json.loads(p.read_text())
                a = np.load(p.with_suffix(".npz"))
                reference = canonical / folder / case / backend
                metadata = json.loads((reference / "result.json").read_text())
                b = np.load(reference / "reconstruction.npz")
                differences = {}
                for name in (
                    "source_values",
                    "potential_values",
                    "nonlinear_image_active",
                ):
                    np.testing.assert_allclose(a[name], b[name], atol=1e-9, rtol=1e-9)
                    differences[name] = float(np.max(abs(a[name] - b[name])))
                delta = row["result"]["log_evidence"] - metadata["log_evidence"]
                assert abs(delta) < 1e-7
                rows.append(
                    dict(
                        algorithm=algorithm,
                        case=case,
                        backend=backend,
                        max_absolute_differences=differences,
                        log_evidence_difference=delta,
                        worker_sha256=sha(p),
                    )
                )
    destination.write_text(json.dumps(rows, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--canonical-output", type=Path, required=True)
    p.add_argument(
        "--phase",
        choices=(
            "fixtures",
            "small-reference",
            "small-precision",
            "scientific",
            "scaling",
            "search",
            "source-axis",
            "summary",
        ),
        required=True,
    )
    p.add_argument("--summary", type=Path)
    args = p.parse_args()
    out = args.output
    (out / "fixtures").mkdir(parents=True, exist_ok=True)
    (out / "workers").mkdir(exist_ok=True)
    references(args.canonical_output, out)
    if args.phase == "fixtures":
        for case in ("coupled37", "scientific32", "scientific64", "scientific200"):
            for observation in (
                ("subhalo", "control") if case == "scientific200" else ("subhalo",)
            ):
                name = (
                    case + ("" if case == "coupled37" else "-" + observation) + ".npz"
                )
                command = ["--prepare", "--case", case, "--observation", observation]
                if case == "scientific200":
                    command += [
                        "--reference",
                        str(
                            args.canonical_output
                            / "iterative-scientific/validated-reference"
                        ),
                    ]
                invoke(command, out / "fixtures" / name)
    elif args.phase == "small-reference":
        for alg in ("single_step", "iterative"):
            for backend in ("dense", "operator"):
                fit(out, "coupled37", alg, backend, "float64", "strict")
    elif args.phase in ("small-precision", "search"):
        for alg in ("single_step", "iterative"):
            for b in ("dense", "operator"):
                r = json.loads(
                    (
                        out / "workers" / f"coupled37-{alg}-{b}-float64-strict.json"
                    ).read_text()
                )
                assert r["result"]["success"]
        for dtype in ("float64", "float32"):
            for alg in ("single_step", "iterative"):
                for backend in ("dense", "operator"):
                    fit(
                        out,
                        "coupled37",
                        alg,
                        backend,
                        dtype,
                        "small32",
                        repeats=5 if args.phase == "small-precision" else 0,
                        search=25 if args.phase == "search" else 0,
                    )
    elif args.phase == "source-axis":
        fixture = out / "fixtures/scientific64-source65-subhalo.npz"
        invoke(
            [
                "--prepare",
                "--case",
                "scientific64",
                "--source-n",
                "65",
                "--reference-inputs",
                str(out / "fixtures/scientific64-subhalo.npz"),
            ],
            fixture,
        )
        fit(
            out,
            "scientific64-source65",
            "single_step",
            "operator",
            "float64",
            "scientific",
            repeats=1,
        )
    elif args.phase in ("scientific", "scaling"):
        cases = (
            ("scientific200",)
            if args.phase == "scientific"
            else ("scientific32", "scientific64")
        )
        for dtype in (
            ("float64", "float32") if args.phase == "scientific" else ("float64",)
        ):
            for case in cases:
                for observation in (
                    ("subhalo", "control") if case == "scientific200" else ("subhalo",)
                ):
                    for alg in ("single_step", "iterative"):
                        for backend in ("dense", "operator"):
                            fit(
                                out,
                                case,
                                alg,
                                backend,
                                dtype,
                                "scientific",
                                observation=observation,
                                repeats=1,
                            )
    else:
        if args.summary is None:
            p.error("--summary is required for summary phase")
        summarize(out, args.summary)
        reference_reproduction(
            args.canonical_output,
            out,
            args.summary.with_name("reference-reproduction.json"),
        )


if __name__ == "__main__":
    main()
