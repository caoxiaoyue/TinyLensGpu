"""Repeat the physical mock's single-step precision comparison, Dense first.

Uses the existing isolated workflow benchmark without changing library defaults.
An explicit optional residual-tolerance probe changes only solve_rtol in a copy
of the fixture metadata. All physical inputs and regularization stay fixed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.potential_correction_workflow import write_json  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_rounding(fixture):
    """Audit whether adding the physical amplitude prior survives float32."""
    with np.load(fixture) as a:
        cfg = json.loads(str(a["metadata"]))
        nodes = a["node_indices"]
    lookup = {tuple(n): i for i, n in enumerate(nodes)}
    diagonal = np.zeros(len(nodes))
    for dx, dy in ((1, 0), (0, 1)):
        for x, y in nodes:
            window = [(x + i * dx, y + i * dy) for i in range(5)]
            if all(n in lookup for n in window):
                for node, square in zip(window, (1, 16, 36, 16, 1)):
                    diagonal[lookup[node]] += square
    smoothing = np.asarray(cfg["lambda_potential"] * diagonal, dtype=np.float32)
    amplitude = np.float32(cfg["lambda_amplitude"])
    return dict(
        max_D4_gram_diagonal=diagonal.max(),
        max_float32_smoothing_diagonal=smoothing.max(),
        float32_ulp_at_max_diagonal=np.spacing(smoothing.max()),
        lambda_amplitude=cfg["lambda_amplitude"],
        amplitude_lost_diagonal_entries=np.count_nonzero(
            smoothing + amplitude == smoothing
        ),
        potential_nodes=len(nodes),
    )


def fixture_values(path):
    with np.load(path) as archive:
        return dict(archive)


def probe_fixture(derived, original, tolerance, read_only):
    """Reuse or create a solve_rtol-probe copy without changing physics."""
    values = fixture_values(original)
    expected = json.loads(str(values["metadata"]))
    expected["solve_rtol"] = tolerance
    if derived.exists():
        cached = fixture_values(derived)
        if set(cached) != set(values):
            raise ValueError(f"Cached probe inputs changed keys: {derived}")
        for key, original_array in values.items():
            if key == "metadata":
                if json.loads(str(cached[key])) != expected:
                    raise ValueError(
                        f"Cached probe metadata is not the original fixture with "
                        f"only solve_rtol={tolerance}: {derived}"
                    )
            else:
                try:
                    np.testing.assert_array_equal(cached[key], original_array)
                except AssertionError as error:
                    raise ValueError(
                        f"Cached probe array changed: {key} in {derived}"
                    ) from error
        return derived
    if read_only:
        raise FileNotFoundError(
            f"Missing probe inputs are not created under --summarize-only: {derived}"
        )
    values["metadata"] = json.dumps(expected)
    np.savez_compressed(derived, **values)
    return derived


def load_worker(path, *, backend, observation, dtype, fixture):
    """Read a worker record and validate it against its declared inputs."""
    row = json.loads(path.read_text())
    metadata = json.loads(str(fixture_values(fixture)["metadata"]))
    audit = row.get("dtype_audit", {})
    expected_fft = "complex64" if dtype == "float32" else "complex128"
    problems = []
    for key, wanted in (
        ("worker_returncode", 0),
        ("algorithm", "single_step"),
        ("backend", backend),
        ("case", metadata.get("name")),
        ("observation", observation),
        ("dtype", dtype),
        ("configuration", metadata),
        ("input_sha256", sha(fixture)),
    ):
        if row.get(key) != wanted:
            problems.append(f"{key}: expected {wanted!r}, found {row.get(key)!r}")
    solve_rtol = row.get("result", {}).get("configuration", {}).get("solve_rtol")
    if solve_rtol != metadata["solve_rtol"]:
        problems.append(
            f"result solve_rtol {solve_rtol!r} != fixture {metadata['solve_rtol']!r}"
        )
    if any(v not in (dtype, None) for k, v in audit.items() if k != "psf_fft"):
        problems.append(f"dtype_audit arrays are not {dtype}: {audit}")
    if audit.get("psf_fft") != expected_fft:
        problems.append(f"psf_fft {audit.get('psf_fft')!r} != {expected_fft!r}")
    if problems:
        raise ValueError(
            f"Worker record failed validation {path}: " + "; ".join(problems)
        )
    return row, audit


def compare(reference, candidate, *, reference_fixture, candidate_fixture):
    r, c = (json.loads(p.read_text()) for p in (reference, candidate))
    with np.load(reference_fixture) as ri, np.load(candidate_fixture) as ci:
        for key in ri.files:
            if key != "metadata":
                np.testing.assert_array_equal(ri[key], ci[key])
        rc, cc = (json.loads(str(a["metadata"])) for a in (ri, ci))
        rc.pop("solve_rtol", None)
        cc.pop("solve_rtol", None)
        assert rc == cc, "A physical or prior setting changed"
    with np.load(reference.with_suffix(".npz")) as a, np.load(
        candidate.with_suffix(".npz")
    ) as b:
        np.testing.assert_array_equal(
            a["potential_node_indices"], b["potential_node_indices"]
        )
        np.testing.assert_array_equal(a["derivative_valid"], b["derivative_valid"])
        valid = a["derivative_valid"]
        fields = {}
        for name in ("source_values", "potential_values", "convergence"):
            x, y = (np.asarray(z[name], dtype=np.float64) for z in (a, b))
            if name == "convergence":
                x, y = x[valid], y[valid]
            finite = np.isfinite(x) & np.isfinite(y)
            if r["result"]["success"] and c["result"]["success"]:
                assert np.all(finite)
            if not np.all(finite):
                fields[name] = dict(
                    finite_pairs=int(finite.sum()),
                    total=int(finite.size),
                    unavailable_reason="failed_fit_nonfinite_fields",
                )
                continue
            fields[name] = dict(
                max_absolute=np.max(abs(y - x)),
                rms=np.sqrt(np.mean((y - x) ** 2)),
                relative_l2=np.linalg.norm(y - x) / np.linalg.norm(x),
            )
        noise = np.asarray(a["noise_active"], dtype=np.float64)
        delta_image = (
            b["nonlinear_image_active"] - a["nonlinear_image_active"]
        ) / noise
        fields["nonlinear_image_difference_rms_noise"] = (
            float(np.sqrt(np.mean(delta_image**2)))
            if np.all(np.isfinite(delta_image))
            else None
        )
        nodes = a["potential_node_indices"]
        spacing = r["configuration"]["potential_spacing"]
        # The workflow renderer uses PotentialGrid's default origin (0, 0).
        xy = nodes * spacing
        peaks = {}
        for label, z in (("float64", a), ("float32", b)):
            if not np.all(np.isfinite(z["convergence"][valid])):
                peaks[label] = None
                continue
            index = np.flatnonzero(valid)[np.argmax(z["convergence"][valid])]
            peaks[label] = dict(
                kappa=float(z["convergence"][index]),
                position_arcsec=xy[index].tolist(),
                distance_to_injection_arcsec=float(
                    np.linalg.norm(xy[index] - [0.75, 0.1])
                ),
            )
    warm = c["timing"]["warm_fit_seconds"]
    rwarm = r["timing"]["warm_fit_seconds"]
    return dict(
        reference=reference.name,
        candidate=candidate.name,
        reference_success=r["result"]["success"],
        candidate_success=c["result"]["success"],
        fields=fields,
        peaks=peaks,
        log_evidence_difference=(
            None
            if c["result"]["log_evidence"] is None
            else c["result"]["log_evidence"] - r["result"]["log_evidence"]
        ),
        warm_speedup=(
            float(np.median(rwarm) / np.median(warm)) if warm and rwarm else None
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--probe-solve-rtol", type=float)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.probe_solve_rtol is not None and not (
        np.isfinite(args.probe_solve_rtol) and args.probe_solve_rtol > 0
    ):
        parser.error("--probe-solve-rtol must be a finite positive tolerance")
    if args.summarize_only:
        # Summarization treats the output directory and fixtures as read-only.
        if not args.output.is_dir():
            parser.error(
                f"--summarize-only requires existing worker records: {args.output}"
            )
    else:
        args.output.mkdir(parents=True, exist_ok=True)
    runs = {}
    worker_fixtures = {}
    probe_rtols = set()
    for backend in ("dense", "operator"):
        for profile in ("strict", "residual_probe"):
            if profile == "residual_probe" and args.probe_solve_rtol is None:
                continue
            for case in ("subhalo", "control"):
                fixture = args.fixtures / f"scientific200-{case}.npz"
                if profile == "residual_probe":
                    fixture = probe_fixture(
                        args.output / f"{case}-residual-probe-inputs.npz",
                        fixture,
                        args.probe_solve_rtol,
                        args.summarize_only,
                    )
                for dtype in ("float64", "float32"):
                    key = f"{backend}-{profile}-{case}-{dtype}"
                    path = args.output / f"{key}.json"
                    if not args.summarize_only:
                        if path.exists():
                            raise FileExistsError(f"Retain existing evidence: {path}")
                        command = [
                            sys.executable,
                            str(ROOT / "benchmarks/potential_correction_workflow.py"),
                            "--inputs",
                            str(fixture.resolve()),
                            "--algorithm",
                            "single_step",
                            "--backend",
                            backend,
                            "--dtype",
                            dtype,
                            "--profile",
                            "scientific",
                            "--repeats",
                            str(args.repeats),
                            "--output",
                            str(path.resolve()),
                        ]
                        print(key, flush=True)
                        subprocess.run(
                            command,
                            check=True,
                            cwd=ROOT,
                            env={
                                **os.environ,
                                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                                "JAX_ENABLE_COMPILATION_CACHE": "false",
                            },
                        )
                    row, audit = load_worker(
                        path,
                        backend=backend,
                        observation=case,
                        dtype=dtype,
                        fixture=fixture,
                    )
                    worker_fixtures[key] = fixture
                    if profile == "residual_probe":
                        probe_rtols.add(row["result"]["configuration"]["solve_rtol"])
                    runs[key] = dict(
                        result={
                            k: v
                            for k, v in row["result"].items()
                            if k not in ("configuration", "history")
                        },
                        timing=row["timing"],
                        counts=row["counts"],
                        dtype_audit=audit,
                        geometry=row["geometry"],
                        software=row["software"],
                        input_sha256=row["input_sha256"],
                        record_sha256=sha(path),
                        array_sha256=sha(path.with_suffix(".npz")),
                    )
    comparisons = {}
    for key in runs:
        if key.endswith("float32"):
            reference_key = key.replace("float32", "float64")
            comparisons[key] = compare(
                args.output / f"{reference_key}.json",
                args.output / f"{key}.json",
                reference_fixture=worker_fixtures[reference_key],
                candidate_fixture=worker_fixtures[key],
            )
    if len(probe_rtols) > 1:
        raise ValueError(
            f"Probe worker records disagree on solve_rtol: {sorted(probe_rtols)}"
        )
    write_json(
        args.summary or args.output / "summary.json",
        dict(
            scope="200x200 physical mock; single-step only; fixed priors; matched data",
            residual_probe_rtol=next(iter(probe_rtols), None),
            driver_sha256=sha(Path(__file__)),
            prior_rounding={
                case: prior_rounding(args.fixtures / f"scientific200-{case}.npz")
                for case in ("subhalo", "control")
            },
            runs=runs,
            comparisons=comparisons,
        ),
    )


if __name__ == "__main__":
    main()
