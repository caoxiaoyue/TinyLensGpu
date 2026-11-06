"""Serial, quality-gated ablations of the standalone Gauss--Newton prototype.

Each variant gets a fresh process; its repeated fits retain compiled kernels
but rebuild numerical metrics/factors. Failed fits remain in the summary.
Run from the repository root in tinylens_gpu. No JAX imports in this parent.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VARIANTS = {
    "baseline": [],
    "candidate": ["--gate-policy", "candidate"],
    "tol1e8": ["--lsmr-tol", "1e-8"],
    "tol1e6": ["--lsmr-tol", "1e-6"],
    "cached": ["--linearization", "cached"],
    "combined": [
        "--gate-policy",
        "candidate",
        "--linearization",
        "cached",
        "--lsmr-tol",
        "1e-8",
    ],
    "bridge": ["--bridge-profile"],
    "explicit": ["--whitening", "explicit"],
    "compiled_every": [
        "--linearization",
        "cached",
        "--lsmr-tol",
        "1e-6",
        "--whitening",
        "explicit",
        "--diagonal-backend",
        "compiled",
    ],
    "compiled_candidate": [
        "--linearization",
        "cached",
        "--lsmr-tol",
        "1e-6",
        "--whitening",
        "explicit",
        "--diagonal-backend",
        "compiled",
        "--gate-policy",
        "candidate",
    ],
    "combined6": [
        "--gate-policy",
        "candidate",
        "--linearization",
        "cached",
        "--lsmr-tol",
        "1e-6",
    ],
    "fast": [
        "--gate-policy",
        "candidate",
        "--linearization",
        "cached",
        "--lsmr-tol",
        "1e-6",
        "--whitening",
        "explicit",
    ],
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def decomposition(row):
    t = row["timings"]
    actions = sum(t[k + "_seconds"] for k in ("matvec", "rmatvec", "matmat"))
    linearization = t.get("linearization_seconds", 0.0)
    return dict(
        total=t["total_seconds"],
        setup=t["setup_seconds"],
        jacobian_actions=actions,
        linearization=linearization,
        callback=t["external_gradient_seconds"],
        residual=t["residual_seconds"],
        solver_other=t["scipy_api_seconds"]
        - actions
        - linearization
        - t["external_gradient_seconds"]
        - t["residual_seconds"],
        endpoint=t["certificate_seconds"] + t["product_seconds"],
        outside_accounting=t["total_seconds"]
        - t["setup_seconds"]
        - t["scipy_api_seconds"]
        - t["certificate_seconds"]
        - t["product_seconds"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=HERE / "gauss_newton_performance"
    )
    parser.add_argument(
        "--variants",
        choices=tuple(VARIANTS),
        nargs="+",
        default=["baseline", "candidate", "tol1e8", "tol1e6", "cached", "combined"],
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("subhalo", "control"),
        default=["subhalo", "control"],
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=45.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "study.json").exists():
        raise RuntimeError("Use a fresh output directory")
    runner = HERE / "run_gauss_newton.py"
    runner_sha256 = hashlib.sha256(runner.read_bytes()).hexdigest()
    shutil.copyfile(Path(__file__), args.output / "profile_snapshot.py")
    write(
        args.output / "study.json",
        dict(
            runner_sha256=runner_sha256,
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            policy="Serial isolated variants; warm fits recompute numerical state; no overlapping GPU work",
        ),
    )
    summary = {}
    for variant in args.variants:
        directory = args.output / variant
        directory.mkdir()
        command = [
            sys.executable,
            "-u",
            str(runner),
            "--cases",
            *args.cases,
            "--repeats",
            str(args.repeats),
            "--max-seconds",
            str(args.max_seconds),
            "--output",
            str(directory),
            *VARIANTS[variant],
        ]
        write(directory / "command.json", command)
        print("START", variant, flush=True)
        with (directory / "console.log").open("w") as log:
            process = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        result_file = directory / "summary.json"
        if not result_file.exists():
            summary[variant] = dict(
                returncode=process.returncode,
                error="No numerical summary; see console.log",
            )
            write(args.output / "summary.json", summary)
            print("ERROR", variant, "no summary", flush=True)
            continue
        rows = json.loads(result_file.read_text())
        env = json.loads((directory / "environment.json").read_text())
        actual = hashlib.sha256(
            (directory / "runner_snapshot.py").read_bytes()
        ).hexdigest()
        assert (
            actual
            == env["source_sha256"]["workspace/pt_new_algorithm/run_gauss_newton.py"]
        )
        assert (
            actual == hashlib.sha256(runner.read_bytes()).hexdigest()
        ), "Runner changed during study"
        assert actual == runner_sha256, "Runner changed between variants"
        by_case = {}
        for case in args.cases:
            subset = {k: r for k, r in rows.items() if k.startswith(case + "/")}
            warm = {k: r for k, r in subset.items() if k != case + "/0"}
            passed = [r for r in warm.values() if r["accepted_reconstruction"]]
            by_case[case] = dict(
                attempts=len(subset),
                accepted=sum(r["accepted_reconstruction"] for r in subset.values()),
                warm_attempts=len(warm),
                warm_accepted=len(passed),
                successful_warm_median=(
                    None
                    if not passed
                    else statistics.median(
                        r["timings"]["total_seconds"] for r in passed
                    )
                ),
                rows={
                    k: dict(
                        accepted=r["accepted_reconstruction"],
                        objective=r["objective"],
                        scaled_gradient=r["scaled_gradient"],
                        threshold=r["gradient_threshold"],
                        decomposition=decomposition(r),
                        counts=r["counts"],
                        iterations=r["iterations"],
                        bridge={
                            k: r["timings"].get(k)
                            for k in (
                                "host_to_device_seconds",
                                "device_call_seconds",
                                "device_to_host_seconds",
                            )
                        },
                    )
                    for k, r in subset.items()
                },
            )
            print(
                "RESULT",
                variant,
                case,
                f"accepted {by_case[case]['accepted']}/{len(subset)}",
                "warm median",
                by_case[case]["successful_warm_median"],
                flush=True,
            )
        summary[variant] = dict(returncode=process.returncode, cases=by_case)
        write(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
