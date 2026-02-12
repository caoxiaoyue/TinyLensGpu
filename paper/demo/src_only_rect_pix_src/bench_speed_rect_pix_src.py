"""Benchmark rectangular bilinear pixelized-source backend performance.

This script benchmarks the rectangular-source-grid workflow across selected
regularization schemes, grid resolutions, and inversion backends.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from demo_rect_pix_src import (
    reconstruct_source,
    setup_rectangular_pixelized_model,
    simulate_lensing_data,
)


def _time_call(fn, n_runs: int) -> dict:
    vals = []
    for _ in range(int(n_runs)):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        t1 = time.perf_counter()
        vals.append(t1 - t0)
    arr = np.array(vals, dtype=np.float64)
    return {
        "n_runs": int(n_runs),
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=1) if int(n_runs) > 1 else 0.0),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def _current_peak_mem_mib() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        vals = [float(x.strip()) for x in out.splitlines() if x.strip()]
        return max(vals) if vals else None
    except Exception:
        return None


class _GpuMemorySampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = float(interval_s)
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self):
        while not self._stop.is_set():
            mem = _current_peak_mem_mib()
            if mem is not None:
                self.samples.append(mem)
            time.sleep(self.interval_s)

    def start(self) -> bool:
        if shutil.which("nvidia-smi") is None:
            return False
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return True

    def stop_and_peak(self) -> float | None:
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        if not self.samples:
            return None
        return float(max(self.samples))


def _parse_grid_shapes(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"Invalid grid shape token '{token}', expected format NxM (e.g. 32x32).")
        sx, sy = token.split("x", 1)
        nx = int(sx)
        ny = int(sy)
        if nx <= 0 or ny <= 0:
            raise ValueError(f"Grid shape must be positive, got {token}.")
        out.append((nx, ny))
    if not out:
        raise ValueError("No valid grid shapes were parsed.")
    return out


def _parse_reg_types(text: str) -> List[str]:
    valid = {"zero", "gradient", "curvature"}
    out = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("No regularization schemes provided.")
    for scheme in out:
        if scheme not in valid:
            raise ValueError(f"Invalid rect_reg_type '{scheme}', choose from {sorted(valid)}.")
    return out


def _solve_once(prob_model):
    data_vector = prob_model.image_data[~prob_model.mask]
    noise_variance = prob_model.noise_map[~prob_model.mask] ** 2
    inverter = prob_model.simulator.build_inverter(
        data_vector=data_vector,
        noise_variance=noise_variance,
        reg_scale=prob_model.pix_src_model.reg_scale.value,
        reg_coefficient=prob_model.pix_src_model.reg_coefficient.value,
    )
    return inverter.solve()


def _run_case(
    *,
    data_dict: Dict[str, Any],
    inversion_backend: str,
    source_grid_nx: int,
    source_grid_ny: int,
    source_grid_margin_frac: float,
    rect_reg_type: str,
    n_runs: int,
    nonnegative: bool,
    cg_tol: float,
    cg_maxiter: int,
    slq_probes: int,
    slq_steps: int,
    evidence_mode: str,
    operator_cache_policy: str,
) -> Dict[str, Any]:
    model = setup_rectangular_pixelized_model(
        data_dict,
        inversion_backend=inversion_backend,
        source_grid_nx=source_grid_nx,
        source_grid_ny=source_grid_ny,
        source_grid_margin_frac=source_grid_margin_frac,
        rect_reg_type=rect_reg_type,
        nonnegative=nonnegative,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        slq_probes=slq_probes,
        slq_steps=slq_steps,
        evidence_mode=evidence_mode,
        operator_cache_policy=operator_cache_policy,
    )

    # Warmup
    _ = model().block_until_ready()
    _ = reconstruct_source(model)

    sampler = _GpuMemorySampler()
    sampler_started = sampler.start()

    t_log_ev = _time_call(lambda: model(), n_runs)
    t_solve = _time_call(lambda: _solve_once(model), n_runs)
    t_reconstruct = _time_call(lambda: reconstruct_source(model), n_runs)

    peak_mem_abs = sampler.stop_and_peak() if sampler_started else _current_peak_mem_mib()

    result = reconstruct_source(model)

    return {
        "inversion_backend": str(inversion_backend),
        "source_grid_nx": int(source_grid_nx),
        "source_grid_ny": int(source_grid_ny),
        "rect_reg_type": rect_reg_type,
        "log_evidence": float(result["log_evidence"]),
        "n_source": int(result["source_intensities"].shape[0]),
        "n_data": int(np.sum(~data_dict["mask"])),
        "timing": {
            "log_evidence_only": t_log_ev,
            "solve_only": t_solve,
            "reconstruct_source_total": t_reconstruct,
        },
        "vram_peak_abs_mib": peak_mem_abs,
    }


def _run_benchmark(
    *,
    n_runs: int,
    seed: int,
    inversion_backends: Iterable[str],
    grid_shapes: Iterable[Tuple[int, int]],
    rect_reg_types: Iterable[str],
    source_grid_margin_frac: float,
    nonnegative: bool,
    cg_tol: float,
    cg_maxiter: int,
    slq_probes: int,
    slq_steps: int,
    evidence_mode: str,
    operator_cache_policy: str,
) -> Dict[str, Any]:
    np.random.seed(seed)
    data_dict = simulate_lensing_data(seed=seed)

    baseline_mem = _current_peak_mem_mib()

    cases = []
    for backend in inversion_backends:
        for (nx, ny) in grid_shapes:
            for scheme in rect_reg_types:
                case = _run_case(
                    data_dict=data_dict,
                    inversion_backend=str(backend),
                    source_grid_nx=nx,
                    source_grid_ny=ny,
                    source_grid_margin_frac=source_grid_margin_frac,
                    rect_reg_type=scheme,
                    n_runs=n_runs,
                    nonnegative=nonnegative,
                    cg_tol=cg_tol,
                    cg_maxiter=cg_maxiter,
                    slq_probes=slq_probes,
                    slq_steps=slq_steps,
                    evidence_mode=evidence_mode,
                    operator_cache_policy=operator_cache_policy,
                )
                if case["vram_peak_abs_mib"] is not None and baseline_mem is not None:
                    case["vram_peak_delta_mib"] = float(max(0.0, case["vram_peak_abs_mib"] - baseline_mem))
                else:
                    case["vram_peak_delta_mib"] = None
                cases.append(case)

    # Best-case summary by solve-only mean
    best_case = min(cases, key=lambda c: c["timing"]["solve_only"]["mean_s"])

    payload = {
        "seed": int(seed),
        "n_runs": int(n_runs),
        "inversion_backends": [str(b) for b in inversion_backends],
        "source_grid_margin_frac": float(source_grid_margin_frac),
        "nonnegative": bool(nonnegative),
        "evidence_mode": evidence_mode,
        "operator_cache_policy": operator_cache_policy,
        "cg_tol": float(cg_tol),
        "cg_maxiter": int(cg_maxiter),
        "slq_probes": int(slq_probes),
        "slq_steps": int(slq_steps),
        "vram_baseline_abs_mib": baseline_mem,
        "cases": cases,
        "best_by_solve_only": {
            "inversion_backend": str(best_case["inversion_backend"]),
            "source_grid_nx": int(best_case["source_grid_nx"]),
            "source_grid_ny": int(best_case["source_grid_ny"]),
            "rect_reg_type": str(best_case["rect_reg_type"]),
            "solve_only_mean_s": float(best_case["timing"]["solve_only"]["mean_s"]),
            "reconstruct_mean_s": float(best_case["timing"]["reconstruct_source_total"]["mean_s"]),
            "log_evidence_mean_s": float(best_case["timing"]["log_evidence_only"]["mean_s"]),
        },
    }

    return payload


def main():
    parser = argparse.ArgumentParser(description="Benchmark rectangular bilinear pixelized-source backend speed")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inversion-backends", type=str, default="operator,matrix")
    parser.add_argument("--grid-shapes", type=str, default="24x24,32x32,48x48")
    parser.add_argument("--rect-reg-types", type=str, default="zero,gradient,curvature")
    parser.add_argument("--source-grid-margin-frac", type=float, default=0.10)
    parser.add_argument("--nonnegative", action="store_true")
    parser.add_argument("--cg_tol", type=float, default=1e-4)
    parser.add_argument("--cg_maxiter", type=int, default=120)
    parser.add_argument("--slq_probes", type=int, default=32)
    parser.add_argument("--slq_steps", type=int, default=60)
    parser.add_argument("--evidence-mode", choices=["fast", "accurate"], default="accurate")
    parser.add_argument("--operator-cache-policy", choices=["off", "safe", "unsafe_static"], default="safe")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("bench_speed_rect_pix_src_results.json"),
    )
    args = parser.parse_args()

    inversion_backends = [x.strip().lower() for x in args.inversion_backends.split(",") if x.strip()]
    if not inversion_backends:
        raise ValueError("No inversion backends were provided.")
    for backend in inversion_backends:
        if backend not in {"matrix", "operator"}:
            raise ValueError(f"Invalid inversion backend '{backend}', choose from ['matrix', 'operator'].")

    grid_shapes = _parse_grid_shapes(args.grid_shapes)
    rect_reg_types = _parse_reg_types(args.rect_reg_types)

    payload = _run_benchmark(
        n_runs=args.n_runs,
        seed=args.seed,
        inversion_backends=inversion_backends,
        grid_shapes=grid_shapes,
        rect_reg_types=rect_reg_types,
        source_grid_margin_frac=args.source_grid_margin_frac,
        nonnegative=bool(args.nonnegative),
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
        slq_probes=args.slq_probes,
        slq_steps=args.slq_steps,
        evidence_mode=args.evidence_mode,
        operator_cache_policy=args.operator_cache_policy,
    )

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
