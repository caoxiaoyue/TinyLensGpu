"""Benchmark matrix vs operator backend speed and VRAM usage.

This benchmark runs each backend in an isolated subprocess to improve
comparability of GPU memory measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _time_call(fn, n_runs: int) -> dict:
    vals = []
    for _ in range(n_runs):
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
        "std_s": float(arr.std(ddof=1) if n_runs > 1 else 0.0),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def _current_peak_mem() -> float | None:
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
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            mem = _current_peak_mem()
            if mem is not None:
                self.samples.append(mem)
            time.sleep(self.interval_s)

    def start(self):
        if shutil.which("nvidia-smi") is None:
            return False
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return True

    def stop_and_peak(self):
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        if not self.samples:
            return None
        return float(max(self.samples))


def _run_backend_inproc(
    *,
    backend: str,
    n_runs: int,
    seed: int,
    nonnegative: bool,
    evidence_mode: str,
    operator_cache_policy: str,
    reg_operator_mode: str,
    reg_sparse_k_neighbors: int,
) -> Dict[str, Any]:
    from demo_pix_src import simulate_lensing_data, setup_pixelized_model, reconstruct_source

    np.random.seed(seed)
    data_dict = simulate_lensing_data()

    model = setup_pixelized_model(
        data_dict,
        backend=backend,
        nonnegative=nonnegative,
        evidence_mode=evidence_mode,
        operator_cache_policy=operator_cache_policy,
        reg_operator_mode=reg_operator_mode,
        reg_sparse_k_neighbors=reg_sparse_k_neighbors,
    )

    # Warmup
    _ = model().block_until_ready()
    _ = reconstruct_source(model)

    # Timings
    t_log_ev = _time_call(lambda: model(), n_runs)
    t_total = _time_call(lambda: reconstruct_source(model), n_runs)

    # Solve-only
    data_vector = model.image_data[~model.mask]
    noise_variance = model.noise_map[~model.mask] ** 2
    reg_scale = model.pix_src_model.reg_scale.value
    reg_coefficient = model.pix_src_model.reg_coefficient.value

    def _solve_once():
        inverter = model.simulator.build_inverter(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
        )
        return inverter.solve()

    t_solve = _time_call(_solve_once, n_runs)

    result = reconstruct_source(model)

    return {
        "backend": backend,
        "nonnegative": bool(nonnegative),
        "log_evidence": float(result["log_evidence"]),
        "n_source": int(result["source_intensities"].shape[0]),
        "timing": {
            "log_evidence_only": t_log_ev,
            "reconstruct_source_total": t_total,
            "solve_only": t_solve,
        },
        "result": {
            "source_intensities": result["source_intensities"].tolist(),
            "model_image": result["model_image"].tolist(),
        },
        "n_data": int(np.sum(~data_dict["mask"])),
    }


def _run_backend_subprocess(
    *,
    backend: str,
    n_runs: int,
    seed: int,
    nonnegative: bool,
    evidence_mode: str,
    operator_cache_policy: str,
    reg_operator_mode: str,
    reg_sparse_k_neighbors: int,
) -> Dict[str, Any]:
    cmd = [
        os.environ.get("PYTHON", "python"),
        str(Path(__file__).resolve()),
        "--mode",
        "worker",
        "--backend",
        backend,
        "--n-runs",
        str(n_runs),
        "--seed",
        str(seed),
        "--evidence-mode",
        evidence_mode,
        "--operator-cache-policy",
        operator_cache_policy,
        "--reg-operator-mode",
        reg_operator_mode,
        "--reg-sparse-k-neighbors",
        str(int(reg_sparse_k_neighbors)),
    ]
    if nonnegative:
        cmd.append("--nonnegative")

    env = dict(os.environ)
    # Disable large upfront reservation for fairer peak VRAM comparison.
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    out = proc.stdout.strip().splitlines()
    payload_line = None
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            payload_line = line
            break
    if payload_line is None:
        # Fallback: try parse full stdout
        payload_line = proc.stdout.strip()
    return json.loads(payload_line)


def _run_parent(
    *,
    n_runs: int,
    seed: int,
    nonnegative: bool,
    evidence_mode: str,
    operator_cache_policy: str,
    reg_operator_mode: str,
    reg_sparse_k_neighbors: int,
) -> Dict[str, Any]:
    base_mem = _current_peak_mem()

    sampler_matrix = _GpuMemorySampler()
    has_matrix_sampler = sampler_matrix.start()
    matrix = _run_backend_subprocess(
        backend="matrix",
        n_runs=n_runs,
        seed=seed,
        nonnegative=nonnegative,
        evidence_mode=evidence_mode,
        operator_cache_policy=operator_cache_policy,
        reg_operator_mode=reg_operator_mode,
        reg_sparse_k_neighbors=reg_sparse_k_neighbors,
    )
    matrix_peak_abs = sampler_matrix.stop_and_peak() if has_matrix_sampler else _current_peak_mem()

    sampler_operator = _GpuMemorySampler()
    has_operator_sampler = sampler_operator.start()
    operator = _run_backend_subprocess(
        backend="operator",
        n_runs=n_runs,
        seed=seed,
        nonnegative=nonnegative,
        evidence_mode=evidence_mode,
        operator_cache_policy=operator_cache_policy,
        reg_operator_mode=reg_operator_mode,
        reg_sparse_k_neighbors=reg_sparse_k_neighbors,
    )
    operator_peak_abs = sampler_operator.stop_and_peak() if has_operator_sampler else _current_peak_mem()

    def _delta(abs_peak):
        if abs_peak is None or base_mem is None:
            return None
        return float(max(0.0, abs_peak - base_mem))

    matrix_peak_delta = _delta(matrix_peak_abs)
    operator_peak_delta = _delta(operator_peak_abs)

    src_matrix = np.asarray(matrix["result"]["source_intensities"], dtype=np.float32)
    src_operator = np.asarray(operator["result"]["source_intensities"], dtype=np.float32)
    model_matrix = np.asarray(matrix["result"]["model_image"], dtype=np.float32)
    model_operator = np.asarray(operator["result"]["model_image"], dtype=np.float32)

    src_diff = np.max(np.abs(src_matrix - src_operator))
    model_diff = np.max(np.abs(model_matrix - model_operator))
    logev_diff = abs(float(matrix["log_evidence"]) - float(operator["log_evidence"]))

    payload = {
        "seed": int(seed),
        "n_runs": int(n_runs),
        "nonnegative": bool(nonnegative),
        "evidence_mode": evidence_mode,
        "operator_cache_policy": operator_cache_policy,
        "reg_operator_mode": reg_operator_mode,
        "reg_sparse_k_neighbors": int(reg_sparse_k_neighbors),
        "n_data": int(matrix["n_data"]),
        "matrix": {
            "log_evidence": float(matrix["log_evidence"]),
            "timing": matrix["timing"],
        },
        "operator": {
            "log_evidence": float(operator["log_evidence"]),
            "timing": operator["timing"],
        },
        "consistency_metrics": {
            "max_abs_source_diff": float(src_diff),
            "max_abs_model_diff": float(model_diff),
            "abs_log_evidence_diff": float(logev_diff),
        },
        "vram_mib": {
            "baseline_abs": base_mem,
            "matrix_peak_abs": matrix_peak_abs,
            "operator_peak_abs": operator_peak_abs,
            "matrix_peak_delta": matrix_peak_delta,
            "operator_peak_delta": operator_peak_delta,
        },
        "speed_ratio": {
            "operator_over_matrix_log_evidence": float(
                operator["timing"]["log_evidence_only"]["mean_s"]
                / matrix["timing"]["log_evidence_only"]["mean_s"]
            ),
            "operator_over_matrix_reconstruct_total": float(
                operator["timing"]["reconstruct_source_total"]["mean_s"]
                / matrix["timing"]["reconstruct_source_total"]["mean_s"]
            ),
            "operator_over_matrix_solve_only": float(
                operator["timing"]["solve_only"]["mean_s"]
                / matrix["timing"]["solve_only"]["mean_s"]
            ),
        },
    }

    out = Path(__file__).with_name("bench_speed_pix_src_results.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parent", "worker"], default="parent")
    parser.add_argument("--backend", choices=["matrix", "operator"], default="matrix")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nonnegative", action="store_true")
    parser.add_argument("--evidence-mode", choices=["fast", "accurate"], default="accurate")
    parser.add_argument("--operator-cache-policy", choices=["off", "safe", "unsafe_static"], default="safe")
    parser.add_argument("--reg-operator-mode", choices=["dense_gp", "sparse_knn"], default="dense_gp")
    parser.add_argument("--reg-sparse-k-neighbors", type=int, default=16)
    args = parser.parse_args()

    if args.mode == "worker":
        payload = _run_backend_inproc(
            backend=args.backend,
            n_runs=args.n_runs,
            seed=args.seed,
            nonnegative=bool(args.nonnegative),
            evidence_mode=args.evidence_mode,
            operator_cache_policy=args.operator_cache_policy,
            reg_operator_mode=args.reg_operator_mode,
            reg_sparse_k_neighbors=args.reg_sparse_k_neighbors,
        )
        print(json.dumps(payload, sort_keys=True))
        return

    payload = _run_parent(
        n_runs=args.n_runs,
        seed=args.seed,
        nonnegative=bool(args.nonnegative),
        evidence_mode=args.evidence_mode,
        operator_cache_policy=args.operator_cache_policy,
        reg_operator_mode=args.reg_operator_mode,
        reg_sparse_k_neighbors=args.reg_sparse_k_neighbors,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
