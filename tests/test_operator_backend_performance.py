"""Performance comparison for matrix vs operator backends."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import jax

from paper.demo.src_only_pix_src.demo_pix_src import simulate_lensing_data, setup_pixelized_model


def _time_call(fn, n_runs: int = 3) -> float:
    vals = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        t1 = time.perf_counter()
        vals.append(t1 - t0)
    return float(np.mean(vals))


def _time_reconstruct(prob_model, n_runs: int = 3) -> float:
    data_vector = prob_model.image_data[~prob_model.mask]
    noise_variance = prob_model.noise_map[~prob_model.mask] ** 2
    reg_scale = prob_model.pix_src_model.reg_scale.value
    reg_coefficient = prob_model.pix_src_model.reg_coefficient.value

    def _one():
        return prob_model.simulator.reconstruct_source(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
            return_2d=False,
        )

    vals = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        source_intensities, source_mesh_beta, model_data, inverter = _one()
        source_intensities.block_until_ready()
        source_mesh_beta.block_until_ready()
        model_data.block_until_ready()
        if hasattr(inverter, "d"):
            inverter.d.block_until_ready()
        t1 = time.perf_counter()
        vals.append(t1 - t0)
    return float(np.mean(vals))


def _time_solve_only(prob_model, n_runs: int = 3) -> float:
    data_vector = prob_model.image_data[~prob_model.mask]
    noise_variance = prob_model.noise_map[~prob_model.mask] ** 2
    reg_scale = prob_model.pix_src_model.reg_scale.value
    reg_coefficient = prob_model.pix_src_model.reg_coefficient.value

    def _one():
        inverter = prob_model.simulator.build_inverter(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
        )
        return inverter.solve()

    vals = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        x = _one()
        x.block_until_ready()
        t1 = time.perf_counter()
        vals.append(t1 - t0)
    return float(np.mean(vals))


def _max_used_gpu_memory_mib() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        used = [float(line.strip()) for line in out if line.strip()]
        if not used:
            return None
        return float(max(used))
    except Exception:
        return None


@pytest.mark.performance
@pytest.mark.slow
def test_operator_backend_speed_and_vram(tmp_path):
    if jax.default_backend() != "gpu":
        pytest.skip("JAX GPU backend not available; skipping GPU benchmark")

    mem_before = _max_used_gpu_memory_mib()
    if mem_before is None:
        pytest.skip("nvidia-smi unavailable; skipping GPU VRAM benchmark")

    np.random.seed(0)
    data = simulate_lensing_data()

    model_matrix = setup_pixelized_model(
        data,
        backend="matrix",
        evidence_mode="accurate",
        operator_cache_policy="safe",
    )
    model_operator = setup_pixelized_model(
        data,
        backend="operator",
        evidence_mode="accurate",
        operator_cache_policy="safe",
    )

    # Warmup
    _ = model_matrix().block_until_ready()
    _ = model_operator().block_until_ready()
    _ = _time_reconstruct(model_matrix, n_runs=1)
    _ = _time_reconstruct(model_operator, n_runs=1)

    t_matrix = _time_reconstruct(model_matrix, n_runs=3)
    mem_mid = _max_used_gpu_memory_mib() or mem_before

    t_operator = _time_reconstruct(model_operator, n_runs=3)
    mem_after = _max_used_gpu_memory_mib() or mem_mid

    t_matrix_solve = _time_solve_only(model_matrix, n_runs=3)
    t_operator_solve = _time_solve_only(model_operator, n_runs=3)

    # Also report evidence timing for reference, but don't hard-fail on it.
    t_matrix_log_ev = _time_call(lambda: model_matrix(), n_runs=3)
    t_operator_log_ev = _time_call(lambda: model_operator(), n_runs=3)

    peak_matrix = max(mem_before, mem_mid)
    peak_operator = max(mem_mid, mem_after)

    # Primary criterion: solve-only operator path should not be materially slower.
    assert t_operator_solve <= 1.10 * t_matrix_solve, (
        f"operator backend slower in solve_only: operator={t_operator_solve:.4f}s matrix={t_matrix_solve:.4f}s"
    )
    assert peak_operator <= peak_matrix + 512.0, (
        f"operator VRAM not improved: operator={peak_operator:.1f} MiB matrix={peak_matrix:.1f} MiB"
    )

    report = {
        "matrix_reconstruct_time_s": t_matrix,
        "operator_reconstruct_time_s": t_operator,
        "matrix_solve_only_time_s": t_matrix_solve,
        "operator_solve_only_time_s": t_operator_solve,
        "matrix_log_evidence_time_s": t_matrix_log_ev,
        "operator_log_evidence_time_s": t_operator_log_ev,
        "matrix_peak_vram_mib": peak_matrix,
        "operator_peak_vram_mib": peak_operator,
    }
    out = Path(tmp_path) / "operator_backend_perf_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
