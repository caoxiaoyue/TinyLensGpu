"""Observable isolated workflow measurements, separate from core correctness."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.performance
def test_isolated_workflow_reports_synchronized_phases_and_preparation_peak(tmp_path):
    script = Path(__file__).with_name("potential_correction_workflow.py")
    fixture = tmp_path / "coupled37.npz"
    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
    }
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--prepare",
            "--case",
            "coupled37",
            "--output",
            str(fixture),
        ],
        check=True,
        env=env,
        timeout=90,
    )
    output = tmp_path / "fit.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--inputs",
            str(fixture),
            "--algorithm",
            "single_step",
            "--backend",
            "operator",
            "--dtype",
            "float64",
            "--repeats",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        env=env,
        timeout=120,
    )
    row = json.loads(output.read_text())
    assert row["result"]["success"]
    assert row["counts"]["data"] == 16 and row["counts"]["joint"] == 37
    assert row["timing"]["setup_seconds"] > 0
    assert row["timing"]["first_fit_compile_inclusive_seconds"] > 0
    assert len(row["timing"]["warm_fit_seconds"]) == 2
    assert all(t > 0 for t in row["timing"]["warm_fit_seconds"])
    assert row["timing"]["cold_workflow_seconds"] >= (
        row["timing"]["setup_seconds"]
        + row["timing"]["first_fit_compile_inclusive_seconds"]
    )
    assert [s["phase"] for s in row["allocator_snapshots"]] == [
        "device_initialized",
        "setup",
        "first_fit",
        "warm_fit",
    ]
    peak = row["allocator_snapshots"][-1]["stats"].get("peak_bytes_in_use")
    assert peak is None or peak > 0
    assert row["host_peak_rss_bytes"] > 0
    assert row["software"]["platform"] == "gpu"
    assert row["dtype_audit"]["psf_fft"] == "complex128"
    assert row["dtype_audit"]["prior_action"] == "float64"
    assert row["precision_defaults"]["before"] == row["precision_defaults"]["after"]
    assert row["input_sha256"] and row["geometry"]["integer_topology_matches_fixture"]
    assert (
        row["process_gpu_memory"]["semantics"] == "sampled_worker_residency_lower_bound"
    )


@pytest.mark.performance
def test_actual_fixed_image_source_axis_avoids_live_full_joint_matrices(tmp_path):
    script = Path(__file__).with_name("potential_correction_workflow.py")
    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
    }
    base, axis = tmp_path / "base.npz", tmp_path / "axis.npz"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--prepare",
            "--case",
            "scientific64",
            "--output",
            str(base),
        ],
        check=True,
        env=env,
        timeout=90,
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--prepare",
            "--case",
            "scientific64",
            "--source-n",
            "65",
            "--reference-inputs",
            str(base),
            "--output",
            str(axis),
        ],
        check=True,
        env=env,
        timeout=90,
    )
    a, b = np.load(base), np.load(axis)
    for name in ("image_data", "noise_map", "psf", "mask"):
        np.testing.assert_array_equal(a[name], b[name])
    output = tmp_path / "axis.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--inputs",
            str(axis),
            "--algorithm",
            "single_step",
            "--backend",
            "operator",
            "--dtype",
            "float64",
            "--repeats",
            "0",
            "--output",
            str(output),
        ],
        check=True,
        env=env,
        timeout=180,
    )
    row = json.loads(output.read_text())
    assert row["counts"]["data"] == 2252 and row["counts"]["source"] == 4225
    assert row["counts"]["support"] == 592
    assert row["configuration"]["source_axis_observations_match"]
    assert row["live_arrays"]["prohibited_live_shapes"] == []
    assert row["result"]["success"], row["result"]["termination_reason"]
    peak = row["allocator_snapshots"][-1]["stats"].get("peak_bytes_in_use")
    assert peak is None or peak < 11 * 1024**3
