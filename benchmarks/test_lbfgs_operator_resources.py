"""Exercise the public operator fit and audit its compiled array shapes."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.performance
def test_large_lbfgs_operator_fit_uses_bounded_blocks_through_evidence():
    script = Path(__file__).with_name("lbfgs_operator_resources.py")
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    record = json.loads(completed.stdout)
    fit, audit, resources = (
        record["fit"],
        record["allocation_audit"],
        record["resources"],
    )
    assert fit["backend"] == "operator"
    assert fit["success"] and fit["iterations"] > 0
    assert fit["stationarity_solver"]["converged"]
    assert fit["stationarity_solver"]["iterations"] > 0
    assert fit["determinant_method"] == "principal_block_cholesky"
    assert fit["log_evidence"] is not None
    assert audit["compiled_modules"] > 0
    assert audit["bounded_block_shapes"]
    assert audit["violations"] == []
    assert resources["precision"] == "float64"
    assert resources["cold_fit_seconds_including_compilation"] > 0
    assert {
        "fixed_reference_seconds",
        "whitening_seconds",
        "optimizer_seconds",
        "endpoint_seconds",
        "result_seconds",
    } <= resources["fit_stage_seconds"].keys()
    assert "peak_device_bytes_in_use" in resources
