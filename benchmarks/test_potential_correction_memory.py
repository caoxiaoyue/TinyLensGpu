"""Operator storage evidence in an isolated process without old dense arrays."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.performance
@pytest.mark.parametrize("algorithm", ["single_step", "iterative"])
def test_operator_preparation_has_bounded_observable_allocation(algorithm):
    script = Path(__file__).with_name("potential_correction_memory.py")
    completed = subprocess.run(
        [sys.executable, str(script), "--algorithm", algorithm],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        timeout=180,
    )
    record = json.loads(completed.stdout.splitlines()[-1])
    assert record["algorithm"] == algorithm
    if algorithm == "iterative":
        assert record["completed_linear_solves"] >= 2
    source, potential, joint, data = (
        record[key] for key in ("n_source", "n_potential", "n_joint", "n_data")
    )
    prohibited = {
        (source, source),
        (potential, potential),
        (joint, joint),
        (source, potential),
        (potential, source),
        (joint, data),
        (data, joint),
    }
    assert not prohibited.intersection(map(tuple, record["live_shapes"]))
    assert record["largest_live_array_elements"] < 32 * joint
    peak = record["peak_bytes_in_use"]
    if peak is not None:
        # A single dense source or potential square would already exceed this
        # generous budget. This includes compilation and solve preparation.
        assert peak < min(source**2, potential**2) * 8 / 2
    print({key: value for key, value in record.items() if key != "live_shapes"})
