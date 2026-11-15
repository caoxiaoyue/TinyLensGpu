"""Checksum-validated M3 macro handoff reader for the J0946 corrections."""

import hashlib
import json
from pathlib import Path

import numpy as np


def load_handoff(macro_dir):
    """Read a complete, checksum-validated M3 handoff without preparing a solve."""
    macro_dir = Path(macro_dir)
    metadata = json.loads((macro_dir / "m3_handoff.json").read_text())
    digest = hashlib.sha256((macro_dir / "m3_handoff.npz").read_bytes()).hexdigest()
    if not metadata["complete"] or digest != metadata["handoff_sha256"]:
        raise RuntimeError("Incomplete or changed M3 handoff")
    with np.load(macro_dir / "m3_handoff.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return metadata, arrays
