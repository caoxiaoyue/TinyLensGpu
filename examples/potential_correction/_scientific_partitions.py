"""Shared partition metadata for the scientific correction examples."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from TinyLensGpu.ObservationModel.LensImage._potential_correction_operator_common import (
    _partition as validate_partition,
)
from TinyLensGpu.utils.lensing.mapping import source_boundary_for_interpolation


SOURCE_INCIDENCE_VOTE_RULE = "absolute_source_neighbor_weight_v1"
_LEGACY_PARTITION_SHA256 = (
    "203e0ccc7bfde571c44cd5bf0e02c07ef4b903f182c513e55b40ada9c4f32486"
)


def geometry_fingerprint(renderer):
    """Fingerprint fixed source and potential geometry independently of mode."""
    digest = hashlib.sha256()
    grid = renderer.potential_grid
    for values in (
        renderer.source_x_axis,
        renderer.source_y_axis,
        grid.node_indices,
        grid.core_indices,
        grid.x,
        grid.y,
        grid.query_x,
        grid.query_y,
    ):
        array = np.asarray(values)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validated_blocks(renderer, cap, indices):
    n_source = renderer.n_source
    n_potential = renderer.potential_grid.n_unknown
    geometry = SimpleNamespace(
        n_source=n_source, n_potential=n_potential, n_joint=n_source + n_potential
    )
    blocks, _, _ = validate_partition(geometry, cap, indices)
    return [list(block) for block in blocks]


def _load_automatic_partition(renderer, path, expected):
    candidate = {"file": str(path)}
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return None, {**candidate, "reason": "partition_file_missing"}
    candidate["file_sha256"] = hashlib.sha256(payload).hexdigest()
    try:
        record = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, {**candidate, "reason": "invalid_partition_record"}
    if isinstance(record, dict):
        record = record.get("block_partition", record)
    if not isinstance(record, dict):
        return None, {**candidate, "reason": "invalid_partition_record"}
    if candidate["file_sha256"] == _LEGACY_PARTITION_SHA256:
        # The archived bilinear stencil was nonnegative. Its original votes
        # equal magnitude votes for the same geometry, tile and cap.
        record.update(
            source="single_step",
            method="source_incidence",
            source_interpolation="bilinear",
            source_boundary=source_boundary_for_interpolation("bilinear"),
            vote_rule=SOURCE_INCIDENCE_VOTE_RULE,
            source_tile=24,
            cap=1024,
        )
        candidate["legacy_bilinear"] = True
    mismatches = [
        field
        for field, value in expected.items()
        if field not in record or record[field] != value
    ]
    if mismatches:
        return None, {
            **candidate,
            "reason": "partition_provenance_mismatch",
            "mismatched_fields": mismatches,
        }
    try:
        if record.get("indices") is None:
            raise ValueError("The saved partition has no indices")
        blocks = _validated_blocks(renderer, expected["cap"], record["indices"])
    except ValueError as exc:
        return None, {**candidate, "reason": "invalid_block_indices", "message": str(exc)}
    candidate["source"] = record.get("source", "saved_partition")
    if "reference" in record:
        candidate["reference"] = record["reference"]
    return blocks, candidate


def prepare_partition(
    renderer, config, build_blocks, *, source, method, default_file=None
):
    """Select validated fixed, compatible saved, or newly generated blocks."""
    provenance = {
        "method": method,
        "geometry_sha256": geometry_fingerprint(renderer),
        "source_interpolation": renderer.source_interpolation,
        "source_boundary": renderer.source_boundary,
        "vote_rule": SOURCE_INCIDENCE_VOTE_RULE if method == "source_incidence" else None,
        "source_tile": config.get("source_tile") if method == "source_incidence" else None,
        "cap": config["block_size"],
    }
    blocks = None
    path = config.get("partition_file", default_file)
    fixed_indices = config.get("block_indices")
    if fixed_indices is not None:
        blocks = _validated_blocks(renderer, provenance["cap"], fixed_indices)
        provenance.update(
            origin="explicit_fixed",
            source="configuration",
            method="fixed",
            vote_rule=None,
            source_tile=None,
        )
    elif path is not None:
        blocks, candidate = _load_automatic_partition(renderer, Path(path), provenance)
        if blocks is None:
            provenance["reuse_rejected"] = candidate
        else:
            provenance.update(candidate, origin="automatic_reuse")
    if blocks is None:
        blocks = [list(map(int, block)) for block in build_blocks()]
        provenance.update(origin="generated", source=source)
    provenance.update(indices=blocks, actual_sizes=[len(block) for block in blocks])
    return blocks, provenance
