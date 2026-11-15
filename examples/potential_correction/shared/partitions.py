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

# Regenerated from scientific_configuration.json on CUDA, where the recorded
# grid hash and source-incidence blocks match the bundled calibration exactly.
_LEGACY_MACRO_RAYS_SHA256 = (
    "6c9b99d998332c6a4d900a2852c386a0a385349bd0b5ab6e329355993e859adb"
)
_LEGACY_INCIDENCE_GRADIENT_SHA256 = (
    "e26f1d21fef1f5048a300dcb3d472eb5c5a6c7c8fd8af653f28c084bfa7ab663"
)
# The paired iterative_scientific_configuration.json uses the same partition
# with its own initial nodal gradient on that traced geometry.
_LEGACY_INITIAL_NODAL_GRADIENT_SHA256 = (
    "47803ae571abc1b926302e0b6471d6abeaf5eb9b7d90e3bab13c09a5b26b78ba"
)


def _arrays_fingerprint(arrays):
    digest = hashlib.sha256()
    for values in arrays:
        array = np.asarray(values)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def geometry_fingerprint(renderer):
    """Fingerprint fixed source and potential geometry independently of mode."""
    grid = renderer.potential_grid
    return _arrays_fingerprint((
        renderer.source_x_axis,
        renderer.source_y_axis,
        grid.node_indices,
        grid.core_indices,
        grid.x,
        grid.y,
        grid.query_x,
        grid.query_y,
    ))


def _validated_blocks(renderer, cap, indices):
    n_source = renderer.n_source
    n_potential = renderer.potential_grid.n_unknown
    geometry = SimpleNamespace(
        n_source=n_source, n_potential=n_potential, n_joint=n_source + n_potential
    )
    blocks, _, _ = validate_partition(geometry, cap, indices)
    return [list(block) for block in blocks]


def _load_automatic_partition(renderer, path, expected, *, source):
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
            macro_rays_sha256=_LEGACY_MACRO_RAYS_SHA256,
            incidence_gradient_sha256=_LEGACY_INCIDENCE_GRADIENT_SHA256,
            source_tile=24,
            cap=1024,
        )
        candidate["legacy_bilinear"] = True
        # The historical iterative comparison deliberately keeps the fixed
        # partition constructed with the single-step analytic gradient.
        if (
            source == "initial_nodal"
            and expected.get("incidence_gradient_sha256")
            == _LEGACY_INITIAL_NODAL_GRADIENT_SHA256
        ):
            expected = {
                **expected,
                "incidence_gradient_sha256": _LEGACY_INCIDENCE_GRADIENT_SHA256,
            }
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
    if "incidence_gradient_sha256" in record:
        candidate["incidence_gradient_sha256"] = record["incidence_gradient_sha256"]
    return blocks, candidate


def prepare_partition(
    renderer, config, build_blocks, *, source, method, default_file=None,
    incidence_gradients=None,
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
    if method == "source_incidence" and fixed_indices is None:
        if incidence_gradients is None:
            raise ValueError("Source-incidence partitioning requires its initial gradients")
        provenance["macro_rays_sha256"] = _arrays_fingerprint(renderer.macro_mapping)
        provenance["incidence_gradient_sha256"] = _arrays_fingerprint(incidence_gradients)
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
        blocks, candidate = _load_automatic_partition(
            renderer, Path(path), provenance, source=source,
        )
        if blocks is None:
            provenance["reuse_rejected"] = candidate
        else:
            provenance.update(candidate, origin="automatic_reuse")
    if blocks is None:
        blocks = [list(map(int, block)) for block in build_blocks()]
        provenance.update(origin="generated", source=source)
    provenance.update(indices=blocks, actual_sizes=[len(block) for block in blocks])
    return blocks, provenance


def _morton_order(coordinates):
    coordinates = np.asarray(coordinates, dtype=np.int64)
    coordinates = coordinates - coordinates.min(axis=0)
    keys = np.zeros(len(coordinates), dtype=np.int64)
    for bit in range(31):
        keys |= ((coordinates[:, 0] >> bit) & 1) << (2 * bit)
        keys |= ((coordinates[:, 1] >> bit) & 1) << (2 * bit + 1)
    return np.argsort(keys, kind="stable")


def spatial_blocks(renderer, block_size):
    """Fixed Morton-ordered field partitions, with no dense numerical probing."""
    blocks = []
    coordinates = np.stack(
        np.meshgrid(np.arange(renderer.source_n), np.arange(renderer.source_n)), axis=-1
    ).reshape(-1, 2)
    for offset, coords in (
        (0, coordinates),
        (renderer.n_source, np.asarray(renderer.potential_grid.node_indices)),
    ):
        order = _morton_order(coords) + offset
        width = min(block_size, max(1, len(order) - 1))
        blocks.extend(
            order[start : start + width].tolist()
            for start in range(0, len(order), width)
        )
    return blocks


def source_incidence_blocks(
    renderer, source_mapping, gradient_x, gradient_y, *, source_tile, block_size
):
    """Sparse query incidence only; no dense source-by-potential matrix.

    Vote for source tiles using the magnitude of each source weight times
    each linear-potential-response magnitude at the same subpixel. Attach
    unilluminated support nodes to their nearest voted potential node.
    This ignores PSF spread deliberately.
    Inputs are explicit initial mapping and gradient arrays, evaluated once;
    no truth function or reconstruction problem is retained. Iterative callers
    may supply gradients of their initial pixelized source nodes and keep this
    partition fixed while refreshing the numerical Jacobian independently.
    The builder uses sparse incidence dictionaries, never a dense cross matrix.
    """
    from scipy.spatial import cKDTree

    for name, value in (("source_tile", source_tile), ("block_size", block_size)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    cap = block_size
    n = renderer.source_n
    ns = renderer.n_source
    grid = renderer.potential_grid
    nt = (n + source_tile - 1) // source_tile
    tiles = (np.arange(ns) // n // source_tile) * nt + (
        np.arange(ns) % n // source_tile
    )
    votes = [dict() for _ in range(grid.n_unknown)]
    di = np.asarray(grid.derivative_indices)
    ii = np.asarray(grid.interpolation_indices)
    iw = np.asarray(grid.interpolation_weights)
    si = np.asarray(source_mapping.indices)
    sw = np.asarray(source_mapping.weights)
    valid = np.asarray(source_mapping.valid)
    gx, gy = np.asarray(gradient_x), np.asarray(gradient_y)
    ray_shape = grid.query_x.shape
    if (
        gx.shape != ray_shape
        or gy.shape != ray_shape
        or grid.query_y.shape != ray_shape
        or valid.shape != ray_shape
        or si.ndim != 2
        or si.shape != sw.shape
        or si.shape[0] != len(grid.query_x)
        or si.shape[1] == 0
        or ii.ndim != 2
        or ii.shape != iw.shape
        or ii.shape[0] != len(grid.query_x)
        or ii.shape[1] == 0
    ):
        raise ValueError(
            "Mapping and gradients must match the renderer query positions"
        )
    if not np.isfinite(gx).all() or not np.isfinite(gy).all():
        raise ValueError("Partition gradients must be finite")
    gx = np.where(valid, gx, 0.0)
    gy = np.where(valid, gy, 0.0)
    for q in range(len(ii)):
        for corner in range(ii.shape[1]):
            core = ii[q, corner]
            for neighbour, gradient in ((1, gx[q]), (2, gx[q]), (3, gy[q]), (4, gy[q])):
                p = di[core, neighbour]
                factor = abs(iw[q, corner] * gradient) / (2 * grid.spacing)
                if factor == 0:
                    continue
                for src, weight in zip(si[q], sw[q]):
                    if weight == 0:
                        continue
                    tile = int(tiles[src])
                    vote = votes[p]
                    vote[tile] = vote.get(tile, 0.0) + factor * abs(float(weight))
    assigned = np.array(
        [max(vote, key=lambda key: (vote[key], -key)) if vote else -1 for vote in votes]
    )
    voted = np.flatnonzero(assigned >= 0)
    missing = np.flatnonzero(assigned < 0)
    coords = np.asarray(grid.node_indices)
    if not len(voted):
        raise ValueError(
            "source_incidence partition requires a nonzero source gradient"
        )
    if len(missing):
        nearest = cKDTree(coords[voted]).query(coords[missing], k=1)[1]
        assigned[missing] = assigned[voted[nearest]]
    result = []
    for tile in sorted(set(tiles) | set(assigned)):
        source = list(np.flatnonzero(tiles == tile))
        p = np.flatnonzero(assigned == tile)
        potential = list(ns + p[_morton_order(coords[p])]) if len(p) else []
        interleaved = []
        # Round-robin gives both fields room in early chunks if a tile spills.
        for k in range(max(len(source), len(potential))):
            if k < len(source):
                interleaved.append(source[k])
            if k < len(potential):
                interleaved.append(potential[k])
        result.extend(
            np.asarray(interleaved[k : k + cap], dtype=np.int32)
            for k in range(0, len(interleaved), cap)
        )
    return result
