"""Replay a saved J0946 iterative source at its archived fields and geometry."""

# ruff: noqa: E402  -- configure JAX and the local checkout before importing it.

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.example_support import source_interpolation_metadata
from workspace.iteration import pt, source_cubic
from workspace.iteration.source_boundary import ZeroRingRenderer

_HARD = source_interpolation_metadata("bilinear")["source_boundary"]
_HERMITE = source_interpolation_metadata("cubic_hermite")["source_boundary"]
_ZERO_RING = ZeroRingRenderer.source_boundary
_ABS_TOL = 1e-11


def _recorded_conventions(record):
    fields = ("source_interpolation", "source_boundary")
    scopes = (
        ("result", record),
        ("configuration", record.get("configuration", {})),
        ("initialization", record.get("initialization", {})),
    )
    return {
        name: {key: scope[key] for key in fields if key in scope}
        for name, scope in scopes
    }


def _archive_label(records, key, aliases):
    labels = set()
    for values in records.values():
        if key in values:
            label = values[key]
            if not isinstance(label, str) or label not in aliases:
                raise ValueError(f"Unknown archived {key} {label!r}")
            labels.add(aliases[label])
    if len(labels) > 1:
        raise ValueError(f"Conflicting archived {key} values {sorted(labels)}")
    return next(iter(labels), None)


def archive_renderer(base, record):
    """Use recorded reconstruction conventions independently of the M3 handoff."""
    records = _recorded_conventions(record)
    mode = _archive_label(records, "source_interpolation", {
        "bilinear": "bilinear",
        "cubic": "cubic_hermite",
        "C1_cardinal_cubic": "cubic_hermite",
        "cubic_hermite": "cubic_hermite",
    })
    boundary = _archive_label(records, "source_boundary", {
        "hard": _HARD,
        _HARD: _HARD,
        "zero_ring": _ZERO_RING,
        "one_fixed_zero_ghost_ring": _ZERO_RING,
        _ZERO_RING: _ZERO_RING,
        _HERMITE: _HERMITE,
    })
    if mode is None and boundary is None and record.get("protocol") != pt.PROTOCOL:
        raise ValueError("Archive does not identify its source interpolation or boundary")
    if mode is None:
        mode = "cubic_hermite" if boundary == _HERMITE else "bilinear"
    canonical = source_interpolation_metadata(mode)["source_boundary"]
    if boundary is None:
        boundary = canonical
    if boundary == _ZERO_RING and mode == "bilinear":
        renderer = ZeroRingRenderer
    elif boundary == canonical:
        renderer = PotentialCorrectionRenderer
    else:
        raise ValueError(f"Archived interpolation {mode!r} conflicts with boundary {boundary!r}")
    return renderer(
        base.macro_model, base.sim_config, base.source_n, base.source_bbox,
        potential_grid=base.potential_grid, dtype=base.dtype.name,
        source_interpolation=mode,
    )


def _compare(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=_ABS_TOL, equal_nan=False)
    difference = actual - expected
    return dict(
        max_abs=float(np.max(np.abs(difference))),
        rms=float(np.sqrt(np.mean(difference**2))),
    )


def _check_geometry(renderer, state, handoff, config):
    grid = renderer.potential_grid
    for key, actual in (
        ("source_bbox", renderer.source_bbox),
        ("potential_node_indices", grid.node_indices),
        ("potential_x", grid.x),
        ("potential_y", grid.y),
        ("derivative_valid", grid.derivative_valid),
        ("mask", handoff["mask"]),
        ("source_scale", handoff["source_scale"]),
        ("lens_light_model", handoff["lens_light_model"]),
    ):
        np.testing.assert_array_equal(actual, state[key], err_msg=key)
    for key in ("source_x_axis", "source_y_axis"):
        np.testing.assert_allclose(
            getattr(renderer, key), state[key], rtol=0, atol=1e-14, err_msg=key
        )
    for key, actual in (
        ("nsub", renderer.nsub),
        ("image_shape", list(renderer.image_shape)),
        ("n_data", renderer.n_active),
        ("n_source", renderer.n_source),
        ("n_potential", grid.n_unknown),
        ("n_core", grid.n_core),
    ):
        if actual != config[key]:
            raise ValueError(f"Replayed {key} differs from the archive")
    if state["source_values"].dtype != np.float64 or state["potential_values"].dtype != np.float64:
        raise ValueError("Expected the original Float64 source and potential coefficients")
    np.testing.assert_array_equal(
        np.r_[state["source_values"], state["potential_values"]], state["joint_values"],
    )


def replay(macro_dir, trial_dir, protocol_path):
    """Measure a fixed-state replay and return checksums and numerical errors."""
    record_path = trial_dir / "result.json"
    reconstruction_path = trial_dir / "reconstruction.npz"
    record = json.loads(record_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if record["protocol"] != protocol["protocol"]:
        raise ValueError("Trial protocol does not match the supplied protocol")
    expected_inputs = {
        Path(path).name: digest
        for field in ("input_hashes", "handoff_hashes")
        for path, digest in protocol.get(field, {}).items()
    }
    hashes = {
        name: pt.sha256(path) for name, path in (
            ("m3_handoff.json", macro_dir / "m3_handoff.json"),
            ("m3_handoff.npz", macro_dir / "m3_handoff.npz"),
            ("result.json", record_path),
            ("reconstruction.npz", reconstruction_path),
            ("protocol.json", protocol_path),
        )
    }
    for name in ("m3_handoff.json", "m3_handoff.npz"):
        if hashes[name] != expected_inputs.get(name):
            raise ValueError(f"Fixed handoff {name} differs from the archived input")
    config = record["configuration"]
    if config["dtype"] != "float64":
        raise ValueError("This replay expects an iterative Float64 reconstruction")

    with jax.experimental.enable_x64(), jax.default_matmul_precision("highest"):
        problem, metadata, handoff = pt.load_problem(macro_dir, config["potential_spacing"])
        if record["fixed_macro_parameters"] != metadata["macro_parameters"]:
            raise ValueError("Saved macro parameters differ from the handoff")
        if record["initialization"]["handoff_sha256"] != metadata["handoff_sha256"]:
            raise ValueError("Trial refers to a different fixed M3 handoff")
        with np.load(reconstruction_path) as archive:
            state = {key: archive[key] for key in archive.files}
        base = problem.renderer
        renderer = archive_renderer(base, record)
        _check_geometry(renderer, state, handoff, config)
        source = jnp.asarray(state["source_values"])
        potential = jnp.asarray(state["potential_values"])
        rays = base.macro_mapping
        rendered = renderer.render(source, potential, macro_mapping=rays)
        saved_image = state["nonlinear_source_model_image"]
        errors = {"replay_image_vs_archive": _compare(rendered.model_image, saved_image)}
        if renderer.source_interpolation == "cubic_hermite":
            reference_hash = pt.sha256(source_cubic.__file__)
            if reference_hash != protocol["source_hashes"].get(
                "workspace/iteration/source_cubic.py"
            ):
                raise ValueError("Experimental Hermite source differs from archived code")
            hashes["source_cubic.py"] = reference_hash
            reference = source_cubic.CubicSourceRenderer(
                base.macro_model, base.sim_config, base.source_n, base.source_bbox,
                potential_grid=base.potential_grid, dtype="float64",
                source_interpolation="cubic_hermite",
            )
            original = reference.source_mapping(potential, macro_mapping=rays)
            current = renderer.source_mapping(potential, macro_mapping=rays)
            np.testing.assert_array_equal(current.beta_x, original.beta_x)
            np.testing.assert_array_equal(current.beta_y, original.beta_y)
            np.testing.assert_array_equal(current.valid, original.valid)
            original_light = jnp.sum(original.weights * source[original.indices], axis=1)
            current_light = jnp.sum(current.weights * source[current.indices], axis=1)
            original_gx, original_gy = reference.source_spatial_gradient(source, original)
            current_gx, current_gy = renderer.source_spatial_gradient(source, current)
            original_image = reference.image_forward(original_light, full_image=True)
            errors.update(
                ray_brightness=_compare(current_light, original_light),
                derivative_x=_compare(current_gx, original_gx),
                derivative_y=_compare(current_gy, original_gy),
                integrated_psf_image=_compare(rendered.model_image, original_image),
                experimental_image_vs_archive=_compare(original_image, saved_image),
            )
    return dict(
        protocol=record["protocol"],
        trial_directory=record["directory"],
        profile=record.get("profile"),
        repeat=record.get("repeat"),
        input_sha256=hashes,
        recorded_source_conventions=_recorded_conventions(record),
        handoff_interpolation=base.source_interpolation,
        handoff_boundary=base.source_boundary,
        source_interpolation=renderer.source_interpolation,
        source_boundary=renderer.source_boundary,
        preconditioner_reference_boundary=record.get("preconditioner_reference_boundary"),
        source_bbox=state["source_bbox"].tolist(),
        recorded_source_support_bbox=record.get("source_support_bbox"),
        source_n=renderer.source_n,
        n_potential=renderer.potential_grid.n_unknown,
        n_core=renderer.potential_grid.n_core,
        potential_spacing=renderer.potential_grid.spacing,
        n_active=renderer.n_active,
        n_rays=int(renderer.query_x.size),
        image_shape=renderer.image_shape,
        psf_shape=handoff["psf_kernel"].shape,
        dpix=metadata["dpix"],
        nsub=renderer.nsub,
        dtype=renderer.dtype.name,
        absolute_tolerance=_ABS_TOL,
        differences=errors,
        jax=jax.__version__,
        device=jax.devices()[0].device_kind,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-dir", type=Path, required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.macro_dir, args.trial_dir, args.protocol)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
