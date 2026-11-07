"""Qualify a fresh J0946 L-BFGS-B refit against the local trial 497 archive."""

import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import (
    LBFGSPotentialCorrection,
    PotentialCorrectionStrengths,
    PotentialCorrectionWhiteningReference,
)
from TinyLensGpu.PhysicalModel import EPL, PhysicalModel, Shear
from examples.potential_correction.result_products import (
    RESULT_ARRAYS,
    json_safe,
    write_json,
)

_TRIAL_497_SHA256 = {
    "macro/m3_handoff.json": "ef9c77f40fd055854e011bae1bdb4b4d49f70f9db99e8e0f6283906a25f0fb03",
    "macro/m3_handoff.npz": "d09a1102a64f74dcf540530d850ed633430313009d44bc0f5b5b551e18b10bda",
    "archive/protocol.json": "43820e3f4b0ea29813d859ef3e048780d8de80bfe91e27f48f34896f4b08e038",
    "archive/initial_values.npz": "6004fb3427ce453bab3109211cbe05ec02cdf78f9336467105771f010ff5dc09",
    "archive/trials/0497/result.json": "a24d9a1a39b552c4c03c4f0ae8d9c6093d3fc9907263a5e0d7e2400ca8b8251e",
    "archive/trials/0497/reconstruction.npz": "aad0ff87e957f84d0e4fd8c0975285cb029adb512e5cd8afece821565d3a1c5c",
}


_GN_STATIONARITY_TOL = 1e-3
_DENSE_TOLERANCES = {"objective": 1e-3, "log_evidence": 0.01, "convergence": 1e-4}


def _read_verified_files(macro_dir, archive_dir):
    roots = {"macro": macro_dir, "archive": archive_dir}
    contents, identity = {}, {}
    for name, expected in _TRIAL_497_SHA256.items():
        section, relative = name.split("/", 1)
        path = roots[section] / relative
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {path}")
        contents[name] = data
        identity[name] = {"path": str(path), "sha256": digest}
    return contents, identity


def _potential_topology(grid):
    return {
        "potential_node_indices": np.asarray(grid.node_indices),
        "potential_x": np.asarray(grid.x),
        "potential_y": np.asarray(grid.y),
        "derivative_valid": np.asarray(grid.derivative_valid),
    }


def _match_potential_topology(reference, candidate):
    for name in (
        "potential_node_indices",
        "potential_x",
        "potential_y",
        "derivative_valid",
    ):
        if not np.array_equal(reference[name], candidate[name]):
            raise ValueError(
                f"Potential topology or derivative-valid domain differs in {name}"
            )
    valid = np.asarray(reference["derivative_valid"])
    if valid.ndim != 1 or valid.dtype != bool or not valid.any():
        raise ValueError("Comparison requires a nonempty derivative-valid domain")
    return valid


def _convergence_peak(fields, valid):
    values = np.asarray(fields["convergence"])
    if (
        values.shape != valid.shape
        or not np.isfinite(values[valid]).all()
        or not np.isnan(values[~valid]).all()
    ):
        raise ValueError(
            "Convergence must be finite on the valid domain and NaN outside it"
        )
    index = np.flatnonzero(valid)[values[valid].argmax()]
    return {
        "value": float(values[index]),
        "position_arcsec": [
            float(fields["potential_x"][index]),
            float(fields["potential_y"][index]),
        ],
    }


def compare_convergence(reference, candidate):
    """Compare the complete field on the shared derivative-valid potential nodes."""
    valid = _match_potential_topology(reference, candidate)
    peaks = [_convergence_peak(fields, valid) for fields in (reference, candidate)]
    difference = (
        np.asarray(candidate["convergence"])[valid]
        - np.asarray(reference["convergence"])[valid]
    )
    return {
        "derivative_valid_count": int(valid.sum()),
        "max_absolute_difference": float(np.max(np.abs(difference))),
        "reference_peak": peaks[0],
        "candidate_peak": peaks[1],
    }


def compare_dense_to_trial(result, trial, trial_fields):
    """Score the dense trial 497 gates using a freshly fitted public result."""
    candidate = {
        **_potential_topology(result.potential_grid),
        "derivative_valid": np.asarray(result.derivative_valid),
        "convergence": np.asarray(result.convergence),
    }
    if result.success:
        convergence = compare_convergence(trial_fields, candidate)
    else:
        _match_potential_topology(trial_fields, candidate)
        convergence = None
    info = result.optimizer_info
    qualified = bool(
        result.success
        and result.backend == "dense"
        and info.native_success
        and info.gn_stationarity_tol == _GN_STATIONARITY_TOL
        and info.gn_stationarity is not None
        and 0 <= info.gn_stationarity <= _GN_STATIONARITY_TOL
    )
    differences = {
        "objective": abs(float(result.objective) - trial["objective"]),
        "log_evidence": (
            None
            if result.log_evidence is None
            else abs(float(result.log_evidence) - trial["log_evidence"])
        ),
        "convergence": (
            None if convergence is None else convergence["max_absolute_difference"]
        ),
    }
    gates = {
        "qualified": qualified,
        **{
            name: differences[name] is not None and differences[name] <= limit
            for name, limit in _DENSE_TOLERANCES.items()
        },
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "absolute_differences": differences,
        "tolerances": dict(_DENSE_TOLERANCES),
        "convergence": convergence,
    }


def _npz(data):
    with np.load(BytesIO(data), allow_pickle=False) as saved:
        return dict(saved)


def _prepare_fit(handoff, observation, initial, trial, trial_fields):
    # Preserve the archived native sampling before the renderer promotes rays.
    with jax.experimental.enable_x64(False):
        simulation = SimulatorConfig(
            handoff["dpix"],
            observation["mask"].shape[0],
            jnp.asarray(observation["psf_kernel"], dtype=jnp.float32),
            nsub=handoff["nsub"],
            mask=observation["mask"],
        )
    macro_params = handoff["macro_parameters"]
    macro = PhysicalModel(
        lens_mass=[
            EPL(
                theta_E=macro_params["theta_E"],
                gamma=macro_params["gamma"],
                e1=macro_params["e1_mass"],
                e2=macro_params["e2_mass"],
                center_x=macro_params["center_x_mass"],
                center_y=macro_params["center_y_mass"],
            ),
            Shear(gamma1=macro_params["gamma1"], gamma2=macro_params["gamma2"]),
        ],
        source_light=[],
        lens_light=[],
    )
    renderer = PotentialCorrectionRenderer(
        macro,
        simulation,
        handoff["source_n"],
        observation["source_bbox"],
        potential_spacing=trial["configuration"]["potential_spacing"],
        dtype="float64",
        source_interpolation="cubic_hermite",
    )
    _match_potential_topology(
        trial_fields, _potential_topology(renderer.potential_grid)
    )
    for name in ("source_x_axis", "source_y_axis"):
        if not np.allclose(
            getattr(renderer, name), trial_fields[name], rtol=0, atol=1e-14
        ):
            raise ValueError(f"Reconstructed {name} differs from the archive")
    for name, actual in {
        "image_shape": list(renderer.image_shape),
        "nsub": renderer.nsub,
        "n_data": renderer.n_active,
        "n_source": renderer.n_source,
        "n_potential": renderer.potential_grid.n_unknown,
        "n_core": renderer.potential_grid.n_core,
    }.items():
        if actual != trial["configuration"][name]:
            raise ValueError(f"Reconstructed {name} differs from the archive")

    saved_start = initial["joint_values"]
    n_source = renderer.n_source
    if (
        saved_start.shape != (n_source + renderer.potential_grid.n_unknown,)
        or saved_start.dtype != np.float64
        or np.any(saved_start[n_source:] != 0)
    ):
        raise ValueError(
            "Trial 497 requires the archived Float64 start with zero potential"
        )
    source_start = saved_start[:n_source]
    source_reference = initial["saved_M3_source"]
    if (
        not np.array_equal(source_start, trial_fields["source_initial"])
        or not np.array_equal(source_reference, observation["source_values"])
        or np.array_equal(source_start, source_reference)
    ):
        raise ValueError(
            "Archived source start and M3 whitening reference do not match"
        )
    problem = LBFGSPotentialCorrection(
        renderer,
        observation["lens_subtracted_image"],
        observation["noise_map"],
        source_start,
        source_regularization=handoff["source_regularization"],
        source_scale=observation["source_scale"],
        initialization_provenance={
            "handoff_sha256": handoff["handoff_sha256"],
            "m3_sample_index": handoff["selected_sample_index"],
            "source_start_archive": "initial_values.npz joint_values source entries",
            "whitening_reference_archive": "initial_values.npz saved_M3_source",
            "adaptive_reg_rho": handoff["adaptive_reg_rho"],
        },
        whitening_reference=PotentialCorrectionWhiteningReference(
            source_reference, "bilinear"
        ),
    )
    jax.block_until_ready(
        (
            renderer.psf_fft,
            renderer.macro_mapping,
            problem.initial_values,
            problem.reference_values,
            problem.source_scale,
        )
    )
    return problem


def _fit_record(result):
    return {
        "algorithm": result.algorithm,
        "backend": result.backend,
        "success": result.success,
        "termination_reason": result.termination_reason,
        "failure_reason": result.failure_reason,
        "strengths": asdict(result.strengths),
        "objective": result.objective,
        "chi2": result.chi2,
        "regularization_penalty": result.regularization_penalty,
        "log_evidence": result.log_evidence,
        "evidence_method": result.evidence_method,
        "determinant_method": result.determinant_method,
        "evidence_is_exact": result.evidence_is_exact,
        "prior_logdet": result.prior_logdet,
        "posterior_logdet": result.posterior_logdet,
        "noise_logdet": result.noise_logdet,
        "source_bbox": result.source_bbox,
        "potential_origin_arcsec": result.potential_grid.origin,
        "configuration": result.configuration,
        "initialization_provenance": result.initialization_provenance,
        "optimizer_info": asdict(result.optimizer_info),
    }


def _retain_result(output, report, result):
    output.mkdir(parents=True, exist_ok=False)
    fields = {
        name: np.asarray(getattr(result, name))
        for name in RESULT_ARRAYS
        if getattr(result, name) is not None
    }
    fields.update(_potential_topology(result.potential_grid))
    fields["source_bbox"] = np.asarray(result.source_bbox)
    reconstruction = output / "reconstruction.npz"
    np.savez_compressed(reconstruction, **fields)
    report["reconstruction"] = {
        "path": reconstruction.name,
        "sha256": hashlib.sha256(reconstruction.read_bytes()).hexdigest(),
    }
    write_json(output / "result.json", report)


def run_j0946_validation(macro_dir, archive_dir, output):
    """Fit at the archived strengths and retain the dense qualification report."""
    macro_dir, archive_dir, output = (
        Path(p).resolve() for p in (macro_dir, archive_dir, output)
    )
    if (
        output.exists()
        or output.is_relative_to(macro_dir)
        or output.is_relative_to(archive_dir)
    ):
        raise ValueError("Use a new output directory outside the input archives")

    started = perf_counter()
    files, identity = _read_verified_files(macro_dir, archive_dir)
    handoff = json.loads(files["macro/m3_handoff.json"])
    observation = _npz(files["macro/m3_handoff.npz"])
    protocol = json.loads(files["archive/protocol.json"])
    initial = _npz(files["archive/initial_values.npz"])
    trial = json.loads(files["archive/trials/0497/result.json"])
    trial_fields = _npz(files["archive/trials/0497/reconstruction.npz"])
    load_seconds = perf_counter() - started
    weights = trial["strengths"]
    strengths = PotentialCorrectionStrengths(
        weights["lambda_s"], weights["lambda4"], weights["lambda0"]
    )

    with jax.experimental.enable_x64(), jax.default_matmul_precision("highest"):
        started = perf_counter()
        problem = _prepare_fit(handoff, observation, initial, trial, trial_fields)
        preparation_seconds = perf_counter() - started
        started = perf_counter()
        result = problem.fit(
            strengths,
            max_iter=protocol["optimizer"]["maxiter"],
            gn_stationarity_tol=_GN_STATIONARITY_TOL,
        )
        jax.block_until_ready(
            (result.source_values, result.convergence, result.log_evidence)
        )
        fit_seconds = perf_counter() - started
        device = next(iter(result.source_values.devices()))
        memory_stats = device.memory_stats() or {}

    comparison = compare_dense_to_trial(result, trial, trial_fields)
    report = {
        "lens": trial["lens"],
        "archive": {
            "trial_index": trial["index"],
            "protocol": protocol["protocol"],
            "files": identity,
            "objective": trial["objective"],
            "log_evidence": trial["log_evidence"],
            "gn_stationarity": trial["gn_stationarity_proxy"],
            "macro_parameters": handoff["macro_parameters"],
            "image_pixel_spacing_arcsec": handoff["dpix"],
            "psf_shape": observation["psf_kernel"].shape,
            "native_sampling_dtype": "float32",
        },
        "result": _fit_record(result),
        "comparison": comparison,
        "resources": {
            "device": device.device_kind,
            "platform": device.platform,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "archive_read_seconds": load_seconds,
            "synchronized_preparation_seconds": preparation_seconds,
            "cold_fit_seconds_including_compilation": fit_seconds,
            "peak_device_bytes_in_use": memory_stats.get("peak_bytes_in_use"),
            "peak_memory_method": "JAX device allocator peak_bytes_in_use through the fit",
        },
    }
    _retain_result(output, report, result)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_j0946_validation(args.macro_dir, args.archive_dir, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    summary = {
        "passed": report["comparison"]["passed"],
        "absolute_differences": report["comparison"]["absolute_differences"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(json_safe(summary), allow_nan=False))
    if not report["comparison"]["passed"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
