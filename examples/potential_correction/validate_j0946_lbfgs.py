"""Qualify a fresh J0946 L-BFGS-B refit against the local trial 497 archive."""

import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import (
    LBFGSPotentialCorrection,
    PotentialCorrectionStrengths,
    PotentialCorrectionWhiteningReference,
)
from TinyLensGpu.PhysicalModel import EPL, PhysicalModel, Shear
from examples.potential_correction._lbfgs_resources import measure_lbfgs_fit
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
        "node_index": np.asarray(fields["potential_node_indices"])[index].tolist(),
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


def _result_convergence_fields(result):
    return {
        **_potential_topology(result.potential_grid),
        "derivative_valid": np.asarray(result.derivative_valid),
        "convergence": np.asarray(result.convergence),
    }


def _qualified_fit(fit, backend, determinant_method):
    info = fit["optimizer_info"]
    return bool(
        fit["success"]
        and fit["algorithm"] == "lbfgs"
        and fit["backend"] == backend
        and fit["evidence_method"] == "local_gauss_newton"
        and fit["evidence_is_exact"] is False
        and fit["determinant_method"] == determinant_method
        and fit["log_evidence"] is not None
        and np.isfinite(fit["log_evidence"])
        and info["native_success"]
        and info["gn_stationarity_tol"] == _GN_STATIONARITY_TOL
        and info["gn_stationarity"] is not None
        and 0 <= info["gn_stationarity"] <= _GN_STATIONARITY_TOL
    )


def _comparison_report(differences, tolerances, convergence, **gates):
    gates.update(
        {
            name: differences[name] is not None and differences[name] <= limit
            for name, limit in tolerances.items()
        }
    )
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "absolute_differences": differences,
        "tolerances": tolerances,
        "convergence": convergence,
    }


def _convergence_if_success(reference_fields, fit, fields):
    if fit["success"]:
        return compare_convergence(reference_fields, fields)
    _match_potential_topology(reference_fields, fields)
    return None


def _compare_dense_record(fit, fields, trial, trial_fields):
    convergence = _convergence_if_success(trial_fields, fit, fields)
    evidence = fit["log_evidence"]
    differences = {
        "objective": abs(float(fit["objective"]) - trial["objective"]),
        "log_evidence": (
            None if evidence is None else abs(float(evidence) - trial["log_evidence"])
        ),
        "convergence": (
            None if convergence is None else convergence["max_absolute_difference"]
        ),
    }
    return _comparison_report(
        differences,
        dict(_DENSE_TOLERANCES),
        convergence,
        qualified=_qualified_fit(fit, "dense", "full_data_space"),
    )


def compare_dense_to_trial(result, trial, trial_fields):
    """Score the dense trial 497 gates using a freshly fitted public result."""
    return _compare_dense_record(
        _fit_record(result), _result_convergence_fields(result), trial, trial_fields
    )


def _certified_operator_residual(fit):
    solver = fit["optimizer_info"]["diagnostics"].get("stationarity_solver", {})
    if not (
        solver.get("method") == "pcg"
        and solver.get("converged")
        and solver.get("failed") is False
    ):
        return False
    rhs = solver["rhs_norm"]
    residual = solver["true_residual_norm"]
    reported_threshold = solver["true_residual_threshold"]
    config = fit["configuration"]
    threshold = config["solve_atol"] + config["solve_rtol"] * rhs
    return bool(
        np.isfinite([rhs, residual, reported_threshold, threshold]).all()
        and rhs >= 0
        and np.isclose(reported_threshold, threshold, atol=0, rtol=1e-12)
        and 0 <= residual <= threshold
    )


def compare_operator_to_dense(operator, operator_fields, dense, dense_fields):
    """Score operator qualification against the matched dense fit products."""
    convergence = _convergence_if_success(dense_fields, operator, operator_fields)
    spacing = operator["configuration"]["potential_spacing"]
    peak_separation = None
    positive_peaks = False
    if convergence is not None:
        reference_peak = convergence["reference_peak"]
        candidate_peak = convergence["candidate_peak"]
        positive_peaks = reference_peak["value"] > 0 and candidate_peak["value"] > 0
        # Integer lattice offsets avoid rounding an adjacent pair outside the gate.
        offset = np.asarray(candidate_peak["node_index"]) - reference_peak["node_index"]
        peak_separation = float(np.linalg.norm(offset)) * spacing
    differences = {
        "objective": abs(float(operator["objective"]) - dense["objective"]),
        "convergence": (
            None if convergence is None else convergence["max_absolute_difference"]
        ),
        "peak_separation_arcsec": peak_separation,
    }
    report = _comparison_report(
        differences,
        {"objective": 0.01, "convergence": 0.005, "peak_separation_arcsec": spacing},
        convergence,
        qualified=(
            _qualified_fit(dense, "dense", "full_data_space")
            and _qualified_fit(operator, "operator", "principal_block_cholesky")
        ),
        true_residual=_certified_operator_residual(operator),
        positive_peaks=positive_peaks,
    )
    evidence = {
        name: {
            "log_evidence": fit["log_evidence"],
            "evidence_method": fit["evidence_method"],
            "determinant_method": fit["determinant_method"],
        }
        for name, fit in (("dense", dense), ("operator", operator))
    }
    estimate = operator["log_evidence"]
    evidence["operator_minus_dense"] = (
        None if estimate is None else float(estimate) - dense["log_evidence"]
    )
    report["evidence"] = evidence
    return report


def _npz(data):
    with np.load(BytesIO(data), allow_pickle=False) as saved:
        return dict(saved)


def _archive_identity(archive):
    return (
        archive["trial_index"],
        archive["protocol"],
        {name: entry["sha256"] for name, entry in archive["files"].items()},
    )


def load_dense_reference(directory, archive, trial_fields, expected_fit):
    """Read dense products and recheck identity, fit controls, and archive gates."""
    directory = Path(directory).resolve()
    report_bytes = (directory / "result.json").read_bytes()
    report = json.loads(report_bytes)
    try:
        if _archive_identity(report["archive"]) != _archive_identity(archive):
            raise ValueError(
                "Dense reference archive identity differs from the selected trial"
            )
        fit = report["result"]
        for name in ("strengths", "initialization_provenance"):
            if fit[name] != expected_fit[name]:
                raise ValueError(f"Dense reference {name} differs from this fit")
        for name, expected in expected_fit["configuration"].items():
            if fit["configuration"][name] != expected:
                raise ValueError(f"Dense reference configuration differs in {name}")

        reconstruction = report["reconstruction"]
        if reconstruction["path"] != "reconstruction.npz":
            raise ValueError("Dense reference requires the retained reconstruction.npz")
        arrays = (directory / "reconstruction.npz").read_bytes()
        array_digest = hashlib.sha256(arrays).hexdigest()
        if array_digest != reconstruction["sha256"]:
            raise ValueError("Dense reference reconstruction SHA-256 mismatch")
        fields = _npz(arrays)
        for name in ("source_bbox", "source_x_axis", "source_y_axis"):
            if not np.allclose(fields[name], trial_fields[name], rtol=0, atol=1e-14):
                raise ValueError(f"Dense reference {name} differs from the archive")
        comparison = _compare_dense_record(fit, fields, archive, trial_fields)
        if report["comparison"]["passed"] is not True or not comparison["passed"]:
            raise ValueError("Dense reference does not pass the archive comparison")
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid dense reference products") from exc
    return (
        fit,
        fields,
        {
            "directory": str(directory),
            "result_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "reconstruction_sha256": array_digest,
            "comparison": comparison,
        },
    )


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
        solve_rtol=trial["configuration"]["solve_rtol"],
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


def run_j0946_validation(
    macro_dir,
    archive_dir,
    output,
    *,
    backend="dense",
    dense_reference=None,
    pcg_max_iter=2000,
):
    """Fit at the archived strengths and retain the qualification report."""
    if backend not in ("dense", "operator"):
        raise ValueError("backend must be dense or operator")
    if backend == "operator" and dense_reference is None:
        raise ValueError("operator qualification requires --dense-reference")
    if backend == "dense" and dense_reference is not None:
        raise ValueError("--dense-reference applies only to operator qualification")
    macro_dir, archive_dir, output = (
        Path(p).resolve() for p in (macro_dir, archive_dir, output)
    )
    inputs = [macro_dir, archive_dir]
    if dense_reference is not None:
        inputs.append(Path(dense_reference).resolve())
    if output.exists() or any(output.is_relative_to(path) for path in inputs):
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
    archive = {
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
    }
    weights = trial["strengths"]
    strengths = PotentialCorrectionStrengths(
        weights["lambda_s"], weights["lambda4"], weights["lambda0"]
    )
    optimizer_options = {
        name: value for name, value in protocol["optimizer"].items() if name != "method"
    }
    fit_options = {
        "max_iter": optimizer_options["maxiter"],
        "gn_stationarity_tol": _GN_STATIONARITY_TOL,
    }

    with jax.experimental.enable_x64(), jax.default_matmul_precision("highest"):
        started = perf_counter()
        problem = _prepare_fit(handoff, observation, initial, trial, trial_fields)
        preparation_seconds = perf_counter() - started
        reference_provenance = None
        if backend == "operator":
            expected_fit = {
                "strengths": asdict(strengths),
                "initialization_provenance": problem.initialization_provenance,
                "configuration": {
                    **trial["configuration"],
                    **fit_options,
                    "optimizer_options": optimizer_options,
                    "source_interpolation": problem.renderer.source_interpolation,
                    "source_boundary": problem.renderer.source_boundary,
                    "solve_atol": problem.solve_atol,
                },
            }
            dense_fit, dense_fields, reference_provenance = load_dense_reference(
                dense_reference, archive, trial_fields, expected_fit
            )
            fit_options.update(
                block_size=64,
                pcg_max_iter=pcg_max_iter,
                pcg_rtol=1e-10,
                pcg_atol=0.0,
            )
        result, resources = measure_lbfgs_fit(
            problem, strengths, backend=backend, **fit_options
        )

    fit = _fit_record(result)
    fields = _result_convergence_fields(result)
    if backend == "operator":
        comparison = compare_operator_to_dense(fit, fields, dense_fit, dense_fields)
    else:
        comparison = _compare_dense_record(fit, fields, trial, trial_fields)
    report = {
        "lens": trial["lens"],
        "archive": archive,
        "result": fit,
        "comparison": comparison,
        "resources": {
            **resources,
            "archive_read_seconds": load_seconds,
            "synchronized_preparation_seconds": preparation_seconds,
        },
    }
    if reference_provenance is not None:
        report["dense_reference"] = reference_provenance
    _retain_result(output, report, result)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("dense", "operator"), default="dense")
    parser.add_argument(
        "--dense-reference", type=Path, help="Qualified dense output directory"
    )
    parser.add_argument(
        "--pcg-max-iter",
        type=int,
        default=2000,
        help="Operator stationarity solve budget",
    )
    args = parser.parse_args(argv)
    try:
        report = run_j0946_validation(
            args.macro_dir,
            args.archive_dir,
            args.output,
            backend=args.backend,
            dense_reference=args.dense_reference,
            pcg_max_iter=args.pcg_max_iter,
        )
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
