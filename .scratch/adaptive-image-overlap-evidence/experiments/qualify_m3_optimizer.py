"""Qualify full-score Float64 mass optimization on the archived Euclid mock."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, minimize

from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.ObservationModel.LensImage._pixelized_regularization import HERMITE_COVERAGE_OFFSET

# Import after TinyLensGpu selects Caskade's JAX backend.
import caskade as ck

from euclid_mock_replay import build_m3, first_order_precision_banded, load_mock

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "m3_optimizer_config.json"


def _read_json(path):
    return json.loads(path.read_text())


def _write_json(path, record):
    path.write_text(json.dumps(record, allow_nan=False, indent=2) + "\n")


def _build_qualification(archive, mock_output, config):
    mock = load_mock(archive)
    provenance = _read_json(mock_output / "configuration.json")
    coverage = config["hermite_coverage_offset"]
    if (coverage != HERMITE_COVERAGE_OFFSET
            or provenance["m2_m3"]["hermite_coverage_offset"] != coverage
            or provenance["inputs_sha256"] != mock.hashes):
        raise ValueError("Qualification requires a matching mock calibration and coverage offset.")
    calibration = _read_json(mock_output / "m2_calibration.json")
    fit = _read_json(mock_output / "m3_fit.json")
    comparison = _read_json(mock_output / "matched_evidence.json")
    strengths = calibration["selected"]
    if (calibration["evidence_logdet"] != config["evidence_logdet"]
            or fit["fixed_strengths"] != strengths or comparison["fixed_strengths"] != strengths):
        raise ValueError("M2, M3 and matched comparison must use the same calibrated strengths.")
    model = build_m3(mock, strengths, dtype=np.dtype(config["dtype"]),
                     evidence_logdet=config["evidence_logdet"])
    model.pcg_max_iter = config["solver"]["max_iterations"]
    model.pcg_rtol = config["solver"]["relative_tolerance"]
    _, specs = make_prior_transformation(model)
    names = [spec.name for spec in specs]
    actual_bounds = {spec.name: list(spec.limits) for spec in specs}
    if actual_bounds != config["physical_bounds"]:
        raise ValueError(f"M3 physical bounds changed: {actual_bounds}")
    record = {
        "config_sha256": sha256(CONFIG.read_bytes()).hexdigest(),
        "jax_version": jax.__version__,
        "devices": [device.device_kind for device in jax.devices()],
        "input_sha256": mock.hashes,
        "fixed_strengths": strengths,
        "parameter_names": names,
        "position_likelihood": mock.position_likelihood,
        "block_size": model.block_size,
        "evidence_block_size": model.evidence_block_size,
    }
    return mock, model, names, record


def _point(config, names, label):
    return np.array([config["starts"][label][name] for name in names], dtype=np.float64)


def _position_evaluator(model):
    @jax.jit
    def position(theta):
        with ck.ActiveContext(model):
            model.fill_params(theta)
            x, y = model.phys_model.deflection(model._pos_px, model._pos_py)
            separation = jnp.max(jnp.hypot(x[:, None] - x, y[:, None] - y))
            return model._position_likelihood_penalty_jax(), separation

    return position


def _conditional_energy(model, source, strength):
    # Hold the reconstructed coefficients at the injection proposal while
    # allowing each proposed mass to change the projected source precision.
    def energy(theta):
        with ck.ActiveContext(model):
            model.fill_params(theta)
            bbox, mapping, _, _ = model._mapping_terms()
            scale = model._get_reg_scale(mapping)
            weighted_source = model.reg_builder.matvec_free(source, *bbox, scale=scale)
            return 0.5 * strength * jnp.dot(source, weighted_source)

    return energy


def _independent_energy(source, scale, mock, strength):
    spacing = (mock.bbox[1] - mock.bbox[0]) / (mock.n - 1)
    bands = first_order_precision_banded(scale, mock.n, strength, spacing)
    source = np.asarray(source)
    diagonal = np.dot(bands[0], source**2)
    off_diagonal = sum(np.dot(bands[offset, :-offset], source[:-offset] * source[offset:])
                       for offset in (1, mock.n))
    return 0.5 * diagonal + off_diagonal


def _find_ray_crossing(model, mock, center, gamma_index, radius):
    @jax.jit
    def traced(theta):
        with ck.ActiveContext(model):
            model.fill_params(theta)
            return jnp.stack(model.phys_model.deflection(
                model.sim_obj.image_x_active_sub, model.sim_obj.image_y_active_sub,
            ), axis=1)

    low_gamma = center[gamma_index] - radius
    high_gamma = center[gamma_index] + radius
    origin = np.array(mock.bbox)[[0, 2]]
    step = (np.array(mock.bbox)[[1, 3]] - origin) / (mock.n - 1)

    def grid_coordinates(gamma):
        theta = center.copy()
        theta[gamma_index] = gamma
        return (np.asarray(traced(theta)) - origin) / step

    low, high = grid_coordinates(low_gamma), grid_coordinates(high_gamma)
    knots = np.ceil(np.minimum(low, high))
    changed_cell = (
        ((low - knots) * (high - knots) < 0)
        & (knots > 0) & (knots < mock.n - 1)
        & (low[:, ::-1] > 0) & (low[:, ::-1] < mock.n - 1)
        & (high[:, ::-1] > 0) & (high[:, ::-1] < mock.n - 1)
    )
    candidates, axes = np.nonzero(changed_cell)
    if not candidates.size:
        raise RuntimeError("No active subpixel ray crosses an interior knot in the declared bracket.")
    # Pick the brightest archived native parent among crossing rays. Ray
    # enumeration and axis order break ties deterministically.
    parents = np.asarray(model.sim_obj._agg_segment_ids)
    active_brightness = mock.adapt_image[~mock.mask]
    selected = int(np.argmax(active_brightness[parents[candidates]]))
    ray, axis = int(candidates[selected]), int(axes[selected])
    knot = int(knots[ray, axis])

    def offset(gamma):
        return grid_coordinates(gamma)[ray, axis] - knot

    crossing = brentq(offset, low_gamma, high_gamma, xtol=1e-14)
    return crossing, offset, {
        "gamma": crossing,
        "bracket": [low_gamma, high_gamma],
        "axis": ("x", "y")[axis],
        "source_knot_index": knot,
        "subpixel_ray_index": ray,
        "native_parent_flat_index": int(np.flatnonzero(~mock.mask)[parents[ray]]),
        "archived_native_brightness": float(active_brightness[parents[ray]]),
        "bracket_offsets_in_grid_pixels": [float(low[ray, axis] - knot), float(high[ray, axis] - knot)],
    }


def _derivative_probe(theta, index, step, functions, tolerances):
    row = {}
    low, high = theta.copy(), theta.copy()
    low[index] -= step
    high[index] += step
    for label, (value, value_grad) in functions.items():
        _, gradient = value_grad(theta)
        automatic = float(gradient[index])
        finite_difference = (float(value(high)) - float(value(low))) / (2 * step)
        row[label] = {
            "automatic": automatic,
            "central_difference": finite_difference,
            "absolute_error": abs(automatic - finite_difference),
            "passes": bool(np.isclose(automatic, finite_difference, **tolerances)),
        }
    return row


def run_probes(mock, model, names, config, provenance, output):
    start = perf_counter()
    controls = config["probes"]
    theta = _point(config, names, config["scaling_origin"])
    result, diagnostics = model.evaluate_with_diagnostics(jnp.asarray(theta))
    if not diagnostics.valid or not diagnostics.evidence_valid:
        raise RuntimeError("The conditional source requires a valid injection reconstruction.")
    source = result.source_pixels
    strength = float(np.exp(provenance["fixed_strengths"]["log_lambda_reg"]))
    energy = _conditional_energy(model, source, strength)
    prior_value = jax.jit(energy)
    expected = _independent_energy(source, diagnostics.regularization_scale, mock, strength)
    np.testing.assert_allclose(prior_value(theta), expected, rtol=1e-11, atol=1e-11)
    evaluate = jax.jit(model.evaluate)
    position = _position_evaluator(model)
    gamma_index = names.index("gamma")
    slope_rows = []
    for gamma in controls["slope_scan"]:
        trial = theta.copy()
        trial[gamma_index] = gamma
        inversion = evaluate(trial)
        penalty, separation = map(float, position(trial))
        score = float(inversion.log_evidence)
        slope_rows.append({
            "gamma": gamma,
            "conditional_prior_energy": float(prior_value(trial)),
            "complete_score": score,
            "primary_valid": bool(inversion.solver_info.converged),
            "evidence_valid": bool(np.isfinite(score) and score > -1e10),
            "position_penalty": penalty,
            "maximum_source_separation_arcsec": separation,
        })
    if not all(row["primary_valid"] and row["evidence_valid"] for row in slope_rows):
        raise RuntimeError("The slope scan encountered an invalid fit.")
    print(f"Completed {len(slope_rows)} slope probes", flush=True)

    def score(parameters):
        return model(parameters)

    functions = {
        "conditional_prior_energy": (prior_value, jax.jit(jax.value_and_grad(energy))),
        "complete_score": (jax.jit(score), jax.jit(jax.value_and_grad(score))),
    }
    tolerances = dict(rtol=controls["relative_derivative_tolerance"],
                      atol=controls["absolute_derivative_tolerance"])
    derivative_rows = []
    for index, name in enumerate(names):
        step = controls["central_step_in_scaled_units"] * config["parameter_scales"][name]
        derivative_rows.append({
            "probe": "injection",
            "parameter": name,
            "step": step,
            **_derivative_probe(theta, index, step, functions, tolerances),
        })

    gamma, offset, crossing = _find_ray_crossing(
        model, mock, theta, gamma_index, controls["ray_crossing_bracket_radius"],
    )
    for label, shift in (("left", -controls["ray_crossing_flank"]), ("knot", 0.0),
                         ("right", controls["ray_crossing_flank"])):
        trial = theta.copy()
        trial[gamma_index] = gamma + shift
        step = controls["ray_crossing_step"]
        derivative_rows.append({
            "probe": f"ray_crossing_{label}",
            "parameter": "gamma",
            "gamma": float(trial[gamma_index]),
            "ray_offset_in_grid_pixels": offset(trial[gamma_index]),
            "step": step,
            **_derivative_probe(trial, gamma_index, step, functions, tolerances),
        })

    # Record penalty regimes for each central difference and require that
    # its stencil stays on one branch of the configured penalty.
    for row in derivative_rows:
        trial = theta.copy()
        trial[gamma_index] = row.get("gamma", theta[gamma_index])
        idx = names.index(row["parameter"])
        values = []
        for direction in (-1, 0, 1):
            shifted = trial.copy()
            shifted[idx] += direction * row["step"]
            values.append(tuple(map(float, position(shifted))))
        row["position_penalties"] = [pair[0] for pair in values]
        row["maximum_source_separations_arcsec"] = [pair[1] for pair in values]
        row["same_position_regime"] = len({pair[0] == 0.0 for pair in values}) == 1

    passed = all(
        row["same_position_regime"]
        and row["conditional_prior_energy"]["passes"]
        and row["complete_score"]["passes"]
        for row in derivative_rows
    )
    _write_json(output / "probes.json", {
        **provenance,
        "passed": passed,
        "conditional_prior_source": config["scaling_origin"],
        "conditional_prior_energy_independent_at_injection": expected,
        "slope_scan": slope_rows,
        "ray_crossing": crossing,
        "derivative_probes": derivative_rows,
        "elapsed_seconds": perf_counter() - start,
    })
    if not passed:
        raise RuntimeError("Float64 derivative or position-regime qualification failed.")
    print(f"Passed {len(derivative_rows)} two-objective derivative probes", flush=True)


def _finite_value(value):
    return float(value) if np.isfinite(value) else None


def _parameter_record(names, values):
    return {name: _finite_value(value) for name, value in zip(names, values)}


def _projection(gradient, point, lower, upper):
    # The unit projected step for maximization is zero at a bound-constrained
    # stationary point even when the physical gradient is nonzero there.
    return np.clip(point + gradient, lower, upper) - point


def _endpoint_diagnostics(model, theta):
    result, diag = model.evaluate_with_diagnostics(jnp.asarray(theta))
    return {
        "primary_valid": bool(diag.valid),
        "evidence_valid": bool(diag.evidence_valid),
        "primary_invalid_reason": diag.invalid_reason,
        "evidence_invalid_reason": diag.evidence_invalid_reason,
        "pcg_iterations": int(result.solver_info.n_iter),
        "chi2": _finite_value(diag.chi2),
        "score": _finite_value(result.log_evidence),
        "position_penalty": _finite_value(diag.position.penalty),
        "maximum_source_separation_arcsec": _finite_value(diag.position.max_separation),
    }


def run_optimizer(model, names, config, provenance, output):
    probes_file = output / "probes.json"
    if not probes_file.exists():
        raise RuntimeError("Run the slope and derivative probes before starting L-BFGS-B.")
    probes = _read_json(probes_file)
    if not probes["passed"] or any(probes[key] != value for key, value in provenance.items()):
        raise RuntimeError("L-BFGS-B requires matching, passing slope and derivative probes.")

    criteria = config["pass_criteria"]
    origin = _point(config, names, config["scaling_origin"])
    scale = np.array([config["parameter_scales"][name] for name in names])
    physical_lower, physical_upper = np.array([config["physical_bounds"][name] for name in names]).T
    lower = (physical_lower - origin) / scale
    upper = (physical_upper - origin) / scale
    bounds = list(zip(lower, upper))
    value_grad = jax.jit(jax.value_and_grad(lambda theta: model(theta)))
    basins = []
    rows = []
    elapsed_start = perf_counter()

    for label in config["starts"]:
        theta0 = _point(config, names, label)
        z0 = (theta0 - origin) / scale
        if np.any(z0 < lower) or np.any(z0 > upper):
            raise ValueError(f"Optimizer start {label} is outside the M3 physical bounds.")
        initial = _endpoint_diagnostics(model, theta0)
        if not initial["primary_valid"] or not initial["evidence_valid"]:
            raise RuntimeError(f"Invalid optimizer start {label}: {initial}")
        invalid_trials = 0

        def loss(scaled):
            nonlocal invalid_trials
            score, gradient = value_grad(origin + scale * scaled)
            score, gradient = float(score), np.asarray(gradient)
            if not np.isfinite(score) or score <= -1e10 or not np.all(np.isfinite(gradient)):
                invalid_trials += 1
            return -score, -scale * gradient

        started = perf_counter()
        fit = minimize(loss, z0, method="L-BFGS-B", jac=True,
                       bounds=bounds, options=config["lbfgsb_options"])
        theta = origin + scale * fit.x
        endpoint = _endpoint_diagnostics(model, theta)
        endpoint_score, physical_grad = value_grad(theta)
        endpoint_score, physical_grad = float(endpoint_score), np.asarray(physical_grad)
        scaled_grad = scale * physical_grad
        scaled_projected = _projection(scaled_grad, fit.x, lower, upper)
        physical_projected = _projection(physical_grad, theta, physical_lower, physical_upper)
        scaled_norm = float(np.linalg.norm(scaled_projected, ord=np.inf))
        score_gain = endpoint_score - initial["score"]
        finite = bool(np.isfinite(endpoint_score) and np.all(np.isfinite(physical_grad)))
        in_bounds = bool(np.all(lower <= fit.x) and np.all(fit.x <= upper))
        passed = bool(
            (fit.success or not criteria["require_scipy_success"])
            and (finite or not criteria["require_finite_endpoint_and_gradient"])
            and ((endpoint["primary_valid"] and endpoint["evidence_valid"])
                 or not criteria["require_valid_primary_and_evidence"])
            and in_bounds and score_gain >= criteria["min_score_gain"]
            and scaled_norm <= criteria["max_scaled_projected_gradient_inf"]
        )
        basin = None
        if finite and in_bounds and endpoint["primary_valid"] and endpoint["evidence_valid"]:
            for index, (reference, reference_score) in enumerate(basins, 1):
                if (np.linalg.norm(reference - fit.x, ord=np.inf)
                        <= criteria["basin_scaled_parameter_inf_tolerance"]
                        and abs(reference_score - endpoint_score) <= criteria["basin_score_tolerance"]):
                    basin = index
                    break
            if basin is None:
                basins.append((fit.x.copy(), endpoint_score))
                basin = len(basins)
        row = {
            "start": label,
            "initial_parameters": _parameter_record(names, theta0),
            "initial_diagnostics": initial,
            "scipy_success": bool(fit.success),
            "scipy_status": int(fit.status),
            "scipy_termination_message": str(fit.message),
            "iterations": int(fit.nit),
            "objective_evaluations": int(fit.nfev),
            "gradient_evaluations": int(fit.njev),
            "invalid_trials": invalid_trials,
            "endpoint_parameters": _parameter_record(names, theta),
            "endpoint_diagnostics": endpoint,
            "endpoint_score_from_value_and_grad": _finite_value(endpoint_score),
            "physical_score_gradient": _parameter_record(names, physical_grad),
            "physical_projected_gradient": _parameter_record(names, physical_projected),
            "scaled_projected_gradient": _parameter_record(names, scaled_projected),
            "scaled_projected_gradient_inf": _finite_value(scaled_norm),
            "score_gain": _finite_value(score_gain),
            "endpoint_finite": finite,
            "endpoint_in_bounds": in_bounds,
            "endpoint_basin": basin,
            "elapsed_seconds": perf_counter() - started,
            "passed": passed,
        }
        rows.append(row)
        print(f"{label}: {fit.message} | score={endpoint_score:.9f} "
              f"| projected gradient={scaled_norm:.3e} | basin={basin} | passed={passed}", flush=True)
        _write_json(output / "optimizer.json", {
            **provenance,
            "probe_sha256": sha256(probes_file.read_bytes()).hexdigest(),
            "completed_starts": len(rows),
            "passed": len(rows) == len(config["starts"]) and all(entry["passed"] for entry in rows),
            "endpoint_basins": len(basins),
            "elapsed_seconds": perf_counter() - elapsed_start,
            "starts": rows,
        })

    if not all(row["passed"] for row in rows):
        raise RuntimeError("One or more L-BFGS-B starts failed qualification.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("probe", "optimize"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--mock-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    config = _read_json(CONFIG)
    mock, model, names, provenance = _build_qualification(args.archive, args.mock_output, config)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "probe":
        run_probes(mock, model, names, config, provenance, args.output)
    else:
        run_optimizer(model, names, config, provenance, args.output)


if __name__ == "__main__":
    main()
