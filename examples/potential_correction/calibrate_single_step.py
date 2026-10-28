"""Small physical-mock calibration using public fits and bounded DIRECT searches.

Run with an explicit JSON configuration in tinylens_gpu:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/calibrate_single_step.py --config CONFIG.json --output OUTPUT

Dense references intentionally materialize small matrices. Operator fits use
bounded principal blocks. Raw fields are saved unchanged; affine alignment is
a separately named diagnostic, never substituted in images or evidence.
"""

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.Inference import (
    PotentialCorrectionEvidence,
    optimize_potential_correction,
)
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from examples.potential_correction.mock import MockConfig, generate_mock_pair


@dataclass(frozen=True)
class CalibrationRun:
    """Scalar record and selected raw public results keyed by mock/prior run."""

    record: dict
    reconstructions: dict


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

    Vote for source tiles through absolute linear-potential-response stencil
    weights at the same subpixel, then attach unilluminated support nodes to
    their nearest voted potential node. This ignores PSF spread deliberately.
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
    gx, gy = np.asarray(gradient_x), np.asarray(gradient_y)
    if (
        gx.shape != grid.query_x.shape
        or gy.shape != grid.query_y.shape
        or si.shape != iw.shape
        or sw.shape != iw.shape
    ):
        raise ValueError(
            "Mapping and gradients must match the renderer query positions"
        )
    if not np.isfinite(gx).all() or not np.isfinite(gy).all():
        raise ValueError("Partition gradients must be finite")
    gx = np.where(np.asarray(source_mapping.valid), gx, 0.0)
    gy = np.where(np.asarray(source_mapping.valid), gy, 0.0)
    for q in range(len(ii)):
        for corner in range(4):
            core = ii[q, corner]
            for neighbour, gradient in ((1, gx[q]), (2, gx[q]), (3, gy[q]), (4, gy[q])):
                p = di[core, neighbour]
                factor = abs(iw[q, corner] * gradient) / (2 * grid.spacing)
                if factor == 0:
                    continue
                for src, weight in zip(si[q], sw[q]):
                    tile = int(tiles[src])
                    votes[p][tile] = votes[p].get(tile, 0.0) + factor * float(weight)
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


def _rms(values):
    values = np.asarray(values)
    return None if not values.size else float(np.sqrt(np.mean(values**2)))


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _truth(renderer, observation):
    grid = renderer.potential_grid
    sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
    source = observation.source_truth.brightness(sx, sy).ravel()
    potential = observation.subhalo_truth.potential(grid.x, grid.y)
    ax, ay, kappa = grid.derivatives(potential)
    core = np.asarray(grid.core_indices)
    cx, cy = grid.x[core], grid.y[core]
    bx, by = observation.macro.deflection(cx, cy)
    illumination = np.asarray(observation.source_truth.brightness(bx, by))
    radius = np.hypot(np.asarray(cx), np.asarray(cy))
    illuminated = (illumination >= 0.05 * illumination.max()) & (
        (radius >= observation.config.mask_inner)
        & (radius <= observation.config.mask_outer)
    )
    roi_radius = max(
        2 * np.sqrt(observation.config.b_sub * observation.config.theta_E),
        2 * grid.spacing,
    )
    local = (
        np.hypot(
            np.asarray(cx) - observation.config.subhalo_x,
            np.asarray(cy) - observation.config.subhalo_y,
        )
        <= roi_radius
    )
    return dict(
        source=np.asarray(source),
        potential=np.asarray(potential),
        alpha_x=np.asarray(ax),
        alpha_y=np.asarray(ay),
        kappa=np.asarray(kappa),
        analytic_kappa=np.asarray(observation.subhalo_truth.convergence(cx, cy)),
        illuminated_core=illuminated,
        local_core=local,
        local_radius=roi_radius,
    )


def _source_box_margins(grid, observation, bbox, potential=None):
    bx, by = observation.macro.deflection(grid.query_x, grid.query_y)
    if potential is not None:
        ax, ay = grid.deflection(potential)
        bx, by = bx - ax, by - ay
    x, y = np.asarray(bx), np.asarray(by)
    xmin, xmax, ymin, ymax = bbox
    margins = np.minimum.reduce((x - xmin, xmax - x, y - ymin, ymax - y))
    outside = margins < 0
    brightness = np.asarray(observation.source_truth.brightness(bx, by))
    return dict(
        query_count=len(x),
        minimum_signed_margin=float(margins.min()),
        margin_percentiles=np.percentile(margins, [1, 5, 50]).tolist(),
        outside_count=int(outside.sum()),
        outside_fraction=float(outside.mean()),
        outside_analytic_brightness_fraction=float(
            brightness[outside].sum() / brightness.sum()
        ),
        outside_analytic_brightness_max=(
            float(brightness[outside].max()) if outside.any() else 0.0
        ),
    )


def _fit_record(result, truth, observation):
    record = dict(
        success=result.success,
        failure_reason=result.failure_reason,
        strengths=asdict(result.strengths),
        solver_info=asdict(result.solver_info),
        dtype=result.dtype,
        evidence_method=result.evidence_method,
        determinant_method=result.determinant_method,
        evidence_is_exact=result.evidence_is_exact,
        configuration={
            k: v for k, v in result.configuration.items() if k != "block_indices"
        },
    )
    if not result.success:
        return record
    s, p = np.asarray(result.source_values), np.asarray(result.potential_values)
    grid = result.potential_grid
    core = np.asarray(grid.core_indices)
    kappa = np.asarray(result.convergence)[core]
    ax, ay = (
        np.asarray(result.deflection_x)[core],
        np.asarray(result.deflection_y)[core],
    )
    raw_truth = truth["potential"]
    affine = np.column_stack((np.ones(len(p)), np.asarray(grid.x), np.asarray(grid.y)))
    coefficients = np.linalg.lstsq(affine, p - raw_truth, rcond=None)[0]
    peak = int(np.argmax(kappa))
    peak_xy = [
        float(np.asarray(grid.x)[core[peak]]),
        float(np.asarray(grid.y)[core[peak]]),
    ]
    convergence = {}
    for name, mask in dict(
        all_core=np.ones(len(core), dtype=bool),
        illuminated_core=truth["illuminated_core"],
        local_core=truth["local_core"],
    ).items():
        finite = mask & np.isfinite(truth["analytic_kappa"])
        convergence[name] = dict(
            count=int(mask.sum()),
            rmse_discrete_truth=_rms((kappa - truth["kappa"])[mask]),
            truth_discrete_rms=_rms(truth["kappa"][mask]),
            rmse_analytic_point_truth=_rms((kappa - truth["analytic_kappa"])[finite]),
            analytic_valid_count=int(finite.sum()),
            recovered_rms=_rms(kappa[mask]),
            recovered_max=float(kappa[mask].max()) if mask.any() else None,
            truth_discrete_max=(
                float(truth["kappa"][mask].max()) if mask.any() else None
            ),
        )
    record.update(
        log_evidence=float(result.log_evidence),
        prior_logdet=float(result.prior_logdet),
        posterior_logdet=float(result.posterior_logdet),
        chi2=float(result.chi2),
        regularization_penalty=float(result.regularization_penalty),
        objective=float(result.objective),
        amplitude_penalty=float(result.strengths.lambda_amplitude * (p @ p)),
        source_relative_l2=float(
            np.linalg.norm(s - truth["source"]) / np.linalg.norm(truth["source"])
        ),
        source_rmse=_rms(s - truth["source"]),
        potential_rmse_raw_center_zero_truth=_rms(p - raw_truth),
        potential_rmse_mean_centered_truth=_rms(p - (raw_truth - raw_truth.mean())),
        potential_rms=_rms(p),
        potential_mean=float(p.mean()),
        deflection_vector_rmse_discrete_truth=_rms(
            np.hypot(ax - truth["alpha_x"], ay - truth["alpha_y"])
        ),
        affine_diagnostic_only=dict(
            fitted_difference_constant_x_y=coefficients.tolist(),
            potential_rmse=_rms(p - affine @ coefficients - raw_truth),
            deflection_vector_rmse=_rms(
                np.hypot(
                    ax - coefficients[1] - truth["alpha_x"],
                    ay - coefficients[2] - truth["alpha_y"],
                )
            ),
            applied_to_raw_fields_or_images_or_evidence=False,
        ),
        convergence=convergence,
        convergence_peak_xy=peak_xy,
        corrected_ray_source_box=_source_box_margins(
            grid, observation, result.source_bbox, result.potential_values
        ),
        peak_distance_to_subhalo=float(
            np.hypot(
                peak_xy[0] - observation.config.subhalo_x,
                peak_xy[1] - observation.config.subhalo_y,
            )
        ),
        linear_residual_rms_noise=float(
            np.sqrt(float(result.chi2) / len(result.data_active))
        ),
        nonlinear_residual_rms_noise=_rms(
            (np.asarray(result.data_active) - np.asarray(result.nonlinear_image_active))
            / np.asarray(result.noise_active)
        ),
        fitted_linearization_rms_noise=_rms(
            (
                np.asarray(result.nonlinear_image_active)
                - np.asarray(result.linear_image_active)
            )
            / np.asarray(result.noise_active)
        ),
    )
    return record


def _compare(left, right):
    if not left.success or not right.success:
        return dict(
            success=False, failure_reasons=[left.failure_reason, right.failure_reason]
        )
    result = dict(success=True)
    for field in (
        "source_values",
        "potential_values",
        "convergence",
        "linear_image_active",
        "nonlinear_image_active",
    ):
        a, b = np.asarray(getattr(left, field)), np.asarray(getattr(right, field))
        valid = np.isfinite(a) & np.isfinite(b)
        result[field + "_max_abs_difference"] = float(
            np.max(np.abs(a[valid] - b[valid]))
        )
        result[field + "_rmse_difference"] = _rms(a[valid] - b[valid])
    result.update(
        objective_difference=float(right.objective - left.objective),
        chi2_difference=float(right.chi2 - left.chi2),
        log_evidence_difference=float(right.log_evidence - left.log_evidence),
        determinant_contribution_to_log_evidence_difference=-0.5
        * float(
            right.posterior_logdet
            - right.prior_logdet
            - left.posterior_logdet
            + left.prior_logdet
        ),
    )
    return result


def _supplied_truth_diagnostics(problem, observation, truth):
    renderer = problem.renderer
    s, p = jnp.asarray(truth["source"]), jnp.asarray(truth["potential"])
    physical = np.asarray(observation.model_image).ravel()[
        np.asarray(renderer.flat_indices)
    ]
    supplied = np.asarray(renderer.render(s, p).image_active)
    linear = np.asarray(problem.forward(jnp.concatenate((s, p))))
    ax, ay = renderer.potential_grid.deflection(p)
    bx, by = renderer.macro_mapping
    analytic = np.asarray(
        renderer.image_forward(observation.source_truth.brightness(bx - ax, by - ay))
    )
    gx, gy = observation.source_truth.gradient(bx, by)
    analytic_linear = np.asarray(
        renderer.image_forward(
            observation.source_truth.brightness(bx, by) - gx * ax - gy * ay
        )
    )
    noise = observation.config.noise_sigma
    return dict(
        macro_ray_source_box=_source_box_margins(
            renderer.potential_grid, observation, renderer.source_bbox
        ),
        truth_corrected_ray_source_box=_source_box_margins(
            renderer.potential_grid, observation, renderer.source_bbox, p
        ),
        supplied_truth_vs_analytic_mock_rms_noise=_rms((supplied - physical) / noise),
        source_interpolation_at_discrete_p_rms_noise=_rms(
            (supplied - analytic) / noise
        ),
        potential_discretization_with_analytic_source_rms_noise=_rms(
            (analytic - physical) / noise
        ),
        truth_linearization_vs_nonlinear_rms_noise=_rms((linear - supplied) / noise),
        analytic_source_taylor_remainder_rms_noise=_rms(
            (analytic_linear - analytic) / noise
        ),
    )


def _search_record(search, truth, observation):
    result = dict(
        has_valid_fit=search.has_valid_fit,
        optimizer_success=search.optimizer_success,
        optimizer_status=search.optimizer_status,
        optimizer_message=search.message,
        nfev=search.nfev,
        log_strengths=search.log_strengths,
        boundary_sides=search.boundary_sides,
        boundary_distances=search.boundary_distances,
        evaluations=[asdict(item) for item in search.evaluations],
    )
    if search.has_valid_fit:
        result.update(_fit_record(search.reconstruction, truth, observation))
    return result


def _order_check(problem, points, fit_options):
    objective = PotentialCorrectionEvidence(
        problem, backend="operator", fit_options=fit_options
    )
    first = []
    for logs in points:
        first.append((objective(logs), objective.last_reconstruction))
    obj_error, field_error, matched = 0.0, 0.0, True
    for i in reversed(range(len(points))):
        value = objective(points[i])
        old_value, previous = first[i]
        current = objective.last_reconstruction
        matched &= (
            current.success == previous.success
            and current.failure_reason == previous.failure_reason
        )
        if current.success:
            obj_error = max(obj_error, abs(value - old_value))
            field_error = max(
                field_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(current.source_values - previous.source_values)
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            np.asarray(
                                current.potential_values - previous.potential_values
                            )
                        )
                    )
                ),
            )
    return dict(
        passed=bool(matched and obj_error <= 1e-8 and field_error <= 1e-8),
        candidate_log_strengths=points,
        objective_atol=1e-8,
        raw_field_atol=1e-8,
        max_objective_difference=obj_error,
        max_raw_field_difference=field_error,
        evaluations=[asdict(item) for item in objective.evaluations],
    )


def run_calibration(configuration, *, output):
    """Scan and independently optimize both backends for every fixed lambda0.

    Scan strengths are physical, DIRECT bounds are natural logarithms. All
    candidates share fixed geometry and truth-derived gradients. Failed fits,
    DIRECT termination and boundaries remain visible. No numerical acceptance
    is inferred from finite evidence or scalar ranking agreement.
    """
    config = dict(configuration)
    mock = MockConfig.from_dict(config["mock"])
    if mock.dtype != "float64":
        raise ValueError("Backend calibration requires a float64 reference")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pair = generate_mock_pair(mock)
    renderer = PotentialCorrectionRenderer(
        pair.control.macro,
        pair.control.sim_config,
        config["source_n"],
        tuple(config["source_bbox"]),
        potential_spacing=config["potential_spacing"],
        dtype=mock.dtype,
    )
    layout = config.get("block_layout", "morton")
    if layout not in ("morton", "source_incidence"):
        raise ValueError("block_layout must be morton or source_incidence")
    blocks = None
    options = dict(
        block_size=config["block_size"],
        pcg_max_iter=config["pcg_max_iter"],
        pcg_rtol=config["pcg_rtol"],
        pcg_atol=0.0,
    )
    record = dict(
        selection_comparison_reference="best bounded dense DIRECT candidate; optimizer termination retained, no proven global maximum",
        conditional_nonlinear_discrepancy="frozen analytic truth gradient versus nonlinear fitted pixelized source; not solely a Taylor remainder",
        purpose="Small physical-mock backend calibration; no detection or mass-precision claim",
        configuration=config,
        software=dict(
            jax=jax.__version__,
            numpy=np.__version__,
            devices=[str(d) for d in jax.devices()],
        ),
        sizes=dict(
            data=renderer.n_active,
            source=renderer.n_source,
            potential_core=renderer.potential_grid.n_core,
            potential_support=renderer.potential_grid.n_unknown,
        ),
        block_partition=dict(method=layout, indices=blocks),
        truth_conventions=dict(
            potential="analytic nodal psi, psi(subhalo center)=0; fitted fields raw",
            discrete_derivatives="same centered physical differences on nodal analytic psi; core only",
            analytic_convergence="point samples, no averaging or invented core; R<1e-8 excluded",
            illuminated_core="within observation annulus and analytic macro source brightness >=5% of core maximum",
            local_core="disk max(2*sqrt(mock b_sub*theta_E),2*potential spacing); matched between cases",
            affine_diagnostic_only="least-squares constant+x+y difference on U; never applied to fitted products",
        ),
        runs=[],
    )
    cases = config.get("cases", ["subhalo", "control"])
    if (
        not cases
        or len(set(cases)) != len(cases)
        or any(case not in ("subhalo", "control") for case in cases)
    ):
        raise ValueError(
            "cases must be a nonempty unique selection of subhalo and control"
        )
    inputs, saved, reconstructions = {}, {}, {}
    with jax.experimental.enable_x64():
        for case in cases:
            obs = getattr(pair, case)
            truth = _truth(renderer, obs)
            inputs.update(
                {
                    f"{case}_{k}": v
                    for k, v in dict(
                        image=np.asarray(obs.image_data),
                        noiseless_image=np.asarray(obs.model_image),
                        noise=np.asarray(obs.noise_realization),
                        source_truth=truth["source"],
                        potential_truth=truth["potential"],
                        discrete_kappa_truth=truth["kappa"],
                        analytic_kappa_truth=truth["analytic_kappa"],
                        illuminated_core=truth["illuminated_core"],
                        local_core=truth["local_core"],
                    ).items()
                }
            )
            for index, amplitude in enumerate(config["lambda_amplitudes"]):
                key = f"{case}_amplitude{amplitude:g}"
                problem = SingleStepPotentialCorrection(
                    renderer,
                    obs.image_data,
                    obs.noise_map,
                    obs.source_truth.gradient,
                    lambda_amplitude=amplitude,
                    solve_rtol=config["solve_rtol"],
                    initialization_provenance={
                        "source": "analytic mock truth gradient; fixed macro rays"
                    },
                )
                if blocks is None:
                    blocks = (
                        spatial_blocks(renderer, config["block_size"])
                        if layout == "morton"
                        else [
                            block.tolist()
                            for block in source_incidence_blocks(
                                renderer,
                                problem.mapping,
                                problem.gradient_x,
                                problem.gradient_y,
                                block_size=config["block_size"],
                                source_tile=config["source_tile"],
                            )
                        ]
                    )
                    options["block_indices"] = blocks
                    record["block_partition"].update(
                        indices=blocks,
                        actual_sizes=[len(block) for block in blocks],
                        version=1,
                        construction_gradient="explicit initial mapping/gradient arrays; no retained truth callable",
                    )
                row = dict(
                    key=key,
                    case=case,
                    lambda_amplitude=amplitude,
                    supplied_truth=_supplied_truth_diagnostics(problem, obs, truth),
                    scan=[],
                )
                order_points = []
                for ls in config["source_strengths"]:
                    for lp in config["potential_strengths"]:
                        dense = problem.fit(ls, lp)
                        operator = problem.fit(ls, lp, backend="operator", **options)
                        row["scan"].append(
                            dict(
                                lambda_source=ls,
                                lambda_potential=lp,
                                dense=_fit_record(dense, truth, obs),
                                operator=_fit_record(operator, truth, obs),
                                fixed_strength_comparison=_compare(dense, operator),
                            )
                        )
                        if operator.success:
                            order_points.append(np.log([ls, lp]).tolist())
                selected = {
                    backend: optimize_potential_correction(
                        problem,
                        log_bounds=config["log_bounds"],
                        backend=backend,
                        fit_options=options if backend == "operator" else None,
                        maxfun=config["direct_maxfun"],
                        len_tol=config["direct_len_tol"],
                        boundary_rtol=config["boundary_rtol"],
                    )
                    for backend in ("dense", "operator")
                }
                row["selected"] = {
                    b: _search_record(v, truth, obs) for b, v in selected.items()
                }
                fits = {b: search.reconstruction for b, search in selected.items()}
                if all(fit is not None for fit in fits.values()):
                    d, o = fits["dense"], fits["operator"]
                    od = problem.fit(
                        d.strengths.lambda_source,
                        d.strengths.lambda_potential,
                        backend="operator",
                        **options,
                    )
                    do = problem.fit(
                        o.strengths.lambda_source, o.strengths.lambda_potential
                    )
                    fits.update(operator_at_dense=od, dense_at_operator=do)
                    row.update(
                        at_dense_selected_strengths=_compare(d, od),
                        at_operator_selected_strengths=_compare(do, o),
                        selected_field_comparison=_compare(d, o),
                        dense_at_operator=_fit_record(do, truth, obs),
                        dense_selected_minus_complete_at_operator_log_evidence=(
                            float(d.log_evidence - do.log_evidence)
                            if do.success
                            else None
                        ),
                    )
                    failure = PotentialCorrectionEvidence(
                        problem,
                        backend="operator",
                        fit_options={**options, "pcg_max_iter": 1},
                    )
                    failure(selected["dense"].log_strengths)
                    row["failure_check"] = asdict(failure.evaluations[-1])
                else:
                    row["failure_check"] = dict(
                        success=False, failure_reason="no_valid_selected_fit"
                    )
                points = (
                    order_points[:1]
                    + order_points[len(order_points) // 2 : len(order_points) // 2 + 1]
                    + order_points[-1:]
                )
                row["order_check"] = (
                    _order_check(problem, points, options)
                    if points
                    else dict(passed=False)
                )
                reconstructions[key] = fits
                for backend, fit in fits.items():
                    if fit is None or not fit.success:
                        continue
                    for field in (
                        "source_values",
                        "source_x_axis",
                        "source_y_axis",
                        "potential_values",
                        "deflection_x",
                        "deflection_y",
                        "convergence",
                        "derivative_valid",
                        "potential_window_coverage",
                        "linear_model_image",
                        "nonlinear_model_image",
                        "linear_residual",
                        "nonlinear_residual",
                    ):
                        saved[f"{key}_{backend}_{field}"] = np.asarray(
                            getattr(fit, field)
                        )
                record["runs"].append(row)
                np.savez_compressed(output / "inputs.npz", **inputs)
                np.savez_compressed(output / "reconstructions.npz", **saved)
                (output / "calibration.json").write_text(
                    json.dumps(_json_safe(record), indent=2, allow_nan=False) + "\n"
                )
                print(
                    json.dumps(
                        dict(
                            run=key,
                            amplitude=amplitude,
                            exact_regret=row.get(
                                "dense_selected_minus_complete_at_operator_log_evidence"
                            ),
                            selected={
                                b: {
                                    k: row["selected"][b].get(k)
                                    for k in (
                                        "log_strengths",
                                        "boundary_sides",
                                        "has_valid_fit",
                                        "optimizer_success",
                                    )
                                }
                                for b in ("dense", "operator")
                            },
                        )
                    ),
                    flush=True,
                )
        grid = renderer.potential_grid
        inputs.update(
            potential_x=np.asarray(grid.x),
            potential_y=np.asarray(grid.y),
            potential_node_indices=np.asarray(grid.node_indices),
            potential_core_indices=np.asarray(grid.core_indices),
            derivative_valid=np.asarray(grid.derivative_valid),
            source_x_axis=np.asarray(renderer.source_x_axis),
            source_y_axis=np.asarray(renderer.source_y_axis),
            noise_map=np.asarray(pair.control.noise_map),
            psf=np.asarray(pair.control.psf_kernel),
            mask=np.asarray(pair.control.mask),
        )
    np.savez_compressed(output / "inputs.npz", **inputs)
    np.savez_compressed(output / "reconstructions.npz", **saved)
    return CalibrationRun(_json_safe(record), reconstructions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_calibration(json.loads(args.config.read_text()), output=args.output)


if __name__ == "__main__":
    main()
