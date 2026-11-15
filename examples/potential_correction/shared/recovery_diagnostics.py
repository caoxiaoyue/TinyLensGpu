"""Shared consumer math on public fields; no reconstruction or reference solve.

Truth derivatives and selections here are core-sized. Export adapters retain
their own schemas, including full support arrays with NaN derivative padding.
The analytic profile itself remains independently defined in ``mock.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _rms(values):
    values = np.asarray(values)
    return float(np.sqrt(np.mean(values**2))) if values.size else None


def _sample_truth(source_x_axis, source_y_axis, grid, observation, dtype):
    """Sample the same truth and diagnostic domains for every result consumer."""
    with jax.experimental.enable_x64(dtype == "float64"):
        sx, sy = jnp.meshgrid(source_x_axis, source_y_axis)
        source = observation.source_truth.brightness(sx, sy).ravel()
        potential = observation.subhalo_truth.potential(grid.x, grid.y)
        ax, ay, kappa = grid.derivatives(potential)
        core = np.asarray(grid.core_indices)
        cx, cy = grid.x[core], grid.y[core]
        aax, aay = observation.subhalo_truth.deflection(cx, cy)
        analytic_kappa = observation.subhalo_truth.convergence(cx, cy)
        bx, by = observation.macro.deflection(cx, cy)
        illumination = np.asarray(observation.source_truth.brightness(bx, by))
    radius = np.hypot(np.asarray(cx), np.asarray(cy))
    illuminated = (illumination >= 0.05 * illumination.max()) & (
        (radius >= observation.config.mask_inner)
        & (radius <= observation.config.mask_outer)
    )
    # Use the nominal configured subhalo even in the matched zero-strength
    # control, so its diagnostic disk does not shrink with the actual truth.
    local_radius = max(
        2 * np.sqrt(observation.config.b_sub * observation.config.theta_E),
        2 * grid.spacing,
    )
    local = (
        np.hypot(
            np.asarray(cx) - observation.config.subhalo_x,
            np.asarray(cy) - observation.config.subhalo_y,
        )
        <= local_radius
    )
    return dict(
        source=np.asarray(source),
        potential=np.asarray(potential),
        alpha_x=np.asarray(ax),
        alpha_y=np.asarray(ay),
        kappa=np.asarray(kappa),
        analytic_alpha_x=np.asarray(aax),
        analytic_alpha_y=np.asarray(aay),
        analytic_kappa=np.asarray(analytic_kappa),
        illuminated_core=illuminated,
        local_core=local,
        local_radius=local_radius,
    )


def _truth_arrays(truth, grid, dtype):
    """Adapt core samples to scientific exports without extending validity."""
    core = np.asarray(grid.core_indices)
    arrays = {
        "truth_source_values": truth["source"],
        "truth_potential_values": truth["potential"],
    }
    for name, key in (
        ("discrete_deflection_x", "alpha_x"),
        ("discrete_deflection_y", "alpha_y"),
        ("discrete_convergence", "kappa"),
        ("analytic_deflection_x", "analytic_alpha_x"),
        ("analytic_deflection_y", "analytic_alpha_y"),
        ("analytic_convergence", "analytic_kappa"),
    ):
        values = np.full(grid.n_unknown, np.nan, dtype=dtype)
        values[core] = truth[key]
        arrays[f"truth_{name}"] = values
    for name in ("illuminated_core", "local_core"):
        arrays[name] = np.zeros(grid.n_unknown, dtype=bool)
        arrays[name][core] = truth[name]
    return arrays


def _ray_domain(grid, observation, bbox, potential=None, *, analytic_truth=False):
    """Shared ray positions, source-box margins and analytic brightness."""
    bx, by = observation.macro.deflection(grid.query_x, grid.query_y)
    if analytic_truth:
        ax, ay = observation.subhalo_truth.deflection(grid.query_x, grid.query_y)
        bx, by = bx - ax, by - ay
    elif potential is not None:
        ax, ay = grid.deflection(potential)
        bx, by = bx - ax, by - ay
    x, y = np.asarray(bx), np.asarray(by)
    xmin, xmax, ymin, ymax = bbox
    margin = np.minimum.reduce((x - xmin, xmax - x, y - ymin, ymax - y))
    brightness = np.asarray(observation.source_truth.brightness(bx, by))
    return margin, brightness


def _recovery_metrics(result, truth, observation, *, affine_diagnostics=False):
    """Field/domain metrics shared by calibration and scientific schemas."""
    grid = result.potential_grid
    core = np.asarray(grid.core_indices)
    s, p = np.asarray(result.source_values), np.asarray(result.potential_values)
    kappa = np.asarray(result.convergence)[core]
    ax, ay = (
        np.asarray(result.deflection_x)[core],
        np.asarray(result.deflection_y)[core],
    )
    peak = core[np.argmax(kappa)]
    peak_xy = [float(np.asarray(grid.x)[peak]), float(np.asarray(grid.y)[peak])]
    convergence = {}
    for name, domain in (
        ("all_core", np.ones(len(core), dtype=bool)),
        ("illuminated_core", truth["illuminated_core"]),
        ("local_core", truth["local_core"]),
    ):
        finite = domain & np.isfinite(truth["analytic_kappa"])
        convergence[name] = {
            "count": int(domain.sum()),
            "rmse_discrete_truth": _rms((kappa - truth["kappa"])[domain]),
            "truth_discrete_rms": _rms(truth["kappa"][domain]),
            "rmse_analytic_point_truth": _rms(
                (kappa - truth["analytic_kappa"])[finite]
            ),
            "analytic_valid_count": int(finite.sum()),
            "recovered_rms": _rms(kappa[domain]),
            "recovered_mean": float(kappa[domain].mean()) if domain.any() else None,
            "truth_discrete_mean": (
                float(truth["kappa"][domain].mean()) if domain.any() else None
            ),
            "recovered_max": float(kappa[domain].max()) if domain.any() else None,
            "truth_discrete_max": (
                float(truth["kappa"][domain].max()) if domain.any() else None
            ),
        }
    summary = {
        "source_relative_l2": float(
            np.linalg.norm(s - truth["source"]) / np.linalg.norm(truth["source"])
        ),
        "source_rmse": _rms(s - truth["source"]),
        "source_negative_node_fraction": float(np.mean(s < 0)),
        "source_negative_to_positive_sum": (
            float(-s[s < 0].sum() / s[s > 0].sum()) if (s > 0).any() else None
        ),
        "potential_rmse_raw_center_zero_truth": _rms(p - truth["potential"]),
        "potential_rmse_mean_centered_truth": _rms(
            p - (truth["potential"] - truth["potential"].mean())
        ),
        "potential_rms": _rms(p),
        "potential_mean": float(p.mean()),
        "deflection_vector_rmse_discrete_truth": _rms(
            np.hypot(ax - truth["alpha_x"], ay - truth["alpha_y"])
        ),
        "convergence": convergence,
        "convergence_peak_xy_arcsec": peak_xy,
        "peak_distance_to_injected_position_arcsec": float(
            np.hypot(
                peak_xy[0] - observation.config.subhalo_x,
                peak_xy[1] - observation.config.subhalo_y,
            )
        ),
        "local_radius_arcsec": truth["local_radius"],
    }
    aligned = None
    if affine_diagnostics:
        affine = np.column_stack(
            (np.ones(len(p)), np.asarray(grid.x), np.asarray(grid.y))
        )
        coefficients = np.linalg.lstsq(affine, p - truth["potential"], rcond=None)[0]
        aligned = p - affine @ coefficients
        summary["affine_diagnostic_only"] = {
            "fitted_difference_constant_x_y": coefficients.tolist(),
            "potential_rmse": _rms(aligned - truth["potential"]),
            "deflection_vector_rmse": _rms(
                np.hypot(
                    ax - coefficients[1] - truth["alpha_x"],
                    ay - coefficients[2] - truth["alpha_y"],
                )
            ),
            "applied_to_raw_fields_or_images_or_evidence": False,
        }
    return summary, aligned
