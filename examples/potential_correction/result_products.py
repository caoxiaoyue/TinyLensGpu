"""Result-based exports and diagnostics shared by scientific correction examples.

Only public reconstruction fields and observation/truth context enter this
module. It never constructs a reconstruction problem or invokes a fit.
"""

from dataclasses import asdict
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

RESULT_ARRAYS = (
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
    "linear_image_active",
    "nonlinear_image_active",
    "linear_residual",
    "nonlinear_residual",
    "linear_normalized_residual",
    "nonlinear_normalized_residual",
    "data_active",
    "noise_active",
)
PLOT_NAMES = ("images", "fields", "source", "history")


def json_safe(value):
    """Convert numerical metadata to strict JSON, using null for nonfinite scalars."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return json_safe(np.asarray(value).tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, record):
    Path(path).write_text(
        json.dumps(json_safe(record), indent=2, allow_nan=False) + "\n"
    )


def export_failure(record, output):
    """Replace this exporter's known success artifacts with a diagnostic failure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename in (
        "result.json",
        "reconstruction.npz",
        *(f"{name}.{ext}" for name in PLOT_NAMES for ext in ("png", "pdf")),
    ):
        (output / filename).unlink(missing_ok=True)
    write_json(output / "failure.json", record)


def _rms(values):
    values = np.asarray(values)
    return float(np.sqrt(np.mean(values**2))) if values.size else None


def _ray_margins(grid, observation, bbox, potential=None, *, analytic_truth=False):
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
    outside = margin < 0
    return {
        "query_count": len(x),
        "outside_count": int(outside.sum()),
        "minimum_signed_margin_arcsec": float(margin.min()),
        "outside_analytic_brightness_fraction": (
            float(brightness[outside].sum() / brightness.sum())
            if brightness.sum()
            else 0.0
        ),
    }


def truth_products(result, observation):
    """Sample truth on the public result's fixed geometry, without a new solve.

    Discrete truth uses the same physical centered differences of analytic
    nodal psi. Analytic point samples are retained separately. Support-only
    derivatives are NaN, including when the analytic field is defined there.
    """
    grid = result.potential_grid
    with jax.experimental.enable_x64(result.dtype == "float64"):
        sx, sy = jnp.meshgrid(result.source_x_axis, result.source_y_axis)
        source = observation.source_truth.brightness(sx, sy).ravel()
        potential = observation.subhalo_truth.potential(grid.x, grid.y)
        dax, day, kappa = grid.derivatives(potential)
        aax, aay = observation.subhalo_truth.deflection(grid.x, grid.y)
        analytic_kappa = observation.subhalo_truth.convergence(grid.x, grid.y)
        core = np.asarray(grid.core_indices)
        bx, by = observation.macro.deflection(grid.x[core], grid.y[core])
        illumination = np.asarray(observation.source_truth.brightness(bx, by))
    valid = np.asarray(result.derivative_valid)
    def full():
        return np.full(grid.n_unknown, np.nan, dtype=result.dtype)
    arrays = {
        "truth_source_values": np.asarray(source),
        "truth_potential_values": np.asarray(potential),
        "truth_discrete_deflection_x": full(),
        "truth_discrete_deflection_y": full(),
        "truth_discrete_convergence": full(),
        "truth_analytic_deflection_x": np.where(valid, np.asarray(aax), np.nan),
        "truth_analytic_deflection_y": np.where(valid, np.asarray(aay), np.nan),
        "truth_analytic_convergence": np.where(
            valid, np.asarray(analytic_kappa), np.nan
        ),
    }
    for name, values in (
        ("deflection_x", dax),
        ("deflection_y", day),
        ("convergence", kappa),
    ):
        arrays[f"truth_discrete_{name}"][core] = np.asarray(values)
    radius = np.hypot(np.asarray(grid.x)[core], np.asarray(grid.y)[core])
    illuminated = (illumination >= 0.05 * illumination.max()) & (
        (radius >= observation.config.mask_inner)
        & (radius <= observation.config.mask_outer)
    )
    local_radius = max(
        2 * np.sqrt(observation.config.b_sub * observation.config.theta_E),
        2 * grid.spacing,
    )
    local = (
        np.hypot(
            np.asarray(grid.x)[core] - observation.config.subhalo_x,
            np.asarray(grid.y)[core] - observation.config.subhalo_y,
        )
        <= local_radius
    )
    for name, selection in (("illuminated_core", illuminated), ("local_core", local)):
        arrays[name] = np.zeros(grid.n_unknown, dtype=bool)
        arrays[name][core] = selection
    return arrays, local_radius


def summarize_result(result, observation, *, affine_diagnostics=False):
    """Return raw recovery, localization and residual summaries on named domains."""
    if not result.success:
        raise ValueError("A failed reconstruction has no scientific recovery summary")
    truth, local_radius = truth_products(result, observation)
    valid = np.asarray(result.derivative_valid)
    s, p = np.asarray(result.source_values), np.asarray(result.potential_values)
    source = truth["truth_source_values"]
    grid = result.potential_grid
    kappa = np.asarray(result.convergence)
    peak_index = np.flatnonzero(valid)[np.argmax(kappa[valid])]
    peak_xy = [
        float(np.asarray(grid.x)[peak_index]),
        float(np.asarray(grid.y)[peak_index]),
    ]
    convergence = {}
    for name, domain in (
        ("all_core", valid),
        ("illuminated_core", truth["illuminated_core"]),
        ("local_core", truth["local_core"]),
    ):
        analytic_valid = domain & np.isfinite(truth["truth_analytic_convergence"])
        convergence[name] = {
            "count": int(domain.sum()),
            "rmse_discrete_truth": _rms(
                (kappa - truth["truth_discrete_convergence"])[domain]
            ),
            "truth_discrete_rms": _rms(truth["truth_discrete_convergence"][domain]),
            "rmse_analytic_point_truth": _rms(
                (kappa - truth["truth_analytic_convergence"])[analytic_valid]
            ),
            "analytic_valid_count": int(analytic_valid.sum()),
            "recovered_rms": _rms(kappa[domain]),
            "recovered_mean": float(kappa[domain].mean()) if domain.any() else None,
            "truth_discrete_mean": (
                float(truth["truth_discrete_convergence"][domain].mean())
                if domain.any()
                else None
            ),
            "recovered_max": float(kappa[domain].max()) if domain.any() else None,
            "truth_discrete_max": (
                float(truth["truth_discrete_convergence"][domain].max())
                if domain.any()
                else None
            ),
        }
    summary = {
        "source_relative_l2": float(
            np.linalg.norm(s - source) / np.linalg.norm(source)
        ),
        "source_rmse": _rms(s - source),
        "source_negative_node_fraction": float(np.mean(s < 0)),
        "source_negative_to_positive_sum": (
            float(-s[s < 0].sum() / s[s > 0].sum()) if (s > 0).any() else None
        ),
        "potential_rmse_raw_center_zero_truth": _rms(
            p - truth["truth_potential_values"]
        ),
        "potential_rmse_mean_centered_truth": _rms(
            p
            - (truth["truth_potential_values"] - truth["truth_potential_values"].mean())
        ),
        "potential_rms": _rms(p),
        "potential_mean": float(p.mean()),
        "deflection_vector_rmse_discrete_truth": _rms(
            np.hypot(
                np.asarray(result.deflection_x)[valid]
                - truth["truth_discrete_deflection_x"][valid],
                np.asarray(result.deflection_y)[valid]
                - truth["truth_discrete_deflection_y"][valid],
            )
        ),
        "convergence": convergence,
        "convergence_peak_xy_arcsec": peak_xy,
        "peak_distance_to_injected_position_arcsec": float(
            np.hypot(
                peak_xy[0] - observation.config.subhalo_x,
                peak_xy[1] - observation.config.subhalo_y,
            )
        ),
        "local_radius_arcsec": local_radius,
        "nonlinear_residual_rms_noise": _rms(
            np.asarray(result.nonlinear_normalized_residual)[
                ~np.asarray(observation.mask)
            ]
        ),
        "source_box": {},
    }
    if result.linear_image_active is not None:
        summary.update(
            linear_residual_rms_noise=_rms(
                np.asarray(result.linear_normalized_residual)[
                    ~np.asarray(observation.mask)
                ]
            ),
            fitted_linearization_rms_noise=_rms(
                (
                    np.asarray(result.linear_image_active)
                    - np.asarray(result.nonlinear_image_active)
                )
                / np.asarray(result.noise_active)
            ),
        )
    with jax.experimental.enable_x64(result.dtype == "float64"):
        summary["source_box"] = {
            "macro_rays": _ray_margins(grid, observation, result.source_bbox),
            "analytic_truth_corrected_rays": _ray_margins(
                grid, observation, result.source_bbox, analytic_truth=True
            ),
            "fitted_corrected_rays": _ray_margins(
                grid, observation, result.source_bbox, result.potential_values
            ),
        }
    if affine_diagnostics:
        affine = np.column_stack(
            (np.ones(len(p)), np.asarray(grid.x), np.asarray(grid.y))
        )
        coefficients = np.linalg.lstsq(
            affine, p - truth["truth_potential_values"], rcond=None
        )[0]
        aligned = p - affine @ coefficients
        summary["affine_diagnostic_only"] = {
            "fitted_difference_constant_x_y": coefficients.tolist(),
            "potential_rmse": _rms(aligned - truth["truth_potential_values"]),
            "applied_to_raw_fields_or_images_or_evidence": False,
        }
        truth["diagnostic_affine_potential_values"] = aligned
    return summary, truth


def export_result(
    result,
    observation,
    output,
    *,
    configuration,
    search=None,
    make_plots=True,
    affine_diagnostics=False,
):
    """Persist a public result, or a failure record without success artifacts.

    Iterative callers may use the same exporter: absent linear products are
    omitted and their evidence image is explicitly the final nonlinear image.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "success": result.success,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "termination_reason": result.termination_reason,
        "algorithm": result.algorithm,
        "backend": result.backend,
        "dtype": result.dtype,
        "evidence_method": result.evidence_method,
        "determinant_method": result.determinant_method,
        "evidence_is_exact": result.evidence_is_exact,
        "evidence_image": (
            "linear_model_image"
            if result.algorithm == "single_step"
            else "nonlinear_model_image"
        ),
        "log_evidence": result.log_evidence,
        "chi2": result.chi2,
        "regularization_penalty": result.regularization_penalty,
        "objective": result.objective,
        "prior_logdet": result.prior_logdet,
        "posterior_logdet": result.posterior_logdet,
        "noise_logdet": result.noise_logdet,
        "strengths": asdict(result.strengths),
        "solver_info": asdict(result.solver_info),
        "initialization_provenance": result.initialization_provenance,
        "configuration": configuration,
        "result_configuration": result.configuration,
        "mock_configuration": asdict(observation.config),
        "observation": "subhalo" if observation.has_subhalo else "control",
        "integration": {
            "mock_nsub": observation.config.nsub,
            "inversion_nsub": result.configuration["nsub"],
        },
        "source_bbox": result.source_bbox,
        "potential_geometry": {
            "spacing": result.potential_grid.spacing,
            "origin": result.potential_grid.origin,
            "unknown_count": result.potential_grid.n_unknown,
            "core_count": result.potential_grid.n_core,
            "support_only_count": result.potential_grid.n_unknown
            - result.potential_grid.n_core,
        },
        "units": {
            "coordinates": "arcsec",
            "potential": "arcsec^2",
            "deflection": "arcsec",
            "convergence": "dimensionless",
        },
        "truth_conventions": {
            "potential": "analytic nodal singular PseudoJaffeSubhalo; psi(center)=0",
            "discrete_derivatives": "centered physical differences of nodal analytic psi; derivative_valid core only",
            "analytic_derivatives": "point samples, no pixel averaging; kappa NaN for R<1e-8 arcsec; zero strength identically zero",
            "illuminated_core": "observation annulus and macro-source brightness >=5% of core maximum",
            "local_core": "matched disk around injected position; radius max(2*sqrt(mock b_sub*theta_E),2*h)",
            "raw_fields": "signed, unclipped and unaligned; support-only derivatives undefined/NaN",
        },
        "search": search,
        "history": result.history,
        "scope": "fixed true macro reconstruction; no detection significance or mass confidence intervals",
    }
    if not result.success:
        export_failure(metadata, output)
        return metadata
    summary, truth = summarize_result(
        result, observation, affine_diagnostics=affine_diagnostics
    )
    metadata["summary"] = summary
    arrays = {
        name: np.asarray(getattr(result, name))
        for name in RESULT_ARRAYS
        if getattr(result, name) is not None
    }
    grid = result.potential_grid
    for name in (
        "node_indices",
        "x",
        "y",
        "core_indices",
        "derivative_indices",
        "interpolation_indices",
        "interpolation_weights",
        "query_x",
        "query_y",
    ):
        arrays[f"potential_{name}"] = np.asarray(getattr(grid, name))
    arrays.update(source_bbox=np.asarray(result.source_bbox), **truth)
    for name in (
        "image_data",
        "noise_map",
        "noise_realization",
        "model_image",
        "psf_kernel",
        "mask",
    ):
        arrays[name if name != "model_image" else "truth_model_image"] = np.asarray(
            getattr(observation, name)
        )
    arrays["image_xgrid"] = np.asarray(observation.sim_config.xgrid)
    arrays["image_ygrid"] = np.asarray(observation.sim_config.ygrid)
    np.savez_compressed(output / "reconstruction.npz", **arrays)
    write_json(output / "result.json", metadata)
    (output / "failure.json").unlink(missing_ok=True)
    if make_plots:
        plot_result_products(result, observation, output, truth=truth)
    return metadata


def plot_result_products(result, observation, output, *, truth=None):
    """Plot raw public products, masking undefined domains rather than filling them."""
    if not result.success:
        raise ValueError("Cannot plot a failed reconstruction as scientific recovery")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if truth is None:
        truth, _ = truth_products(result, observation)
    label = f"{'Subhalo' if observation.has_subhalo else 'Matched control'} | {result.backend} | {result.algorithm}"
    plt.rcParams.update(
        {"font.size": 9, "pdf.fonttype": 42, "axes.titleweight": "normal"}
    )

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=150)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)

    def image(ax, values, title, extent, *, diverging=False, limit=None):
        values = np.asarray(values)
        if diverging:
            vmax = limit or max(float(np.nanmax(np.abs(values))), 1e-12)
            options = {"vmin": -vmax, "vmax": vmax, "cmap": "RdBu_r"}
        else:
            options = {"cmap": "magma"}
        im = ax.imshow(
            values, origin="lower", extent=extent, interpolation="none", **options
        )
        ax.set(title=title, xlabel="x (arcsec)", ylabel="y (arcsec)")
        fig = ax.figure
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    native_extent = (
        -observation.config.npix * observation.config.dpix / 2,
        observation.config.npix * observation.config.dpix / 2,
    ) * 2
    mask = np.asarray(observation.mask)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    fig.suptitle(
        label
        + "; evidence uses "
        + (
            "linear prediction"
            if result.algorithm == "single_step"
            else "local nonlinear curvature"
        )
    )
    image(
        axes[0, 0],
        np.where(mask, np.nan, observation.image_data),
        "Observed image",
        native_extent,
    )
    if result.linear_model_image is not None:
        image(
            axes[0, 1],
            np.where(mask, np.nan, result.linear_model_image),
            "Fitted linear prediction",
            native_extent,
        )
        image(
            axes[1, 0],
            result.linear_normalized_residual,
            "Linear residual / noise",
            native_extent,
            diverging=True,
            limit=4,
        )
        image(
            axes[1, 2],
            np.where(
                mask,
                np.nan,
                (
                    np.asarray(result.nonlinear_model_image)
                    - np.asarray(result.linear_model_image)
                )
                / np.asarray(observation.noise_map),
            ),
            "(Nonlinear - linear) / noise",
            native_extent,
            diverging=True,
        )
    else:
        image(
            axes[0, 1],
            np.where(mask, np.nan, observation.model_image),
            "Noiseless analytic truth",
            native_extent,
        )
        image(
            axes[1, 0],
            np.where(
                mask,
                np.nan,
                (
                    np.asarray(observation.model_image)
                    - np.asarray(result.nonlinear_model_image)
                )
                / np.asarray(observation.noise_map),
            ),
            "(Truth - nonlinear) / noise",
            native_extent,
            diverging=True,
        )
        axes[1, 2].set_visible(False)
    image(
        axes[0, 2],
        np.where(mask, np.nan, result.nonlinear_model_image),
        "Nonlinear re-render",
        native_extent,
    )
    image(
        axes[1, 1],
        result.nonlinear_normalized_residual,
        "Nonlinear residual / noise",
        native_extent,
        diverging=True,
        limit=4,
    )
    save(fig, "images")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    fig.suptitle(label + "; signed source coefficients, no clipping")
    shape = (len(result.source_y_axis), len(result.source_x_axis))
    for ax, value, title in zip(
        axes,
        (
            truth["truth_source_values"],
            np.asarray(result.source_values),
            np.asarray(result.source_values) - truth["truth_source_values"],
        ),
        ("Analytic source at nodes", "Recovered source", "Recovered - truth"),
    ):
        image(
            ax,
            value.reshape(shape),
            title,
            result.source_bbox,
            diverging=title == "Recovered - truth",
        )
    save(fig, "source")

    grid = result.potential_grid
    indices = np.asarray(grid.node_indices)
    lo, hi = indices.min(axis=0), indices.max(axis=0)
    shape = (hi[1] - lo[1] + 1, hi[0] - lo[0] + 1)
    extent = (
        float(np.min(grid.x)) - grid.spacing / 2,
        float(np.max(grid.x)) + grid.spacing / 2,
        float(np.min(grid.y)) - grid.spacing / 2,
        float(np.max(grid.y)) + grid.spacing / 2,
    )

    def raster(values):
        data = np.full(shape, np.nan)
        data[indices[:, 1] - lo[1], indices[:, 0] - lo[0]] = values
        return data

    fig, axes = plt.subplots(4, 3, figsize=(12, 15), layout="constrained")
    fig.suptitle(label + "; raw fields; white pixels have no derivative support")
    for row, field, truth_field, unit in (
        (0, "potential_values", "truth_potential_values", "psi (arcsec^2)"),
        (1, "deflection_x", "truth_discrete_deflection_x", "alpha_x (arcsec)"),
        (2, "deflection_y", "truth_discrete_deflection_y", "alpha_y (arcsec)"),
        (3, "convergence", "truth_discrete_convergence", "kappa"),
    ):
        recovered, true = np.asarray(getattr(result, field)), truth[truth_field]
        common = max(
            float(np.nanmax(np.abs(true))), float(np.nanmax(np.abs(recovered))), 1e-12
        )
        for column, values, heading in (
            (0, true, "Truth"),
            (1, recovered, "Recovered"),
            (2, recovered - true, "Recovered - truth"),
        ):
            image(
                axes[row, column],
                raster(values),
                heading + ": " + unit,
                extent,
                diverging=True,
                limit=common if column < 2 else None,
            )
            if row == 3:
                axes[row, column].plot(
                    observation.config.subhalo_x,
                    observation.config.subhalo_y,
                    marker="+",
                    color="black",
                    ms=8,
                )
    save(fig, "fields")
    if result.history:
        history = [
            row
            for row in result.history
            if isinstance(row, dict) and "objective" in row
        ]
        if history:
            fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
            ax.plot(range(len(history)), [row["objective"] for row in history], "o-")
            ax.set(
                xlabel="Recorded iteration",
                ylabel="Nonlinear regularized objective",
                title=label,
            )
            save(fig, "history")
