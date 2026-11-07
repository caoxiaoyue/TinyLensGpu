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

from examples.potential_correction._recovery_diagnostics import (
    _ray_domain,
    _recovery_metrics,
    _rms,
    _sample_truth,
    _truth_arrays,
)

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


def optimization_record(search):
    """Persist public search status separately from its retained inner fit."""
    return dict(
        has_valid_fit=search.has_valid_fit,
        optimizer_success=search.optimizer_success,
        optimizer_status=search.optimizer_status,
        message=search.message,
        nfev=search.nfev,
        nit=search.nit,
        log_bounds=search.log_bounds,
        selected_log_strengths=search.log_strengths,
        strengths=None if search.strengths is None else asdict(search.strengths),
        boundary_sides=search.boundary_sides,
        boundary_distances=search.boundary_distances,
        configuration=search.configuration,
        evaluations=[asdict(row) for row in search.evaluations],
    )


def export_failure(record, output):
    """Replace this exporter's known success artifacts with a diagnostic failure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename in (
        "result.json",
        "reconstruction.npz",
        "failure-history.png",
        "failure-history.pdf",
        *(f"{name}.{ext}" for name in PLOT_NAMES for ext in ("png", "pdf")),
    ):
        (output / filename).unlink(missing_ok=True)
    write_json(output / "failure.json", record)


def _ray_margins(grid, observation, bbox, potential=None, *, analytic_truth=False):
    margin, brightness = _ray_domain(
        grid, observation, bbox, potential, analytic_truth=analytic_truth
    )
    outside = margin < 0
    return {
        "query_count": len(margin),
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
    truth = _sample_truth(
        result.source_x_axis,
        result.source_y_axis,
        result.potential_grid,
        observation,
        result.dtype,
    )
    return (
        _truth_arrays(truth, result.potential_grid, result.dtype),
        truth["local_radius"],
    )


def summarize_result(result, observation, *, affine_diagnostics=False):
    """Return raw recovery, localization and residual summaries on named domains."""
    if not result.success:
        raise ValueError("A failed reconstruction has no scientific recovery summary")
    grid = result.potential_grid
    samples = _sample_truth(
        result.source_x_axis, result.source_y_axis, grid, observation, result.dtype
    )
    truth = _truth_arrays(samples, grid, result.dtype)
    summary, aligned = _recovery_metrics(
        result, samples, observation, affine_diagnostics=affine_diagnostics
    )
    summary.update(
        {
            "nonlinear_residual_rms_noise": _rms(
                np.asarray(result.nonlinear_normalized_residual)[
                    ~np.asarray(observation.mask)
                ]
            ),
            "source_box": {},
        }
    )
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
        # This schema historically exports only the aligned potential metric.
        summary["affine_diagnostic_only"].pop("deflection_vector_rmse")
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
            "local_core": "matched disk around specified subhalo position (nominal for control); radius max(2*sqrt(mock b_sub*theta_E),2*h)",
            "raw_fields": "signed, unclipped and unaligned; support-only derivatives undefined/NaN",
        },
        "search": search,
        "history": result.history,
        "scope": "fixed true macro reconstruction; no detection significance or mass confidence intervals",
    }
    if not result.success:
        export_failure(metadata, output)
        if make_plots and result.history:
            plot_iteration_history(result, output, diagnostic_failure=True)
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
    for suffix in ("png", "pdf"):
        (output / f"failure-history.{suffix}").unlink(missing_ok=True)
    if make_plots:
        plot_result_products(result, observation, output, truth=truth)
    return metadata


def load_result(output):
    """Restore a successful public result from safe NPZ/JSON, without fitting.

    Saved geometry and fields are restored directly. Failure records cannot
    be loaded through this success boundary. Observation/truth context stays
    explicit when replotting or computing a new diagnostic.
    """
    from TinyLensGpu.ObservationModel import (
        PotentialCorrectionResult,
        PotentialCorrectionStrengths,
        PotentialSolverInfo,
    )
    from TinyLensGpu.utils.lensing.mapping import source_boundary_for_interpolation
    from TinyLensGpu.utils.lensing.potential import PotentialGrid

    output = Path(output)
    if (output / "failure.json").exists():
        raise ValueError("Cannot load a failure record as a successful reconstruction")
    metadata = json.loads((output / "result.json").read_text())
    if not metadata["success"]:
        raise ValueError("Cannot load a failure record as a successful reconstruction")
    with np.load(output / "reconstruction.npz", allow_pickle=False) as saved:
        with jax.experimental.enable_x64(metadata["dtype"] == "float64"):
            arrays = {
                name: jnp.asarray(saved[name]) if name in saved else None
                for name in RESULT_ARRAYS
            }
            geometry = {
                name: jnp.asarray(saved[f"potential_{name}"])
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
                )
            }
            grid = PotentialGrid(
                spacing=metadata["potential_geometry"]["spacing"],
                origin=tuple(metadata["potential_geometry"]["origin"]),
                derivative_valid=arrays["derivative_valid"],
                **geometry,
            )
            configuration = dict(metadata["result_configuration"])
            source_interpolation = configuration.setdefault(
                "source_interpolation", "bilinear"
            )
            if "source_boundary" not in configuration:
                configuration["source_boundary"] = source_boundary_for_interpolation(
                    source_interpolation
                )
            configuration["image_shape"] = tuple(configuration["image_shape"])
            if "block_sizes" in configuration:
                configuration["block_sizes"] = tuple(configuration["block_sizes"])
            if "block_indices" in configuration:
                configuration["block_indices"] = tuple(
                    tuple(block) for block in configuration["block_indices"]
                )
            scalar_names = (
                "log_evidence",
                "chi2",
                "regularization_penalty",
                "objective",
                "prior_logdet",
                "posterior_logdet",
                "noise_logdet",
            )
            scalars = {
                name: jnp.asarray(metadata[name], dtype=metadata["dtype"])
                for name in scalar_names
            }
            names = (
                "dtype",
                "backend",
                "algorithm",
                "status",
                "success",
                "failure_reason",
                "termination_reason",
                "evidence_method",
                "determinant_method",
                "evidence_is_exact",
                "initialization_provenance",
            )
            return PotentialCorrectionResult(
                **arrays,
                **scalars,
                **{name: metadata[name] for name in names},
                source_bbox=tuple(metadata["source_bbox"]),
                potential_grid=grid,
                strengths=PotentialCorrectionStrengths(**metadata["strengths"]),
                solver_info=PotentialSolverInfo(**metadata["solver_info"]),
                configuration=configuration,
                history=tuple(metadata["history"]),
            )


def compare_reconstructions(reference, candidate):
    """Compare matched public raw results without conflating evidence models.

    Inference objectives retain their original image definition. A separate
    nonlinear objective permits a physical single-step/iterative comparison.
    All arithmetic after device transfer uses host float64 values.
    """
    if not (reference.success and candidate.success):
        return dict(
            both_successful=False,
            termination_reasons=[
                reference.termination_reason,
                candidate.termination_reason,
            ],
        )
    for field in (
        "source_x_axis",
        "source_y_axis",
        "source_bbox",
        "data_active",
        "noise_active",
        "derivative_valid",
    ):
        if not np.array_equal(
            np.asarray(getattr(reference, field)), np.asarray(getattr(candidate, field))
        ):
            raise ValueError(f"Reconstruction comparison requires identical {field}")
    for field in ("node_indices", "core_indices", "x", "y", "query_x", "query_y"):
        if not np.array_equal(
            np.asarray(getattr(reference.potential_grid, field)),
            np.asarray(getattr(candidate.potential_grid, field)),
        ):
            raise ValueError(
                f"Reconstruction comparison requires identical potential {field}"
            )

    def nonlinear_objective(result):
        residual = (
            np.asarray(result.data_active) - np.asarray(result.nonlinear_image_active)
        ) / np.asarray(result.noise_active)
        return 0.5 * float(residual @ residual) + 0.5 * float(
            result.regularization_penalty
        )

    left_strengths, right_strengths = asdict(reference.strengths), asdict(
        candidate.strengths
    )
    left_partition = reference.configuration.get("block_indices")
    right_partition = candidate.configuration.get("block_indices")
    report = dict(
        both_successful=True,
        same_strengths=all(
            np.isclose(left_strengths[key], right_strengths[key], rtol=1e-12, atol=0)
            for key in left_strengths
        ),
        same_source_regularization=reference.configuration["source_regularization"]
        == candidate.configuration["source_regularization"],
        source_interpolations=[
            fit.configuration.get("source_interpolation", "bilinear")
            for fit in (reference, candidate)
        ],
        source_boundaries=[
            fit.configuration.get("source_boundary", "hard_zero_outside_bbox")
            for fit in (reference, candidate)
        ],
        same_static_partition=(
            None
            if left_partition is None or right_partition is None
            else left_partition == right_partition
        ),
        evidence_models=[reference.evidence_method, candidate.evidence_method],
        determinant_methods=[
            reference.determinant_method,
            candidate.determinant_method,
        ],
        evidence_exact_flags=[reference.evidence_is_exact, candidate.evidence_is_exact],
        initialization_provenance=[
            reference.initialization_provenance,
            candidate.initialization_provenance,
        ],
        reference_inference_objective=float(reference.objective),
        candidate_inference_objective=float(candidate.objective),
        reference_nonlinear_objective=nonlinear_objective(reference),
        candidate_nonlinear_objective=nonlinear_objective(candidate),
        raw_fields_aligned=False,
        field_differences={},
    )
    valid = np.asarray(reference.derivative_valid)
    for field in (
        "source_values",
        "potential_values",
        "deflection_x",
        "deflection_y",
        "convergence",
    ):
        left, right = np.asarray(getattr(reference, field)), np.asarray(
            getattr(candidate, field)
        )
        delta = (
            (right - left)[valid]
            if field in ("deflection_x", "deflection_y", "convergence")
            else right - left
        )
        report["field_differences"][field] = dict(
            max_absolute_difference=float(np.max(np.abs(delta))),
            rms_difference=_rms(delta),
        )
    report["nonlinear_image_rms_difference_noise"] = _rms(
        (
            np.asarray(candidate.nonlinear_image_active)
            - np.asarray(reference.nonlinear_image_active)
        )
        / np.asarray(reference.noise_active)
    )
    report["candidate_minus_reference_log_evidence"] = float(
        candidate.log_evidence
    ) - float(reference.log_evidence)
    report["inference_objective_component"] = -float(candidate.objective) + float(
        reference.objective
    )
    report["determinant_component"] = -0.5 * (
        (float(candidate.posterior_logdet) - float(reference.posterior_logdet))
        - (float(candidate.prior_logdet) - float(reference.prior_logdet))
    )
    report["noise_component"] = -0.5 * (
        float(candidate.noise_logdet) - float(reference.noise_logdet)
    )
    return report


def plot_iteration_history(result, output, *, diagnostic_failure=False):
    """Plot public trial history, with failed runs visibly diagnostic only.

    The retained objective includes rejected attempts without accepting them.
    History gradients describe pre-step points; the refreshed final gradient
    is a separate marker when the result supplies it.
    """
    if not result.success and not diagnostic_failure:
        raise ValueError("A failed history requires explicit diagnostic_failure=True")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(result.history)
    if not rows:
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rows))
    before = np.array(
        [row.get("objective_before", np.nan) for row in rows], dtype=float
    )
    after = np.array([row.get("objective_after", np.nan) for row in rows], dtype=float)
    trial = np.array([row.get("objective_trial", np.nan) for row in rows], dtype=float)
    accepted = np.array([row.get("accepted", False) for row in rows])
    rejected = (~accepted) & (x > 0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    role = "DIAGNOSTIC FAILURE" if not result.success else "Iterative reconstruction"
    fig.suptitle(f"{role} | {result.backend} | {result.termination_reason}")
    objective, damping_ax, gradient_ax, residual_ax = axes.ravel()
    objective.plot(x, before, ":", color="gray", label="Before trial")
    objective.plot(x, after, color="black", label="Retained objective")
    objective.plot(
        x[accepted], trial[accepted], "o", ms=4, color="#28795a", label="Accepted trial"
    )
    objective.plot(
        x[rejected], trial[rejected], "x", ms=5, color="#b54c43", label="Rejected trial"
    )
    objective.set(ylabel="Nonlinear regularized objective")
    if np.isfinite(after).all() and np.all(after > 0):
        objective.set_yscale("log")
    objective.legend(fontsize=8)
    damping_ax.semilogy(x, [row["damping"] for row in rows], "o-", ms=3)
    damping_ax.set(ylabel="Damping used for trial")
    gradient_x = [
        i for i, row in enumerate(rows) if row.get("scaled_gradient") is not None
    ]
    gradient_ax.semilogy(
        gradient_x,
        [rows[i]["scaled_gradient"] for i in gradient_x],
        ".-",
        label="Pre-step gradient",
    )
    initial = result.configuration.get("initial_scaled_gradient")
    if initial is not None:
        threshold = (
            result.configuration["gradient_atol"]
            + result.configuration["gradient_rtol"] * initial
        )
        gradient_ax.axhline(
            threshold, color="gray", ls="--", label="Gradient stop threshold"
        )
    final = result.configuration.get("final_scaled_gradient")
    if final is not None:
        gradient_ax.plot(
            len(rows), final, "*", ms=10, color="black", label="Final current gradient"
        )
    gradient_ax.set(ylabel="Physical scaled gradient")
    if gradient_x or final is not None:
        gradient_ax.legend(fontsize=8)
    inner = [
        (i, row["solver_info"]) for i, row in enumerate(rows) if "solver_info" in row
    ]
    if inner:
        residual_ax.semilogy(
            [i for i, _ in inner],
            [info["residual_norm"] for _, info in inner],
            ".-",
            label="True full-system residual",
        )
        thresholds = []
        recursive = []
        for _, info in inner:
            diagnostic = info.get("diagnostics", {})
            threshold = diagnostic.get("true_residual_threshold")
            if threshold is None and info["relative_residual"] > 0:
                rhs_norm = info["residual_norm"] / info["relative_residual"]
                threshold = (
                    result.configuration.get("solve_atol", 0)
                    + result.configuration["solve_rtol"] * rhs_norm
                )
            thresholds.append(threshold if threshold is not None else np.nan)
            phases = diagnostic.get("phases", [])
            recursive.append(phases[0]["recursive_residual_norm"] if phases else np.nan)
        residual_ax.semilogy(
            [i for i, _ in inner],
            thresholds,
            "--",
            color="gray",
            label="True residual threshold",
        )
        if np.isfinite(recursive).any():
            residual_ax.semilogy(
                [i for i, _ in inner],
                recursive,
                ":",
                label="Initial PCG recursive residual",
            )
        residual_ax.legend(fontsize=8)
    residual_ax.set(ylabel="Inner linear residual norm")
    for ax in axes.ravel():
        ax.set_xlabel("Trial record (including retries)")
    name = "failure-history" if not result.success else "history"
    fig.savefig(output / f"{name}.png", dpi=150)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


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

    def image(
        ax, values, title, extent, *, diverging=False, limit=None, intensity_limits=None
    ):
        values = np.asarray(values)
        if diverging:
            vmax = limit or max(float(np.nanmax(np.abs(values))), 1e-12)
            options = {"vmin": -vmax, "vmax": vmax, "cmap": "RdBu_r"}
        else:
            options = {"cmap": "magma"}
            if intensity_limits is not None:
                options.update(vmin=intensity_limits[0], vmax=intensity_limits[1])
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
    comparison_images = [
        observation.image_data,
        result.nonlinear_model_image,
        (
            result.linear_model_image
            if result.linear_model_image is not None
            else observation.model_image
        ),
    ]
    image_limits = (
        min(float(np.min(np.asarray(values)[~mask])) for values in comparison_images),
        max(float(np.max(np.asarray(values)[~mask])) for values in comparison_images),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    fig.suptitle(
        label
        + "; evidence uses "
        + (
            "linear prediction"
            if result.algorithm == "single_step"
            else "local Gauss–Newton curvature"
        )
    )
    image(
        axes[0, 0],
        np.where(mask, np.nan, observation.image_data),
        "Observed image",
        native_extent,
        intensity_limits=image_limits,
    )
    if result.linear_model_image is not None:
        image(
            axes[0, 1],
            np.where(mask, np.nan, result.linear_model_image),
            "Fitted linear prediction",
            native_extent,
            intensity_limits=image_limits,
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
            intensity_limits=image_limits,
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
        (
            "Nonlinear re-render"
            if result.algorithm == "single_step"
            else "Final nonlinear image"
        ),
        native_extent,
        intensity_limits=image_limits,
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
    source_dx = float(np.diff(np.asarray(result.source_x_axis))[0])
    source_dy = float(np.diff(np.asarray(result.source_y_axis))[0])
    source_extent = (
        float(result.source_x_axis[0]) - source_dx / 2,
        float(result.source_x_axis[-1]) + source_dx / 2,
        float(result.source_y_axis[0]) - source_dy / 2,
        float(result.source_y_axis[-1]) + source_dy / 2,
    )
    source_limits = (
        min(
            float(np.min(truth["truth_source_values"])),
            float(np.min(result.source_values)),
        ),
        max(
            float(np.max(truth["truth_source_values"])),
            float(np.max(result.source_values)),
        ),
    )
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
            source_extent,
            diverging=title == "Recovered - truth",
            intensity_limits=source_limits,
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
    fig.suptitle(
        label
        + "\npsi on U; deflection and convergence on core C; cross = "
        + (
            "injected position"
            if observation.has_subhalo
            else "nominal subhalo position"
        )
    )
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
        plot_iteration_history(result, output)
