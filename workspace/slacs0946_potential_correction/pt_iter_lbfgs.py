"""Select three J0946 source–potential strengths with DIRECT and L-BFGS-B.

Run with the local M3 handoff in the tinylens_gpu environment. The fitted
image uses Float64 Hermite interpolation. DIRECT maximizes the dense local
GN evidence over log10 strengths, using a fixed start for every candidate.
Use --plot-only to redraw the diagnostic panels from a completed run.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parents[1]))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.optimize import direct

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import (
    LBFGSPotentialCorrection,
    PotentialCorrectionStrengths,
    PotentialCorrectionWhiteningReference,
)
from TinyLensGpu.PhysicalModel import EPL, PhysicalModel, Shear
from TinyLensGpu.example_support import resolve_source_interpolation
from examples.potential_correction.result_products import (
    RESULT_ARRAYS,
    json_safe,
    write_json,
)
from workspace.slacs0946_potential_correction.pt import load_handoff


STRENGTH_ORDER = ("lambda_s", "lambda_4", "lambda_0")
# Potential bounds from the archived J0946 three-strength Hermite search.
J0946_LOG10_LAMBDA4 = (-1.2051919733248864, 7.794808026675113)
J0946_LOG10_LAMBDA0 = (0.9378228269292088, 6.937822826929208)
REFINE_RADIUS_DEX = 0.3
DIRECT_OPTIONS = {
    "maxiter": 100,
    "eps": 1e-4,
    "locally_biased": False,
    "len_tol": 0.002,
    "vol_tol": 1e-16,
}


def prepare_problem(macro_dir, potential_spacing):
    metadata, observation = load_handoff(macro_dir)
    parameters = metadata["macro_parameters"]
    # Preserve native Float32 image sampling before promoting the rays.
    with jax.experimental.enable_x64(False):
        simulation = SimulatorConfig(
            metadata["dpix"],
            observation["mask"].shape[0],
            jnp.asarray(observation["psf_kernel"], dtype=jnp.float32),
            nsub=metadata["nsub"],
            mask=observation["mask"],
        )
    macro = PhysicalModel(
        lens_mass=[
            EPL(
                theta_E=parameters["theta_E"],
                gamma=parameters["gamma"],
                e1=parameters["e1_mass"],
                e2=parameters["e2_mass"],
                center_x=parameters["center_x_mass"],
                center_y=parameters["center_y_mass"],
            ),
            Shear(gamma1=parameters["gamma1"], gamma2=parameters["gamma2"]),
        ],
        source_light=[],
        lens_light=[],
    )
    renderer = PotentialCorrectionRenderer(
        macro,
        simulation,
        metadata["source_n"],
        observation["source_bbox"],
        potential_spacing=potential_spacing,
        dtype="float64",
        source_interpolation="cubic_hermite",
    )
    reference_interpolation = resolve_source_interpolation(
        metadata=metadata, context="M3 handoff"
    )
    source = observation["source_values"]
    problem = LBFGSPotentialCorrection(
        renderer,
        observation["lens_subtracted_image"],
        observation["noise_map"],
        source,
        source_regularization=metadata["source_regularization"],
        source_scale=observation["source_scale"],
        initialization_provenance={
            "source": "saved M3 source intensities",
            "handoff_sha256": metadata["handoff_sha256"],
            "m3_sample_index": metadata["selected_sample_index"],
        },
        whitening_reference=PotentialCorrectionWhiteningReference(
            source, reference_interpolation
        ),
    )
    return problem, metadata, observation


def log10_bounds(args, metadata):
    center = np.log10(metadata["lambda_source"])
    bounds = np.asarray(
        (
            args.log10_lambda_s or (center - 2, center + 2),
            args.log10_lambda4 or J0946_LOG10_LAMBDA4,
            args.log10_lambda0 or J0946_LOG10_LAMBDA0,
        ),
        dtype=np.float64,
    )
    with np.errstate(over="ignore", under="ignore"):
        strengths = 10.0**bounds
    if not (
        np.all(np.isfinite(bounds))
        and np.all(bounds[:, 0] < bounds[:, 1])
        and np.all(np.isfinite(strengths) & (strengths > 0))
    ):
        raise ValueError(
            "Log10 bounds must be increasing and give finite positive strengths"
        )
    return bounds


def search(problem, bounds, output, *, maxfun, refine_maxfun, fit_options):
    """Retain the highest-evidence qualified result visited by either DIRECT pass."""
    scores = {}
    best = best_result = None
    qualified_count = 0
    passes = []
    with (output / "evaluations.jsonl").open("x", buffering=1) as stream:

        def evaluate(log_strengths):
            nonlocal best, best_result, qualified_count
            key = tuple(map(float, log_strengths))
            if key in scores:
                return scores[key]
            strengths = PotentialCorrectionStrengths(
                *map(float, 10.0 ** np.asarray(key))
            )
            result = problem.fit(strengths, **fit_options)
            evidence = (
                None if result.log_evidence is None else float(result.log_evidence)
            )
            qualified = (
                result.success and evidence is not None and np.isfinite(evidence)
            )
            row = {
                "index": len(scores),
                "phase": phase,
                "log10_strengths": key,
                "strengths": asdict(strengths),
                "qualified": qualified,
                "failure_reason": result.failure_reason,
                "objective": result.objective,
                "log_evidence": evidence,
                "optimizer_info": asdict(result.optimizer_info),
                "timings": result.configuration["timings"],
            }
            scores[key] = -evidence if qualified else np.inf
            stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
            qualified_count += int(qualified)
            if qualified and (best is None or evidence > best["log_evidence"]):
                best, best_result = row, result
                print(
                    f"[{phase}] fit {row['index']} best log evidence {evidence:.8f}",
                    flush=True,
                )
            elif len(scores) % 10 == 0:
                print(
                    f"[{phase}] {len(scores)} fits, {qualified_count} qualified",
                    flush=True,
                )
            return scores[key]

        for phase, budget in (("global", maxfun), ("refine", refine_maxfun)):
            if phase == "global":
                phase_bounds = bounds
            elif budget and best is not None:
                center = np.asarray(best["log10_strengths"])
                phase_bounds = np.column_stack(
                    (
                        np.maximum(bounds[:, 0], center - REFINE_RADIUS_DEX),
                        np.minimum(bounds[:, 1], center + REFINE_RADIUS_DEX),
                    )
                )
            else:
                continue
            before = len(scores)
            started = perf_counter()
            native = direct(
                evaluate, phase_bounds.tolist(), maxfun=budget, **DIRECT_OPTIONS
            )
            passes.append(
                {
                    "phase": phase,
                    "log10_bounds": phase_bounds,
                    "maxfun": budget,
                    "success": bool(native.success),
                    "status": int(native.status),
                    "message": str(native.message),
                    "nfev": int(native.nfev),
                    "nit": int(native.nit),
                    "x": native.x,
                    "negative_log_evidence": native.fun,
                    "new_fits": len(scores) - before,
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            print(f"[{phase}] {native.message}", flush=True)

    return (
        best,
        best_result,
        {
            "success": best is not None,
            "evaluations": len(scores),
            "qualified": qualified_count,
            "passes": passes,
        },
    )


def save_reconstruction(destination, result, selected, observation):
    destination.mkdir()
    fields = {
        name: np.asarray(getattr(result, name))
        for name in RESULT_ARRAYS
        if getattr(result, name) is not None
    }
    grid = result.potential_grid
    fields.update(
        source_bbox=result.source_bbox,
        source_initial=observation["source_values"],
        potential_x=np.asarray(grid.x),
        potential_y=np.asarray(grid.y),
        potential_node_indices=np.asarray(grid.node_indices),
        image_data=observation["image_data"],
        lens_subtracted_image=observation["lens_subtracted_image"],
        lens_light_model=observation["lens_light_model"],
        macro_model_image=observation["macro_model_image"],
        mask=observation["mask"],
        noise_map=observation["noise_map"],
        psf_kernel=observation["psf_kernel"],
        source_scale=observation["source_scale"],
    )
    valid = fields["derivative_valid"]
    peak = np.flatnonzero(valid)[np.argmax(fields["convergence"][valid])]
    reconstruction = destination / "reconstruction.npz"
    np.savez_compressed(reconstruction, **fields)
    write_json(
        destination / "result.json",
        {
            **selected,
            "algorithm": result.algorithm,
            "backend": result.backend,
            "dtype": result.dtype,
            "chi2": result.chi2,
            "nonlinear_residual_rms_noise": np.sqrt(
                float(result.chi2) / result.data_active.size
            ),
            "regularization_penalty": result.regularization_penalty,
            "evidence_method": result.evidence_method,
            "evidence_is_exact": result.evidence_is_exact,
            "determinant_method": result.determinant_method,
            "prior_logdet": result.prior_logdet,
            "posterior_logdet": result.posterior_logdet,
            "noise_logdet": result.noise_logdet,
            "configuration": result.configuration,
            "initialization_provenance": result.initialization_provenance,
            "potential_origin_arcsec": grid.origin,
            "delta_kappa_peak": fields["convergence"][peak],
            "delta_kappa_peak_xy_arcsec": [
                fields["potential_x"][peak],
                fields["potential_y"][peak],
            ],
            "reconstruction_sha256": hashlib.sha256(
                reconstruction.read_bytes()
            ).hexdigest(),
        },
    )


def plot_saved_diagnostics(output, *, require_matplotlib_panel_alignment=None):
    """Render the six reconstruction panels from the retained result."""
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    destination = output / "maximum_evidence"
    result = json.loads((destination / "result.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    with np.load(destination / "reconstruction.npz", allow_pickle=False) as saved:
        mask = saved["mask"]
        observed = np.where(mask, np.nan, saved["lens_subtracted_image"])
        model = np.where(mask, np.nan, saved["nonlinear_model_image"])
        residual = np.where(mask, np.nan, saved["nonlinear_normalized_residual"])
        sx, sy = saved["source_x_axis"], saved["source_y_axis"]
        source = saved["source_values"].reshape(sy.size, sx.size)
        nodes = saved["potential_node_indices"]
        ix, iy = (nodes - nodes.min(axis=0)).T
        shape = (int(iy.max()) + 1, int(ix.max()) + 1)
        potential, convergence = np.full(shape, np.nan), np.full(shape, np.nan)
        potential[iy, ix] = saved["potential_values"]
        valid = saved["derivative_valid"]
        convergence[iy[valid], ix[valid]] = saved["convergence"][valid]
        px, py = saved["potential_x"], saved["potential_y"]

    dpix = manifest["fixed_macro"]["dpix"]
    height, width = mask.shape
    image_extent = np.array([-width, width, -height, height]) * dpix / 2
    source_extent = np.array(
        [
            sx[0] - (sx[1] - sx[0]) / 2,
            sx[-1] + (sx[-1] - sx[-2]) / 2,
            sy[0] - (sy[1] - sy[0]) / 2,
            sy[-1] + (sy[-1] - sy[-2]) / 2,
        ]
    )
    half_spacing = result["configuration"]["potential_spacing"] / 2
    potential_extent = np.array(
        [
            px.min() - half_spacing,
            px.max() + half_spacing,
            py.min() - half_spacing,
            py.max() + half_spacing,
        ]
    )
    row, col = np.nonzero(~mask)
    selected_x = (col - (width - 1) / 2) * dpix
    selected_y = (row - (height - 1) / 2) * dpix
    view_radius = (
        max(
            np.max(np.abs(selected_x)) + dpix / 2,
            np.max(np.abs(selected_y)) + dpix / 2,
            np.max(np.abs(potential_extent)),
        )
        + dpix
    )
    image_view = ((-view_radius, view_radius),) * 2
    source_intervals = source_extent.reshape(2, 2)
    source_radius = np.ptp(source_intervals, axis=1).max() / 2
    source_view = tuple(
        (center - source_radius, center + source_radius)
        for center in source_intervals.mean(axis=1)
    )

    brightness = np.r_[observed[~mask], model[~mask]]
    brightness_limits = (brightness.min(), brightness.max())
    residual_limit = max(3.0, np.nanmax(np.abs(residual)))
    potential_limit = max(np.nanmax(np.abs(potential)), np.finfo(float).eps)
    convergence_limit = max(np.nanmax(np.abs(convergence)), np.finfo(float).eps)
    blank = "#e8eaee"
    flux_map = mpl.colormaps["magma"].with_extremes(bad=blank)
    signed_map = mpl.colormaps["RdBu_r"].with_extremes(bad=blank)
    panels = (
        (
            "Lens-light-subtracted data",
            observed,
            image_extent,
            image_view,
            flux_map,
            brightness_limits,
            "Brightness",
        ),
        (
            "Nonlinear model image",
            model,
            image_extent,
            image_view,
            flux_map,
            brightness_limits,
            "Brightness",
        ),
        (
            "Normalized residual",
            residual,
            image_extent,
            image_view,
            signed_map,
            (-residual_limit, residual_limit),
            "(Data − model) / σ",
        ),
        (
            "Reconstructed source",
            source,
            source_extent,
            source_view,
            flux_map,
            (source.min(), source.max()),
            "Source brightness",
        ),
        (
            "Potential correction",
            potential,
            potential_extent,
            image_view,
            signed_map,
            (-potential_limit, potential_limit),
            "δψ (arcsec²)",
        ),
        (
            "Convergence correction",
            convergence,
            potential_extent,
            image_view,
            signed_map,
            (-convergence_limit, convergence_limit),
            "δκ",
        ),
    )

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure = Figure(figsize=(7.2, 6.65), dpi=300)
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(
            4,
            3,
            height_ratios=(1, 0.06, 1, 0.06),
            left=0.10,
            right=0.965,
            bottom=0.105,
            top=0.92,
            wspace=0.28,
            hspace=0.42,
        )
        axes = []
        for index, (title, values, extent, view, cmap, limits, units) in enumerate(
            panels
        ):
            row, column = divmod(index, 3)
            axis = figure.add_subplot(grid[2 * row, column])
            axes.append(axis)
            color_axis = figure.add_subplot(grid[2 * row + 1, column])
            axis.set_facecolor(blank)
            image = axis.imshow(
                values,
                extent=extent,
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
            )
            axis.set(xlim=view[0], ylim=view[1])
            axis.set_title(title, fontsize=8.5, pad=5)
            axis.text(
                -0.12,
                1.03,
                "abcdef"[index],
                transform=axis.transAxes,
                fontsize=9,
                fontweight="bold",
                va="bottom",
            )
            axis.set_xlabel("x (arcsec)", labelpad=2)
            if column == 0:
                axis.set_ylabel("y (arcsec)", labelpad=2)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.tick_params(length=2.5, pad=2)
            bar = figure.colorbar(
                image,
                cax=color_axis,
                orientation="horizontal",
                ticks=MaxNLocator(nbins=3),
            )
            bar.set_label(units, labelpad=1)
            color_axis.tick_params(length=2, pad=2)
        figure.suptitle("J0946+1006 L-BFGS-B reconstruction", fontsize=11.5, y=0.982)
        figure.text(
            0.10,
            0.025,
            f"{int((~mask).sum()):,} fitted image pixels. "
            f"δψ on {len(nodes):,} nodes and δκ on {int(valid.sum()):,} valid nodes. "
            "Gray marks cells outside each panel’s domain.",
            fontsize=7.5,
        )
        figure.canvas.draw()
        if require_matplotlib_panel_alignment is not None:
            require_matplotlib_panel_alignment(
                figure,
                axes=axes,
                require_panel_labels=True,
                strict=True,
                json_out=destination / "diagnostics.alignment.json",
            )
        figure.savefig(destination / "diagnostics.pdf", dpi=600)
        figure.savefig(destination / "diagnostics.svg", dpi=600)
        figure.savefig(destination / "diagnostics.png", dpi=600)
        figure.clear()
    print(f"[diagnostics] {destination / 'diagnostics.png'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-dir", type=Path, default=HERE / "macro")
    parser.add_argument("--output", type=Path, default=HERE / "potential_lbfgs")
    parser.add_argument("--potential-spacing", type=float, default=0.1)
    parser.add_argument("--log10-lambda-s", type=float, nargs=2)
    parser.add_argument("--log10-lambda4", type=float, nargs=2)
    parser.add_argument("--log10-lambda0", type=float, nargs=2)
    parser.add_argument(
        "--maxfun", type=int, default=160, help="Nominal global DIRECT budget"
    )
    parser.add_argument(
        "--refine-maxfun",
        type=int,
        default=90,
        help="Nominal refinement budget (0 skips it)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=3000, help="L-BFGS-B iterations per candidate"
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Plot a completed output"
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.plot_only:
        plot_saved_diagnostics(output)
        return
    if args.maxfun < 1 or args.refine_maxfun < 0 or args.max_iter < 1:
        parser.error(
            "Use positive global and inner budgets and a nonnegative refinement budget"
        )
    macro_dir = args.macro_dir.resolve()
    if output.exists() or output.is_relative_to(macro_dir):
        parser.error("Use a new output directory outside the M3 handoff")

    fit_options = {"max_iter": args.max_iter, "gn_stationarity_tol": 1e-3}
    with jax.experimental.enable_x64(), jax.default_matmul_precision("highest"):
        problem, metadata, observation = prepare_problem(
            macro_dir, args.potential_spacing
        )
        bounds = log10_bounds(args, metadata)
        output.mkdir(parents=True, exist_ok=False)
        write_json(
            output / "manifest.json",
            {
                "macro_dir": macro_dir,
                "m3_handoff_json_sha256": hashlib.sha256(
                    (macro_dir / "m3_handoff.json").read_bytes()
                ).hexdigest(),
                "fixed_macro": metadata,
                "initialization_provenance": problem.initialization_provenance,
                "potential_spacing_arcsec": args.potential_spacing,
                "native_sampling_dtype": "float32",
                "fit_dtype": "float64",
                "backend": "dense",
                "strength_order": STRENGTH_ORDER,
                "log10_bounds": bounds,
                "direct_options": DIRECT_OPTIONS,
                "global_maxfun": args.maxfun,
                "refine_maxfun": args.refine_maxfun,
                "refine_radius_dex": REFINE_RADIUS_DEX,
                "fit_options": fit_options,
                "selection": "Highest qualified dense local GN evidence among visited candidates",
                "versions": {
                    "jax": jax.__version__,
                    "scipy": scipy.__version__,
                    "numpy": np.__version__,
                },
                "device": jax.devices()[0].device_kind,
            },
        )
        best, result, summary = search(
            problem,
            bounds,
            output,
            maxfun=args.maxfun,
            refine_maxfun=args.refine_maxfun,
            fit_options=fit_options,
        )

    if result is not None:
        destination = output / "maximum_evidence"
        save_reconstruction(destination, result, best, observation)
        point = np.asarray(best["log10_strengths"])
        summary["selected"] = best
        summary["distance_to_global_bounds_dex"] = dict(
            zip(STRENGTH_ORDER, np.minimum(point - bounds[:, 0], bounds[:, 1] - point))
        )
        summary["result"] = "maximum_evidence/result.json"
    write_json(output / "completion.json", summary)
    if result is None:
        raise SystemExit(
            "DIRECT found no qualified L-BFGS-B candidate; see evaluations.jsonl"
        )
    plot_saved_diagnostics(output)
    print(
        json.dumps(
            {"strengths": best["strengths"], "log_evidence": best["log_evidence"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
