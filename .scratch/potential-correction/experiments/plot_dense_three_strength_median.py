"""Weighted hyperposterior and the conditional reconstruction at its medians.

Figure contract: the corner shows all three marginals and pairwise dependence;
the image plate tests image agreement, source recovery and localized subhalo
convergence against discrete truth. The fields are fitted at marginal median
strengths, not posterior-mean fields or a maximum-evidence solution. Undefined
potential derivatives are masked, and all raw valid coefficients are retained.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_three_strength_posterior import quantile, moments


def default_audit_dirs():
    home = Path.home()
    return (
        home / ".agents/skills/nature-figure/scripts",
        home / ".config/devin/skills/nature-figure/scripts",
    )


def load_auditor(scripts_dir=None):
    """Load the external audit_panel_alignment module by file location."""
    candidates = (
        [Path(scripts_dir)] if scripts_dir is not None else list(default_audit_dirs())
    )
    for directory in candidates:
        path = directory / "audit_panel_alignment.py"
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("audit_panel_alignment", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        audit = getattr(module, "require_matplotlib_panel_alignment", None)
        if callable(audit):
            return audit
    hint = "install the auditor scripts or pass --audit-scripts-dir DIRECTORY"
    raise ValueError(
        f"No usable audit_panel_alignment.py under {[str(c) for c in candidates]}; "
        + hint
    )


def save(fig, base, *, axes, auditor):
    auditor(fig, axes=axes, json_out=base.with_suffix(".alignment.json"), strict=True)
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)


def nodal_image(values, nodes, spacing):
    lower, upper = nodes.min(axis=0), nodes.max(axis=0)
    image = np.full((upper[1] - lower[1] + 1, upper[0] - lower[0] + 1), np.nan)
    image[nodes[:, 1] - lower[1], nodes[:, 0] - lower[0]] = values
    extent = [
        (lower[0] - 0.5) * spacing,
        (upper[0] + 0.5) * spacing,
        (lower[1] - 0.5) * spacing,
        (upper[1] + 0.5) * spacing,
    ]
    return image, extent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory for summaries, figures and audits; "
        "defaults to writing beside the run inputs.",
    )
    parser.add_argument(
        "--audit-scripts-dir",
        type=Path,
        help="Directory containing audit_panel_alignment.py; "
        "overrides the default nature-figure skill locations.",
    )
    args = parser.parse_args()
    auditor = load_auditor(args.audit_scripts_dir)
    run = args.run
    if args.output is None:
        out = run
    else:
        out = args.output.resolve()
        out.mkdir(parents=True)  # must be new: never overwrite existing output
    complete = json.loads((run / "completion.json").read_text())
    assert complete["complete"]
    manifest = json.loads((run / "manifest.json").read_text())
    with np.load(run / "posterior.npz") as a:
        points, weights = a["log10_strengths"], a["weights"]
        names = a["parameter_order"].tolist()
    weights = weights / weights.sum()
    qs = np.array(
        [quantile(points[:, i], weights, [0.16, 0.5, 0.84]) for i in range(3)]
    )
    correlation = moments(points, weights)[2]
    summary = dict(
        parameter_order=names,
        log10_quantiles_16_50_84=qs.tolist(),
        strength_quantiles_16_50_84=(10.0**qs).tolist(),
        pearson_log10=correlation.tolist(),
        kish_effective_sample_size=float(1 / (weights @ weights)),
        n_live=complete["n_live"],
        prior_log10_bounds=manifest["sampler_settings"]["log10_prior_bounds"],
        plotted_contours=[0.68, 0.95],
        displayed_marginal_probability=0.999,
        weights="All normalized importance weights; no equal-weight resampling",
        reconstruction="Conditional fitted fields at each hyperparameter's marginal median",
    )
    (out / "posterior-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "sans-serif",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    order = [2, 1, 0]
    labels = [r"$\log_{10}\lambda_0$", r"$\log_{10}\lambda_4$", r"$\log_{10}\lambda_s$"]
    fig = corner.corner(
        points[:, order],
        weights=weights,
        labels=labels,
        range=[0.999] * 3,
        bins=42,
        smooth=0.8,
        smooth1d=0.8,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        color="#277b8b",
        truths=qs[order, 1],
        truth_color="#ba6a31",
        levels=[0.68, 0.95],
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=4,
        label_kwargs={"fontsize": 12},
        fig=plt.figure(figsize=(8.5, 8.5)),
    )
    grid = np.array(fig.axes).reshape(3, 3)
    for i in range(3):
        q = qs[order[i]]
        grid[i, i].set_title(
            f"{q[1]:.3f}  [{q[0]:.3f}, {q[2]:.3f}]", fontsize=10, pad=10
        )
    fig.subplots_adjust(
        left=0.12, right=0.96, bottom=0.11, top=0.86, hspace=0.16, wspace=0.16
    )
    fig.suptitle("Dense single-step regularization posterior", y=0.98, fontsize=14)
    fig.text(
        0.5,
        0.94,
        f"Nautilus: {complete['n_live']} live points; effective samples {summary['kish_effective_sample_size']:.0f}",
        ha="center",
    )
    fig.text(
        0.5,
        0.91,
        "68% / 95% joint contours; orange: marginal medians; titles: median [16%, 84%]",
        ha="center",
        fontsize=9,
    )
    save(
        fig,
        out / "posterior_corner",
        axes=[grid[i, j] for i in range(3) for j in range(i + 1)],
        auditor=auditor,
    )

    path = run / "posterior-median"
    metadata = json.loads((path / "result.json").read_text())
    assert metadata["success"] and metadata["backend"] == "dense"
    with np.load(path / "reconstruction.npz") as a:
        arrays = dict(a)
    cfg = metadata["mock_configuration"]
    half = cfg["npix"] * cfg["dpix"] / 2
    native = [-half, half, -half, half]
    mask = arrays["mask"]
    observed = np.where(mask, np.nan, arrays["image_data"])
    linear = np.where(mask, np.nan, arrays["linear_model_image"])
    residual = arrays["linear_normalized_residual"]
    source = arrays["source_values"].reshape(len(arrays["source_y_axis"]), -1)
    truth_source = arrays["truth_source_values"].reshape(source.shape)
    source_x = arrays["source_x_axis"]
    source_y = arrays["source_y_axis"]
    source_dx = float(np.diff(source_x)[0])
    source_dy = float(np.diff(source_y)[0])
    # Saved axes locate node centers; imshow's extent locates pixel edges.
    source_extent = (
        float(source_x[0]) - source_dx / 2,
        float(source_x[-1]) + source_dx / 2,
        float(source_y[0]) - source_dy / 2,
        float(source_y[-1]) + source_dy / 2,
    )
    valid = arrays["derivative_valid"]
    recovered = np.where(valid, arrays["convergence"], np.nan)
    truth = np.where(valid, arrays["truth_discrete_convergence"], np.nan)
    nodes, spacing = (
        arrays["potential_node_indices"],
        metadata["potential_geometry"]["spacing"],
    )
    kappa, extent = nodal_image(recovered, nodes, spacing)
    ktruth, _ = nodal_image(truth, nodes, spacing)
    difference, _ = nodal_image(recovered - truth, nodes, spacing)
    fig, axes = plt.subplots(3, 3, figsize=(11, 11.8), layout="constrained")
    strength = 10.0 ** qs[:, 1]
    fig.suptitle(
        "Reconstruction at posterior-median hyperparameters\n"
        + f"lambda_s = {strength[0]:.3g}; lambda_4 = {strength[1]:.3g}; lambda_0 = {strength[2]:.3g}",
        fontsize=12,
    )
    panels = [
        (observed, native, "Observed image", "magma", None),
        (linear, native, "Fitted linear prediction", "magma", None),
        (residual, native, "Linear residual / noise", "RdBu_r", 5.0),
        (truth_source, source_extent, "True source", "magma", None),
        (source, source_extent, "Recovered source", "magma", None),
        (
            source - truth_source,
            source_extent,
            "Source error",
            "RdBu_r",
            float(np.nanmax(abs(source - truth_source))),
        ),
        (
            ktruth,
            extent,
            "Discrete true subhalo kappa",
            "RdBu_r",
            float(np.nanmax(abs(ktruth))),
        ),
        (
            kappa,
            extent,
            "Recovered subhalo kappa",
            "RdBu_r",
            float(np.nanmax(abs(ktruth))),
        ),
        (
            difference,
            extent,
            "Kappa error",
            "RdBu_r",
            float(np.nanmax(abs(difference))),
        ),
    ]
    for index, (ax, panel) in enumerate(zip(axes.flat, panels)):
        data, box, title, cmap, limit = panel
        kwargs = dict(cmap=cmap)
        if limit is not None:
            kwargs.update(vmin=-limit, vmax=limit)
        elif index in (0, 1):
            kwargs.update(vmin=np.nanmin(observed), vmax=np.nanmax(observed))
        elif index in (3, 4):
            kwargs.update(
                vmin=min(truth_source.min(), source.min()),
                vmax=max(truth_source.max(), source.max()),
            )
        im = ax.imshow(data, origin="lower", extent=box, interpolation="none", **kwargs)
        ax.set(title=title, xlabel="x (arcsec)", ylabel="y (arcsec)")
        ax.set_title(title, fontsize=10, pad=9)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
        if index >= 6:
            ax.plot(
                cfg["subhalo_x"],
                cfg["subhalo_y"],
                marker="+",
                color="black",
                markersize=9,
                mew=1.4,
            )
    save(
        fig,
        out / "posterior_median_recovery",
        axes=list(axes.flat),
        auditor=auditor,
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), layout="constrained")
    fig.suptitle(
        "Subhalo at posterior-median hyperparameters; black +: injected position",
        fontsize=12,
    )
    limit = float(np.nanmax(abs(ktruth)))
    for ax, data, title, bound in zip(
        axes,
        (ktruth, kappa, difference),
        ("Discrete true kappa", "Recovered kappa", "Recovered - truth"),
        (limit, limit, float(np.nanmax(abs(difference)))),
    ):
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            interpolation="none",
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
        )
        ax.set(title=title, xlabel="x (arcsec)", ylabel="y (arcsec)")
        ax.plot(
            cfg["subhalo_x"],
            cfg["subhalo_y"],
            marker="+",
            color="black",
            markersize=9,
            mew=1.4,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
    save(fig, out / "subhalo_median_recovery", axes=list(axes), auditor=auditor)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
