"""Diagnostic images from saved full-mock endpoints; never refits a model."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def potential_image(arrays, values):
    nodes = arrays["potential_node_indices"]
    minimum, maximum = nodes.min(axis=0), nodes.max(axis=0)
    shape = tuple((maximum - minimum + 1)[::-1])
    image = np.full(shape, np.nan)
    image[nodes[:, 1] - minimum[1], nodes[:, 0] - minimum[0]] = values
    xs, ys = arrays["potential_x"], arrays["potential_y"]
    h = np.min(np.diff(np.unique(xs)))
    extent = (xs.min() - h / 2, xs.max() + h / 2, ys.min() - h / 2, ys.max() + h / 2)
    return image, extent


def show(ax, values, title, extent, *, limits, cmap):
    artist = ax.imshow(
        values,
        origin="lower",
        extent=extent,
        interpolation="none",
        vmin=limits[0],
        vmax=limits[1],
        cmap=cmap,
    )
    ax.set(title=title, xlabel="x (arcsec)", ylabel="y (arcsec)")
    return artist


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_case(directory, output):
    config = json.loads((directory / "configuration.json").read_text())
    report = json.loads((directory / "diagnostics.json").read_text())
    method = report.get("options", {}).get("method")
    if method == "lmfit.Minimizer.leastsq":
        method_label = "JAX AD + lmfit LM"
    elif method == "scipy.optimize.least_squares":
        method_label = "JAX AD + SciPy GN (TRF/LSMR)"
    else:
        method_label = "JAX AD + L-BFGS"
    with np.load(directory / "diagnostic_fields.npz", allow_pickle=False) as saved:
        arrays = {name: np.array(saved[name]) for name in saved.files}
    initialization = config["initialization"]
    case = config["case"]
    status = (
        "PASSED original gradient gate"
        if report["accepted_reconstruction"]
        else "UNCONVERGED endpoint diagnostics"
    )
    repeat_label = (
        f" | {directory.name}" if directory.name.startswith("repeat_") else ""
    )
    title = (
        f"{case.capitalize()} | source init: {initialization}{repeat_label} | {status}"
    )
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{initialization}_{case}"
    extent = tuple(arrays["source_bbox"])
    plt.rcParams.update(
        {"font.size": 9, "pdf.fonttype": 42, "axes.titleweight": "normal"}
    )

    # Comparable kappa panels share a color scale; only derivative-valid core
    # values are shown. The truth convention is the existing centered stencil.
    kappas = [
        arrays[key]
        for key in ("truth_discrete_convergence", "lm_convergence", "convergence")
    ]
    limit = max(float(np.nanmax(np.abs(k))) for k in kappas)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    fig.suptitle(title)
    for ax, kappa, label in zip(
        axes,
        kappas,
        (
            (
                "Injected discrete convergence"
                if case == "subhalo"
                else "True convergence: zero"
            ),
            "Original LM",
            method_label,
        ),
    ):
        image, pextent = potential_image(arrays, kappa)
        artist = show(ax, image, label, pextent, limits=(-limit, limit), cmap="RdBu_r")
        ax.plot(
            config["original_configuration"]["mock"]["subhalo_x"],
            config["original_configuration"]["mock"]["subhalo_y"],
            "+",
            color="black",
            ms=8,
            mew=1,
        )
    fig.colorbar(
        artist,
        ax=axes,
        label="Convergence correction (dimensionless)",
        shrink=0.85,
        pad=0.02,
    )
    fig.supxlabel(
        "Cross: injection position"
        if case == "subhalo"
        else "Cross: nominal reference position; no injection"
    )
    save(fig, output / (prefix + "_perturbation"))

    sources = [
        arrays[key]
        for key in (
            "truth_source_values",
            "initial_source",
            "lm_source_values",
            "source_values",
        )
    ]
    limits = (
        min(float(s.min()) for s in sources),
        max(float(s.max()) for s in sources),
    )
    side = len(arrays["source_x_axis"])
    spacing = (extent[1] - extent[0]) / (side - 1)
    source_extent = (
        extent[0] - spacing / 2,
        extent[1] + spacing / 2,
        extent[2] - spacing / 2,
        extent[3] + spacing / 2,
    )
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), layout="constrained")
    fig.suptitle(title + " | signed source, no clipping")
    for ax, source, label in zip(
        axes,
        sources,
        ("True source at nodes", "Initial source", "Original LM", method_label),
    ):
        artist = show(
            ax,
            source.reshape(side, side),
            label,
            source_extent,
            limits=limits,
            cmap="magma",
        )
    fig.colorbar(artist, ax=axes, label="Source intensity", shrink=0.85, pad=0.02)
    save(fig, output / (prefix + "_source"))

    mask = arrays["mask"]
    images = [
        arrays[key]
        for key in ("image_data", "lm_nonlinear_model_image", "nonlinear_model_image")
    ]
    limits = (
        min(float(im[~mask].min()) for im in images),
        max(float(im[~mask].max()) for im in images),
    )
    mock = config["original_configuration"]["mock"]
    half = mock["npix"] * mock["dpix"] / 2
    image_extent = (-half, half, -half, half)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), layout="constrained")
    fig.suptitle(title)
    for ax, image, label in zip(
        axes[0], images, ("Observed image", "Original LM fit", method_label + " fit")
    ):
        artist = show(
            ax,
            np.where(mask, np.nan, image),
            label,
            image_extent,
            limits=limits,
            cmap="magma",
        )
    fig.colorbar(
        artist, ax=axes[0].tolist(), label="Image intensity", shrink=0.85, pad=0.02
    )
    difference = np.where(mask, np.nan, (images[2] - images[1]) / arrays["noise_map"])
    diff_limit = max(float(np.nanmax(np.abs(difference))), 1e-12)
    artist = show(
        axes[1, 0],
        difference,
        "(New - LM) / noise",
        image_extent,
        limits=(-diff_limit, diff_limit),
        cmap="RdBu_r",
    )
    fig.colorbar(
        artist, ax=axes[1, 0], label="Difference / noise", shrink=0.85, pad=0.02
    )
    for ax, key, label in zip(
        axes[1, 1:],
        ("lm_nonlinear_normalized_residual", "nonlinear_normalized_residual"),
        ("LM residual / noise", "New residual / noise"),
    ):
        artist = show(
            ax, arrays[key], label, image_extent, limits=(-4, 4), cmap="RdBu_r"
        )
    fig.colorbar(
        artist, ax=axes[1, 1:].tolist(), label="Residual / noise", shrink=0.85, pad=0.02
    )
    save(fig, output / (prefix + "_images"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Repeat to plot when a run contains repeats",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "scientific_figures",
    )
    args = parser.parse_args()
    for run in args.runs:
        for metadata in sorted(run.glob("*/*/diagnostics.json")):
            if (
                metadata.parent.name.startswith("repeat_")
                and metadata.parent.name != f"repeat_{args.repeat}"
            ):
                continue
            plot_case(metadata.parent, args.output)
            print(metadata.parent, flush=True)


if __name__ == "__main__":
    main()
