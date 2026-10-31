"""Compare saved fits on their original matched mock; never rerun inference."""

import argparse
import hashlib
import json
from pathlib import Path
import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / ".scratch/potential-correction/output"
RUNS = {
    "single_step": OUTPUT / "single-step-scientific/reference/subhalo/dense",
    "iterative": OUTPUT / "iterative-scientific/validated-reference/subhalo/dense",
}


def read_run(directory):
    metadata = json.loads((directory / "result.json").read_text())
    with np.load(directory / "reconstruction.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    assert metadata["success"] and metadata["backend"] == "dense"
    return metadata, arrays


def raster(arrays, values, spacing):
    indices = arrays["potential_node_indices"]
    i, j = (indices - indices.min(axis=0)).T
    image = np.full((j.max() + 1, i.max() + 1), np.nan)
    image[j, i] = values
    extent = [
        arrays["potential_x"].min() - spacing / 2,
        arrays["potential_x"].max() + spacing / 2,
        arrays["potential_y"].min() - spacing / 2,
        arrays["potential_y"].max() + spacing / 2,
    ]
    return image, extent


def finish(fig, axes, directory, name):
    for index, ax in enumerate(axes.flat):
        ax.set_gid(string.ascii_lowercase[index])
        ax.text(
            -0.16, 1.07, string.ascii_lowercase[index],
            transform=ax.transAxes, weight="bold", fontsize=9,
        )
        ax.tick_params(length=2.5, pad=2)
        if index % axes.shape[1] == 0:
            ax.set_ylabel("y (arcsec)", labelpad=2)
        if index // axes.shape[1] == axes.shape[0] - 1:
            ax.set_xlabel("x (arcsec)", labelpad=2)
    require_matplotlib_panel_alignment(
        fig,
        axes=list(axes.flat),
        json_out=directory / f"{name}.alignment.json",
        overlay_svg=directory / f"{name}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(directory / f"{name}.png", dpi=300)
    fig.savefig(directory / f"{name}.pdf", dpi=300)
    fig.savefig(directory / f"{name}.svg", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "visual-comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = {key: read_run(path) for key, path in RUNS.items()}
    sm, s = records["single_step"]
    im, it = records["iterative"]
    assert sm["mock_configuration"] == im["mock_configuration"]
    assert sm["potential_geometry"] == im["potential_geometry"]
    assert sm["strengths"] == im["strengths"]
    matched_keys = [
        "image_data", "noise_map", "noise_realization", "truth_model_image",
        "mask", "potential_node_indices", "potential_x", "potential_y",
        "derivative_valid", "truth_potential_values", "truth_discrete_convergence",
    ]
    for key in matched_keys:
        np.testing.assert_equal(s[key], it[key], err_msg=f"Unmatched {key}")
    config = sm["mock_configuration"]
    subhalo_xy = (config["subhalo_x"], config["subhalo_y"])
    valid = s["derivative_valid"]
    assert valid.sum() == 724 and s["potential_values"].size == 956
    assert (~s["mask"]).sum() == s["data_active"].size == 22016

    # Establish the evidence and image-integrity contract before any drawing.
    contract = {
        "backend": "Python / matplotlib; existing experiment workflow",
        "archetype": "quantitative image grid",
        "question": "What data were fitted, and how do the two recovered fields compare with the injected subhalo?",
        "claim": "The methods localize the perturbation with a suppressed convergence peak; raw potential low modes differ from truth.",
        "data_panels": "Noiseless and noisy full mock, followed by both final nonlinear images in the fitted annulus.",
        "field_panels": "Columns: injected truth, single-step, iterative. Rows: raw potential, convergence, convergence local zoom.",
        "counts": {"full_image_pixels": 40000, "fitted_pixels": 22016, "potential_nodes": 956, "derivative_valid_nodes": 724},
        "mask": "Full raw mock retained. Model images restricted to the original fitted annulus. Derivatives undefined outside their valid core remain NaN/white.",
        "processing": "No refitting, smoothing, clipping, affine alignment, recentering, or per-panel normalization. Nearest-neighbour pixel display only.",
        "truth": "Raw analytic nodal potential with psi(subhalo center)=0; convergence from centered differences on the identical potential grid.",
        "local_view": {"center_arcsec": subhalo_xy, "half_width_arcsec": 0.35, "role": "Explicit crop to judge localization and peak suppression"},
        "statistics": "One shared mock/noise seed (17); deterministic fit comparison. No replicate uncertainty or population inference.",
        "scales": "All four image intensities share a scale. All potential panels share a scale. All full and zoom convergence panels share a scale including signed values.",
        "exports": "PNG 300 dpi for viewing; editable-text PDF and SVG; 183 mm width; minimum 5 pt rendered glyph; alignment/text/collision QA.",
        "markers": "Purple crosses mark the physical injection coordinate (0.75, 0.10) arcsec.",
        "sources": {
            key: {
                name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                for name in ("result.json", "reconstruction.npz")
            }
            for key, path in RUNS.items()
        },
        "matched_arrays_checked": matched_keys,
    }
    (args.output / "figure-contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.linewidth": 0.6, "figure.facecolor": "white",
    })
    sequential = plt.get_cmap("viridis").copy()
    signed = plt.get_cmap("RdBu_r").copy()
    sequential.set_bad("white")
    signed.set_bad("white")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.3))
    fig.subplots_adjust(left=0.10, right=0.84, bottom=0.10, top=0.88, hspace=0.32, wspace=0.28)
    fig.suptitle("Simulated data and nonlinear image reconstructions", y=0.97, fontsize=12)
    fig.text(0.10, 0.925, "200 × 200 pixels · 0.012 arcsec/pixel · subhalo strength b = 0.04 arcsec", fontsize=8)
    images = [s["truth_model_image"], s["image_data"],
              np.where(s["mask"], np.nan, s["nonlinear_model_image"]),
              np.where(s["mask"], np.nan, it["nonlinear_model_image"])]
    norm = Normalize(min(np.nanmin(x) for x in images), max(np.nanmax(x) for x in images))
    titles = ["Noiseless simulation", "Observed simulation (noise σ = 0.01)",
              "Single-step reconstruction", "Iterative reconstruction"]
    half_field = config["npix"] * config["dpix"] / 2
    image_extent = [-half_field, half_field, -half_field, half_field]
    for ax, values, title in zip(axes.flat, images, titles):
        artist = ax.imshow(values, origin="lower", extent=image_extent, interpolation="nearest", cmap=sequential, norm=norm)
        ax.set_title(title, pad=9)
        ax.plot(*subhalo_xy, marker="+", color="#CC79A7", markersize=7, markeredgewidth=1.2)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
    cax = fig.add_axes([0.89, 0.26, 0.018, 0.45])
    fig.colorbar(artist, cax=cax, label="Image intensity (shared scale)")
    fig.text(0.10, 0.035, "White in fitted images: outside the fitted annulus. Cross: injected subhalo position.", fontsize=7)
    finish(fig, axes, args.output, "simulated-data-and-fits")

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 8.0))
    fig.subplots_adjust(left=0.10, right=0.85, bottom=0.09, top=0.87, hspace=0.33, wspace=0.28)
    fig.suptitle("Injected subhalo and recovered perturbation", y=0.97, fontsize=12)
    fig.text(0.10, 0.925, "Same data, grid and regularization · raw signed fields · no affine alignment", fontsize=8)
    fields = [
        [s["truth_potential_values"], s["potential_values"], it["potential_values"]],
        [s["truth_discrete_convergence"], s["convergence"], it["convergence"]],
    ]
    limits = [max(np.nanmax(np.abs(x)) for x in row) for row in fields]
    spacing = sm["potential_geometry"]["spacing"]
    for row in range(3):
        field_index = min(row, 1)
        norm = TwoSlopeNorm(vmin=-limits[field_index], vcenter=0, vmax=limits[field_index])
        for column in range(3):
            ax = axes[row, column]
            values = fields[field_index][column]
            if row:
                values = np.where(valid, values, np.nan)
            values, extent = raster(s, values, spacing)
            artist = ax.imshow(values, origin="lower", extent=extent, interpolation="nearest", cmap=signed, norm=norm)
            ax.plot(*subhalo_xy, marker="+", color="#CC79A7", markersize=6, markeredgewidth=1.1)
            if row == 0:
                ax.set_title(["Injected truth", "Single-step", "Iterative"][column], pad=10)
            if row < 2:
                ax.set_xticks([-1, 0, 1])
                ax.set_yticks([-1, 0, 1])
            else:
                ax.set_xlim(subhalo_xy[0] - 0.35, subhalo_xy[0] + 0.35)
                ax.set_ylim(subhalo_xy[1] - 0.35, subhalo_xy[1] + 0.35)
                ax.set_xticks([0.5, 0.75, 1.0])
                ax.set_yticks([-0.2, 0, 0.2, 0.4])
        fig.canvas.draw()
        position = axes[row, -1].get_position()
        cax = fig.add_axes([0.89, position.y0, 0.015, position.height])
        labels = ["Raw δψ (arcsec²)", "δκ (dimensionless)", "δκ, local zoom"]
        fig.colorbar(artist, cax=cax, label=labels[row])
    fig.text(0.10, 0.035, "White: undefined / unsupported nodes. κ truth uses the same finite-difference grid.", fontsize=7)
    finish(fig, axes, args.output, "subhalo-field-comparison")

    summary = {
        "truth_discrete_kappa_peak": float(np.nanmax(fields[1][0])),
        "single_step_kappa_peak": float(np.nanmax(fields[1][1])),
        "iterative_kappa_peak": float(np.nanmax(fields[1][2])),
        "single_step_residual_rms_noise": float(np.sqrt(np.mean(s["nonlinear_normalized_residual"][~s["mask"]] ** 2))),
        "iterative_residual_rms_noise": float(np.sqrt(np.mean(it["nonlinear_normalized_residual"][~s["mask"]] ** 2))),
    }
    (args.output / "comparison-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
