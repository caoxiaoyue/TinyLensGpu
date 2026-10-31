"""Plot raw development-pair recovery and null controls from saved products."""

import argparse
import json
from pathlib import Path
import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / ".scratch/potential-correction/output"


def load(directory):
    with np.load(directory / "reconstruction.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads((directory / "result.json").read_text())
    assert metadata["success"]
    return arrays, metadata


def image(arrays, values):
    nodes = arrays["potential_node_indices"]
    i, j = (nodes - nodes.min(axis=0)).T
    result = np.full((j.max() + 1, i.max() + 1), np.nan)
    result[j, i] = values
    half = 0.075 / 2
    extent = [arrays["potential_x"].min() - half, arrays["potential_x"].max() + half,
              arrays["potential_y"].min() - half, arrays["potential_y"].max() + half]
    return result, extent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "peak-recovery/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bases = [OUTPUT / "iterative-scientific/validated-reference",
             OUTPUT / "peak-recovery/iterative-d4-probes/ls-0.0006_lp-400000",
             OUTPUT / "peak-recovery/selected-iterative-final"]
    rows = {case: [load(base / case / "dense") for base in bases]
            for case in ("subhalo", "control")}
    for case, products in rows.items():
        a, metadata = products[0]
        assert metadata["result_configuration"]["potential_spacing"] == 0.075
        for b, mb in products[1:]:
            np.testing.assert_equal(a["potential_node_indices"], b["potential_node_indices"])
            np.testing.assert_equal(a["image_data"], b["image_data"])
            np.testing.assert_equal(a["truth_discrete_convergence"], b["truth_discrete_convergence"])
            assert metadata["mock_configuration"] == mb["mock_configuration"]
    contract = {
        "question": "Can peak recovery improve without adding null-control structure?",
        "claim": "The development-pair finite evidence selection sharpens the true peak and retains the original small control; universally weak D4 introduces null structure.",
        "backend": "Python / matplotlib", "archetype": "quantitative image grid",
        "columns": ["Discrete injected truth", "Original iterative demo", "Exploratory universally weak D4, rejected", "Strict converged evidence-selected fits"],
        "rows": ["Subhalo local 0.7 arcsec window", "No-subhalo full core"],
        "integrity": "Original signed arrays, no smoothing, thresholding, affine alignment or independent panel scaling. Nearest pixel display. White is zero; grey is undefined derivative support. Identical geometry/data verified across each row.",
        "counts": {"valid_core_nodes": 724, "potential_nodes": 956, "native_image_pixels": 40000, "fitted_image_pixels": 22016},
        "uncertainty": "One development noise realization (seed 17). Raw images, no replicate aggregate or uncertainty claim. Separate seeds validate the fixed selection policy.",
        "exports": "183 mm width; PNG 300 dpi; editable PDF/SVG text; minimum 5 pt glyph; alignment/text/collision QA",
        "caution": "The exploratory weak-prior column uses the original gradient tolerance and fails stricter stopping; it illustrates a rejected failure mode, not an adopted reconstruction.",
    }
    (args.output / "figure-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.6,
    })
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.92")
    limit = np.nanmax(np.abs(rows["subhalo"][0][0]["truth_discrete_convergence"]))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.55))
    fig.subplots_adjust(left=0.09, right=0.86, bottom=0.20, top=0.80, hspace=0.50, wspace=0.28)
    fig.suptitle("Sharper subhalo peak with a quiet selected control", y=0.97, fontsize=11)
    fig.text(0.09, 0.90, "Same analytic mock, geometry, noise and colour scale · raw signed κ", fontsize=7.5)
    headers = ["Injected truth", "Original demo", "Uniform weak D4*", "Evidence-selected"]
    for row, case in enumerate(("subhalo", "control")):
        products = rows[case]
        reference = products[0][0]
        fields = [reference["truth_discrete_convergence"]] + [a["convergence"] for a, _ in products]
        for column, field in enumerate(fields):
            ax = axes[row, column]
            raster, extent = image(reference, field)
            artist = ax.imshow(raster, origin="lower", extent=extent, interpolation="nearest", cmap=cmap, norm=norm)
            ax.plot(0.75, 0.10, marker="+", color="#CC79A7", markersize=5, markeredgewidth=0.9)
            if row == 0:
                ax.set_xlim(0.40, 1.10)
                ax.set_ylim(-0.25, 0.45)
                ax.set_xticks([0.5, 0.75, 1.0])
                ax.set_yticks([-0.2, 0, 0.2, 0.4])
                ax.set_title(f"{headers[column]}\npeak {np.nanmax(field):.3f}", pad=8)
            else:
                ax.set_xticks([-1, 0, 1])
                ax.set_yticks([-1, 0, 1])
                ax.set_title(f"max |κ| = {np.nanmax(np.abs(field)):.3g}", pad=8)
                ax.set_xlabel("x (arcsec)", labelpad=2)
            if column == 0:
                ax.set_ylabel(["Subhalo: y (arcsec)", "Control: y (arcsec)"][row], labelpad=2)
            ax.tick_params(length=2, pad=2)
            panel = string.ascii_lowercase[row * 4 + column]
            ax.set_gid(panel)
            ax.text(-0.18, 1.15, panel, transform=ax.transAxes, weight="bold", fontsize=8)
    cax = fig.add_axes([0.90, 0.29, 0.014, 0.42])
    fig.colorbar(artist, cax=cax, label="δκ (dimensionless)")
    fig.text(0.09, 0.105, "Top: local view. Bottom: full core. Cross: injection / nominal reference position.", fontsize=6.5)
    fig.text(0.09, 0.065, "*Exploratory λ4 = 400,000: original tolerance; fails stricter stopping and adds null structure.", fontsize=6.5)
    fig.text(0.09, 0.025, "Selected: same finite evidence rule and strict numerical controls for both observations.", fontsize=6.5)
    require_matplotlib_panel_alignment(fig, axes=list(axes.flat), strict=True,
        require_panel_labels=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        json_out=args.output / "peak-recovery.alignment.json",
        overlay_svg=args.output / "peak-recovery.alignment.svg")
    fig.savefig(args.output / "peak-recovery.png", dpi=300)
    fig.savefig(args.output / "peak-recovery.pdf", dpi=300)
    fig.savefig(args.output / "peak-recovery.svg", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
