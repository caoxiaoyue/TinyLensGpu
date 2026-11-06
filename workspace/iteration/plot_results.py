"""Plot saved, numerically accepted J0946 fields without running any fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = args.run / "maximum_evidence"
    result = json.loads((output / "result.json").read_text())
    if not result["accepted"]:
        raise ValueError("Only a numerically accepted reconstruction may be displayed as the result")
    with np.load(output / "reconstruction.npz") as archive:
        data = {name: archive[name] for name in archive.files}
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.7})
    dpix = result["configuration"].get("dpix", 0.05)
    image_n = data["mask"].shape[0]
    edge = image_n * dpix / 2
    extent = [-edge, edge, -edge, edge]
    mask = data["mask"]
    active_y, active_x = np.where(~mask)
    centers_x = (active_x - (image_n - 1)/2) * dpix
    centers_y = (active_y - (image_n - 1)/2) * dpix
    half = float(max(np.max(np.abs(centers_x)), np.max(np.abs(centers_y)),
                     np.max(np.abs(data["potential_x"])),
                     np.max(np.abs(data["potential_y"]))) + dpix)
    source_model = data["nonlinear_source_model_image"]
    flux_limits = np.percentile(np.r_[data["lens_subtracted_image"][~mask], source_model[~mask]], [0.5, 99.5])
    macro_residual = np.where(mask, np.nan,
        (data["lens_subtracted_image"].astype(float) - data["macro_source_model_image"]) / data["noise_map"])
    nodes = data["potential_node_indices"]
    ix, iy = nodes[:, 0] - nodes[:, 0].min(), nodes[:, 1] - nodes[:, 1].min()
    kappa_image = np.full((int(iy.max())+1, int(ix.max())+1), np.nan)
    kappa_image[iy, ix] = np.where(data["derivative_valid"], data["convergence"], np.nan)
    spacing = result["configuration"]["potential_spacing"]
    kextent = [data["potential_x"].min()-spacing/2, data["potential_x"].max()+spacing/2,
               data["potential_y"].min()-spacing/2, data["potential_y"].max()+spacing/2]
    kmax = float(np.nanmax(np.abs(kappa_image)))
    source_n = result["configuration"]["source_n"]
    source = data["source_values"].reshape(source_n, source_n)
    sx, sy = data["source_x_axis"], data["source_y_axis"]
    dx, dy = sx[1]-sx[0], sy[1]-sy[0]
    sextent = [sx[0]-dx/2, sx[-1]+dx/2, sy[0]-dy/2, sy[-1]+dy/2]
    fig = plt.figure(figsize=(10.4, 9.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, .055]*2, wspace=.42, hspace=.55,
                         left=.075, right=.96, bottom=.11, top=.90)
    panels = [
        ("Lens-light-subtracted data", np.where(mask, np.nan, data["lens_subtracted_image"]),
         extent, "magma", *flux_limits, "Image brightness"),
        ("Iterative model", np.where(mask, np.nan, source_model),
         extent, "magma", *flux_limits, "Image brightness"),
        ("Iterative residual", data["normalized_residual"], extent, "RdBu_r", -5, 5, "Residual / noise"),
        ("Saved M3 residual", macro_residual, extent, "RdBu_r", -5, 5, "Residual / noise"),
        ("Potential correction: delta kappa", kappa_image, kextent, "RdBu_r", -kmax, kmax, "Delta kappa"),
        ("Recovered source", source, sextent, "magma", float(source.min()), float(source.max()), "Source brightness"),
    ]
    axes = []
    colorbar_axes = []
    for index, (title, values, bounds, cmap, lower, upper, label) in enumerate(panels):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(gs[2*row, col])
        ax.set_gid("panel_" + chr(97+index))
        cax = fig.add_subplot(gs[2*row+1, col])
        cax.set_gid("colorbar_" + chr(97+index))
        ax.set_facecolor("#eeeeee")
        mesh = ax.imshow(values, extent=bounds, origin="lower", interpolation="none",
                         cmap=cmap, vmin=lower, vmax=upper, rasterized=True)
        if index < 5:
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
        else:
            cx, cy = (sx[0]+sx[-1])/2, (sy[0]+sy[-1])/2
            shalf = (sx[-1]-sx[0]+dx)/2
            ax.set_xlim(cx-shalf, cx+shalf)
            ax.set_ylim(cy-shalf, cy+shalf)
        ax.set_xlabel("x (arcsec)")
        ax.set_ylabel("y (arcsec)")
        ax.set_title(title, pad=9)
        ax.text(-.17, 1.05, chr(97+index), transform=ax.transAxes,
                fontsize=11, fontweight="bold", ha="left", va="bottom")
        bar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        bar.set_label(label, fontsize=8)
        bar.ax.tick_params(labelsize=8)
        axes.append(ax)
        colorbar_axes.append(cax)
    fig.suptitle("J0946 / Jackpot: iterative reconstruction candidate", fontsize=12, y=.98)
    status = result.get("reproducibility_status", "not_checked")
    if status == "not_checked" and (args.run / "evaluations.jsonl").exists():
        records = [json.loads(line) for line in (args.run / "evaluations.jsonl").read_text().splitlines()]
        repeats = [row for row in records if row["phase"] == "selected_audit"]
        if repeats:
            status = "passed" if repeats[-1]["accepted"] else "repeat_gate_failed"
    repeat_label = {"passed": "PASSED", "repeat_gate_failed": "NOT PASSED",
                    "not_checked": "not checked"}[status]
    fig.text(.075, .042, "Fixed M3 macro model; Float64; full local-GN evidence. Real data: no injected truth.",
             fontsize=9)
    fig.text(.075, .02, "Selected trial; cold-start repeat gate: " + repeat_label + ".", fontsize=9)
    scripts = Path.home() / ".agents/skills/nature-figure/scripts"
    sys.path.insert(0, str(scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    for ax, cax in zip(axes, colorbar_axes):
        position, color_position = ax.get_position(), cax.get_position()
        cax.set_position([position.x0, color_position.y0, position.width, color_position.height])
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,
        json_out=output / "diagnostics.alignment.json",
        overlay_svg=output / "diagnostics.alignment.svg", axes=axes,
        panel_ids=list("abcdef"), strict=True)
    fig.savefig(output / "diagnostics.png", dpi=300)
    fig.savefig(output / "diagnostics.pdf")
    fig.savefig(output / "diagnostics.svg")
    plt.close(fig)
    print(output / "diagnostics.png")


if __name__ == "__main__":
    main()
