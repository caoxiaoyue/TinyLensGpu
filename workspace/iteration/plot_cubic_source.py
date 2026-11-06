"""Fixed-lambda J0946 fields: predeclared repeat 0, all fits in source CSV."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def grid_image(archive, values):
    indices = archive["potential_node_indices"]
    ix, iy = (indices-indices.min(axis=0)).T
    image = np.full((iy.max()+1, ix.max()+1), np.nan)
    image[iy, ix] = np.where(archive["derivative_valid"], values, np.nan)
    x, y = archive["potential_x"], archive["potential_y"]
    dx, dy = np.diff(np.unique(x)).min(), np.diff(np.unique(y)).min()
    extent = (x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2)
    return image, extent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    analysis = json.loads((args.run/"analysis.json").read_text())
    summary = json.loads((args.run/"summary.json").read_text())
    rows = summary["rows"]
    out = args.run/"figure"; out.mkdir(exist_ok=True)
    columns = ("index", "profile", "method", "repeat", "accepted", "native_success", "iterations",
        "gn_stationarity_proxy", "objective", "chi2", "diagnostic_log_evidence", "inner_seconds",
        "coordinate_seconds", "endpoint_seconds", "total_seconds", "delta_kappa_peak")
    with (out/"source_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    profiles = ("previous_fixed_M3_selected_lens", "single_step_lens")
    names = ("Prior selected lens weights", "Single-step lens weights")
    fields = []; exports = {}; vmax = 0.; dmax = 0.
    for profile in profiles:
        pair = []
        for method in ("bilinear", "cubic"):
            row = next(r for r in rows if r["profile"] == profile and r["method"] == method and r["repeat"] == 0)
            with np.load(args.run/row["directory"]/"reconstruction.npz") as archive:
                archive = {k:archive[k] for k in archive.files}
            values = archive["convergence"]
            vmax = max(vmax, float(np.nanmax(abs(values))))
            pair.append((row, archive, values))
            exports[f"{profile}_{method}"] = values
        delta = pair[1][2]-pair[0][2]
        dmax = max(dmax, float(np.nanmax(abs(delta))))
        exports[f"{profile}_cubic_minus_bilinear"] = delta
        fields.append((pair, delta))
    np.savez_compressed(out/"field_source_data.npz", **exports)
    vmax, dmax = max(vmax, 1e-12), max(dmax, 1e-12)
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["DejaVu Sans"],
        "font.size":8, "axes.titlesize":9, "axes.labelsize":8, "xtick.labelsize":8, "ytick.labelsize":8,
        "pdf.fonttype":42, "svg.fonttype":"none", "axes.spines.right":False, "axes.spines.top":False})
    fig = plt.figure(figsize=(9., 9.))
    grid = fig.add_gridspec(3, 3, left=.09, right=.97, top=.88, bottom=.22,
        height_ratios=[1., 1., .055], wspace=.34, hspace=.65)
    axes = []; meshes = []
    for r, (pair, delta) in enumerate(fields):
        for c in range(3):
            ax = fig.add_subplot(grid[r, c]); axes.append(ax)
            label = chr(97+r*3+c)
            ax.set_gid("panel_"+label)
            ax.set_box_aspect(1.)
            ax.text(-.15, 1.12, label, transform=ax.transAxes, fontweight="bold", fontsize=10,
                ha="left", va="bottom")
            if c < 2:
                row, archive, values = pair[c]
                method = ("Bilinear", "C1 cubic")[c]
                group = analysis["groups"][f"{profiles[r]}/{row['method']}"]
                title = f"{method}: {group['accepted']}/{group['attempts']} pass\nPeak {row['delta_kappa_peak']:.5f}"
            else:
                row, archive, _ = pair[1]; values = delta
                title = "C1 cubic − Bilinear\nSame fixed regularization"
            image, extent = grid_image(archive, values)
            bound = vmax if c < 2 else dmax
            mesh = ax.imshow(image, origin="lower", extent=extent, cmap="RdBu_r", vmin=-bound,
                vmax=bound, interpolation="none", aspect="auto", rasterized=True)
            meshes.append(mesh)
            ax.set_facecolor("#eeeeee")
            half = max(extent[1]-extent[0], extent[3]-extent[2])/2
            cx, cy = (extent[1]+extent[0])/2, (extent[3]+extent[2])/2
            ax.set_xlim(cx-half, cx+half); ax.set_ylim(cy-half, cy+half)
            ax.set_title(title, pad=10)
            ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
            ax.set_xlabel("x (arcsec)")
            if c == 0:
                ax.set_ylabel(names[r]+"\ny (arcsec)")
    for c in range(3):
        cax = fig.add_subplot(grid[2, c])
        bar = fig.colorbar(meshes[c], cax=cax, orientation="horizontal")
        bar.set_label("δκ" if c < 2 else "Difference in δκ", fontsize=8)
        fig.canvas.draw()
        apos, cpos = axes[c].get_position(), cax.get_position()
        cax.set_position([apos.x0, cpos.y0, apos.width, cpos.height])
    fig.suptitle("J0946: fixed-prior source interpolation comparison", y=.975, fontsize=12)
    fig.text(.09, .934, "Identical starts; source intensities free; unchanged whitening and convergence gate", fontsize=9)
    fig.text(.09, .092, "Maps: predeclared repeat 0. Counts: all three numerical repeats per method and lens pair.", fontsize=9)
    fig.text(.09, .062, "M3 source hyperparameters fixed; Float64; full finite potential core; no injected real-data truth.", fontsize=9)
    fig.text(.09, .032, "These fixed-lambda fields do not establish a subhalo mass or an evidence-optimal regularization.", fontsize=9)
    sys.path.insert(0, str(Path.home()/".agents/skills/nature-figure/scripts"))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=list("abcdef"), strict=True,
        require_panel_labels=True, json_out=out/"kappa.alignment.json", overlay_svg=out/"kappa.alignment.svg")
    fig.savefig(out/"kappa.png", dpi=300)
    fig.savefig(out/"kappa.pdf")
    fig.savefig(out/"kappa.svg")
    plt.close(fig)
    print(out/"kappa.png")


if __name__ == "__main__":
    main()
