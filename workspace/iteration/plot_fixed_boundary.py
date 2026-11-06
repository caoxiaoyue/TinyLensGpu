"""Fixed-lambda J0946: candidate field and all technical-repeat diagnostics."""

from __future__ import annotations

import argparse
import csv
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
    summary = json.loads((args.run / "summary.json").read_text())
    rows = summary["runs"]
    candidate = next(r for r in rows if r["boundary"] == "zero_ring"
                     and r["initialization_case"] == "single_step" and r["repeat"] == 0)
    out = args.run / "figure"
    out.mkdir(exist_ok=True)
    cases = [(b, i) for i in ("source_only", "single_step") for b in ("hard", "zero_ring")]
    fields = ["boundary", "initialization_case", "repeat", "objective", "iterations",
              "gn_stationarity_proxy", "diagnostic_log_evidence", "inner_seconds",
              "delta_kappa_peak", "accepted"]
    with (out / "source_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with np.load(args.run / "zero_ring/single_step/repeat_00/reconstruction.npz") as archive:
        x, y, ij, k, valid = [archive[name] for name in
                       ("potential_x", "potential_y", "potential_node_indices", "convergence", "derivative_valid")]
    ix, iy = (ij-ij.min(axis=0)).T
    image = np.full((iy.max()+1, ix.max()+1), np.nan)
    image[iy, ix] = np.where(valid, k, np.nan)
    dx, dy = np.diff(np.unique(x)).min(), np.diff(np.unique(y)).min()
    extent = [x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2]
    kmax = float(np.nanmax(abs(k)))
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.right": False, "axes.spines.top": False})
    fig = plt.figure(figsize=(11, 4.7))
    grid = fig.add_gridspec(2, 3, left=.065, right=.98, bottom=.23, top=.83,
                            width_ratios=[1, 1, 1], height_ratios=[1, .045],
                            wspace=.45, hspace=.4)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for index, ax in enumerate(axes):
        ax.set_box_aspect(1)
        ax.set_gid("panel_"+chr(97+index))
        ax.text(-.16, 1.10, chr(97+index), transform=ax.transAxes,
                fontsize=11, fontweight="bold", ha="left", va="bottom")
    ax = axes[0]
    ax.set_facecolor("#eeeeee")
    mesh = ax.imshow(image, extent=extent, origin="lower", interpolation="none",
                      vmin=-kmax, vmax=kmax, cmap="RdBu_r", rasterized=True,
                      aspect="auto")
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    half = max(extent[1]-extent[0], extent[3]-extent[2])/2
    cx, cy = (extent[0]+extent[1])/2, (extent[2]+extent[3])/2
    ax.set_xlim(cx-half, cx+half)
    ax.set_ylim(cy-half, cy+half)
    ax.set_title("Candidate δκ: Zero / Warm", pad=12)
    cax = fig.add_subplot(grid[1, 0])
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    gate_label = "passed" if candidate["accepted"] else "failed"
    bar.set_label(f"δκ (repeat 0; convergence gate {gate_label})", fontsize=8)
    labels = ["Hard\nSource", "Zero\nSource", "Hard\nWarm", "Zero\nWarm"]
    for index, (boundary, initial) in enumerate(cases):
        group = sorted([r for r in rows if r["boundary"] == boundary
                        and r["initialization_case"] == initial], key=lambda r: r["repeat"])
        positions = index+np.linspace(-.12, .12, len(group))
        color = "#52758c" if boundary == "hard" else "#b56a39"
        for ax, key, offset in [(axes[1], "gn_stationarity_proxy", 0),
                                 (axes[2], "diagnostic_log_evidence", 6305)]:
            values = [r[key]-offset for r in group]
            if key == "gn_stationarity_proxy" and not np.all(np.asarray(values) > 0):
                raise ValueError("The logarithmic diagnostic axis requires positive values")
            ax.plot(positions, values, marker="o", markersize=4, linewidth=.8,
                    color=color)
    for ax in axes[1:]:
        ax.set_xticks(range(4), labels)
        ax.set_xlim(-.5, 3.5)
        ax.tick_params(axis="x", length=0, pad=7)
    axes[1].set_yscale("log")
    axes[1].set_ylim(5e-4, 3e-2)
    axes[1].axhline(.001, color="#444444", linestyle="--", linewidth=.8)
    axes[1].set_ylabel("0.5 gᵀ H(GN)⁻¹ g")
    axes[1].set_title("Stationarity: required ≤ 0.001", pad=12)
    axes[2].set_ylabel("Diagnostic log-evidence − 6305")
    axes[2].set_ylim(.04, .23)
    axes[2].set_title("Evidence: three technical repeats", pad=12)
    passed = sum(r["accepted"] for r in rows)
    fig.suptitle(f"J0946: fixed single-step λ; {passed} / {len(rows)} endpoints pass the unchanged gate",
                 fontsize=12, y=.97)
    fig.text(.065, .070, "Source = source-only MAP at zero correction; Warm = saved single-step source + potential.", fontsize=9)
    fig.text(.065, .025, "Float64; same data and fixed metric; dots are individual repeats. Diagnostic evidence only; no injected truth.", fontsize=9)
    scripts = Path.home() / ".agents/skills/nature-figure/scripts"
    sys.path.insert(0, str(scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    pos, cpos = axes[0].get_position(), cax.get_position()
    cax.set_position([pos.x0, cpos.y0, pos.width, cpos.height])
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=list("abc"),
        strict=True, json_out=out / "diagnostics.alignment.json",
        overlay_svg=out / "diagnostics.alignment.svg")
    fig.savefig(out / "diagnostics.png", dpi=300)
    fig.savefig(out / "diagnostics.pdf")
    fig.savefig(out / "diagnostics.svg")
    plt.close(fig)
    print(out / "diagnostics.png")


if __name__ == "__main__":
    main()
