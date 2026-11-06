"""Joint source/lens C1 evidence fit against the previous fixed-source optimum."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_cubic_source import grid_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    analysis = json.loads((args.run/"analysis.json").read_text())
    summary = json.loads((args.run/"summary.json").read_text())
    best, baseline = analysis["best"], analysis["baseline_fixed_source_best"]
    def read(row):
        with np.load(args.run/row["directory"]/"reconstruction.npz") as a:
            return {k:a[k] for k in a.files}
    b, s = read(baseline), read(best)
    out = args.run/"figure"; out.mkdir(exist_ok=True)
    columns = ("index", "phase", "repeat", "accepted", "native_success", "iterations", "scipy_nfev",
        "gn_stationarity_proxy", "objective", "chi2", "diagnostic_log_evidence", "log_evidence",
        "coordinate_seconds", "inner_seconds", "endpoint_seconds", "total_seconds", "delta_kappa_peak",
        "lambda_s", "lambda4", "lambda0")
    with (out/"source_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow(dict(row, **row.get("strengths", {})))
    k0, extent = grid_image(s, b["convergence"])
    k1, _ = grid_image(s, s["convergence"])
    kd = k1-k0
    config = json.loads((args.run/"protocol.json").read_text())["configuration"]
    # Saved square native sampling: image pixel center range +/- (N-1)*dpix/2.
    metadata = json.loads((Path(json.loads((args.run/"protocol.json").read_text())["arguments"]["macro_dir"])/"m3_handoff.json").read_text())
    dpix = metadata["dpix"]; ny, nx = s["mask"].shape
    image_extent = (-nx*dpix/2, nx*dpix/2, -ny*dpix/2, ny*dpix/2)
    active = ~s["mask"]
    iy, ix = np.nonzero(active)
    xx = (ix-(nx-1)/2)*dpix; yy = (iy-(ny-1)/2)*dpix
    view = (min(extent[0],float(xx.min()-dpix/2)), max(extent[1],float(xx.max()+dpix/2)),
            min(extent[2],float(yy.min()-dpix/2)), max(extent[3],float(yy.max()+dpix/2)))
    half = max(view[1]-view[0], view[3]-view[2])/2
    cx, cy = (view[1]+view[0])/2, (view[3]+view[2])/2
    view = cx-half, cx+half, cy-half, cy+half
    assert np.all((xx >= view[0]) & (xx <= view[1]) & (yy >= view[2]) & (yy <= view[3]))
    data = np.where(active, s["lens_subtracted_image"], np.nan)
    model = np.where(active, s["nonlinear_source_model_image"], np.nan)
    residual = np.where(active, s["normalized_residual"], np.nan)
    assert np.isfinite(data).sum() == config["n_data"] == 2293
    np.savez_compressed(out/"plotted_arrays.npz", data=data, model=model, normalized_residual=residual,
        baseline_kappa=k0, selected_kappa=k1, kappa_difference=kd, image_extent=image_extent,
        potential_extent=extent, view=view)
    bright = np.r_[data[active], model[active]]
    kmax = max(float(np.nanmax(abs(k0))), float(np.nanmax(abs(k1))), 1e-12)
    rmax = max(float(np.nanmax(abs(residual))), 1e-12)
    dmax = max(float(np.nanmax(abs(kd))), 1e-12)
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["DejaVu Sans"], "font.size":9,
        "axes.titlesize":10, "axes.labelsize":9, "xtick.labelsize":8, "ytick.labelsize":8,
        "pdf.fonttype":42, "svg.fonttype":"none", "axes.spines.right":False, "axes.spines.top":False})
    fig = plt.figure(figsize=(10., 10.5))
    grid = fig.add_gridspec(4, 3, left=.08, right=.97, top=.86, bottom=.20,
        height_ratios=[1., .055, 1., .055], wspace=.32, hspace=.80)
    arrays = (data, model, residual, k0, k1, kd)
    titles = ("Lens-subtracted data", "Evidence-selected C1 model", "Normalized residual",
        f"C1: previous fixed λs optimum\nPeak {baseline['delta_kappa_peak']:.5f}",
        f"C1: λs + lens weights optimized\nPeak {best['delta_kappa_peak']:.5f}", "Selected − Baseline\nDifference in perturbation field")
    limits = ((float(bright.min()), float(bright.max())),)*2+((-rmax,rmax),(-kmax,kmax),(-kmax,kmax),(-dmax,dmax))
    axes = []
    for i,(array,title,limit) in enumerate(zip(arrays,titles,limits)):
        row, col = divmod(i,3)
        ax = fig.add_subplot(grid[2*row,col]); axes.append(ax)
        ax.set_gid("panel_"+chr(97+i)); ax.set_box_aspect(1.)
        ax.text(-.15, 1.12, chr(97+i), transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")
        mesh = ax.imshow(array, origin="lower", extent=image_extent if row == 0 else extent,
            vmin=limit[0], vmax=limit[1], cmap="viridis" if i < 2 else "RdBu_r",
            interpolation="none", aspect="auto", rasterized=True)
        ax.set_facecolor("#eeeeee")
        ax.set_xlim(view[:2]); ax.set_ylim(view[2:])
        ax.set_xticks([-2,0,2]); ax.set_yticks([-2,0,2])
        ax.set_xlabel("x (arcsec)")
        if col == 0: ax.set_ylabel("y (arcsec)")
        ax.set_title(title, pad=11)
        cax = fig.add_subplot(grid[2*row+1,col])
        bar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        bar.set_label(("Brightness (input units)" if i < 2 else "Residual / noise σ" if i == 2
            else "δκ" if i < 5 else "Difference in δκ"), fontsize=8)
        fig.canvas.draw()
        apos,cpos = ax.get_position(),cax.get_position()
        cax.set_position([apos.x0,cpos.y0,apos.width,cpos.height])
    strengths = best["strengths"]; validation = analysis["selected_validation"]
    fig.suptitle("J0946: C1 evidence optimization of source and lens priors", fontsize=13, y=.975)
    fig.text(.08,.935,f"λs = {strengths['lambda_s']:.5g}, λ4 = {strengths['lambda4']:.5g}, λ0 = {strengths['lambda0']:.5g}; Float64",fontsize=10)
    fig.text(.08,.909,f"Selected endpoint + cold repeats: {sum(r['accepted'] for r in validation['records'])}/{validation['attempts']} pass; repeat gate {validation['status']}",fontsize=10)
    fig.text(.08,.105,"Baseline: previous fixed M3 λs optimum; selected: λs, λ4 and λ0 jointly optimized.",fontsize=9)
    fig.text(.08,.075,"All 2293 likelihood pixels and 802 finite potential-core values are displayed; no map smoothing.",fontsize=9)
    fig.text(.08,.045,"Local-GN evidence; fixed macro and adaptive prior shape; no injected truth or global-optimum certificate.",fontsize=9)
    sys.path.insert(0,str(Path.home()/".agents/skills/nature-figure/scripts"))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list("abcdef"),strict=True,require_panel_labels=True,
        json_out=out/"diagnostics.alignment.json",overlay_svg=out/"diagnostics.alignment.svg")
    fig.savefig(out/"diagnostics.png",dpi=300)
    fig.savefig(out/"diagnostics.pdf")
    fig.savefig(out/"diagnostics.svg")
    plt.close(fig)
    print(out/"diagnostics.png")


if __name__ == "__main__":
    main()
