"""Plot saved three-strength exploratory results without running any fit.

The scientific comparison uses centered nodal-truth convergence, one shared
colour scale and the same core domain. Evidence curves are conditional slices,
not profiles or hyperparameter credible intervals.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def core_image(arrays, values):
    nodes = arrays["potential_node_indices"]
    valid = arrays["derivative_valid"]
    low, high = nodes.min(axis=0), nodes.max(axis=0)
    image = np.full(tuple((high - low + 1)[::-1]), np.nan)
    x, y = (nodes - low).T
    image[y[valid], x[valid]] = values[valid]
    spacing = float(np.median(np.diff(np.unique(arrays["potential_x"])) ))
    extent = [arrays["potential_x"].min() - spacing/2,
              arrays["potential_x"].max() + spacing/2,
              arrays["potential_y"].min() - spacing/2,
              arrays["potential_y"].max() + spacing/2]
    return image, extent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "summary.json").read_text())
    names = ["original", "fixed_amplitude_optimized", "joint_optimized"]
    titles = ["Original hand selection", "Optimize source + D4", "Optimize source + D4 + amplitude"]
    arrays = {}
    for name in names:
        with np.load(args.run / name / "reconstruction.npz") as archive:
            arrays[name] = {key: archive[key] for key in archive.files}
    reference = arrays["original"]
    truth = reference["truth_discrete_convergence"]
    valid = reference["derivative_valid"]
    vmax = max(float(np.nanmax(np.abs(truth[valid]))),
               *(float(np.nanmax(np.abs(arrays[name]["convergence"][valid]))) for name in names))
    if vmax == 0:
        vmax = 1e-6
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), layout="constrained", sharex=True, sharey=True)
    data = [(reference, truth), *((arrays[name], arrays[name]["convergence"]) for name in names)]
    for i, (ax, (values, field)) in enumerate(zip(axes, data)):
        image, extent = core_image(values, field)
        artist = ax.imshow(image, origin="lower", extent=extent, cmap="RdBu_r",
                           norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), interpolation="nearest")
        ax.plot(0.75, 0.1, marker="x", color="black", ms=6, mew=1)
        ax.set_xlabel("x (arcsec)")
        ax.set_title("Nodal truth" if i == 0 else titles[i-1], fontsize=10)
        if i:
            row = report["exports"][names[i-1]]
            recovery = row["recovery_at_truth_peak"]
            text = (f"Peak recovery: {100*recovery:.1f}%" if recovery is not None
                    else f"RMS: {row['summary']['convergence']['all_core']['recovered_rms']:.3g}")
            ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    axes[0].set_ylabel("y (arcsec)")
    fig.colorbar(artist, ax=axes, label=r"$\delta\kappa$", shrink=.85)
    fig.savefig(args.run / "convergence_comparison.png", dpi=180)
    fig.savefig(args.run / "convergence_comparison.pdf")
    plt.close(fig)

    curves = report["conditional_slices"]["curves"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    best = report["selected"]["log_evidence"]
    labels = [r"$\log_{10}\lambda_s$", r"$\log_{10}\lambda_4$", r"$\log_{10}\lambda_0$"]
    for ax, (name, rows), label in zip(axes, curves.items(), labels):
        finite = [row for row in rows if row["log_evidence"] is not None]
        differences = [row["log_evidence"] - best for row in finite]
        ax.plot([row["log10_strength"] for row in finite],
                differences, "o-", ms=4, color="#287a8f")
        dimension = list(curves).index(name)
        ax.axvline(report["selected"]["log10_strengths"][dimension], color="#c87940", ls="--", lw=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(min(-1, 1.1*min(differences)), max(1, max(differences)+.1))
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\log Z-\log Z_{\rm best}$")
        ax.grid(alpha=.2)
    fig.suptitle("Conditional evidence slices; other strengths fixed at joint optimum", fontsize=10)
    fig.savefig(args.run / "conditional_evidence.png", dpi=180)
    fig.savefig(args.run / "conditional_evidence.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
