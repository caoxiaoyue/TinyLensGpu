"""Plot frozen DIRECT/L-BFGS products; never fit or select a candidate.

All core nodes and finite candidate scores are retained. Colour ranges are
paired within case; the control has a separate, labelled scale. See the local
figure contract for inference boundaries and data-integrity rules.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np


def core_image(data):
    nodes = data["node_indices"]
    lower, upper = nodes.min(axis=0), nodes.max(axis=0)
    image = np.full(tuple((upper - lower + 1)[::-1]), np.nan)
    valid = data["derivative_valid"].astype(bool)
    assert int(valid.sum()) == len(data["core_indices"])
    x, y = (nodes - lower).T
    image[y[valid], x[valid]] = data["convergence"][valid]
    h = float(np.median(np.diff(np.unique(data["potential_x"]))))
    extent = (
        data["potential_x"].min() - h / 2,
        data["potential_x"].max() + h / 2,
        data["potential_y"].min() - h / 2,
        data["potential_y"].max() + h / 2,
    )
    return image, extent


def label(ax, letter):
    ax.text(
        0,
        1,
        letter,
        transform=ax.transAxes
        + ScaledTranslation(-12 / 72, 6 / 72, ax.figure.dpi_scale_trans),
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--audit-scripts",
        type=Path,
        default=Path.home() / ".agents/skills/nature-figure/scripts",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.audit_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "legend.fontsize": 7,
            "legend.frameon": False,
        }
    )
    output = args.run / "figures"
    output.mkdir(exist_ok=True)
    cases = ("subhalo", "control")
    reports = {
        case: json.loads((args.run / case / "summary.json").read_text())
        for case in cases
    }

    def save(fig, name, exclude=()):
        prefix = output / name
        fig.canvas.draw()
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(prefix) + ".alignment.json",
            overlay_svg=str(prefix) + ".alignment.svg",
            exclude_axes=list(exclude),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
        fig.savefig(str(prefix) + ".pdf")
        fig.savefig(str(prefix) + ".svg")
        fig.savefig(str(prefix) + ".png", dpi=300)
        fig.savefig(str(prefix) + ".tiff", dpi=600)
        plt.close(fig)

    fig = plt.figure(figsize=(180 / 25.4, 142 / 25.4), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.05))
    colorbars = []
    manifest = {}
    for row, case in enumerate(cases):
        arrays = {}
        for state in ("baseline", "selected"):
            with np.load(args.run / case / f"{state}_fields.npz") as saved:
                arrays[state] = {key: np.array(saved[key]) for key in saved.files}
        images = [core_image(arrays[state]) for state in ("baseline", "selected")]
        limit = max(float(np.nanmax(np.abs(image))) for image, _ in images)
        limit = max(limit, np.finfo(float).eps)
        manifest[case] = dict(
            colour_limit=limit,
            valid_core_count=int(arrays["baseline"]["derivative_valid"].sum()),
            unknown_count=len(arrays["baseline"]["potential_values"]),
        )
        for col, ((image, extent), state) in enumerate(
            zip(images, ("baseline", "selected"))
        ):
            ax = fig.add_subplot(grid[row, col])
            artist = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                interpolation="none",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            if case == "subhalo":
                mock = reports[case]["protocol"]["config"]["mock"]
                ax.plot(
                    mock["subhalo_x"],
                    mock["subhalo_y"],
                    "x",
                    color="black",
                    markersize=5,
                    markeredgewidth=0.8,
                )
            ax.set_title(
                "Fixed regularization"
                if state == "baseline"
                else "DIRECT-selected regularization"
            )
            ax.set_xlabel("x (arcsec)")
            ax.set_ylabel(
                ("Subhalo" if case == "subhalo" else "No-subhalo control")
                + ": y (arcsec)"
            )
            label(ax, "abcd"[row * 2 + col])
        cax = fig.add_subplot(grid[row, 2])
        fig.colorbar(artist, cax=cax, label=r"$\delta\kappa$")
        colorbars.append(cax)
    save(fig, "correction_comparison", colorbars)
    (output / "correction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(180 / 25.4, 72 / 25.4), layout="constrained"
    )
    for ax, case, letter in zip(axes, cases, "ab"):
        rows = [
            json.loads(line)
            for line in (args.run / case / "evaluations.jsonl").read_text().splitlines()
        ]
        baseline = reports[case]["baseline"]["log_evidence"]
        valid = [r for r in rows if r["accepted"]]
        invalid = [r for r in rows if not r["accepted"]]
        ax.scatter(
            [r["index"] for r in valid],
            [r["log_evidence"] - baseline for r in valid],
            s=6,
            color="#B4BDC3",
            alpha=0.7,
            label="Valid candidate",
            rasterized=True,
        )
        running = []
        best = baseline
        for r in rows:
            if r["accepted"]:
                best = max(best, r["log_evidence"])
            running.append(best - baseline)
        ax.plot(
            [r["index"] for r in rows],
            running,
            color="#28647D",
            linewidth=1.3,
            label="Best so far",
        )
        if invalid:
            ax.plot(
                [r["index"] for r in invalid],
                np.full(len(invalid), 0.02),
                "|",
                color="#9C574B",
                markersize=4,
                transform=ax.get_xaxis_transform(),
                label="Rejected (bottom strip)",
            )
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("Distinct evaluated candidate")
        ax.set_ylabel("Δ log E (GN)")
        ax.set_title("Subhalo" if case == "subhalo" else "No-subhalo control")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right")
        label(ax, letter)
        with (output / f"{case}_search.csv").open("w") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "index",
                    "phase",
                    "accepted",
                    "lambda_source",
                    "lambda_potential",
                    "lambda_amplitude",
                    "log_evidence",
                    "failure_reason",
                ),
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(
                    {
                        **{
                            k: r.get(k)
                            for k in (
                                "index",
                                "phase",
                                "accepted",
                                "log_evidence",
                                "failure_reason",
                            )
                        },
                        **r["strengths"],
                    }
                )
    save(fig, "search_history")


if __name__ == "__main__":
    main()
