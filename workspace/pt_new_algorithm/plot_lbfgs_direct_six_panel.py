"""Six saved-result panels: data/model/residual and injected/recovered/delta kappa.

Reads saved fits and canonical mock truth; imports no JAX and runs no fit.
Truth and recovery use the same centered discrete potential derivatives.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_lbfgs_direct import core_image, label

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--case", choices=("subhalo", "control"), default="subhalo")
    parser.add_argument("--state", choices=("baseline", "selected"), default="selected")
    parser.add_argument("--truth-archive", type=Path)
    parser.add_argument(
        "--audit-scripts",
        type=Path,
        default=Path.home() / ".agents/skills/nature-figure/scripts",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.audit_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    directory = args.run / args.case
    report = json.loads((directory / "summary.json").read_text())
    path = directory / f"{args.state}_fields.npz"
    truth_path = args.truth_archive or (
        ROOT
        / ".scratch/potential-correction/output/single-step-scientific/reference"
        / args.case
        / "dense/reconstruction.npz"
    )
    with np.load(path) as saved:
        data = {key: np.array(saved[key]) for key in saved.files}
    with np.load(truth_path) as saved:
        reference = {key: np.array(saved[key]) for key in saved.files}

    # Verify the truth really belongs to this image and exactly this grid.
    for key in (
        "image_data",
        "noise_map",
        "potential_x",
        "potential_y",
        "source_x_axis",
        "source_y_axis",
        "derivative_valid",
    ):
        np.testing.assert_array_equal(data[key], reference[key])
    np.testing.assert_array_equal(
        data["node_indices"], reference["potential_node_indices"]
    )
    np.testing.assert_array_equal(
        data["core_indices"], reference["potential_core_indices"]
    )
    active = ~reference["mask"].astype(bool)
    np.testing.assert_array_equal(
        active, np.isfinite(data["nonlinear_normalized_residual"])
    )
    np.testing.assert_array_equal(
        data["nonlinear_model_image"][active], data["nonlinear_image_active"]
    )
    residual = np.full(data["image_data"].shape, np.nan)
    residual[active] = (
        (data["image_data"] - data["nonlinear_model_image"]) / data["noise_map"]
    )[active]
    np.testing.assert_allclose(
        residual, data["nonlinear_normalized_residual"], rtol=1e-12, atol=1e-12
    )

    core = data["core_indices"]
    valid = data["derivative_valid"].astype(bool)
    np.testing.assert_array_equal(np.flatnonzero(valid), core)
    true_kappa = reference["truth_discrete_convergence"]
    recovered = data["convergence"]
    difference = recovered - true_kappa
    cfg = report["protocol"]["config"]
    h = cfg["potential_spacing"]
    indices = reference["potential_derivative_indices"]

    def discrete_kappa(p):
        center, left, right, down, up = p[indices].T
        return 0.5 * (
            (right - 2 * center + left) / h**2 + (up - 2 * center + down) / h**2
        )

    np.testing.assert_allclose(
        discrete_kappa(reference["truth_potential_values"]),
        true_kappa[core],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        discrete_kappa(data["potential_values"]),
        recovered[core],
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isfinite(difference[core]).all()
    residual_rms = float(np.sqrt(np.mean(residual[active] ** 2)))
    difference_rms = float(np.sqrt(np.mean(difference[core] ** 2)))
    recovery = report[f"{args.state}_recovery"]
    np.testing.assert_allclose(
        residual_rms, recovery["residual_rms_in_noise"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        difference_rms,
        recovery["recovery"]["convergence"]["all_core"]["rmse_discrete_truth"],
        rtol=1e-12,
        atol=1e-12,
    )

    image_data = np.where(active, data["image_data"], np.nan)
    model = np.where(active, data["nonlinear_model_image"], np.nan)
    image_limits = (
        min(float(np.nanmin(image_data)), float(np.nanmin(model))),
        max(float(np.nanmax(image_data)), float(np.nanmax(model))),
    )
    residual_limit = max(1, math.ceil(float(np.nanmax(np.abs(residual)))))
    kappa_limit = max(
        float(np.max(np.abs(true_kappa[core]))),
        float(np.max(np.abs(recovered[core]))),
        np.finfo(float).eps,
    )
    difference_limit = max(float(np.max(np.abs(difference[core]))), np.finfo(float).eps)
    kappa_images = []
    for values in (true_kappa, recovered, difference):
        mapped = {**data, "convergence": values}
        kappa_images.append(core_image(mapped))
    half = cfg["mock"]["npix"] * cfg["mock"]["dpix"] / 2
    image_extent = (-half, half, -half, half)
    panels = (
        (
            image_data,
            image_extent,
            "Data image",
            image_limits,
            "magma",
            "Intensity (a.u.)",
        ),
        (model, image_extent, "Model image", image_limits, "magma", "Intensity (a.u.)"),
        (
            residual,
            image_extent,
            "Normalized residual",
            (-residual_limit, residual_limit),
            "RdBu_r",
            "(Data − model) / noise",
        ),
        (
            *kappa_images[0],
            "Injected δκ (same grid)",
            (-kappa_limit, kappa_limit),
            "RdBu_r",
            "δκ",
        ),
        (*kappa_images[1], "Recovered δκ", (-kappa_limit, kappa_limit), "RdBu_r", "δκ"),
        (
            *kappa_images[2],
            "Recovered − injected",
            (-difference_limit, difference_limit),
            "RdBu_r",
            "Δδκ",
        ),
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.2047244094, 6.0236220472), layout="constrained")
    grid = fig.add_gridspec(4, 3, height_ratios=(1, 0.07, 1, 0.07))
    color_axes = []
    for i, (values, extent, title, limits, cmap, units) in enumerate(panels):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(grid[row * 2, col])
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
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        label(ax, "abcdef"[i])
        if row == 1 and args.case == "subhalo":
            ax.plot(
                cfg["mock"]["subhalo_x"],
                cfg["mock"]["subhalo_y"],
                "x",
                color="black",
                markersize=4,
                markeredgewidth=0.7,
            )
        cax = fig.add_subplot(grid[row * 2 + 1, col])
        colorbar = fig.colorbar(artist, cax=cax, orientation="horizontal")
        colorbar.set_label(units)
        colorbar.locator = MaxNLocator(3)
        colorbar.update_ticks()
        color_axes.append(cax)

    output = args.run / "figures"
    output.mkdir(exist_ok=True)
    prefix = output / f"{args.case}_{args.state}_six_panel"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(prefix) + ".alignment.json",
        overlay_svg=str(prefix) + ".alignment.svg",
        exclude_axes=color_axes,
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
    np.savez_compressed(
        str(prefix) + "_source.npz",
        image_data=data["image_data"],
        model_image=data["nonlinear_model_image"],
        normalized_residual=residual,
        image_valid=active,
        injected_kappa=true_kappa,
        recovered_kappa=recovered,
        kappa_difference=difference,
        analytic_injected_kappa=reference["truth_analytic_convergence"],
        derivative_valid=valid,
        core_indices=core,
        node_indices=data["node_indices"],
        potential_x=data["potential_x"],
        potential_y=data["potential_y"],
        image_extent=image_extent,
    )
    manifest = dict(
        case=args.case,
        state=args.state,
        fit_sha256=digest(path),
        truth_sha256=digest(truth_path),
        exact_geometry_data_noise_mask_match=True,
        image_valid_count=int(active.sum()),
        image_excluded_count=int((~active).sum()),
        potential_unknown_count=len(valid),
        kappa_valid_count=int(valid.sum()),
        kappa_excluded_support_count=int((~valid).sum()),
        residual_definition="(data-model)/noise",
        kappa_difference_definition="recovered-injected",
        truth_convention="Centered second differences of injected potential on the identical 0.075 arcsec potential grid",
        image_limits=image_limits,
        residual_limit=residual_limit,
        kappa_limit=kappa_limit,
        kappa_difference_limit=difference_limit,
        residual_rms=residual_rms,
        kappa_difference_rms=difference_rms,
        kappa_injected_peak=float(np.max(true_kappa[core])),
        kappa_recovered_peak=float(np.max(recovered[core])),
        source_arrays_unchanged=True,
        refitted=False,
    )
    Path(str(prefix) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
