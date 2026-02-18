"""
Single-step shapelet source reconstruction demo (source light only).

Reconstructs the source galaxy using a shapelet basis set with a fixed SIE
mass model.  Uses ImageProbModel with use_linear=True, solver_type='normal'
so that shapelet amplitudes (which can be negative) are solved analytically.

Usage
-----
    python sim_data.py          # generate data/
    python single_step_inversion.py --n-max 20 --beta 0.2 --nsub 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from TinyLensGpu.ObservationModel.LensImage import ImageProbModel
from TinyLensGpu.PhysicalModel import (
    PhysicalModel,
    SIE,
    build_shapelet_set,
    build_shapelet_basis_matrix,
)
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


def build_model(
    data_dict: dict,
    *,
    n_max: int,
    beta: float,
    center_x: float,
    center_y: float,
    nsub: int,
) -> ImageProbModel:
    """Construct the shapelet source-only probability model."""
    e1_l, e2_l = phi_q2_ellipticity(90 * np.pi / 180, 0.9)
    sie = SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0)

    shapelet_basis = build_shapelet_set(
        n_max=n_max, beta=beta, center_x=center_x, center_y=center_y
    )
    phys_model = PhysicalModel(
        lens_mass=[sie],
        source_light=shapelet_basis,
        lens_light=[],
    )

    return ImageProbModel(
        image_data=data_dict["noisy_image"],
        noise_map=data_dict["noise_map"],
        psf_kernel=data_dict["psf_kernel"],
        dpix=float(data_dict["dpix"]),
        nsub=nsub,
        phys_model=phys_model,
        use_linear=True,
        mask=data_dict["mask"],
        solver_type="normal",
    )


def run_inversion(
    prob_model: ImageProbModel,
    *,
    n_max: int,
    beta: float,
    center_x: float,
    center_y: float,
    src_npix: int = 100,
    src_half_size: float = 1.5,
) -> dict:
    """Solve for shapelet amplitudes and reconstruct the source image."""
    log_like = float(np.asarray(prob_model()))

    image_model, X_vec = prob_model.forward_model(use_linear=True, return_intensity=True)
    image_model = np.asarray(image_model)
    X_vec = np.asarray(X_vec)

    # Reconstruct source on a fine grid using the fast basis-matrix path
    x_src = np.linspace(-src_half_size, src_half_size, src_npix)
    y_src = np.linspace(-src_half_size, src_half_size, src_npix)
    X_src, Y_src = np.meshgrid(x_src, y_src)

    basis_mat = np.asarray(
        build_shapelet_basis_matrix(
            X_src.ravel(), Y_src.ravel(),
            n_max, beta, center_x, center_y,
        )
    )  # (src_npix^2, n_basis)
    src_image = (basis_mat @ X_vec).reshape(src_npix, src_npix)

    return {
        "log_like": log_like,
        "image_model": image_model,
        "X_vec": X_vec,
        "src_image": src_image,
        "src_extent": (-src_half_size, src_half_size, -src_half_size, src_half_size),
    }


def visualize(
    data_dict: dict,
    results: dict,
    *,
    output_path: Path,
) -> None:
    """Six-panel figure: observed, model, residual, norm-residual, source, coefficients."""
    noisy_image = np.asarray(data_dict["noisy_image"])
    noise_map = np.asarray(data_dict["noise_map"])
    mask = np.asarray(data_dict["mask"])
    image_model = results["image_model"]
    src_image = results["src_image"]
    X_vec = results["X_vec"]
    log_like = results["log_like"]
    x_min, x_max, y_min, y_max = results["src_extent"]

    residual = np.zeros_like(noisy_image)
    residual[~mask] = noisy_image[~mask] - image_model[~mask]
    norm_res = np.zeros_like(noisy_image)
    norm_res[~mask] = residual[~mask] / noise_map[~mask]
    chi2 = float(np.sum((norm_res[~mask]) ** 2))

    fig = plt.figure(figsize=(18, 10))

    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(noisy_image * (~mask).astype(float), origin="lower", cmap="viridis")
    ax1.set_title("Observed Noisy Image")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(image_model, origin="lower", cmap="viridis")
    ax2.set_title(f"Shapelet Model\nlog L = {log_like:.2f}")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 3, 3)
    vmax_res = np.max(np.abs(residual[~mask]))
    im3 = ax3.imshow(residual, origin="lower", cmap="RdBu_r", vmin=-vmax_res, vmax=vmax_res)
    ax3.set_title("Residual")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(norm_res, origin="lower", cmap="RdBu_r", vmin=-3.0, vmax=3.0)
    ax4.set_title(f"Normalized Residual\nchi2 = {chi2:.1f}")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(
        src_image,
        origin="lower",
        cmap="viridis",
        extent=(x_min, x_max, y_min, y_max),
        aspect="auto",
    )
    ax5.set_title("Reconstructed Source (Shapelets)")
    ax5.set_xlabel("beta_x [arcsec]")
    ax5.set_ylabel("beta_y [arcsec]")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = plt.subplot(2, 3, 6)
    ax6.bar(np.arange(len(X_vec)), X_vec, color="steelblue", edgecolor="none")
    ax6.axhline(0, color="k", linewidth=0.8)
    ax6.set_xlabel("Basis index")
    ax6.set_ylabel("Amplitude")
    ax6.set_title(f"Shapelet Coefficients (n_basis={len(X_vec)})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shapelet source-only single-step inversion")
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--center-x", type=float, default=0.0)
    p.add_argument("--center-y", type=float, default=0.3)
    p.add_argument("--nsub", type=int, default=4)
    p.add_argument("--src-npix", type=int, default=100)
    p.add_argument("--src-half-size", type=float, default=1.5)
    p.add_argument("--save-json", type=Path, default=Path("shapelet_src_only_results.json"))
    p.add_argument("--save-figure", type=Path, default=Path("shapelet_src_only.png"))
    return p


def main() -> None:
    args = build_cli().parse_args()

    print("Loading data...")
    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path="data/image.fits",
        noise_path="data/noise.fits",
        psf_path="data/psf.fits",
        mask_path="data/mask.fits",
    )
    data_dict = {
        "noisy_image": image_data,
        "noise_map": noise_map,
        "psf_kernel": psf_kernel,
        "mask": mask,
        "dpix": 0.05,
    }

    n_basis = (args.n_max + 1) * (args.n_max + 2) // 2
    print(
        f"Building shapelet model: n_max={args.n_max}, n_basis={n_basis}, "
        f"beta={args.beta}, nsub={args.nsub}"
    )
    prob_model = build_model(
        data_dict,
        n_max=args.n_max,
        beta=args.beta,
        center_x=args.center_x,
        center_y=args.center_y,
        nsub=args.nsub,
    )

    print("Running inversion...")
    results = run_inversion(
        prob_model,
        n_max=args.n_max,
        beta=args.beta,
        center_x=args.center_x,
        center_y=args.center_y,
        src_npix=args.src_npix,
        src_half_size=args.src_half_size,
    )

    model_image = np.asarray(results["image_model"])
    chi2 = float(np.sum(((image_data[~mask] - model_image[~mask]) / noise_map[~mask]) ** 2))
    dof = max(int(np.sum(~mask)) - n_basis, 1)
    reduced_chi2 = chi2 / dof

    print(f"Saving figure to {args.save_figure}...")
    visualize(data_dict, results, output_path=args.save_figure)

    payload = {
        "n_max": args.n_max,
        "n_basis": n_basis,
        "beta": args.beta,
        "nsub": args.nsub,
        "center_x": args.center_x,
        "center_y": args.center_y,
        "log_like": float(results["log_like"]),
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "n_data": int(np.sum(~mask)),
        "dof": dof,
        "figure_path": str(args.save_figure),
        "X_vec": results["X_vec"].tolist(),
    }
    args.save_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("\nSummary:")
    print(json.dumps({k: v for k, v in payload.items() if k != "X_vec"}, indent=2))


if __name__ == "__main__":
    main()
