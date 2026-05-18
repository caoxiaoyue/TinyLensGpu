"""
Single-step joint shapelet source + MGE lens-light reconstruction demo.

Reconstructs the source galaxy with a shapelet basis and the lens light with
an MGE basis simultaneously, using a fixed SIE + Shear mass model.
solver_type='normal' allows negative shapelet amplitudes.

Usage
-----
    python sim_data.py
    python single_step_inversion.py --n-max 8 --beta 0.2 --n-mge 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from TinyLensGpu.ObservationModel.LensImage import ImageProbModel
from TinyLensGpu.PhysicalModel import (
    GaussianEllipse,
    PhysicalModel,
    SIE,
    Shear,
    build_shapelet_set,
    build_shapelet_basis_matrix,
)
from TinyLensGpu.utils import load_lens_data, generate_radial_basis_knots
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


def build_mge_basis(
    n_components: int, sigma_min: float, sigma_max: float, dpix: float = 0.05, mask: np.ndarray = None
) -> list[GaussianEllipse]:
    """Build MGE lens-light basis, avoiding the lensed arc region if mask is provided."""
    e1_lens, e2_lens = phi_q2_ellipticity(90.0 * np.pi / 180.0, 0.9)
    sigma_list = generate_radial_basis_knots(
        dpix=dpix, n_sigmas=n_components,
        log_rmin=np.log10(sigma_min), log_rmax=np.log10(sigma_max),
        arc_mask=mask
    )
    return [
        GaussianEllipse(
            sigma=float(s), center_x=0.0, center_y=0.0,
            e1=e1_lens, e2=e2_lens, flux=1.0,
        )
        for s in sigma_list
    ]


def build_model(
    data_dict: dict,
    *,
    n_max: int,
    beta: float,
    center_x: float,
    center_y: float,
    n_mge: int,
    sigma_min: float,
    sigma_max: float,
) -> ImageProbModel:
    """Construct the joint shapelet + MGE probability model."""
    e1_l, e2_l = phi_q2_ellipticity(90.0 * np.pi / 180.0, 0.9)
    mass_model = [
        SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0),
        Shear(gamma1=0.05, gamma2=0.05),
    ]

    shapelet_basis = build_shapelet_set(
        n_max=n_max, beta=beta, center_x=center_x, center_y=center_y
    )
    mge_basis = build_mge_basis(
        n_mge, sigma_min, sigma_max,
        dpix=float(data_dict["dpix"]), mask=data_dict["mask"]
    )

    phys_model = PhysicalModel(
        lens_mass=mass_model,
        source_light=shapelet_basis,
        lens_light=mge_basis,
    )

    return ImageProbModel(
        image_data=data_dict["noisy_image"],
        noise_map=data_dict["noise_map"],
        psf_kernel=data_dict["psf_kernel"],
        dpix=float(data_dict["dpix"]),
        nsub=1,
        phys_model=phys_model,
        use_linear=True,
        mask=data_dict["mask"],
        solver_type="normal",
    )


def run_inversion(
    prob_model: ImageProbModel,
    *,
    n_basis: int,
    n_max: int,
    beta: float,
    center_x: float,
    center_y: float,
    src_npix: int = 100,
    src_half_size: float = 1.5,
) -> dict:
    """Solve joint system and return diagnostics."""
    log_like = float(np.asarray(prob_model()))

    # ret_each_plane=True gives separate source and lens-light images
    img_arc, img_lens, X_vec = prob_model.forward_model(
        use_linear=True, ret_each_plane=True, return_intensity=True
    )
    img_arc = np.asarray(img_arc)
    img_lens = np.asarray(img_lens)
    X_vec = np.asarray(X_vec)

    shapelet_amps = X_vec[:n_basis]
    mge_amps = X_vec[n_basis:]

    # Reconstruct source on a fine grid
    x_src = np.linspace(-src_half_size, src_half_size, src_npix)
    y_src = np.linspace(-src_half_size, src_half_size, src_npix)
    X_src, Y_src = np.meshgrid(x_src, y_src)
    basis_mat = np.asarray(
        build_shapelet_basis_matrix(
            X_src.ravel(), Y_src.ravel(),
            n_max, beta, center_x, center_y,
        )
    )
    src_image = (basis_mat @ shapelet_amps).reshape(src_npix, src_npix)

    # Full model image = source arc + lens light
    model_image = img_arc + img_lens

    return {
        "log_like": log_like,
        "model_image": model_image,
        "lens_model_image": img_lens,
        "src_image": src_image,
        "X_vec": X_vec,
        "shapelet_amps": shapelet_amps,
        "mge_amps": mge_amps,
        "src_extent": (-src_half_size, src_half_size, -src_half_size, src_half_size),
    }


def visualize(data_dict: dict, results: dict, *, output_path: Path) -> None:
    """Six-panel figure: observed, model, residual, norm-residual, source, lens-light."""
    noisy_image = np.asarray(data_dict["noisy_image"])
    noise_map = np.asarray(data_dict["noise_map"])
    mask = np.asarray(data_dict["mask"])
    model_image = results["model_image"]
    lens_model_image = results["lens_model_image"]
    src_image = results["src_image"]
    log_like = results["log_like"]
    x_min, x_max, y_min, y_max = results["src_extent"]

    residual = np.zeros_like(noisy_image)
    residual[~mask] = noisy_image[~mask] - model_image[~mask]
    norm_res = np.zeros_like(noisy_image)
    norm_res[~mask] = residual[~mask] / noise_map[~mask]
    chi2 = float(np.sum(norm_res[~mask] ** 2))

    fig = plt.figure(figsize=(18, 10))

    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(noisy_image * (~mask).astype(float), origin="lower", cmap="viridis")
    ax1.set_title("Observed Noisy Image")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(model_image, origin="lower", cmap="viridis")
    ax2.set_title(f"Joint Model (Shapelet + MGE)\nlog L = {log_like:.2f}")
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
        src_image, origin="lower", cmap="viridis",
        extent=(x_min, x_max, y_min, y_max), aspect="auto",
    )
    ax5.set_title("Reconstructed Source (Shapelets)")
    ax5.set_xlabel("beta_x [arcsec]")
    ax5.set_ylabel("beta_y [arcsec]")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(lens_model_image, origin="lower", cmap="magma")
    ax6.set_title("Lens-Light-Only Model (MGE)")
    mge_amps = results["mge_amps"]
    stats_text = (
        f"n_mge = {len(mge_amps)}\n"
        f"Valid pixels = {int(np.sum(~mask))}"
    )
    ax6.text(
        0.03, 0.97, stats_text, transform=ax6.transAxes,
        ha="left", va="top", fontsize=10, color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 4.0},
    )
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shapelet source + MGE lens-light joint inversion")
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--center-x", type=float, default=0.0)
    p.add_argument("--center-y", type=float, default=0.5)
    p.add_argument("--n-mge", type=int, default=10)
    p.add_argument("--sigma-min", type=float, default=0.01)
    p.add_argument("--sigma-max", type=float, default=3.0)
    p.add_argument("--src-npix", type=int, default=100)
    p.add_argument("--src-half-size", type=float, default=1.5)
    p.add_argument("--save-json", type=Path, default=Path("shapelet_joint_results.json"))
    p.add_argument("--save-figure", type=Path, default=Path("shapelet_joint.png"))
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
        "dpix": 0.074,
    }

    n_basis = (args.n_max + 1) * (args.n_max + 2) // 2
    print(
        f"Building joint model: n_max={args.n_max}, n_basis={n_basis}, "
        f"beta={args.beta}, n_mge={args.n_mge}"
    )
    prob_model = build_model(
        data_dict,
        n_max=args.n_max,
        beta=args.beta,
        center_x=args.center_x,
        center_y=args.center_y,
        n_mge=args.n_mge,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )

    print("Running joint inversion...")
    results = run_inversion(
        prob_model,
        n_basis=n_basis,
        n_max=args.n_max,
        beta=args.beta,
        center_x=args.center_x,
        center_y=args.center_y,
        src_npix=args.src_npix,
        src_half_size=args.src_half_size,
    )

    print(f"Saving figure to {args.save_figure}...")
    visualize(data_dict, results, output_path=args.save_figure)

    payload = {
        "n_max": args.n_max,
        "n_basis": n_basis,
        "beta": args.beta,
        "center_x": args.center_x,
        "center_y": args.center_y,
        "n_mge": args.n_mge,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "log_like": float(results["log_like"]),
        "n_data": int(np.sum(~mask)),
        "figure_path": str(args.save_figure),
        "mge_amps": results["mge_amps"].tolist(),
    }
    args.save_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("\nSummary:")
    print(json.dumps({k: v for k, v in payload.items() if k != "mge_amps"}, indent=2))


if __name__ == "__main__":
    main()
