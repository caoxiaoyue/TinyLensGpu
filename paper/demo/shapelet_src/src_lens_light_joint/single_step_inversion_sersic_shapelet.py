"""
Single-step joint Sersic + shapelet source + MGE lens-light reconstruction demo.

Reconstructs the source galaxy with both a Sersic profile and a shapelet basis
(simultaneously, co-centered), and the lens light with an MGE basis,
using a fixed SIE + Shear mass model.
solver_type='normal' allows negative shapelet amplitudes.

Usage
-----
    python sim_data.py
    python single_step_inversion_sersic_shapelet.py --n-max 8 --beta 0.2 --n-mge 10
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
    SersicEllipse,
    build_shapelet_set,
    build_shapelet_basis_matrix,
)
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.visualizer import overlay_critical_and_caustics, generate_radial_basis_knots
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


def build_mge_basis_lens(
    n_components: int, sigma_min: float, sigma_max: float, dpix: float = 0.05, mask: np.ndarray = None
) -> list[GaussianEllipse]:
    """Build MGE lens-light basis, avoiding the lensed arc region if mask is provided."""
    e1_lens, e2_lens = phi_q2_ellipticity(90.0 * np.pi / 180.0, 0.9)
    sigma_list = generate_radial_basis_knots(
        dpix=dpix, n_sigmas=n_components,
        log_rmin=np.log10(sigma_min), log_rmax=np.log10(sigma_max),
        arc_mask=mask, mode='mge'
    )
    return [
        GaussianEllipse(
            sigma=float(s), center_x=0.0, center_y=0.0,
            e1=e1_lens, e2=e2_lens, flux=1.0,
        )
        for s in sigma_list
    ]


def build_sersic_source(
    R_sersic: float,
    n_sersic: float,
    center_x: float,
    center_y: float,
    e1: float,
    e2: float,
) -> SersicEllipse:
    """Build a Sersic profile for source light.

    Parameters
    ----------
    R_sersic : float
        Half-light radius of the Sersic profile.
    n_sersic : float
        Sersic index.
    center_x, center_y, e1, e2 : float
        Geometric parameters for the Sersic component.

    Returns
    -------
    SersicEllipse
        Sersic profile for source light.
    """
    return SersicEllipse(
        R_sersic=ParamU("R_sersic_src", float(R_sersic)),
        n_sersic=ParamU("n_sersic_src", float(n_sersic)),
        center_x=center_x,
        center_y=center_y,
        e1=e1,
        e2=e2,
        Ie=ParamU("Ie_src", 1.0),
    )


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
    R_sersic: float,
    n_sersic: float,
    e1_src: float,
    e2_src: float,
) -> ImageProbModel:
    """Construct the joint Sersic+shapelet source + MGE lens-light probability model."""
    e1_l, e2_l = phi_q2_ellipticity(90.0 * np.pi / 180.0, 0.9)
    mass_model = [
        SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0),
        Shear(gamma1=0.05, gamma2=0.05),
    ]

    center_x_src = float(center_x)
    center_y_src = float(center_y)
    e1_src_param = float(e1_src)
    e2_src_param = float(e2_src)

    sersic_source = build_sersic_source(
        R_sersic, n_sersic,
        center_x_src, center_y_src, e1_src_param, e2_src_param,
    )

    shapelet_basis = build_shapelet_set(
        n_max=n_max, beta=beta, center_x=center_x, center_y=center_y
    )

    source_light = [sersic_source] + shapelet_basis

    lens_mge_basis = build_mge_basis_lens(
        n_mge, sigma_min, sigma_max,
        dpix=float(data_dict["dpix"]), mask=data_dict["mask"]
    )

    phys_model = PhysicalModel(
        lens_mass=mass_model,
        source_light=source_light,
        lens_light=lens_mge_basis,
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


def phi_q2_2_ellipticity(phi: float, q: float) -> tuple[float, float]:
    """Convert position angle phi and axis ratio q to ellipticity components.

    Parameters
    ----------
    phi : float
        Position angle in radians.
    q : float
        Axis ratio (b/a), must be in (0, 1].

    Returns
    -------
    tuple of (e1, e2)
        Ellipticity components.
    """
    e1 = (1.0 - q) / (1.0 + q) * np.cos(2.0 * phi)
    e2 = (1.0 - q) / (1.0 + q) * np.sin(2.0 * phi)
    return e1, e2


def run_inversion(
    prob_model: ImageProbModel,
    *,
    n_basis: int,
    n_max: int,
    beta: float,
    center_x: float,
    center_y: float,
    R_sersic: float,
    n_sersic: float,
    e1_src: float,
    e2_src: float,
    src_npix: int = 100,
    src_half_size: float = 1.5,
) -> dict:
    """Solve joint system and return diagnostics."""
    log_like = float(np.asarray(prob_model()))

    forward_out = prob_model.forward_model(
        use_linear=True, ret_each_plane=True, return_intensity=True
    )
    if not isinstance(forward_out, tuple) or len(forward_out) != 3:
        raise RuntimeError("Expected forward_model to return (img_arc, img_lens, X_vec)")
    img_arc, img_lens, X_vec = forward_out
    img_arc = np.asarray(img_arc)
    img_lens = np.asarray(img_lens)
    X_vec = np.asarray(X_vec)

    # First amplitude is the Sersic Ie, followed by shapelet amplitudes
    sersic_Ie = X_vec[0]
    shapelet_amps = X_vec[1:n_basis]
    lens_mge_amps = X_vec[n_basis:]

    x_src = np.linspace(-src_half_size, src_half_size, src_npix)
    y_src = np.linspace(-src_half_size, src_half_size, src_npix)
    X_src, Y_src = np.meshgrid(x_src, y_src)

    basis_mat = np.asarray(
        build_shapelet_basis_matrix(
            X_src.ravel(), Y_src.ravel(),
            n_max, beta, center_x, center_y,
        )
    )
    shapelet_image = (basis_mat @ shapelet_amps).reshape(src_npix, src_npix)

    sersic_source = build_sersic_source(
        R_sersic=float(R_sersic),
        n_sersic=float(n_sersic),
        center_x=float(center_x),
        center_y=float(center_y),
        e1=float(e1_src),
        e2=float(e2_src),
    )
    sersic_image = np.asarray(
        sersic_source.light(
            x=X_src,
            y=Y_src,
            R_sersic=float(R_sersic),
            n_sersic=float(n_sersic),
            e1=float(e1_src),
            e2=float(e2_src),
            center_x=float(center_x),
            center_y=float(center_y),
            Ie=float(sersic_Ie),
        )
    )
    src_image = sersic_image + shapelet_image

    model_image = img_arc + img_lens

    return {
        "log_like": log_like,
        "model_image": model_image,
        "lens_model_image": img_lens,
        "src_image": src_image,
        "src_sersic_image": sersic_image,
        "src_shapelet_image": shapelet_image,
        "X_vec": X_vec,
        "shapelet_amps": shapelet_amps,
        "sersic_Ie": sersic_Ie,
        "lens_mge_amps": lens_mge_amps,
        "src_extent": (-src_half_size, src_half_size, -src_half_size, src_half_size),
    }


def visualize(data_dict: dict, results: dict, *, output_path: Path) -> None:
    """Six-panel figure: observed, model, residual, norm-residual, source, lens-light."""
    noisy_image = np.asarray(data_dict["noisy_image"])
    noise_map = np.asarray(data_dict["noise_map"])
    mask = np.asarray(data_dict["mask"], dtype=bool)
    model_image = results["model_image"]
    lens_model_image = results["lens_model_image"]
    src_image = results["src_image"]
    src_sersic_image = results["src_sersic_image"]
    src_shapelet_image = results["src_shapelet_image"]
    log_like = results["log_like"]
    x_min, x_max, y_min, y_max = results["src_extent"]

    residual = np.zeros_like(noisy_image)
    residual[~mask] = noisy_image[~mask] - model_image[~mask]
    norm_res = np.zeros_like(noisy_image)
    norm_res[~mask] = residual[~mask] / noise_map[~mask]
    chi2 = float(np.sum(norm_res[~mask] ** 2))

    fig = plt.figure(figsize=(24, 10))

    ax1 = plt.subplot(2, 4, 1)
    im1 = ax1.imshow(noisy_image * (~mask).astype(float), origin="lower", cmap="viridis")
    ax1.set_title("Observed Noisy Image")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 4, 2)
    im2 = ax2.imshow(model_image, origin="lower", cmap="viridis")
    ax2.set_title(f"Joint Model (Sersic+Shapelet Source + MGE Lens)\nlog L = {log_like:.2f}")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 4, 3)
    vmax_res = np.max(np.abs(residual[~mask]))
    im3 = ax3.imshow(residual, origin="lower", cmap="RdBu_r", vmin=-vmax_res, vmax=vmax_res)
    ax3.set_title("Residual")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = plt.subplot(2, 4, 4)
    im4 = ax4.imshow(norm_res, origin="lower", cmap="RdBu_r", vmin=-3.0, vmax=3.0)
    ax4.set_title(f"Normalized Residual\nchi2 = {chi2:.1f}")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = plt.subplot(2, 4, 5)
    im5 = ax5.imshow(
        src_image, origin="lower", cmap="viridis",
        extent=(x_min, x_max, y_min, y_max), aspect="auto",
    )
    ax5.set_title("Reconstructed Source (Sersic + Shapelets)")
    ax5.set_xlabel("beta_x [arcsec]")
    ax5.set_ylabel("beta_y [arcsec]")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = plt.subplot(2, 4, 6)
    im6 = ax6.imshow(
        src_sersic_image, origin="lower", cmap="viridis",
        extent=(x_min, x_max, y_min, y_max), aspect="auto",
    )
    ax6.set_title("Source Sersic Component")
    ax6.set_xlabel("beta_x [arcsec]")
    ax6.set_ylabel("beta_y [arcsec]")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    ax7 = plt.subplot(2, 4, 7)
    im7 = ax7.imshow(
        src_shapelet_image, origin="lower", cmap="viridis",
        extent=(x_min, x_max, y_min, y_max), aspect="auto",
    )
    ax7.set_title("Source Shapelet Component")
    ax7.set_xlabel("beta_x [arcsec]")
    ax7.set_ylabel("beta_y [arcsec]")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    ax8 = plt.subplot(2, 4, 8)
    im8 = ax8.imshow(lens_model_image, origin="lower", cmap="magma")
    ax8.set_title("Lens-Light-Only Model (MGE)")
    lens_mge_amps = results["lens_mge_amps"]
    stats_text = (
        f"n_mge_lens = {len(lens_mge_amps)}\n"
        f"Valid pixels = {int(np.sum(~mask))}"
    )
    ax8.text(
        0.03, 0.97, stats_text, transform=ax8.transAxes,
        ha="left", va="top", fontsize=10, color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 4.0},
    )
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    overlay_critical_and_caustics(
        image_axes=[axes[0], axes[1], axes[2]],
        source_ax=axes[3],
        lens_mass=prob_model.phys_model,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sersic+Shapelet source + MGE lens-light joint inversion"
    )
    p.add_argument("--n-max", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--center-x", type=float, default=0.0)
    p.add_argument("--center-y", type=float, default=0.5)
    p.add_argument("--n-mge", type=int, default=10)
    p.add_argument("--sigma-min", type=float, default=0.01)
    p.add_argument("--sigma-max", type=float, default=3.0)
    p.add_argument("--R-sersic", type=float, default=0.3, help="Sersic half-light radius for source")
    p.add_argument("--n-sersic", type=float, default=1.0, help="Sersic index for source")
    p.add_argument("--e1-src", type=float, default=0.05, help="Sersic e1 for source")
    p.add_argument("--e2-src", type=float, default=0.05, help="Sersic e2 for source")
    p.add_argument("--src-npix", type=int, default=100)
    p.add_argument("--src-half-size", type=float, default=1.5)
    p.add_argument("--save-json", type=Path, default=Path("sersic_shapelet_joint_results.json"))
    p.add_argument("--save-figure", type=Path, default=Path("sersic_shapelet_joint.png"))
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

    n_shapelet_basis = (args.n_max + 1) * (args.n_max + 2) // 2
    # 1 for Sersic Ie + n_shapelet_basis for shapelet amplitudes
    n_total_basis = 1 + n_shapelet_basis
    print(
        f"Building joint model: n_max={args.n_max}, n_shapelet_basis={n_shapelet_basis}, "
        f"n_total_basis={n_total_basis}, "
        f"beta={args.beta}, n_mge_lens={args.n_mge}, "
        f"R_sersic={args.R_sersic}, n_sersic={args.n_sersic}"
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
        R_sersic=args.R_sersic,
        n_sersic=args.n_sersic,
        e1_src=args.e1_src,
        e2_src=args.e2_src,
    )

    print("Running joint inversion...")
    results = run_inversion(
        prob_model,
        n_basis=n_total_basis,
        n_max=args.n_max,
        beta=args.beta,
        center_x=args.center_x,
        center_y=args.center_y,
        R_sersic=args.R_sersic,
        n_sersic=args.n_sersic,
        e1_src=args.e1_src,
        e2_src=args.e2_src,
        src_npix=args.src_npix,
        src_half_size=args.src_half_size,
    )

    print(f"Saving figure to {args.save_figure}...")
    visualize(data_dict, results, output_path=args.save_figure)

    payload = {
        "n_max": args.n_max,
        "n_shapelet_basis": n_shapelet_basis,
        "n_total_basis": n_total_basis,
        "beta": args.beta,
        "center_x": args.center_x,
        "center_y": args.center_y,
        "n_mge_lens": args.n_mge,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "R_sersic": args.R_sersic,
        "n_sersic": args.n_sersic,
        "e1_src": args.e1_src,
        "e2_src": args.e2_src,
        "log_like": float(results["log_like"]),
        "n_data": int(np.sum(~np.asarray(mask, dtype=bool))),
        "figure_path": str(args.save_figure),
        "sersic_Ie": float(results["sersic_Ie"]),
        "lens_mge_amps": results["lens_mge_amps"].tolist(),
    }
    args.save_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("\nSummary:")
    print(json.dumps({k: v for k, v in payload.items() if k not in ["lens_mge_amps"]}, indent=2))


if __name__ == "__main__":
    main()
