"""Single-step joint inversion for rectangular pixelized source + MGE lens light.

This script performs one semi-linear inversion solve with:

- Source light on a rectangular bilinear pixelized grid.
- Lens light as an MGE basis.
- Fixed SIE + external shear mass model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
from TinyLensGpu.PhysicalModel import (
    GaussianEllipse,
    PhysicalModel,
    PixelizedSourceConfig,
    PixelizedSourceModel,
    RectangularGridConfig,
    RegularizationConfig,
    SIE,
    Shear,
    SolverConfig,
)
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


def build_mass_model() -> list[SIE | Shear]:
    """Build fixed mass components matching the simulation setup."""
    e1_lens, e2_lens = phi_q2_ellipticity(90.0 * np.pi / 180.0, 0.9)
    return [
        SIE(theta_E=1.5, e1=e1_lens, e2=e2_lens, center_x=0.0, center_y=0.0),
        Shear(gamma1=0.05, gamma2=0.05),
    ]


def build_lens_light_mge(*, n_components: int, sigma_min: float, sigma_max: float) -> list[GaussianEllipse]:
    """Build MGE lens-light basis functions similar to lens_src_mge/run_model.py."""
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")
    if sigma_min <= 0.0 or sigma_max <= 0.0 or sigma_max <= sigma_min:
        raise ValueError(
            "Invalid sigma range: require 0 < sigma_min < sigma_max, "
            f"got sigma_min={sigma_min}, sigma_max={sigma_max}."
        )

    e1_lens, e2_lens = phi_q2_ellipticity(90.0 * np.pi / 180.0, 0.9)
    sigma_list = 10.0 ** np.linspace(np.log10(sigma_min), np.log10(sigma_max), int(n_components))
    return [
        GaussianEllipse(
            sigma=float(sigma),
            center_x=0.0,
            center_y=0.0,
            e1=e1_lens,
            e2=e2_lens,
            flux=1.0,
        )
        for sigma in sigma_list
    ]


def setup_joint_model(
    data_dict: dict[str, np.ndarray | float],
    *,
    backend: str,
    nonnegative: bool,
    operator_cache_policy: str,
    source_grid_nx: int,
    source_grid_ny: int,
    source_grid_margin_frac: float,
    scheme: str,
    n_lens_gaussians: int,
    lens_sigma_min: float,
    lens_sigma_max: float,
    lens_light_ridge: float,
) -> PixelizedImageProbModel:
    """Construct rectangular-source + MGE-lens-light joint inversion model."""
    pix_config = PixelizedSourceConfig(
        grid=RectangularGridConfig(
            nx=int(source_grid_nx),
            ny=int(source_grid_ny),
            margin_frac=float(source_grid_margin_frac),
        ),
        regularization=RegularizationConfig(
            scheme=scheme,
        ),
        solver=SolverConfig(
            inversion_backend=backend,
            include_lens_light=True,
            nonnegative=nonnegative,
            lens_light_ridge=float(lens_light_ridge),
            operator_cache_policy=operator_cache_policy,
        ),
    )
    pix_src_model = PixelizedSourceModel(config=pix_config, reg_scale=0.05, reg_coefficient=1.0)

    phys_model = PhysicalModel(
        lens_mass=build_mass_model(),
        source_light=[pix_src_model],
        lens_light=build_lens_light_mge(
            n_components=n_lens_gaussians,
            sigma_min=lens_sigma_min,
            sigma_max=lens_sigma_max,
        ),
    )

    sim_config = SimulatorConfig(
        dpix=float(data_dict["dpix"]),
        npix=int(np.asarray(data_dict["noisy_image"]).shape[0]),
        psf_kernel=np.asarray(data_dict["psf_kernel"]),
        mask=np.asarray(data_dict["mask"]),
    )

    return PixelizedImageProbModel(
        image_data=np.asarray(data_dict["noisy_image"]),
        noise_map=np.asarray(data_dict["noise_map"]),
        sim_config=sim_config,
        phys_model=phys_model,
    )


def reconstruct_joint_components(prob_model: PixelizedImageProbModel) -> dict[str, np.ndarray | float | tuple[int, int] | tuple[float, float, float, float]]:
    """Solve the joint inversion and prepare source/lens-light diagnostics."""
    log_evidence = float(np.asarray(prob_model()))
    mask = np.asarray(prob_model.mask)

    data_vector = prob_model.image_data[~prob_model.mask]
    noise_variance = prob_model.noise_map[~prob_model.mask] ** 2

    source_intensities, lens_coefficients, _, model_image, inverter = (
        prob_model.simulator.reconstruct_source_and_lens_light(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=prob_model.pix_src_model.reg_scale.value,
            reg_coefficient=prob_model.pix_src_model.reg_coefficient.value,
            return_2d=True,
        )
    )

    source_intensities = np.asarray(source_intensities)
    lens_coefficients = np.asarray(lens_coefficients)
    model_image = np.asarray(model_image)

    if prob_model.simulator.source_grid_shape is None:
        raise RuntimeError("Rectangular source grid shape is unavailable.")
    if prob_model.simulator.source_grid_bounds is None:
        raise RuntimeError("Rectangular source grid bounds are unavailable.")

    ny, nx = prob_model.simulator.source_grid_shape
    source_image_2d = source_intensities.reshape(ny, nx)
    source_grid_bounds = tuple(float(v) for v in prob_model.simulator.source_grid_bounds)

    x_lens_only = np.concatenate([np.zeros_like(source_intensities), lens_coefficients], axis=0)
    lens_model_data = np.asarray(inverter.model_predict(x_lens_only))
    lens_model_image = np.zeros_like(model_image)
    lens_model_image[~mask] = lens_model_data

    data_image = np.asarray(prob_model.image_data)
    noise_map = np.asarray(prob_model.noise_map)
    chi2 = float(np.sum(((data_image[~mask] - model_image[~mask]) / noise_map[~mask]) ** 2))

    return {
        "log_evidence": log_evidence,
        "chi2": chi2,
        "source_intensities": source_intensities,
        "source_image_2d": source_image_2d,
        "source_grid_shape": (int(ny), int(nx)),
        "source_grid_bounds": source_grid_bounds,
        "lens_coefficients": lens_coefficients,
        "model_image": model_image,
        "lens_model_image": lens_model_image,
    }


def visualize_results(
    data_dict: dict[str, np.ndarray | float],
    results: dict[str, np.ndarray | float | tuple[int, int] | tuple[float, float, float, float]],
    *,
    output_path: Path,
) -> None:
    """Visualize observed/model/residual/source/lens-light maps."""
    noisy_image = np.asarray(data_dict["noisy_image"])
    noise_map = np.asarray(data_dict["noise_map"])
    mask = np.asarray(data_dict["mask"])  # True means masked out.

    model_image = np.asarray(results["model_image"])
    lens_model_image = np.asarray(results["lens_model_image"])
    source_image_2d = np.asarray(results["source_image_2d"])
    lens_coefficients = np.asarray(results["lens_coefficients"])
    log_evidence = float(results["log_evidence"])
    chi2 = float(results["chi2"])
    x_min, x_max, y_min, y_max = results["source_grid_bounds"]

    residual = np.zeros_like(noisy_image)
    residual[~mask] = noisy_image[~mask] - model_image[~mask]

    norm_residual = np.zeros_like(noisy_image)
    norm_residual[~mask] = residual[~mask] / noise_map[~mask]

    fig = plt.figure(figsize=(18, 10))

    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(noisy_image * (~mask).astype(float), origin="lower", cmap="viridis")
    ax1.set_title("Observed Noisy Image")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(model_image, origin="lower", cmap="viridis")
    ax2.set_title(f"Joint Model Image\nLog Evidence = {log_evidence:.2f}")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 3, 3)
    vmax_res = np.max(np.abs(residual[~mask]))
    im3 = ax3.imshow(residual, origin="lower", cmap="RdBu_r", vmin=-vmax_res, vmax=vmax_res)
    ax3.set_title("Residual")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(norm_residual, origin="lower", cmap="RdBu_r", vmin=-3.0, vmax=3.0)
    ax4.set_title("Normalized Residual")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(
        source_image_2d,
        origin="lower",
        cmap="viridis",
        extent=(x_min, x_max, y_min, y_max),
        aspect="auto",
    )
    ax5.set_title("Source Reconstruction (Rectangular)")
    ax5.set_xlabel("beta_x")
    ax5.set_ylabel("beta_y")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(lens_model_image, origin="lower", cmap="magma")
    ax6.set_title("Lens-Light-Only Model (MGE)")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    ny, nx = results["source_grid_shape"]
    stats_text = (
        f"Chi2 = {chi2:.2f}\n"
        f"Valid Pixels = {np.sum(~mask)}\n"
        f"Source Grid = {nx} x {ny}\n"
        f"Lens MGE Components = {lens_coefficients.shape[0]}"
    )
    ax6.text(
        0.03,
        0.97,
        stats_text,
        transform=ax6.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 4.0},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def build_cli_parser() -> argparse.ArgumentParser:
    """Build command-line interface for one-shot joint inversion."""
    parser = argparse.ArgumentParser(
        description="Rectangular-grid pixelized source + MGE lens-light joint inversion"
    )
    parser.add_argument("--backend", choices=["matrix", "operator"], default="operator")
    parser.add_argument("--nonnegative", action="store_true")
    parser.add_argument("--operator-cache-policy", choices=["off", "safe", "unsafe_static"], default="safe")
    parser.add_argument("--source-grid-nx", type=int, default=64)
    parser.add_argument("--source-grid-ny", type=int, default=64)
    parser.add_argument("--source-grid-margin-frac", type=float, default=0.10)
    parser.add_argument(
        "--scheme",
        choices=["rectangular_zero", "rectangular_first", "rectangular_second"],
        default="rectangular_first",
    )
    parser.add_argument("--n-lens-gaussians", type=int, default=10)
    parser.add_argument("--lens-sigma-min", type=float, default=1.0e-2)
    parser.add_argument("--lens-sigma-max", type=float, default=3.0)
    parser.add_argument(
        "--lens-light-ridge",
        type=float,
        default=1.0,
        help=(
            "Ridge regularization for lens-light amplitudes. "
            "A stronger default keeps matrix/operator solves numerically consistent "
            "for the 10-component MGE basis."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-json", type=Path, default=Path("demo_joint_rect_results.json"))
    parser.add_argument("--save-figure", type=Path, default=Path("joint_rect_reconstruction.png"))
    return parser


def main() -> None:
    """Execute one-shot joint inversion and save figure + JSON diagnostics."""
    args = build_cli_parser().parse_args()
    np.random.seed(args.seed)

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

    print(f"Setting up rectangular-grid joint model with backend={args.backend}...")
    prob_model = setup_joint_model(
        data_dict,
        backend=args.backend,
        nonnegative=args.nonnegative,
        operator_cache_policy=args.operator_cache_policy,
        source_grid_nx=args.source_grid_nx,
        source_grid_ny=args.source_grid_ny,
        source_grid_margin_frac=args.source_grid_margin_frac,
        scheme=args.scheme,
        n_lens_gaussians=args.n_lens_gaussians,
        lens_sigma_min=args.lens_sigma_min,
        lens_sigma_max=args.lens_sigma_max,
        lens_light_ridge=args.lens_light_ridge,
    )

    print("Running joint inversion...")
    results = reconstruct_joint_components(prob_model)

    effective_lens_light_ridge = float(args.lens_light_ridge)
    is_matrix_unstable = (
        args.backend == "matrix"
        and not args.nonnegative
        and (
            (not np.isfinite(float(results["log_evidence"])))
            or (float(np.max(np.abs(np.asarray(results["lens_coefficients"])))) > 50.0)
        )
    )
    if is_matrix_unstable and effective_lens_light_ridge < 1.0:
        effective_lens_light_ridge = 1.0
        print(
            "Detected unstable matrix joint inversion (ill-conditioned lens-light amplitudes). "
            f"Re-running with stabilized lens_light_ridge={effective_lens_light_ridge:.1f}."
        )
        prob_model = setup_joint_model(
            data_dict,
            backend=args.backend,
            nonnegative=args.nonnegative,
            operator_cache_policy=args.operator_cache_policy,
            source_grid_nx=args.source_grid_nx,
            source_grid_ny=args.source_grid_ny,
            source_grid_margin_frac=args.source_grid_margin_frac,
            scheme=args.scheme,
            n_lens_gaussians=args.n_lens_gaussians,
            lens_sigma_min=args.lens_sigma_min,
            lens_sigma_max=args.lens_sigma_max,
            lens_light_ridge=effective_lens_light_ridge,
        )
        results = reconstruct_joint_components(prob_model)

    print(f"Saving visualization to {args.save_figure}...")
    visualize_results(data_dict, results, output_path=args.save_figure)

    payload = {
        "backend": args.backend,
        "nonnegative": bool(args.nonnegative),
        "operator_cache_policy": args.operator_cache_policy,
        "source_grid_nx": int(args.source_grid_nx),
        "source_grid_ny": int(args.source_grid_ny),
        "source_grid_margin_frac": float(args.source_grid_margin_frac),
        "scheme": args.scheme,
        "n_lens_gaussians": int(args.n_lens_gaussians),
        "lens_sigma_min": float(args.lens_sigma_min),
        "lens_sigma_max": float(args.lens_sigma_max),
        "lens_light_ridge": float(effective_lens_light_ridge),
        "log_evidence": float(results["log_evidence"]),
        "chi2": float(results["chi2"]),
        "n_data": int(np.sum(~np.asarray(data_dict["mask"]))),
        "n_source": int(np.asarray(results["source_intensities"]).shape[0]),
        "lens_light_coefficients": np.asarray(results["lens_coefficients"]).astype(float).tolist(),
        "figure_path": str(args.save_figure),
    }
    if not np.isclose(effective_lens_light_ridge, float(args.lens_light_ridge)):
        payload["requested_lens_light_ridge"] = float(args.lens_light_ridge)
        payload["matrix_auto_stabilized"] = True
    args.save_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("\nSummary results:")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
