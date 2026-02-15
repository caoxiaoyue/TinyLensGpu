"""
Demo: pixelized source reconstruction with selectable matrix/operator backend.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized import (
    IrregularGridConfig,
    MappingConfig,
    PixelizedSourceConfig,
    PixelizedSourceModel,
    RegularizationConfig,
    SolverConfig,
)
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.utils.geometry import phi_q2_ellipticity
from TinyLensGpu.visualizer import _plot_irregular_source_voronoi


def setup_pixelized_model(
    data_dict,
    *,
    backend: str = "matrix",
    nonnegative: bool = False,
    operator_cache_policy: str = "safe",
    scheme: str = "irregular_gp_exp",
    reg_sparse_k_neighbors: int = 16,
):
    """Setup the pixelized source model."""
    e1_l, e2_l = phi_q2_ellipticity(90 * np.pi / 180, 0.9)
    sie = SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0)

    pix_config = PixelizedSourceConfig(
        grid=IrregularGridConfig(
            n_source_points=1500,
            mesh_alpha=1.5,
            mesh_blur_sigma=0.0,
            mesh_method="random",
            mesh_seed=42,
        ),
        mapping=MappingConfig(
            k_neighbors=5,
            interp_kernel="wendland_c4",
            radius_scale=1.5,
        ),
        regularization=RegularizationConfig(
            scheme=scheme,
            sparse_k_neighbors=reg_sparse_k_neighbors,
        ),
        solver=SolverConfig(
            inversion_backend=backend,
            nonnegative=nonnegative,
            operator_cache_policy=operator_cache_policy,
        ),
    )
    pix_src_model = PixelizedSourceModel(config=pix_config, reg_scale=0.05, reg_coefficient=1.0)
    phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src_model], lens_light=[])

    sim_config = SimulatorConfig(
        dpix=data_dict["dpix"],
        npix=data_dict["noisy_image"].shape[0],
        psf_kernel=data_dict["psf_kernel"],
        mask=data_dict["mask"],
    )

    return PixelizedImageProbModel(
        image_data=data_dict["noisy_image"],
        noise_map=data_dict["noise_map"],
        sim_config=sim_config,
        phys_model=phys_model,
    )


def reconstruct_source(prob_model):
    """Reconstruct the source light distribution."""
    log_ev = float(np.asarray(prob_model()))
    data_vector = prob_model.image_data[~prob_model.mask]
    noise_variance = prob_model.noise_map[~prob_model.mask] ** 2
    reg_scale = prob_model.pix_src_model.reg_scale.value
    reg_coefficient = prob_model.pix_src_model.reg_coefficient.value

    source_intensities, source_mesh_beta, model_image, _ = prob_model.simulator.reconstruct_source(
        data_vector=data_vector,
        noise_variance=noise_variance,
        reg_scale=reg_scale,
        reg_coefficient=reg_coefficient,
        return_2d=True,
    )

    return {
        "log_evidence": log_ev,
        "source_intensities": np.array(source_intensities),
        "source_mesh_beta": np.array(source_mesh_beta),
        "model_image": np.array(model_image),
    }


def visualize_results(data_dict, results, *, output_path: Path):
    """Visualize the reconstruction results."""
    noisy_image = data_dict["noisy_image"]
    noise_map = data_dict["noise_map"]
    mask = data_dict["mask"]
    model_image = results["model_image"]
    source_intensities = results["source_intensities"]
    source_mesh_beta = results["source_mesh_beta"]
    log_evidence = results["log_evidence"]

    fig = plt.figure(figsize=(18, 10))
    ax1 = plt.subplot(2, 3, 1)
    im1 = plt.imshow(noisy_image * (~mask).astype(float), origin="lower", cmap="viridis")
    plt.title("Observed Noisy Image")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 3, 2)
    im2 = plt.imshow(model_image, origin="lower", cmap="viridis")
    plt.title(f"Model Image\nLog Evidence = {log_evidence:.2f}")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    residual_image = np.zeros_like(noisy_image)
    residual_image[~mask] = noisy_image[~mask] - model_image[~mask]
    vmax_res = np.max(np.abs(residual_image))
    ax3 = plt.subplot(2, 3, 3)
    im3 = plt.imshow(residual_image, origin="lower", cmap="RdBu_r", vmin=-vmax_res, vmax=vmax_res)
    plt.title("Residual")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    normalized_residual = np.zeros_like(noisy_image)
    normalized_residual[~mask] = residual_image[~mask] / noise_map[~mask]
    ax4 = plt.subplot(2, 3, 4)
    im4 = plt.imshow(normalized_residual, origin="lower", cmap="RdBu_r", vmin=-3.0, vmax=3.0)
    plt.title("Normalized Residual")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = plt.subplot(2, 3, 5)
    _plot_irregular_source_voronoi(ax5, source_mesh_beta, source_intensities, cmap="viridis")
    ax5.set_title("Source Reconstruction")

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    stats_text = (
        f"Log Evidence: {log_evidence:.2f}\n"
        f"Source Points: {len(source_intensities)}\n"
        f"Valid Pixels: {np.sum(~mask)}\n"
        f"Chi2: {np.sum(((noisy_image[~mask]-model_image[~mask])/noise_map[~mask])**2):.2f}"
    )
    ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment="center", family="monospace")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def build_cli_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Pixelized source demo with matrix/operator backend")
    parser.add_argument("--backend", choices=["matrix", "operator"], default="matrix")
    parser.add_argument("--nonnegative", action="store_true")
    parser.add_argument("--operator-cache-policy", choices=["off", "safe", "unsafe_static"], default="safe")
    parser.add_argument(
        "--scheme",
        choices=[
            "irregular_gp_exp",
            "irregular_gp_gauss",
            "irregular_gp_matern32",
            "irregular_gp_matern52",
            "irregular_knn_exp",
            "irregular_knn_gauss",
            "irregular_knn_matern32",
            "irregular_knn_matern52",
        ],
        default="irregular_gp_exp",
    )
    parser.add_argument("--reg-sparse-k-neighbors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-json", type=Path, default=Path("demo_pix_src_results.json"))
    parser.add_argument(
        "--save-figure",
        type=Path,
        default=Path("pixelized_source_reconstruction.png"),
    )
    return parser


def main():
    # --- Step 1: Parse CLI arguments ---
    parser = build_cli_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)

    # --- Step 2: Load observational data ---
    print("Loading data from data/ directory...")
    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path='data/image.fits',
        noise_path='data/noise.fits',
        psf_path='data/psf.fits',
        mask_path='data/mask.fits'
    )
    
    data_dict = {
        "noisy_image": image_data,
        "noise_map": noise_map,
        "psf_kernel": psf_kernel,
        "mask": mask,
        "dpix": 0.074, # Fixed value from simulation
    }

    # --- Step 3: Setup the irregular pixelized source model ---
    print(f"Setting up model with backend: {args.backend}...")
    prob_model = setup_pixelized_model(
        data_dict,
        backend=args.backend,
        nonnegative=args.nonnegative,
        operator_cache_policy=args.operator_cache_policy,
        scheme=args.scheme,
        reg_sparse_k_neighbors=args.reg_sparse_k_neighbors,
    )

    # --- Step 4: Perform source reconstruction (Inversion) ---
    print("Reconstructing source...")
    results = reconstruct_source(prob_model)

    # --- Step 5: Visualize and save results ---
    print(f"Visualizing results to {args.save_figure}...")
    visualize_results(data_dict, results, output_path=args.save_figure)

    payload = {
        "backend": args.backend,
        "nonnegative": bool(args.nonnegative),
        "log_evidence": float(results["log_evidence"]),
        "n_source": int(results["source_intensities"].shape[0]),
        "n_data": int(np.sum(~data_dict["mask"])),
        "figure_path": str(args.save_figure),
        "operator_cache_policy": args.operator_cache_policy,
        "scheme": args.scheme,
        "reg_sparse_k_neighbors": int(args.reg_sparse_k_neighbors),
    }
    args.save_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("\nSummary results:")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
