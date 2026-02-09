"""Demo: pixelized source reconstruction with selectable matrix/operator backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt

from TinyLensGpu.ForwardSimulation import SimulatorConfig, LensSimulator, make_grid_2d
from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse, GaussianEllipse
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized import PixelizedSourceModel
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.geometry import phi_q2_ellipticity
from TinyLensGpu.visualizer import _plot_irregular_source_voronoi


def simulate_lensing_data():
    e1_l, e2_l = phi_q2_ellipticity(90 * np.pi / 180, 0.9)
    phy_model = PhysicalModel(
        lens_mass=[SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0)],
        source_light=[
            SersicEllipse(
                R_sersic=0.3,
                n_sersic=1.0,
                e1=0.05,
                e2=0.05,
                center_x=0.0,
                center_y=0.3,
                Ie=1.0,
            )
        ],
        lens_light=[],
    )

    npix = 200
    image_size = 10.0
    dpix = image_size / npix

    x_psf, y_psf = make_grid_2d(21, dpix)
    psf_kernel = GaussianEllipse(
        flux=1.0,
        sigma=0.05,
        e1=0.0,
        e2=0.0,
        center_x=0.0,
        center_y=0.0,
    ).light(x=x_psf, y=y_psf)
    psf_kernel /= psf_kernel.sum()
    psf_kernel = np.asarray(psf_kernel)

    sim_config = SimulatorConfig(dpix=dpix, npix=npix, psf_kernel=psf_kernel, nsub=16)
    sim_obj = LensSimulator(phy_model, sim_config)
    img_2d = sim_obj.simulate()

    def mock_lens(ideal_image, back_rms, exp_time):
        noise_map = np.sqrt(ideal_image / exp_time + back_rms**2)
        noisy_image = ideal_image + np.random.normal(0, noise_map)
        return noisy_image, noise_map

    noisy_image, noise_map = mock_lens(img_2d, 0.1, 300)

    xgrid_image, ygrid_image = make_grid_2d(npix, dpix)
    rgrid_image = np.sqrt(xgrid_image**2 + ygrid_image**2)
    mask = rgrid_image > 2.7

    return {
        "noisy_image": noisy_image,
        "noise_map": noise_map,
        "psf_kernel": psf_kernel,
        "mask": mask,
        "dpix": dpix,
    }


def setup_pixelized_model(
    data_dict,
    *,
    backend: str = "matrix",
    nonnegative: bool = False,
    cg_tol: float = 1e-4,
    cg_maxiter: int = 120,
    slq_probes: int = 32,
    slq_steps: int = 60,
    evidence_mode: str = "accurate",
    operator_cache_policy: str = "safe",
    reg_operator_mode: str = "dense_gp",
    reg_sparse_k_neighbors: int = 16,
):
    e1_l, e2_l = phi_q2_ellipticity(90 * np.pi / 180, 0.9)
    sie = SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0)

    pix_src_model = PixelizedSourceModel(
        reg_scale=0.05,
        reg_coefficient=1.0,
        reg_type="exp",
        n_source_points=1500,
        mesh_alpha=1.5,
        mesh_blur_sigma=0.0,
        mesh_method="random",
        mesh_seed=42,
        k_neighbors=5,
        interp_kernel="wendland_c4",
        radius_scale=1.5,
        reg_operator_mode=reg_operator_mode,
        reg_sparse_k_neighbors=reg_sparse_k_neighbors,
    )
    phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src_model], lens_light=[])

    return PixelizedImageProbModel(
        image_data=data_dict["noisy_image"],
        noise_map=data_dict["noise_map"],
        psf_kernel=data_dict["psf_kernel"],
        dpix=data_dict["dpix"],
        phys_model=phys_model,
        mask=data_dict["mask"],
        inversion_backend=backend,
        nonnegative=nonnegative,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        slq_probes=slq_probes,
        slq_steps=slq_steps,
        evidence_mode=evidence_mode,
        operator_cache_policy=operator_cache_policy,
    )


def reconstruct_source(prob_model):
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
        inversion_backend=prob_model.inversion_backend,
        nonnegative=prob_model.nonnegative,
        cg_tol=prob_model.cg_tol,
        cg_maxiter=prob_model.cg_maxiter,
        slq_seed=prob_model.slq_seed,
        slq_probes=prob_model.slq_probes,
        slq_steps=prob_model.slq_steps,
        evidence_mode=prob_model.evidence_mode,
        operator_cache_policy=prob_model.operator_cache_policy,
        nnls_maxiter=prob_model.nnls_maxiter,
        nnls_tol=prob_model.nnls_tol,
        nnls_lipschitz_iters=prob_model.nnls_lipschitz_iters,
    )

    return {
        "log_evidence": log_ev,
        "source_intensities": np.array(source_intensities),
        "source_mesh_beta": np.array(source_mesh_beta),
        "model_image": np.array(model_image),
    }


def visualize_results(data_dict, results, *, output_path: Path):
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


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pixelized source demo with matrix/operator backend")
    parser.add_argument("--backend", choices=["matrix", "operator"], default="matrix")
    parser.add_argument("--nonnegative", action="store_true")
    parser.add_argument("--cg_tol", type=float, default=1e-4)
    parser.add_argument("--cg_maxiter", type=int, default=120)
    parser.add_argument("--slq_probes", type=int, default=32)
    parser.add_argument("--slq_steps", type=int, default=60)
    parser.add_argument("--evidence-mode", choices=["fast", "accurate"], default="accurate")
    parser.add_argument("--operator-cache-policy", choices=["off", "safe", "unsafe_static"], default="safe")
    parser.add_argument("--reg-operator-mode", choices=["dense_gp", "sparse_knn"], default="dense_gp")
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
    parser = build_cli_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_dict = simulate_lensing_data()
    prob_model = setup_pixelized_model(
        data_dict,
        backend=args.backend,
        nonnegative=args.nonnegative,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
        slq_probes=args.slq_probes,
        slq_steps=args.slq_steps,
        evidence_mode=args.evidence_mode,
        operator_cache_policy=args.operator_cache_policy,
        reg_operator_mode=args.reg_operator_mode,
        reg_sparse_k_neighbors=args.reg_sparse_k_neighbors,
    )
    results = reconstruct_source(prob_model)
    visualize_results(data_dict, results, output_path=args.save_figure)

    payload = {
        "backend": args.backend,
        "nonnegative": bool(args.nonnegative),
        "log_evidence": float(results["log_evidence"]),
        "n_source": int(results["source_intensities"].shape[0]),
        "n_data": int(np.sum(~data_dict["mask"])),
        "figure_path": str(args.save_figure),
        "evidence_mode": args.evidence_mode,
        "operator_cache_policy": args.operator_cache_policy,
        "reg_operator_mode": args.reg_operator_mode,
        "reg_sparse_k_neighbors": int(args.reg_sparse_k_neighbors),
    }
    args.save_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
