"""
Pixelized source inversion demo — operator (matrix-free) backend.

Lens mass is fixed to the simulation truth values.
Source is reconstructed on a 40x40 pixel grid with first-order regularization,
using matrix-free operators and PCG instead of explicit Cholesky.

Produces a 4-panel figure:
  1. Lensed arc (data)
  2. Pix-src model image
  3. Normalized residual (sigma units)
  4. Source reconstruction
"""

import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.chdir(Path(__file__).parent)

import numpy as np
import matplotlib.pyplot as plt

from TinyLensGpu.PhysicalModel import PhysicalModel, SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.utils import load_lens_data

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
DPIX     = 0.05   # arcsec / pixel
LAMBDA_LIST = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

# ------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------ #
image_data, noise_map, psf_kernel, mask = load_lens_data(
    image_path="data/image.fits",
    noise_path="data/noise.fits",
    psf_path="data/psf.fits",
    mask_path="data/mask.fits",
)

# ------------------------------------------------------------------ #
# Physical model (lens mass fixed to truth)
# ------------------------------------------------------------------ #
sie = SIE(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)

results_summary = []

for LAMBDA in LAMBDA_LIST:
    print(f"\n" + "="*60)
    print(f"Running pixelized source inversion (λ={LAMBDA}) …")

    pix_src = PixelizedSourceModel(
        nx=40,
        ny=40,
        regularization_type="second-order",
        lambda_reg=max(LAMBDA, 1e-8),
    )

    phys_model = PhysicalModel(
        lens_mass=[sie],
        source_light=[pix_src],
        lens_light=[],
    )

    prob_model = PixelizedImageProbModelOperator(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        phys_model=phys_model,
        mask=mask,
    )

    # ------------------------------------------------------------------ #
    # Solve source via PCG (inside forward_model)
    # ------------------------------------------------------------------ #
    model_image, source_pixels = prob_model.forward_model(return_source=True)
    source_pixels_np = np.array(source_pixels)
    model_image_np   = np.array(model_image)
    data_np          = np.array(image_data)
    noise_np         = np.array(noise_map)
    mask_np          = np.array(mask)

    resid_norm = (data_np - model_image_np) / noise_np
    chi2_nu    = float(np.sum(resid_norm[~mask_np] ** 2) / (~mask_np).sum())

    log_ev = prob_model.likelihood()
    print(f"  log evidence = {log_ev:.2f}")

    results_summary.append({
        'lambda': LAMBDA,
        'log_ev': log_ev,
        'chi2_nu': chi2_nu
    })

    source_image = source_pixels_np.reshape(pix_src.ny, pix_src.nx)

    # ------------------------------------------------------------------ #
    # Source bbox for display
    # ------------------------------------------------------------------ #
    _, _, bx, by = prob_model.sim_obj._get_beta_sub_and_seed()
    xmin, xmax, ymin, ymax = prob_model.sim_obj._infer_and_fix_bbox(bx, by)
    ext_s = [float(xmin), float(xmax), float(ymin), float(ymax)]

    # ------------------------------------------------------------------ #
    # 4-panel figure
    # ------------------------------------------------------------------ #
    npix  = data_np.shape[0]
    ext_i = [-npix * DPIX / 2, npix * DPIX / 2, -npix * DPIX / 2, npix * DPIX / 2]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    # Panel 1: data
    vmax = np.nanpercentile(data_np[~mask_np], 99.5)
    im0 = axes[0].imshow(data_np, origin="lower", extent=ext_i, cmap="viridis",
                         vmin=0, vmax=vmax)
    axes[0].set_title("Lensed arc (data)", fontsize=11)
    axes[0].set_xlabel("arcsec"); axes[0].set_ylabel("arcsec")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: model image
    im1 = axes[1].imshow(model_image_np, origin="lower", extent=ext_i, cmap="viridis",
                         vmin=0, vmax=vmax)
    axes[1].set_title("Pix-src model image\n(operator)", fontsize=11)
    axes[1].set_xlabel("arcsec")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: normalized residual
    resid_display = np.where(mask_np, np.nan, resid_norm)
    im2 = axes[2].imshow(resid_display, origin="lower", extent=ext_i,
                         cmap="RdBu_r", vmin=-5, vmax=5)
    axes[2].set_title(f"Norm. residual (σ)\nχ²/ν = {chi2_nu:.3f}", fontsize=11)
    axes[2].set_xlabel("arcsec")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: source reconstruction
    im3 = axes[3].imshow(source_image, origin="lower", extent=ext_s, cmap="viridis")
    reg_label = f"1st-order, λ={LAMBDA}"
    title_suffix = f"\n({reg_label})\nlog Z = {log_ev:.2f}"
    axes[3].set_title(f"Source reconstruction{title_suffix}", fontsize=11)
    axes[3].set_xlabel("arcsec"); axes[3].set_ylabel("arcsec")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Pixelized source inversion — operator backend (SIE fixed, λ={LAMBDA})",
        fontsize=12,
    )
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    out_name = f"output/demo_inversion_operator_lambda{LAMBDA}.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_name}")
    plt.close(fig)

# ------------------------------------------------------------------ #
# Summary Table
# ------------------------------------------------------------------ #
print("\n" + "="*60)
print(f"{'LAMBDA':>10s} | {'log Evidence':>15s} | {'chi2/nu':>10s}")
print("-" * 60)
for res in results_summary:
    print(f"{res['lambda']:10.1e} | {res['log_ev']:15.2f} | {res['chi2_nu']:10.3f}")
print("="*60)
