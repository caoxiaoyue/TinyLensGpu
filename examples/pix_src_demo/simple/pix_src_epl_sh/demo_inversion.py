"""
Pixelized source inversion demo.

Lens mass is fixed to the simulation truth values.
Source is reconstructed on a 40x40 pixel grid with first-order regularization.

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
from TinyLensGpu.example_support import (
    draw_image_panel,
    require_valid_reconstruction,
)
import matplotlib.pyplot as plt

from TinyLensGpu.PhysicalModel import PhysicalModel, EPL, Shear
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.visualizer import overlay_critical_and_caustics

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
epl = EPL(theta_E=1.0, gamma=2.2, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
shear = Shear(gamma1=0.05, gamma2=0.05)

results_summary = []

for LAMBDA in LAMBDA_LIST:
    print(f"\n" + "="*60)
    print(f"Running pixelized source inversion (λ={LAMBDA}) …")
    
    pix_src = PixelizedSourceModel(n=40,
        regularization_type="first-order",
        log_lambda_reg=np.log(max(LAMBDA, 1e-8)),
    )

    phys_model = PhysicalModel(
        lens_mass=[epl, shear],
        source_light=[pix_src],
        lens_light=[],
    )

    prob_model = PixelizedImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        phys_model=phys_model,
        mask=mask,
    )

    # Solve once for the evidence, source, image and source grid.
    result, diagnostics = prob_model.evaluate_with_diagnostics()
    require_valid_reconstruction(result, diagnostics, context="plotting reconstruction")
    source_pixels = np.asarray(result.source_pixels)
    log_ev = float(result.log_evidence)
    model_image = np.asarray(result.model_image)
    source_bbox = result.source_bbox
    print(f"  log evidence = {log_ev:.2f}")

    resid_norm = np.asarray(diagnostics.normalized_residuals)
    # Retain the demo's chi2 per active pixel (not effective residual DOF).
    chi2_nu    = diagnostics.chi2 / int((~mask).sum())
    print(f"  χ²/N_pix = {chi2_nu:.3f}")

    results_summary.append({
        'lambda': LAMBDA,
        'log_ev': log_ev,
        'chi2_nu': chi2_nu
    })

    source_image = source_pixels.reshape(pix_src.n, pix_src.n)

    # ------------------------------------------------------------------ #
    # 4-panel figure
    # ------------------------------------------------------------------ #
    npix  = image_data.shape[0]
    ext_i = [-npix * DPIX / 2, npix * DPIX / 2, -npix * DPIX / 2, npix * DPIX / 2]
    ext_s = [float(source_bbox[0]), float(source_bbox[1]), float(source_bbox[2]), float(source_bbox[3])]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    # Panel 1: data
    vmax = np.nanpercentile(image_data[~mask], 99.5)
    im0 = draw_image_panel(
        axes[0],
        image_data,
        extent=ext_i,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        title="Lensed arc (data)",
        title_size=11,
        xlabel=None,
    )
    axes[0].set_xlabel("arcsec"); axes[0].set_ylabel("arcsec")

    # Panel 2: model image
    im1 = draw_image_panel(
        axes[1],
        model_image,
        extent=ext_i,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        title="Pix-src model image",
        title_size=11,
        xlabel=None,
    )
    axes[1].set_xlabel("arcsec")

    # Panel 3: normalized residual
    resid_display = np.where(mask, np.nan, resid_norm)
    im2 = draw_image_panel(
        axes[2],
        resid_display,
        extent=ext_i,
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        title=f"Norm. residual (σ)\nχ²/N_pix = {chi2_nu:.3f}",
        title_size=11,
        xlabel=None,
    )
    axes[2].set_xlabel("arcsec")

    # Panel 4: source reconstruction
    reg_label = f"1st-order, λ={LAMBDA}"
    title_suffix = f"\n({reg_label})\nlog Z = {log_ev:.2f}"
    im3 = draw_image_panel(
        axes[3],
        source_image,
        extent=ext_s,
        cmap="viridis",
        title=f"Source reconstruction{title_suffix}",
        title_size=11,
        xlabel=None,
    )
    axes[3].set_xlabel("arcsec"); axes[3].set_ylabel("arcsec")

    plt.suptitle(
        f"Pixelized source inversion (EPL+Shear fixed, λ={LAMBDA})",
        fontsize=12,
    )
    overlay_critical_and_caustics(
        image_axes=[axes[0], axes[1], axes[2]],
        source_ax=axes[3],
        lens_mass=prob_model.phys_model,
    )
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    out_name = f"output/demo_inversion_lambda{LAMBDA}.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_name}")
    plt.close(fig)

# ------------------------------------------------------------------ #
# Summary Table
# ------------------------------------------------------------------ #
print("\n" + "="*60)
print(f"{'LAMBDA':>10s} | {'log Evidence':>15s} | {'chi2/N_pix':>10s}")
print("-" * 60)
for res in results_summary:
    print(f"{res['lambda']:10.1e} | {res['log_ev']:15.2f} | {res['chi2_nu']:10.3f}")
print("="*60)
