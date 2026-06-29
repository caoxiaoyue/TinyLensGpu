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
import matplotlib.pyplot as plt

from TinyLensGpu.PhysicalModel import PhysicalModel, EPL, Shear
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
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
        log_lambda_reg=jnp.log(max(LAMBDA, 1e-8)),
    )

    phys_model = PhysicalModel(
        lens_mass=[epl, shear],
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
    # Semi-linear inversion (operator backend — matrix-free PCG solve)
    # ------------------------------------------------------------------ #
    import jax.numpy as jnp
    xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub, _bx_seed, _by_seed = prob_model._get_bbox()
    reg_data = prob_model._regularization_data(xmin, xmax, ymin, ymax)
    op_data = prob_model.sim_obj.precompute_operator_data(
        xmin, xmax, ymin, ymax, _betas_sub=(beta_x_sub, beta_y_sub),
    )
    lam_j = jnp.asarray(LAMBDA)
    block_chols, block_masks = prob_model.sim_obj.build_block_diag_preconditioner(
        prob_model.noise_1d, xmin, xmax, ymin, ymax, lam_j, prob_model.reg_builder, block_size=prob_model.block_size,
    )
    preconditioner = (block_chols, block_masks)
    sp, _ = prob_model._solve_source(
        xmin, xmax, ymin, ymax, lam_j, reg_data, preconditioner, op_data=op_data,
    )
    source_pixels = np.array(sp)
    log_ev = prob_model.likelihood()
    print(f"  log evidence = {log_ev:.2f}")

    # ------------------------------------------------------------------ #
    # Reconstruct model image via operator forward_model
    # ------------------------------------------------------------------ #
    model_1d  = np.array(prob_model.sim_obj.forward_model(
        sp, xmin, xmax, ymin, ymax, op_data=op_data,
    ))
    model_image = np.zeros(image_data.shape, dtype=np.float32)
    model_image[~mask] = model_1d

    resid_norm = (image_data - model_image) / noise_map
    chi2_nu    = float(np.sum(resid_norm[~mask] ** 2) / (~mask).sum())
    print(f"  χ²/ν = {chi2_nu:.3f}")

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
    ext_s = [float(xmin), float(xmax), float(ymin), float(ymax)]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    # Panel 1: data
    vmax = np.nanpercentile(image_data[~mask], 99.5)
    im0 = axes[0].imshow(image_data, origin="lower", extent=ext_i, cmap="viridis",
                         vmin=0, vmax=vmax)
    axes[0].set_title("Lensed arc (data)", fontsize=11)
    axes[0].set_xlabel("arcsec"); axes[0].set_ylabel("arcsec")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: model image
    im1 = axes[1].imshow(model_image, origin="lower", extent=ext_i, cmap="viridis",
                         vmin=0, vmax=vmax)
    axes[1].set_title("Pix-src model image", fontsize=11)
    axes[1].set_xlabel("arcsec")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: normalized residual
    resid_display = np.where(mask, np.nan, resid_norm)
    im2 = axes[2].imshow(resid_display, origin="lower", extent=ext_i,
                         cmap="RdBu_r", vmin=-3, vmax=3)
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
print(f"{'LAMBDA':>10s} | {'log Evidence':>15s} | {'chi2/nu':>10s}")
print("-" * 60)
for res in results_summary:
    print(f"{res['lambda']:10.1e} | {res['log_ev']:15.2f} | {res['chi2_nu']:10.3f}")
print("="*60)

