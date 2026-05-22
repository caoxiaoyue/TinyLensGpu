"""
Generate simulated strong lensing data for joint pixelized source + lens light demo.

Lens mass  : SIE (theta_E=1.0", e1=0.1, e2=0.0)
Lens light : SersicEllipse (R_sersic=1.0", n_sersic=4.0)
Source     : Gaussian (sigma=0.15", offset from Einstein ring)
Image      : 100x100 px, dpix=0.05 "/px (5"x5" field)
Noise      : Poisson + sky background
Masks      : Dual masking - data_mask + source_seed_mask
"""

import os
from pathlib import Path

os.chdir(Path(__file__).parent)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE, GaussianEllipse, SersicEllipse
from TinyLensGpu.ForwardSimulation import SimulatorConfig, LensSimulator
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d
from TinyLensGpu.utils.misc import arc_mask_from

# ------------------------------------------------------------------ #
# True parameters
# ------------------------------------------------------------------ #
DPIX = 0.05       # arcsec / pixel
NPIX = 100        # image size (pixels)

SIE_TRUE = dict(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
SRC_TRUE = dict(flux=1.0, sigma=0.15, e1=0.0, e2=0.0, center_x=0.1, center_y=0.1)
LENS_LIGHT_TRUE = dict(
    R_sersic=1.0, n_sersic=4.0, Ie=1.0,
    e1=0.1, e2=0.0, center_x=0.0, center_y=0.0,
)

BACK_RMS  = 0.05   # sky background rms (counts / px)
EXP_TIME  = 100.0  # effective exposure time (s)
SEED      = 42

# ------------------------------------------------------------------ #
# Build physical model
# ------------------------------------------------------------------ #
phys_model = PhysicalModel(
    lens_mass=[SIE(**SIE_TRUE)],
    source_light=[GaussianEllipse(**SRC_TRUE)],
    lens_light=[SersicEllipse(**LENS_LIGHT_TRUE)],
)

# ------------------------------------------------------------------ #
# PSF (Gaussian, FWHM ~ 0.09")
# ------------------------------------------------------------------ #
x_psf, y_psf = make_grid_2d(21, DPIX)
psf_raw    = GaussianEllipse(
    flux=1.0, sigma=0.04, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0
).light(x=x_psf, y=y_psf)
psf_kernel = np.array(psf_raw / psf_raw.sum())

# ------------------------------------------------------------------ #
# Forward simulation
# ------------------------------------------------------------------ #
sim_config = SimulatorConfig(dpix=DPIX, npix=NPIX, psf_kernel=psf_kernel, nsub=16)
sim_obj    = LensSimulator(phys_model, sim_config)
img_arc_ideal, img_lens_light_ideal = [np.asarray(x) for x in sim_obj.simulate(ret_each_plane=True)]
img_ideal = img_arc_ideal + img_lens_light_ideal

# ------------------------------------------------------------------ #
# Add noise
# ------------------------------------------------------------------ #
rng       = np.random.default_rng(SEED)
noise_map = np.sqrt(np.maximum(img_ideal, 0.0) / EXP_TIME + BACK_RMS**2)
img_noisy = img_ideal + rng.normal(0.0, noise_map)

# ------------------------------------------------------------------ #
# Dual masking
# ------------------------------------------------------------------ #
xgrid, ygrid = make_grid_2d(NPIX, DPIX)
rgrid        = np.sqrt(np.asarray(xgrid)**2 + np.asarray(ygrid)**2)
# Data mask: circular region covering lensed arc + lens center
data_mask = rgrid > 2.0
# Source seed mask: tighter region only covering the lensed arc
arc_image = img_noisy - img_lens_light_ideal
source_seed_mask= arc_mask_from(arc_image, threshold=2.5)

# Ensure source_seed_mask is a subset of data_mask
source_seed_mask = source_seed_mask | data_mask  # Union: more masked pixels

# ------------------------------------------------------------------ #
# Quick-look plot
# ------------------------------------------------------------------ #
extent = [-NPIX * DPIX / 2, NPIX * DPIX / 2, -NPIX * DPIX / 2, NPIX * DPIX / 2]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

im0 = axes[0, 0].imshow(img_ideal, origin="lower", extent=extent, cmap="viridis")
axes[0, 0].set_title("Noiseless lensed image + lens light")
axes[0, 0].set_xlabel("arcsec")
fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

im1 = axes[0, 1].imshow(img_noisy, origin="lower", extent=extent, cmap="viridis")
axes[0, 1].set_title("Noisy image")
axes[0, 1].set_xlabel("arcsec")
fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

snr_map = img_noisy / noise_map
snr_map_masked = np.where(data_mask, np.nan, snr_map)
im2 = axes[0, 2].imshow(snr_map_masked, origin="lower", extent=extent, cmap="viridis")
axes[0, 2].set_title(f"S/N map (data mask, {int((~data_mask).sum())} px)")
axes[0, 2].set_xlabel("arcsec")
fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

# Mask visualizations
data_mask_viz = np.where(data_mask, 1.0, 0.0)
seed_mask_viz = np.where(source_seed_mask, 1.0, 0.0)

im3 = axes[1, 0].imshow(data_mask_viz, origin="lower", extent=extent, cmap="RdYlGn_r", vmin=0, vmax=1)
axes[1, 0].set_title("Data Mask (red=masked)")
axes[1, 0].set_xlabel("arcsec")

im4 = axes[1, 1].imshow(seed_mask_viz, origin="lower", extent=extent, cmap="RdYlGn_r", vmin=0, vmax=1)
axes[1, 1].set_title(f"Source Seed Mask (red=masked, {int((~source_seed_mask).sum())} px)")
axes[1, 1].set_xlabel("arcsec")

# Overlay
overlay = np.zeros((*data_mask.shape, 3))
overlay[data_mask] = [1, 0, 0]  # Red: data mask
overlay[source_seed_mask & ~data_mask] = [0, 1, 0]  # Green: seed mask only
im5 = axes[1, 2].imshow(overlay, origin="lower", extent=extent)
axes[1, 2].set_title("Mask overlay (red=data, green=seed only)")
axes[1, 2].set_xlabel("arcsec")

plt.suptitle(
    f"SIE + Sersic lens light + Gaussian source simulation  "
    f"(theta_E={SIE_TRUE['theta_E']}\", sigma_src={SRC_TRUE['sigma']}\")",
    fontsize=11,
)
plt.tight_layout()
os.makedirs("data", exist_ok=True)
plt.savefig("data/sim_preview.png", dpi=120, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------ #
# Save FITS
# ------------------------------------------------------------------ #
fits.writeto("data/image.fits", img_noisy.astype(np.float32), overwrite=True)
fits.writeto("data/noise.fits", noise_map.astype(np.float32), overwrite=True)
fits.writeto("data/psf.fits",   psf_kernel.astype(np.float32), overwrite=True)
fits.writeto("data/mask.fits",  data_mask.astype(np.int32),         overwrite=True)
fits.writeto("data/source_seed_mask.fits", source_seed_mask.astype(np.int32), overwrite=True)

print("Saved to data/")
print("\nTrue parameters:")
for k, v in SIE_TRUE.items():
    print(f"  SIE.{k:<12s} = {v}")
for k, v in LENS_LIGHT_TRUE.items():
    print(f"  LensLight.{k:<12s} = {v}")
for k, v in SRC_TRUE.items():
    print(f"  Source.{k:<9s} = {v}")
print(f"\nImage: {NPIX}x{NPIX} px,  dpix={DPIX}\"/px")
print(f"Noise: back_rms={BACK_RMS},  exp_time={EXP_TIME} s")
print(f"Data mask: {int((~data_mask).sum())} active pixels")
print(f"Seed mask: {int((~source_seed_mask).sum())} active pixels")
