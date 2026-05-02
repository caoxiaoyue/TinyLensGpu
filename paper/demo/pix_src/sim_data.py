"""
Generate simulated strong lensing data for pixelized source reconstruction demo.

Lens mass  : SIE  (theta_E=1.0", e1=0.1, e2=0.0)
Source light: Gaussian (sigma=0.15", offset from Einstein ring)
Image      : 100x100 px, dpix=0.05 "/px  (5"×5" field)
Noise      : Poisson + sky background
Mask       : tight annular region wrapping the lensed arc (arc-brightness threshold)
"""

import os
from pathlib import Path

# Run from the script's own directory so relative paths (data/) work correctly.
os.chdir(Path(__file__).parent)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import binary_dilation

from TinyLensGpu.PhysicalModel import PhysicalModel, SIE, GaussianEllipse
from TinyLensGpu.ForwardSimulation import SimulatorConfig, LensSimulator
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d

# ------------------------------------------------------------------ #
# True parameters (saved as reference)
# ------------------------------------------------------------------ #
DPIX = 0.05       # arcsec / pixel
NPIX = 100        # image size (pixels)

SIE_TRUE = dict(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
SRC_TRUE = dict(flux=1.0, sigma=0.15, e1=0.0, e2=0.0, center_x=0.1, center_y=0.1)

BACK_RMS  = 0.05   # sky background rms (counts / px)
EXP_TIME  = 100.0  # effective exposure time (s)
SEED      = 42

# ------------------------------------------------------------------ #
# Build physical model
# ------------------------------------------------------------------ #
phys_model = PhysicalModel(
    lens_mass=[SIE(**SIE_TRUE)],
    source_light=[GaussianEllipse(**SRC_TRUE)],
    lens_light=[],
)

# ------------------------------------------------------------------ #
# PSF (Gaussian, FWHM ≈ 0.09")
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
img_ideal  = np.asarray(sim_obj.simulate())

# ------------------------------------------------------------------ #
# Add noise
# ------------------------------------------------------------------ #
rng       = np.random.default_rng(SEED)
noise_map = np.sqrt(np.maximum(img_ideal, 0.0) / EXP_TIME + BACK_RMS**2)
img_noisy = img_ideal + rng.normal(0.0, noise_map)

# ------------------------------------------------------------------ #
# Annular mask: tight wrap around the lensed arc (tutorial-style)
# Threshold on ideal arc brightness, then dilate 1 px and use the
# radial bounding annulus.  This keeps Nd ≈ Ns ≈ 1600, making the
# lambda=0 inversion underdetermined (→ noisy source, as expected).
# ------------------------------------------------------------------ #
xgrid, ygrid = make_grid_2d(NPIX, DPIX)
rgrid        = np.sqrt(np.asarray(xgrid)**2 + np.asarray(ygrid)**2)

arc_seed = img_ideal > 0.08 * float(img_ideal.max())
arc_seed = binary_dilation(arc_seed.astype(bool), iterations=1)
r_arc    = rgrid[arc_seed]
r_inner  = max(0.0, float(r_arc.min()) - DPIX)
r_outer  = float(r_arc.max()) + DPIX
mask     = (rgrid < r_inner) | (rgrid > r_outer)   # True = masked out

# ------------------------------------------------------------------ #
# Quick-look plot
# ------------------------------------------------------------------ #
extent = [-NPIX * DPIX / 2, NPIX * DPIX / 2, -NPIX * DPIX / 2, NPIX * DPIX / 2]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].imshow(img_ideal, origin="lower", extent=extent, cmap="viridis")
axes[0].set_title("Noiseless lensed image")
axes[0].set_xlabel("arcsec")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(img_noisy, origin="lower", extent=extent, cmap="viridis")
axes[1].set_title("Noisy image")
axes[1].set_xlabel("arcsec")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

snr_map = img_noisy / noise_map
snr_map_masked = np.where(mask, np.nan, snr_map)
im2 = axes[2].imshow(snr_map_masked, origin="lower", extent=extent, cmap="viridis")
axes[2].set_title("S/N map (unmasked region)")
axes[2].set_xlabel("arcsec")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle(
    f"SIE+Gaussian source simulation  "
    f"(θ_E={SIE_TRUE['theta_E']}\", σ_src={SRC_TRUE['sigma']}\")",
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
fits.writeto("data/mask.fits",  mask.astype(np.int32),         overwrite=True)

print("Saved to data/")
print("\nTrue parameters:")
for k, v in SIE_TRUE.items():
    print(f"  SIE.{k:<12s} = {v}")
for k, v in SRC_TRUE.items():
    print(f"  Source.{k:<9s} = {v}")
print(f"\nImage: {NPIX}×{NPIX} px,  dpix={DPIX}\"/px")
print(f"Noise: back_rms={BACK_RMS},  exp_time={EXP_TIME} s")
print(f"Mask:  annular {r_inner:.2f}\"–{r_outer:.2f}\" ({int((~mask).sum())} active pixels)")
