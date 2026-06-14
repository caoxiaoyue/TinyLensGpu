#%%
"""
Simulate a realistic strong-lensing dataset for the pix_src_pipe demo.

Lens mass   : EPL (gamma=2.2) + external shear
Lens light  : one Sersic component
Source light: two overlapping elliptical Gaussians (irregular, non-axisymmetric)
Image       : 100x100 px, dpix=0.05"/px, back_rms=0.05, exp_time=300 s
"""

import os
from pathlib import Path

os.chdir(Path(__file__).parent)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from TinyLensGpu.PhysicalModel import PhysicalModel, Shear, SersicEllipse, GaussianEllipse
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import EPL
from TinyLensGpu.ForwardSimulation import SimulatorConfig, LensSimulator
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d
from TinyLensGpu.utils.geometry import phi_q2_ellipticity

# ------------------------------------------------------------------ #
# Grid / noise config
# ------------------------------------------------------------------ #
DPIX = 0.05
NPIX = 100
BACK_RMS = 0.05
EXP_TIME = 300.0
SEED = 42

# ------------------------------------------------------------------ #
# True parameters
# ------------------------------------------------------------------ #
e1_mass, e2_mass = phi_q2_ellipticity(40 * np.pi / 180, 0.75)
EPL_TRUE = dict(theta_E=1.6, gamma=2.2, e1=float(e1_mass), e2=float(e2_mass),
                center_x=0.0, center_y=0.0)
SHEAR_TRUE = dict(gamma1=0.04, gamma2=-0.03)

e1_ll, e2_ll = phi_q2_ellipticity(40 * np.pi / 180, 0.80)
LENS_LIGHT_TRUE = dict(R_sersic=1.0, n_sersic=3.5, e1=float(e1_ll), e2=float(e2_ll),
                       center_x=0.0, center_y=0.0, Ie=1.0)

SRC_G1_TRUE = dict(flux=1.0, sigma=0.12, e1=0.1, e2=-0.05,
                   center_x=0.05, center_y=0.10)
SRC_G2_TRUE = dict(flux=0.6, sigma=0.08, e1=-0.05, e2=0.05,
                   center_x=0.15, center_y=0.00)

# ------------------------------------------------------------------ #
# Physical model
# ------------------------------------------------------------------ #
phys_model = PhysicalModel(
    lens_mass=[EPL(**EPL_TRUE), Shear(**SHEAR_TRUE)],
    source_light=[
        GaussianEllipse(**SRC_G1_TRUE),
        GaussianEllipse(**SRC_G2_TRUE),
    ],
    lens_light=[SersicEllipse(**LENS_LIGHT_TRUE)],
)

# ------------------------------------------------------------------ #
# PSF (Gaussian, sigma = 0.05")
# ------------------------------------------------------------------ #
x_psf, y_psf = make_grid_2d(21, DPIX)
psf_raw = GaussianEllipse(
    flux=1.0, sigma=0.05, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0
).light(x=x_psf, y=y_psf)
psf_kernel = np.array(psf_raw / psf_raw.sum())

# ------------------------------------------------------------------ #
# Forward simulation
# ------------------------------------------------------------------ #
sim_config = SimulatorConfig(dpix=DPIX, npix=NPIX, psf_kernel=psf_kernel, nsub=16)
sim_obj = LensSimulator(phys_model, sim_config)
img_ideal = np.asarray(sim_obj.simulate())

# ------------------------------------------------------------------ #
# Add noise
# ------------------------------------------------------------------ #
rng = np.random.default_rng(SEED)
noise_map = np.sqrt(np.maximum(img_ideal, 0.0) / EXP_TIME + BACK_RMS ** 2)
img_noisy = img_ideal + rng.normal(0.0, noise_map)

# ------------------------------------------------------------------ #
# Quick-look plot
# ------------------------------------------------------------------ #
extent = [-NPIX * DPIX / 2, NPIX * DPIX / 2, -NPIX * DPIX / 2, NPIX * DPIX / 2]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].imshow(img_ideal, origin="lower", extent=extent, cmap="viridis")
axes[0].set_title("Noiseless image (lens+lensed source)")
axes[0].set_xlabel("arcsec")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(img_noisy, origin="lower", extent=extent, cmap="viridis")
axes[1].set_title("Noisy image")
axes[1].set_xlabel("arcsec")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

snr_map = img_noisy / noise_map
im2 = axes[2].imshow(snr_map, origin="lower", extent=extent, cmap="viridis",
                     vmin=0.0, vmax=np.nanpercentile(snr_map, 99.5))
axes[2].set_title("S/N map")
axes[2].set_xlabel("arcsec")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle(
    f"EPL (gamma={EPL_TRUE['gamma']}) + shear + Sersic lens + 2-Gaussian source",
    fontsize=11,
)
plt.tight_layout()
os.makedirs("data", exist_ok=True)
plt.savefig("data/sim_preview.png", dpi=120, bbox_inches="tight")

# ------------------------------------------------------------------ #
# True lens light (noiseless, PSF-convolved) for Stage L diagnostics
# ------------------------------------------------------------------ #
phys_model_ll = PhysicalModel(
    lens_mass=[],
    source_light=[],
    lens_light=[SersicEllipse(**LENS_LIGHT_TRUE)],
)
sim_obj_ll = LensSimulator(phys_model_ll, sim_config)
img_lens_light_ideal = np.asarray(sim_obj_ll.simulate())
fits.writeto("data/lens_light_true.fits", img_lens_light_ideal.astype(np.float32), overwrite=True)

from matplotlib import pyplot as plt
plt.figure()
plt.imshow((img_noisy - img_lens_light_ideal)/noise_map, origin="lower", extent=extent, cmap="viridis")
plt.colorbar()
plt.show()

# ------------------------------------------------------------------ #
# Save FITS
# ------------------------------------------------------------------ #
fits.writeto("data/image.fits", img_noisy.astype(np.float32), overwrite=True)
fits.writeto("data/noise.fits", noise_map.astype(np.float32), overwrite=True)
fits.writeto("data/psf.fits", psf_kernel.astype(np.float32), overwrite=True)

print("Saved to data/")
print("\nTrue parameters:")
for k, v in EPL_TRUE.items():
    print(f"  EPL.{k:<10s} = {v}")
for k, v in SHEAR_TRUE.items():
    print(f"  Shear.{k:<8s} = {v}")
for k, v in LENS_LIGHT_TRUE.items():
    print(f"  LensLight.{k:<10s} = {v}")
for tag, src in (("Src1", SRC_G1_TRUE), ("Src2", SRC_G2_TRUE)):
    for k, v in src.items():
        print(f"  {tag}.{k:<10s} = {v}")
print(f"\nImage: {NPIX}x{NPIX} px,  dpix={DPIX}\"/px")
print(f"Noise: back_rms={BACK_RMS},  exp_time={EXP_TIME} s,  seed={SEED}")

# %%
