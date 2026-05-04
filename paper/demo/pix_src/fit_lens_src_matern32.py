"""
Fit lens mass + pixelized source with Matern-3/2 GP regularization.

Same setup as fit_lens_src.py, but replaces the first-order
finite-difference regularization with a Matern-3/2 GP covariance prior.
Two regularization hyperparameters are sampled: lambda_reg (overall strength)
and kernel_scale (GP correlation length in arcsec).
"""

import os
import gzip
import pickle
import time
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.chdir(Path(__file__).parent)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from nautilus import Sampler

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.utils import load_lens_data

# ------------------------------------------------------------------ #
# True parameters (must match sim_data.py)
# ------------------------------------------------------------------ #
SIE_TRUE = dict(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
DPIX     = 0.05

# ------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------ #
print("[1] Loading data ...")
image_data, noise_map, psf_kernel, mask = load_lens_data(
    image_path="data/image.fits",
    noise_path="data/noise.fits",
    psf_path="data/psf.fits",
    mask_path="data/mask.fits",
)

# ------------------------------------------------------------------ #
# Physical model
# ------------------------------------------------------------------ #
print("[2] Building model ...")

sie = SIE(
    theta_E=ParamU("theta_E",  SIE_TRUE["theta_E"],
                   prior_type="gaussian", prior_settings=[1.0, 0.1],
                   limits=[0.3, 3.0]),
    e1=ParamU("e1",  SIE_TRUE["e1"],
              prior_type="gaussian", prior_settings=[0.1, 0.1],
              limits=[-0.9, 0.9]),
    e2=ParamU("e2",  SIE_TRUE["e2"],
              prior_type="gaussian", prior_settings=[0.0, 0.1],
              limits=[-0.9, 0.9]),
    center_x=ParamU("center_x", 0.0,
                    prior_type="gaussian", prior_settings=[0.0, 0.05],
                    limits=[-0.5, 0.5]),
    center_y=ParamU("center_y", 0.0,
                    prior_type="gaussian", prior_settings=[0.0, 0.05],
                    limits=[-0.5, 0.5]),
)

pix_src = PixelizedSourceModel(
    nx=40,
    ny=40,
    regularization_type="matern32",
    lambda_reg=ParamU("lambda_reg", 1.0,
                      prior_type="log_uniform", prior_settings=[1e-3, 1e3],
                      limits=[1e-6, 1e6]),
    kernel_scale=ParamU("kernel_scale", 0.3,
                        prior_type="log_uniform", prior_settings=[0.01, 2.0],
                        limits=[1e-3, 10.0]),
)

phys_model = PhysicalModel(
    lens_mass=[sie],
    source_light=[pix_src],
    lens_light=[],
)

sie.theta_E.to_dynamic()
sie.e1.to_dynamic()
sie.e2.to_dynamic()
sie.center_x.to_dynamic()
sie.center_y.to_dynamic()
pix_src.lambda_reg.to_dynamic()
pix_src.kernel_scale.to_dynamic()

# ------------------------------------------------------------------ #
# Probability model
# ------------------------------------------------------------------ #
prob_model = PixelizedImageProbModel(
    image_data=image_data,
    noise_map=noise_map,
    psf_kernel=psf_kernel,
    dpix=DPIX,
    phys_model=phys_model,
    mask=mask,
    nsub=2,
)

# ------------------------------------------------------------------ #
# Prior transformation and likelihood
# ------------------------------------------------------------------ #
print("[3] Building prior and likelihood ...")
prior, prior_specs = make_prior_transformation(prob_model)
param_names = [s.name for s in prior_specs]

print(f"  {len(param_names)} dynamic parameters:")
for s in prior_specs:
    print(f"    {s.name:15s}: {s.describe()}")

loglike = make_likelihood(prob_model, vectorized=False)

# ------------------------------------------------------------------ #
# Nautilus nested sampling
# ------------------------------------------------------------------ #
print("[4] Running Nautilus sampler ...")
sampler = Sampler(
    prior,
    loglike,
    n_dim=len(param_names),
    n_live=200,
    vectorized=False,
)

t0 = time.time()
sampler.run(verbose=True, n_eff=800)
print(f"  Sampling done in {time.time() - t0:.1f} s")

# ------------------------------------------------------------------ #
# Posterior summary
# ------------------------------------------------------------------ #
print("[5] Processing results ...")
samples, log_w, _ = sampler.posterior()
weights = np.exp(log_w - np.max(log_w))
weights /= weights.sum()

print("\n" + "="*60)
print("Posterior Summary")
print("="*60)
q16_list, q50_list, q84_list = [], [], []
for i, name in enumerate(param_names):
    idx   = np.argsort(samples[:, i])
    s_s   = samples[idx, i]
    w_s   = weights[idx]
    cdf   = np.cumsum(w_s); cdf /= cdf[-1]
    q16   = float(np.interp(0.16, cdf, s_s))
    q50   = float(np.interp(0.50, cdf, s_s))
    q84   = float(np.interp(0.84, cdf, s_s))
    q16_list.append(q16); q50_list.append(q50); q84_list.append(q84)
    print(f"  {name:15s} = {q50:.4f}  ({q16-q50:+.4f}, {q84-q50:+.4f})")

log_z = float(np.asarray(sampler.log_z))
print(f"\nlog(Z) = {log_z:.2f}")

# ------------------------------------------------------------------ #
# Save
# ------------------------------------------------------------------ #
print("[6] Saving results ...")
os.makedirs("output_matern32", exist_ok=True)

np.savetxt("output_matern32/fit_samples.csv", samples,
           delimiter=",", header=",".join(param_names))

with open("output_matern32/fit_summary.csv", "w") as f:
    f.write("parameter,median,lower,upper\n")
    for i, name in enumerate(param_names):
        f.write(f"{name},{q50_list[i]:.6f},{q16_list[i]:.6f},{q84_list[i]:.6f}\n")

with gzip.open("output_matern32/fit_results.pkl.gz", "wb") as f:
    pickle.dump({"samples": samples, "weights": weights,
                 "log_z": log_z, "param_names": param_names}, f)

# ------------------------------------------------------------------ #
# Visualization: MAP source reconstruction
# ------------------------------------------------------------------ #
print("[7] Generating MAP visualization ...")

import caskade as ck

best_params = jnp.array(q50_list)
with ck.ActiveContext(prob_model):
    prob_model.fill_params(best_params)
    design_matrix, src_half_size = prob_model.sim_obj.design_matrix()
    reg_matrix = prob_model._regularization_matrix(src_half_size)
    lam = jnp.asarray(pix_src.lambda_reg.value)
    source_pixels, _, _ = prob_model._solve_source(
        design_matrix, reg_matrix, lam
    )

source_pixels_np = np.array(source_pixels)
model_1d         = np.array(design_matrix @ source_pixels)
model_image      = np.zeros(image_data.shape)
model_image[~mask] = model_1d
resid_norm       = (image_data - model_image) / noise_map
chi2_nu          = float(np.sum(resid_norm[~mask]**2) / (~mask).sum())
source_image     = source_pixels_np.reshape(40, 40)

npix   = image_data.shape[0]
ext_i  = [-npix * DPIX / 2,  npix * DPIX / 2,
           -npix * DPIX / 2,  npix * DPIX / 2]
ext_s  = [-float(src_half_size), float(src_half_size),
           -float(src_half_size), float(src_half_size)]

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
vmax = np.nanpercentile(image_data[~mask], 99.5)

im0 = axes[0].imshow(image_data, origin="lower", extent=ext_i,
                     cmap="viridis", vmin=0, vmax=vmax)
axes[0].set_title("Lensed arc (data)", fontsize=11)
axes[0].set_xlabel("arcsec"); axes[0].set_ylabel("arcsec")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(model_image, origin="lower", extent=ext_i,
                     cmap="viridis", vmin=0, vmax=vmax)
axes[1].set_title("Pix-src model image", fontsize=11)
axes[1].set_xlabel("arcsec")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

resid_display = np.where(mask, np.nan, resid_norm)
im2 = axes[2].imshow(resid_display, origin="lower", extent=ext_i,
                     cmap="RdBu_r", vmin=-5, vmax=5)
axes[2].set_title(f"Norm. residual (σ)\nχ²/ν = {chi2_nu:.3f}", fontsize=11)
axes[2].set_xlabel("arcsec")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

lam_med = q50_list[param_names.index("lambda_reg")]
ks_med  = q50_list[param_names.index("kernel_scale")]
im3 = axes[3].imshow(source_image, origin="lower", extent=ext_s, cmap="viridis")
axes[3].set_title(f"Source reconstruction\n(λ={lam_med:.2e}, ℓ={ks_med:.2f}\")", fontsize=11)
axes[3].set_xlabel("arcsec"); axes[3].set_ylabel("arcsec")
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

plt.suptitle(
    f"Fit: SIE + pix src (Matérn-3/2)  (θ_E={q50_list[0]:.3f}\", "
    f"e1={q50_list[1]:.3f}, λ={lam_med:.2e}, ℓ={ks_med:.2f}\")",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("output_matern32/fit_map_visualization.png", dpi=150, bbox_inches="tight")
print("  Saved output_matern32/fit_map_visualization.png")
plt.show()

print("\n" + "="*60)
print("Done.")
print("="*60)
