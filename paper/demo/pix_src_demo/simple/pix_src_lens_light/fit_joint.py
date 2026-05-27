"""
Fit lens mass + pixelized source + lens light with Nautilus nested sampling.

Joint model: SIE mass + Sersic lens light (linear amplitude) + pixelized source
Dual masking: data_mask for likelihood, source_seed_mask for bounding box
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
from astropy.io import fits
from nautilus import Sampler

from TinyLensGpu.Inference import ParamU, nautilus_posterior_summary
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE, SersicEllipse
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.utils import load_lens_data

# ------------------------------------------------------------------ #
# True parameters (for reference)
# ------------------------------------------------------------------ #
SIE_TRUE = dict(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
LENS_LIGHT_TRUE = dict(R_sersic=1.0, n_sersic=4.0, Ie=1.0, e1=0.1, e2=0.0)
SRC_TRUE = dict(center_x=0.1, center_y=0.1)
DPIX = 0.05

# ------------------------------------------------------------------ #
# Solve true lensed image positions for position likelihood
# ------------------------------------------------------------------ #
print("[0] Solving true lensed image positions ...")
_true_mass = PhysicalModel(
    lens_mass=[SIE(**SIE_TRUE)],
    source_light=[],
    lens_light=[],
)
_solver = PointSourceProbModel(
    phys_model=_true_mass,
    observed_positions=[[0.0, 0.0]],   # placeholder, not used for solving
    position_sigma=[0.01],
    source_x=SRC_TRUE["center_x"],
    source_y=SRC_TRUE["center_y"],
    source_position_fixed=True,
    solver="optimization",
    solver_config={
        "initial_range": 3.0,
        "n_x": 200,
        "n_y": 200,
        "k_keep": 30,
        "num_iters": 20,
        "tolerance": 5.0e-4,
        "cluster_tol": 0.08,
    },
)
_img_positions, _ = _solver.solve_image_positions()
_img_positions = np.asarray(_img_positions)
print(f"  Found {len(_img_positions)} lensed image positions:")
for pos in _img_positions:
    print(f"    ({pos[0]:.4f}, {pos[1]:.4f})")

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

# Load source seed mask
source_seed_mask = np.asarray(fits.getdata("data/source_seed_mask.fits"), dtype=bool)

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

# Lens light: SersicEllipse with linear intensity solving
# Ie is static (unit amplitude basis), solved jointly with source pixels
lens_light = SersicEllipse(
    R_sersic=ParamU("R_sersic_lens", LENS_LIGHT_TRUE["R_sersic"],
                    prior_type="uniform", prior_settings=[0.5, 2.0],
                    limits=[0.1, 5.0]),
    n_sersic=ParamU("n_sersic_lens", LENS_LIGHT_TRUE["n_sersic"],
                    prior_type="uniform", prior_settings=[2.0, 6.0],
                    limits=[1.0, 8.0]),
    e1=ParamU("e1_lens", LENS_LIGHT_TRUE["e1"],
              prior_type="gaussian", prior_settings=[0.1, 0.1],
              limits=[-0.9, 0.9]),
    e2=ParamU("e2_lens", LENS_LIGHT_TRUE["e2"],
              prior_type="gaussian", prior_settings=[0.0, 0.1],
              limits=[-0.9, 0.9]),
    center_x=ParamU("center_x_lens", 0.0,
                    prior_type="gaussian", prior_settings=[0.0, 0.05],
                    limits=[-0.5, 0.5]),
    center_y=ParamU("center_y_lens", 0.0,
                    prior_type="gaussian", prior_settings=[0.0, 0.05],
                    limits=[-0.5, 0.5]),
    Ie=1.0,  # Unit amplitude basis, solved linearly
)

pix_src = PixelizedSourceModel(
    nx=40,
    ny=40,
    regularization_type="first-order",
    lambda_reg=ParamU("lambda_reg", 1.0,
                      prior_type="log_uniform", prior_settings=[1e-3, 1e3],
                      limits=[1e-6, 1e6]),
)

phys_model = PhysicalModel(
    lens_mass=[sie],
    source_light=[pix_src],
    lens_light=[lens_light],
)

# Mark dynamic parameters
sie.theta_E.to_dynamic()
sie.e1.to_dynamic()
sie.e2.to_dynamic()
sie.center_x.to_dynamic()
sie.center_y.to_dynamic()
lens_light.R_sersic.to_dynamic()
lens_light.n_sersic.to_dynamic()
lens_light.e1.to_dynamic()
lens_light.e2.to_dynamic()
lens_light.center_x.to_dynamic()
lens_light.center_y.to_dynamic()
pix_src.lambda_reg.to_dynamic()

# ------------------------------------------------------------------ #
# Position likelihood: penalize lens mass models whose deflections
# are inconsistent with the observed multiply-imaged positions.
# ------------------------------------------------------------------ #
position_likelihood = {
    'positions': _img_positions.tolist(),
    'threshold_arcsec': 0.3,
    'min_log_like': -1.0e10,
}

# ------------------------------------------------------------------ #
# Probability model (Bayesian evidence)
# ------------------------------------------------------------------ #
print("[3] Building probability model ...")
prob_model = PixelizedImageProbModel(
    image_data=image_data,
    noise_map=noise_map,
    psf_kernel=psf_kernel,
    dpix=DPIX,
    phys_model=phys_model,
    mask=mask,
    source_seed_mask=source_seed_mask,
    nsub=2,
    position_likelihood=position_likelihood,
)

# ------------------------------------------------------------------ #
# Prior transformation and likelihood
# ------------------------------------------------------------------ #
print("[4] Building prior and likelihood ...")
prior, prior_specs = make_prior_transformation(prob_model)
param_names = [s.name for s in prior_specs]

print(f"  {len(param_names)} dynamic parameters:")
for s in prior_specs:
    print(f"    {s.name:20s}: {s.describe()}")

loglike = make_likelihood(prob_model, vectorized=False)

# ------------------------------------------------------------------ #
# Nautilus nested sampling
# ------------------------------------------------------------------ #
print("[5] Running Nautilus sampler ...")
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
print("[6] Processing results ...")
samples, weights, quantiles, log_z = nautilus_posterior_summary(sampler, param_names)
q16_list = [float(qs[0]) for qs in quantiles.values()]
q50_list = [float(qs[1]) for qs in quantiles.values()]
q84_list = [float(qs[2]) for qs in quantiles.values()]

# ------------------------------------------------------------------ #
# Save
# ------------------------------------------------------------------ #
print("[7] Saving results ...")
os.makedirs("output", exist_ok=True)

np.savetxt("output/fit_samples.csv", samples,
           delimiter=",", header=",".join(param_names))

with open("output/fit_summary.csv", "w") as f:
    f.write("parameter,median,lower,upper\n")
    for i, name in enumerate(param_names):
        f.write(f"{name},{q50_list[i]:.6f},{q16_list[i]:.6f},{q84_list[i]:.6f}\n")

with gzip.open("output/fit_results.pkl.gz", "wb") as f:
    pickle.dump({"samples": samples, "weights": weights,
                 "log_z": log_z, "param_names": param_names}, f)

# ------------------------------------------------------------------ #
# Visualization: MAP source reconstruction and separated components
# ------------------------------------------------------------------ #
print("[8] Generating MAP visualization ...")

import caskade as ck

best_params = jnp.array(q50_list)
with ck.ActiveContext(prob_model):
    prob_model.fill_params(best_params)

    # Get joint design matrix
    design_matrix, src_half_size = prob_model.sim_obj.design_matrix()
    reg_matrix, _ = prob_model._regularization_matrix(src_half_size)
    lam = jnp.asarray(pix_src.lambda_reg.value)

    # Solve joint system
    linear_params, chol, curvature = prob_model._solve_source(
        design_matrix, reg_matrix, lam
    )

    n_source = prob_model.sim_obj.n_source_pixels
    source_pixels = linear_params[:n_source]
    lens_amplitudes = linear_params[n_source:] if prob_model.has_lens_light else None

    # Reconstruct components
    model_1d = design_matrix @ linear_params

    # Source-only image
    F = design_matrix[:, :n_source]
    source_1d = F @ source_pixels

    # Lens light image
    if prob_model.has_lens_light:
        L = design_matrix[:, n_source:]
        lens_1d = L @ lens_amplitudes
    else:
        lens_1d = jnp.zeros_like(source_1d)

# Build 2D images
npix = image_data.shape[0]
flat_indices = prob_model.sim_obj.flat_indices

model_image = np.zeros((npix, npix))
model_image.put(flat_indices, np.array(model_1d))

source_image_2d = np.zeros((npix, npix))
source_image_2d.put(flat_indices, np.array(source_1d))

lens_image_2d = np.zeros((npix, npix))
lens_image_2d.put(flat_indices, np.array(lens_1d))

resid_norm = (image_data - model_image) / noise_map

# Chi-square
dof = int((~mask).sum()) - n_source - (len(lens_amplitudes) if lens_amplitudes is not None else 0)
chi2 = float(np.sum(resid_norm[~mask]**2))
chi2_nu = chi2 / dof if dof > 0 else 0.0

# Source reconstruction on source plane
source_pixels_np = np.array(source_pixels)
source_image = source_pixels_np.reshape(prob_model.sim_obj.source_ny, prob_model.sim_obj.source_nx)

ext_i = [-npix * DPIX / 2,  npix * DPIX / 2,
         -npix * DPIX / 2,  npix * DPIX / 2]
ext_s = [-float(src_half_size), float(src_half_size),
         -float(src_half_size), float(src_half_size)]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
vmax = np.nanpercentile(image_data[~mask], 99.5)

# Row 1: Data, Model, Residual, Source
im0 = axes[0, 0].imshow(image_data, origin="lower", extent=ext_i,
                        cmap="viridis", vmin=0, vmax=vmax)
axes[0, 0].set_title("Observed image", fontsize=11)
axes[0, 0].set_xlabel("arcsec"); axes[0, 0].set_ylabel("arcsec")
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

im1 = axes[0, 1].imshow(model_image, origin="lower", extent=ext_i,
                        cmap="viridis", vmin=0, vmax=vmax)
axes[0, 1].set_title("Joint model (source + lens light)", fontsize=11)
axes[0, 1].set_xlabel("arcsec")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

resid_display = np.where(mask, np.nan, resid_norm)
im2 = axes[0, 2].imshow(resid_display, origin="lower", extent=ext_i,
                        cmap="RdBu_r", vmin=-5, vmax=5)
axes[0, 2].set_title(f"Norm. residual (sigma)\nchi^2/ν = {chi2_nu:.3f}", fontsize=11)
axes[0, 2].set_xlabel("arcsec")
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

lam_med = q50_list[param_names.index("lambda_reg")]
im3 = axes[0, 3].imshow(source_image, origin="lower", extent=ext_s, cmap="viridis")
axes[0, 3].set_title(f"Source reconstruction\n(lambda={lam_med:.2e})", fontsize=11)
axes[0, 3].set_xlabel("arcsec"); axes[0, 3].set_ylabel("arcsec")
plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

# Row 2: Separated components
im4 = axes[1, 0].imshow(source_image_2d, origin="lower", extent=ext_i,
                        cmap="viridis", vmin=0, vmax=vmax)
axes[1, 0].set_title("Lensed source only", fontsize=11)
axes[1, 0].set_xlabel("arcsec"); axes[1, 0].set_ylabel("arcsec")
plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

im5 = axes[1, 1].imshow(lens_image_2d, origin="lower", extent=ext_i,
                        cmap="viridis", vmin=0, vmax=vmax)
axes[1, 1].set_title(f"Lens light only\n(Ie={float(lens_amplitudes[0]):.3f})", fontsize=11)
axes[1, 1].set_xlabel("arcsec")
plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

im6 = axes[1, 2].imshow(image_data - lens_image_2d, origin="lower", extent=ext_i,
                        cmap="viridis", vmin=0, vmax=vmax)
axes[1, 2].set_title("Data - lens light", fontsize=11)
axes[1, 2].set_xlabel("arcsec")
plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

# Mask overlay
mask_overlay = np.zeros((*mask.shape, 3))
mask_overlay[mask] = [0.5, 0.5, 0.5]  # Grey for masked
im7 = axes[1, 3].imshow(mask_overlay, origin="lower", extent=ext_i)
axes[1, 3].set_title("Masks (grey=masked)", fontsize=11)
axes[1, 3].set_xlabel("arcsec")

plt.suptitle(
    f"Joint Fit: SIE + Sersic lens light + pix src  "
    f"(theta_E={q50_list[0]:.3f}\", R_sersic={q50_list[5]:.3f}\")",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("output/fit_map_visualization.png", dpi=150, bbox_inches="tight")
print("  Saved output/fit_map_visualization.png")
plt.show()

print("\n" + "="*60)
print("Done.")
print("="*60)
