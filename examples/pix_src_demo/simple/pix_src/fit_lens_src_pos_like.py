"""
Fit lens mass + pixelized source with position likelihood constraint.

Same setup as fit_lens_src.py, but adds a position likelihood penalty to
quickly reject lens mass models whose deflections are inconsistent with the
observed multiply-imaged positions.

The image-plane positions used for the position likelihood are computed by
solving the lens equation with the true SIE parameters and the true source
position from sim_data.py, so they are physically meaningful.
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
import jax.scipy.linalg as jsl
import matplotlib.pyplot as plt
from nautilus import Sampler

from TinyLensGpu.Inference import ParamU, nautilus_posterior_summary
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.visualizer import overlay_critical_and_caustics

# ------------------------------------------------------------------ #
# True parameters (must match sim_data.py)
# ------------------------------------------------------------------ #
SIE_TRUE = dict(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
SRC_TRUE = dict(center_x=0.1, center_y=0.1)
DPIX     = 0.05

# ------------------------------------------------------------------ #
# Solve true lensed image positions for the position likelihood
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

pix_src = PixelizedSourceModel(n=40,
    regularization_type="first-order",
    log_lambda_reg=ParamU("log_lambda_reg", 0.0,
                      prior_type="uniform", prior_settings=[jnp.log(1e-3), jnp.log(1e3)],
                      limits=[-13.815510557964274, 13.815510557964274]),
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
pix_src.log_lambda_reg.to_dynamic()

# ------------------------------------------------------------------ #
# Position likelihood: use the solved true image positions.
# A correct lens mass model must map all these image-plane positions
# back to the same source position; the penalty fires when the
# max pairwise source-plane separation exceeds the threshold.
# ------------------------------------------------------------------ #
position_likelihood = {
    'positions': _img_positions.tolist(),
    'threshold_arcsec': 0.3,
    'min_log_like': -1.0e10,
}

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
    position_likelihood=position_likelihood,
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
samples, weights, quantiles, log_z = nautilus_posterior_summary(sampler, param_names)
q16_list = [float(qs[0]) for qs in quantiles.values()]
q50_list = [float(qs[1]) for qs in quantiles.values()]
q84_list = [float(qs[2]) for qs in quantiles.values()]

# ------------------------------------------------------------------ #
# Save
# ------------------------------------------------------------------ #
print("[6] Saving results ...")
os.makedirs("output_pos_like", exist_ok=True)

np.savetxt("output_pos_like/fit_samples.csv", samples,
           delimiter=",", header=",".join(param_names))

with open("output_pos_like/fit_summary.csv", "w") as f:
    f.write("parameter,median,lower,upper\n")
    for i, name in enumerate(param_names):
        f.write(f"{name},{q50_list[i]:.6f},{q16_list[i]:.6f},{q84_list[i]:.6f}\n")

with gzip.open("output_pos_like/fit_results.pkl.gz", "wb") as f:
    pickle.dump({"samples": samples, "weights": weights,
                 "log_z": log_z, "param_names": param_names,
                 "image_positions": _img_positions}, f)

# ------------------------------------------------------------------ #
# Visualization: MAP source reconstruction
# ------------------------------------------------------------------ #
print("[7] Generating MAP visualization ...")

import caskade as ck

best_params = jnp.array(q50_list)
with ck.ActiveContext(prob_model):
    prob_model.fill_params(best_params)
    design_matrix, source_bbox = prob_model.sim_obj.design_matrix()
    reg_matrix, _ = prob_model._regularization_matrix(source_bbox)
    lam = jnp.exp(jnp.asarray(pix_src.log_lambda_reg.value))
    source_pixels, chol, curvature = prob_model._solve_source(
        design_matrix, reg_matrix, lam
    )
    
    # Calculate effective degrees of freedom (N_d - N_eff)
    inv_F = jsl.cho_solve((chol, True), jnp.eye(curvature.shape[0]))
    ATA = curvature - lam * reg_matrix
    N_eff = float(jnp.trace(inv_F @ ATA))

source_pixels_np = np.array(source_pixels)
model_1d         = np.array(design_matrix @ source_pixels)
model_image      = np.zeros(image_data.shape)
model_image[~mask] = model_1d
resid_norm       = (image_data - model_image) / noise_map

# Total chi-square and reduced chi-square
chi2             = float(np.sum(resid_norm[~mask]**2))
N_d              = int((~mask).sum())
dof              = N_d - N_eff
chi2_nu          = chi2 / dof if dof > 0 else 0.0

source_image     = source_pixels_np.reshape(40, 40)

npix   = image_data.shape[0]
ext_i  = [-npix * DPIX / 2,  npix * DPIX / 2,
           -npix * DPIX / 2,  npix * DPIX / 2]
ext_s  = [float(source_bbox[0]), float(source_bbox[1]),
           float(source_bbox[2]), float(source_bbox[3])]

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
                     cmap="RdBu_r", vmin=-3, vmax=3)
axes[2].set_title(f"Norm. residual (σ)\nχ²/ν = {chi2_nu:.3f}", fontsize=11)
axes[2].set_xlabel("arcsec")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

lam_med = q50_list[param_names.index("log_lambda_reg")]
im3 = axes[3].imshow(source_image, origin="lower", extent=ext_s, cmap="viridis")
axes[3].set_title(f"Source reconstruction\n(λ={float(jnp.exp(lam_med)):.2e})", fontsize=11)
axes[3].set_xlabel("arcsec"); axes[3].set_ylabel("arcsec")
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

plt.suptitle(
    f"Fit: SIE + pix src + pos-like  (θ_E={q50_list[0]:.3f}\", "
    f"e1={q50_list[1]:.3f}, λ={float(jnp.exp(lam_med)):.2e})",
    fontsize=12,
)
overlay_critical_and_caustics(
    image_axes=[axes[0], axes[1], axes[2]],
    source_ax=axes[3],
    lens_mass=prob_model.phys_model,
)

plt.tight_layout()
plt.savefig("output_pos_like/fit_map_visualization.png", dpi=150, bbox_inches="tight")
print("  Saved output_pos_like/fit_map_visualization.png")
plt.show()

print("\n" + "="*60)
print("Done.")
print("="*60)
