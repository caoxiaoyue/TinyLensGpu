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
import jax.scipy.linalg as jsl
import matplotlib.pyplot as plt
from nautilus import Sampler

from TinyLensGpu.Inference import ParamU, nautilus_posterior_summary
from TinyLensGpu.PhysicalModel import PhysicalModel, EPL, Shear
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.visualizer import overlay_critical_and_caustics

# ------------------------------------------------------------------ #
# True parameters (must match sim_data.py)
# ------------------------------------------------------------------ #
EPL_TRUE = dict(theta_E=1.0, gamma=2.2, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
SHEAR_TRUE = dict(gamma1=0.05, gamma2=0.05)
SRC_TRUE = dict(center_x=0.1, center_y=0.1)
DPIX     = 0.05

# ------------------------------------------------------------------ #
# Solve true lensed image positions for the position likelihood
# ------------------------------------------------------------------ #
print("[0] Solving true lensed image positions ...")
_true_mass = PhysicalModel(
    lens_mass=[EPL(**EPL_TRUE), Shear(**SHEAR_TRUE)],
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

epl = EPL(
    theta_E=ParamU("theta_E", EPL_TRUE["theta_E"],
                   prior_type="gaussian", prior_settings=[EPL_TRUE["theta_E"], 0.1],
                   limits=[0.3, 3.0]),
    gamma=ParamU("gamma", EPL_TRUE["gamma"],
                 prior_type="gaussian", prior_settings=[EPL_TRUE["gamma"], 0.1],
                 limits=[1.5, 3.0]),
    e1=ParamU("e1", EPL_TRUE["e1"],
              prior_type="gaussian", prior_settings=[EPL_TRUE["e1"], 0.1],
              limits=[-0.9, 0.9]),
    e2=ParamU("e2", EPL_TRUE["e2"],
              prior_type="gaussian", prior_settings=[EPL_TRUE["e2"], 0.1],
              limits=[-0.9, 0.9]),
    center_x=ParamU("center_x", EPL_TRUE["center_x"],
                    prior_type="gaussian", prior_settings=[EPL_TRUE["center_x"], 0.05],
                    limits=[-0.5, 0.5]),
    center_y=ParamU("center_y", EPL_TRUE["center_y"],
                    prior_type="gaussian", prior_settings=[EPL_TRUE["center_y"], 0.05],
                    limits=[-0.5, 0.5]),
)

shear = Shear(
    gamma1=ParamU("gamma1", SHEAR_TRUE["gamma1"],
                  prior_type="gaussian", prior_settings=[SHEAR_TRUE["gamma1"], 0.05],
                  limits=[-0.5, 0.5]),
    gamma2=ParamU("gamma2", SHEAR_TRUE["gamma2"],
                  prior_type="gaussian", prior_settings=[SHEAR_TRUE["gamma2"], 0.05],
                  limits=[-0.5, 0.5]),
)

pix_src = PixelizedSourceModel(
    nx=25,
    ny=25,
    regularization_type="matern32",
    log_lambda_reg=ParamU("log_lambda_reg", 2.302585092994046,
                      prior_type="uniform", prior_settings=[-4.605170185988091, 9.210340371976184],
                      limits=[-4.605170185988091, 11.512925464970229]),
    kernel_scale=ParamU("kernel_scale", 0.3,
                        prior_type="log_uniform", prior_settings=[0.02, 2.0],
                        limits=[0.01, 5.0]),
)

phys_model = PhysicalModel(
    lens_mass=[epl, shear],
    source_light=[pix_src],
    lens_light=[],
)

epl.theta_E.to_dynamic()
epl.gamma.to_dynamic()
epl.e1.to_dynamic()
epl.e2.to_dynamic()
epl.center_x.to_dynamic()
epl.center_y.to_dynamic()
shear.gamma1.to_dynamic()
shear.gamma2.to_dynamic()
pix_src.log_lambda_reg.to_dynamic()
pix_src.kernel_scale.to_dynamic()

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
prob_model = PixelizedImageProbModelOperator(
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
os.makedirs("output_matern32", exist_ok=True)

np.savetxt("output_matern32/fit_samples.csv", samples,
           delimiter=",", header=",".join(param_names))

with open("output_matern32/fit_summary.csv", "w") as f:
    f.write("parameter,median,lower,upper\n")
    for i, name in enumerate(param_names):
        f.write(f"{name},{q50_list[i]:.6f},{q16_list[i]:.6f},{q84_list[i]:.6f}\n")

with gzip.open("output_matern32/fit_results.pkl.gz", "wb") as f:
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
    lam = jnp.exp(jnp.asarray(pix_src.log_lambda_reg.value))

    # --- Operator backend: PCG solve without building dense design matrix ---
    xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub = prob_model._get_bbox()
    reg_data, _, reg_matrix_dense = prob_model._regularization_data(xmin, xmax, ymin, ymax)
    op_data = prob_model.sim_obj.precompute_operator_data(
        xmin, xmax, ymin, ymax, _betas_sub=(beta_x_sub, beta_y_sub),
    )
    P, P_chol = prob_model.sim_obj.build_preconditioner(
        prob_model.noise_1d, xmin, xmax, ymin, ymax, lam, reg_matrix_dense,
    )
    source_pixels, pcg_info = prob_model._solve_source(
        xmin, xmax, ymin, ymax, lam, reg_data, P_chol, op_data=op_data,
    )
    model_1d = prob_model.sim_obj.forward_model(
        source_pixels, xmin, xmax, ymin, ymax, op_data=op_data,
    )

    # Effective degrees of freedom (operator approximation: N_eff ≈ Ns - λ Tr(P⁻¹R))
    inv_P_R = jsl.cho_solve((P_chol, True), reg_matrix_dense)
    N_eff = float(reg_matrix_dense.shape[0] - lam * jnp.trace(inv_P_R))

source_pixels_np = np.array(source_pixels)
model_1d_np      = np.array(model_1d)
model_image      = np.zeros(image_data.shape)
model_image[~mask] = model_1d_np
resid_norm       = (image_data - model_image) / noise_map

# Total chi-square and reduced chi-square
chi2             = float(np.sum(resid_norm[~mask]**2))
N_d              = int((~mask).sum())
dof              = N_d - N_eff
chi2_nu          = chi2 / dof if dof > 0 else 0.0

source_image     = source_pixels_np.reshape(pix_src.ny, pix_src.nx)

npix   = image_data.shape[0]
ext_i  = [-npix * DPIX / 2,  npix * DPIX / 2,
           -npix * DPIX / 2,  npix * DPIX / 2]
ext_s  = [float(xmin), float(xmax), float(ymin), float(ymax)]

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

lam_med = q50_list[param_names.index("log_lambda_reg")]
ks_med  = q50_list[param_names.index("kernel_scale")]
im3 = axes[3].imshow(source_image, origin="lower", extent=ext_s, cmap="viridis")
axes[3].set_title(f"Source reconstruction\n(λ={float(jnp.exp(lam_med)):.2e}, ℓ={ks_med:.2f}\")", fontsize=11)
axes[3].set_xlabel("arcsec"); axes[3].set_ylabel("arcsec")
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

plt.suptitle(
    f"Fit: EPL+Shear + pix src (Matérn-3/2) + pos-like  (θ_E={q50_list[0]:.3f}\", "
    f"γ={q50_list[1]:.3f}, λ={float(jnp.exp(lam_med)):.2e}, ℓ={ks_med:.2f}\")",
    fontsize=12,
)
overlay_critical_and_caustics(
    image_axes=[axes[0], axes[1], axes[2]],
    source_ax=axes[3],
    lens_mass=prob_model.phys_model,
)

plt.tight_layout()
plt.savefig("output_matern32/fit_map_visualization.png", dpi=150, bbox_inches="tight")
print("  Saved output_matern32/fit_map_visualization.png")
plt.show()

print("\n" + "="*60)
print("Done.")
print("="*60)
