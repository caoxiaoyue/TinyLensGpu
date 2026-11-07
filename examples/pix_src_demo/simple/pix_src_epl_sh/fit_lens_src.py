"""
Fit lens mass + pixelized source with Nautilus nested sampling.

Lens mass : SIE with Gaussian priors near truth values
Pix source: 40x40 grid, 1st-order regularization
            lambda_reg sampled with log-uniform prior

Uses make_prior_transformation / make_likelihood from TinyLensGpu.Inference,
following the same pattern as examples/lens_src/run_model.py.
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
from TinyLensGpu.PhysicalModel import PhysicalModel, EPL, Shear
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
from TinyLensGpu.example_support import (
    SamplingConfig,
    draw_image_panel,
    posterior_quantiles,
    run_sampling,
    require_valid_reconstruction,
    source_interpolation_metadata,
)
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.visualizer import overlay_critical_and_caustics

# ------------------------------------------------------------------ #
# True parameters (for reference)
# ------------------------------------------------------------------ #
EPL_TRUE = dict(theta_E=1.0, gamma=2.2, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
SHEAR_TRUE = dict(gamma1=0.05, gamma2=0.05)
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

# EPL + Shear — Gaussian priors centred on truth, narrow width
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

# Pixelized source — lambda_reg sampled log-uniformly over [1e-3, 1e3]
pix_src = PixelizedSourceModel(n=40,
    interpolation=os.environ.get("TINYLENS_SOURCE_INTERPOLATION", "bilinear"),
    regularization_type="first-order",
    log_lambda_reg=ParamU("log_lambda_reg", 0.0,
                      prior_type="uniform", prior_settings=[jnp.log(1e-3), jnp.log(1e3)],
                      limits=[-13.815510557964274, 13.815510557964274]),
)

phys_model = PhysicalModel(
    lens_mass=[epl, shear],
    source_light=[pix_src],
    lens_light=[],
)

# Mark dynamic parameters
epl.theta_E.to_dynamic()
epl.gamma.to_dynamic()
epl.e1.to_dynamic()
epl.e2.to_dynamic()
epl.center_x.to_dynamic()
epl.center_y.to_dynamic()
shear.gamma1.to_dynamic()
shear.gamma2.to_dynamic()
pix_src.log_lambda_reg.to_dynamic()

# ------------------------------------------------------------------ #
# Probability model (Bayesian evidence)
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
t0 = time.time()
stage = run_sampling(
    prob_model,
    SamplingConfig(n_live=200, n_eff=800, vectorized=False, n_batch=None,
                   stabilize_weights=True),
    tag="fit", sampler_factory=Sampler,
)
print(f"  Sampling done in {time.time() - t0:.1f} s")
samples, weights = stage.samples, stage.weights
param_names, log_z = stage.param_names, stage.log_z
quantiles = posterior_quantiles(stage)
q16_list = [float(qs[0]) for qs in quantiles.values()]
q50_list = [float(qs[1]) for qs in quantiles.values()]
q84_list = [float(qs[2]) for qs in quantiles.values()]

# ------------------------------------------------------------------ #
# Save
# ------------------------------------------------------------------ #
print("[6] Saving results ...")
os.makedirs("output", exist_ok=True)

np.savetxt("output/fit_samples.csv", samples,
           delimiter=",", header=",".join(param_names))

with open("output/fit_summary.csv", "w") as f:
    f.write("parameter,median,lower,upper\n")
    for i, name in enumerate(param_names):
        f.write(f"{name},{q50_list[i]:.6f},{q16_list[i]:.6f},{q84_list[i]:.6f}\n")

with gzip.open("output/fit_results.pkl.gz", "wb") as f:
    pickle.dump(
        {
            "samples": samples,
            "weights": weights,
            "log_z": log_z,
            "param_names": param_names,
            **source_interpolation_metadata(pix_src.interpolation),
        },
        f,
    )

# ------------------------------------------------------------------ #
# Visualization: MAP source reconstruction
# ------------------------------------------------------------------ #
print("[7] Generating MAP visualization ...")

# Evaluate model at posterior median
import caskade as ck

# Feed posterior median into the model and do a MAP source solve
best_params = jnp.array(q50_list)
result, diagnostics = prob_model.evaluate_with_diagnostics(
    best_params, include_effective_dof=True,
)
require_valid_reconstruction(result, diagnostics, context="plotting reconstruction")

source_bbox      = result.source_bbox
source_pixels_np = np.asarray(result.source_pixels)
model_image      = np.asarray(result.model_image)
resid_norm       = np.asarray(diagnostics.normalized_residuals)

# Total chi-square and reduced chi-square
chi2             = diagnostics.chi2
N_eff            = diagnostics.effective_parameters
dof              = diagnostics.residual_dof
chi2_nu          = diagnostics.reduced_chi2

source_image     = source_pixels_np.reshape(pix_src.n, pix_src.n)

npix   = image_data.shape[0]
ext_i  = [-npix * DPIX / 2,  npix * DPIX / 2,
           -npix * DPIX / 2,  npix * DPIX / 2]
ext_s  = [float(source_bbox[0]), float(source_bbox[1]),
           float(source_bbox[2]), float(source_bbox[3])]

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
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

resid_display = np.where(mask, np.nan, resid_norm)
im2 = draw_image_panel(
    axes[2],
    resid_display,
    extent=ext_i,
    cmap="RdBu_r",
    vmin=-3,
    vmax=3,
    title=f"Norm. residual (σ)\nχ²/ν = {chi2_nu:.3f}",
    title_size=11,
    xlabel=None,
)
axes[2].set_xlabel("arcsec")

lam_val = diagnostics.lambda_reg
im3 = draw_image_panel(
    axes[3],
    source_image,
    extent=ext_s,
    cmap="viridis",
    title=f"Source reconstruction\n(λ={lam_val:.2e})",
    title_size=11,
    xlabel=None,
)
axes[3].set_xlabel("arcsec"); axes[3].set_ylabel("arcsec")

plt.suptitle(
    f"Fit: EPL+Shear + pix src  (θ_E={q50_list[0]:.3f}\", "
    f"γ={q50_list[1]:.3f}, λ={lam_val:.2e})",
    fontsize=12,
)
with ck.ActiveContext(prob_model):
    prob_model.fill_params(best_params)
    overlay_critical_and_caustics(
        image_axes=[axes[0], axes[1], axes[2]],
        source_ax=axes[3],
        lens_mass=prob_model.phys_model,
    )

plt.tight_layout()
plt.savefig("output/fit_map_visualization.png", dpi=150, bbox_inches="tight")
print("  Saved output/fit_map_visualization.png")
plt.show()

print("\n" + "="*60)
print("Done.")
print("="*60)
