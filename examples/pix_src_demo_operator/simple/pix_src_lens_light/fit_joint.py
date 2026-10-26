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

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.example_support import (
    SamplingConfig,
    run_sampling,
    posterior_quantiles,
    draw_image_panel,
    require_valid_reconstruction,
)
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE, SersicEllipse
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
from TinyLensGpu.ObservationModel.LensImage.inversion_result import InvalidReconstructionError
from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.visualizer import overlay_critical_and_caustics

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
    n=80,
    regularization_type="first-order",
    log_lambda_reg=ParamU(
        "log_lambda_reg",
        0.0,
        prior_type="uniform",
        prior_settings=[jnp.log(1e-6), jnp.log(1e3)],
        limits=[-13.815510557964274, 13.815510557964274],
    ),
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
pix_src.log_lambda_reg.to_dynamic()

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
prob_model = PixelizedImageProbModelOperator(
    image_data=image_data,
    noise_map=noise_map,
    psf_kernel=psf_kernel,
    dpix=DPIX,
    phys_model=phys_model,
    mask=mask,
    source_seed_mask=source_seed_mask,
    nsub=4,
    source_bbox_padding=0.2,
    position_likelihood=position_likelihood,
    solver_type="pcg",
)

# ------------------------------------------------------------------ #
# Prior transformation and likelihood
# ------------------------------------------------------------------ #
print("[4] Building prior and likelihood ...")
print("[5] Running Nautilus sampler ...")
t0 = time.time()
stage = run_sampling(
    prob_model,
    SamplingConfig(
        n_live=300, n_eff=800, vectorized=True,
        n_batch=200, vectorized_chunk_size=50,
        stabilize_weights=True,
    ),
    tag="joint-fit", sampler_factory=Sampler,
)
print(f"  Sampling done in {time.time() - t0:.1f} s")
print("[6] Processing results ...")
samples, weights = stage.samples, stage.weights
param_names, log_z = stage.param_names, stage.log_z
quantiles = posterior_quantiles(stage)
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
result, diagnostics = prob_model.evaluate_with_diagnostics(
    best_params, include_components=True,
)
require_valid_reconstruction(result, diagnostics, context="plotting joint fit")
source_pixels = result.source_pixels
lens_amplitudes = result.lens_amplitudes
xmin, xmax, ymin, ymax = result.source_bbox

# Build 2D images
npix = image_data.shape[0]

model_image = np.asarray(result.model_image)
source_image_2d = np.asarray(diagnostics.source_image)
lens_image_2d = np.asarray(diagnostics.lens_light_image)

resid_norm = np.asarray(diagnostics.normalized_residuals)

# The effective degrees of freedom of a regularized semi-linear inversion is
# not ``Ndata - Nsource``. Report chi-square per fitted datum explicitly.
n_data = int((~mask).sum())
chi2 = diagnostics.chi2
chi2_per_data = chi2 / n_data

# Arc-local residual diagnostics. The reconstructed lensed-source component
# defines the arc support, so this remains usable on real data without truth
# images. Correlation with the arc template catches coherent under/over-fitting
# that a global chi-square can hide.
arc_region = (~mask) & (source_image_2d / noise_map > 3.0)
arc_resid = resid_norm[arc_region]
arc_template = source_image_2d[arc_region] / noise_map[arc_region]
if arc_resid.size == 0:
    arc_resid_mean = float("nan")
    arc_resid_std = float("nan")
else:
    arc_resid_mean = float(np.mean(arc_resid))
    arc_resid_std = float(np.std(arc_resid))
if (
    arc_resid.size < 2
    or np.std(arc_resid) == 0.0
    or np.std(arc_template) == 0.0
):
    arc_template_corr = float("nan")
else:
    arc_template_corr = float(np.corrcoef(arc_resid, arc_template)[0, 1])
print(
    "  Residual diagnostics: "
    f"chi2/Ndata={chi2_per_data:.4f}, "
    f"arc mean={arc_resid_mean:.4f}, arc std={arc_resid_std:.4f}, "
    f"arc-template corr={arc_template_corr:.4f}"
)
np.savetxt(
    "output/fit_diagnostics.csv",
    np.asarray([[chi2_per_data, arc_resid_mean, arc_resid_std, arc_template_corr]]),
    delimiter=",",
    header="chi2_per_data,arc_resid_mean,arc_resid_std,arc_template_corr",
    comments="",
)

# Source reconstruction on source plane
source_pixels_np = np.array(source_pixels)
source_image = source_pixels_np.reshape(prob_model.sim_obj.source_n, prob_model.sim_obj.source_n)

ext_i = [-npix * DPIX / 2,  npix * DPIX / 2,
         -npix * DPIX / 2,  npix * DPIX / 2]
ext_s = [float(xmin), float(xmax), float(ymin), float(ymax)]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
vmax = np.nanpercentile(image_data[~mask], 99.5)

# Row 1: Data, Model, Residual, Source
im0 = draw_image_panel(
    axes[0, 0],
    image_data,
    extent=ext_i,
    title='Observed image',
    cmap='viridis',
    vmin=0,
    vmax=vmax,
    ylabel='arcsec',
)

im1 = draw_image_panel(
    axes[0, 1],
    model_image,
    extent=ext_i,
    title='Joint model (source + lens light)',
    cmap='viridis',
    vmin=0,
    vmax=vmax,
)

resid_display = np.where(mask, np.nan, resid_norm)
im2 = draw_image_panel(
    axes[0, 2],
    resid_display,
    extent=ext_i,
    title=f'Norm. residual (sigma)\nchi^2/Ndata = {chi2_per_data:.3f}',
    cmap='RdBu_r',
    vmin=-3,
    vmax=3,
)

im3 = draw_image_panel(
    axes[0, 3],
    source_image,
    extent=ext_s,
    title=f'Source reconstruction\n(lambda={diagnostics.lambda_reg:.2e})',
    cmap='viridis',
    ylabel='arcsec',
)

# Row 2: Separated components
im4 = draw_image_panel(
    axes[1, 0],
    source_image_2d,
    extent=ext_i,
    title='Lensed source only',
    cmap='viridis',
    vmin=0,
    vmax=vmax,
    ylabel='arcsec',
)

im5 = draw_image_panel(
    axes[1, 1],
    lens_image_2d,
    extent=ext_i,
    title=f'Lens light only\n(Ie={float(lens_amplitudes[0]):.3f})',
    cmap='viridis',
    vmin=0,
    vmax=vmax,
)

im6 = draw_image_panel(
    axes[1, 2],
    image_data - lens_image_2d,
    extent=ext_i,
    title='Data - lens light',
    cmap='viridis',
    vmin=0,
    vmax=vmax,
)

# Mask overlay
mask_overlay = np.zeros((*mask.shape, 3))
mask_overlay[mask] = [0.5, 0.5, 0.5]  # Grey for masked
im7 = axes[1, 3].imshow(mask_overlay, origin="lower", extent=ext_i)
axes[1, 3].set_title("Masks (grey=masked)", fontsize=11)
axes[1, 3].set_xlabel("arcsec")

plt.suptitle(
    f"Joint Fit: SIE + Sersic lens light + pix src  "
    f"(theta_E={q50_list[0]:.3f}\", R_sersic={q50_list[6]:.3f}\")",
    fontsize=12,
)
with ck.ActiveContext(prob_model):
    prob_model.fill_params(best_params)
    overlay_critical_and_caustics(
        image_axes=[axes[0, 0], axes[0, 1], axes[1, 0]],
        source_ax=axes[0, 3],
        lens_mass=prob_model.phys_model,
    )

plt.tight_layout()
plt.savefig("output/fit_map_visualization.png", dpi=150, bbox_inches="tight")
print("  Saved output/fit_map_visualization.png")
plt.show()

print("\n" + "="*60)
print("Done.")
print("="*60)
