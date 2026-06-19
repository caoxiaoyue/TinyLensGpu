"""
Four-stage inference pipeline for lensing data with pre-subtracted lens light.

The input ``image_no_lens.fits`` already has the lens light removed, so this
pipeline skips lens-light modelling entirely and focuses on the mass model and
source reconstruction.

Stage a  : SIE + shear + MGE source light (uniform priors, NO lens light)
Stage b  : build an arc feature mask from image_no_lens S/N
Stage m1 : SIE + shear + pixelized source — GPU grid search for lambda_reg
           (SIE+shear fixed at stage-A medians; only lambda_reg free)
Stage m2 : EPL + shear + pixelized source — Nautilus nested sampling
           (Gaussian priors from stage a on EPL+shear; lambda_reg fixed at M1 best)

Each stage pickles its posterior samples/weights to
``output/stage_{a,m1,m2}.pkl`` and is re-runnable via ``--skip-done``.

Usage::

    # From no_lens_light/
    python model.py
    python model.py --skip-done
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.chdir(Path(__file__).parent)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from nautilus import Sampler

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import ImageProbModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
from TinyLensGpu.PhysicalModel import (
    PhysicalModel,
    PixelizedSourceModel,
    Shear,
)
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import GaussianEllipse
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE, EPL
from TinyLensGpu.utils import generate_radial_basis_knots
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.utils.misc import arc_mask_from, weighted_quantile
from TinyLensGpu.visualizer import plot_model_results, overlay_critical_and_caustics

from TinyLensGpu.Inference import GaussianPriorPasser

import caskade as ck
import jax.scipy.linalg as jsl

# ------------------------------------------------------------------ #
NSRCX = 40
NSRCY = 40
DPIX = 0.05
NSUB = 4
NSUB_PIX = 4
N_GAUSSIANS_SRC = 10
MASK_RADIUS = 2.5
NOISE_MASK_THRESHOLD = 1e7  # noise_map pixels above this are pre-masked
OUT_DIR = Path("output")
DATA_DIR = Path("data")


# ------------------------------------------------------------------ #
def _run_sampler(likelihood, n_live: int, n_eff: int, tag: str, vectorized: bool = True):
    prior, prior_specs = make_prior_transformation(likelihood)
    param_names = [spec.name for spec in prior_specs]
    print(f"\n[{tag}] {len(param_names)} dynamic params:")
    for spec in prior_specs:
        print(f"    {spec.name:25s} {spec.describe()}")

    loglike = make_likelihood(likelihood, vectorized=vectorized)
    sampler_kwargs = dict(n_live=n_live, vectorized=vectorized)
    if vectorized:
        sampler_kwargs["n_batch"] = 100
    sampler = Sampler(prior, loglike, n_dim=len(param_names), **sampler_kwargs)
    sampler.run(verbose=True, n_eff=n_eff)

    samples, log_w, _ = sampler.posterior()
    samples = np.asarray(samples, dtype=np.float64)
    weights = np.exp(np.asarray(log_w, dtype=np.float64))
    weights /= weights.sum()

    log_z = float(sampler.log_z)
    print(f"[{tag}] log_z = {log_z:.3f}")
    return samples, weights, param_names, log_z


def _dump_stage(tag: str, samples, weights, param_names, log_z, extra=None):
    OUT_DIR.mkdir(exist_ok=True)
    payload = dict(
        samples=samples, weights=weights,
        param_names=param_names, log_z=log_z,
        extra=extra or {},
    )
    with open(OUT_DIR / f"stage_{tag}.pkl", "wb") as f:
        pickle.dump(payload, f)
    print(f"[{tag}] posterior saved to output/stage_{tag}.pkl")


def _load_stage(tag: str):
    with open(OUT_DIR / f"stage_{tag}.pkl", "rb") as f:
        return pickle.load(f)


def _posterior_median(samples, weights, param_names):
    """Return dict of weighted medians keyed by param name."""
    out = {}
    q = np.array([0.5])
    for i, name in enumerate(param_names):
        out[name] = float(weighted_quantile(
            np.asarray(samples[:, i]), weights, q))
    return out


def _print_summary(tag: str, samples, weights, param_names):
    print(f"\n[{tag}] Posterior summary:")
    q = np.array([0.16, 0.5, 0.84])
    for i, name in enumerate(param_names):
        qs = weighted_quantile(np.asarray(samples[:, i]), weights, q)
        q16, q50, q84 = float(qs[0]), float(qs[1]), float(qs[2])
        print(f"    {name:25s} = {q50:+.4f} ({q16-q50:+.4f}, {q84-q50:+.4f})")


# ------------------------------------------------------------------ #
def _make_circular_mask(image_shape, dpix, radius_arcsec=3.5):
    """Return a boolean mask where True = pixels outside ``radius_arcsec``."""
    ny, nx = image_shape
    y = (np.arange(ny) - (ny - 1) / 2) * dpix
    x = (np.arange(nx) - (nx - 1) / 2) * dpix
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return (xx ** 2 + yy ** 2) > radius_arcsec ** 2


# ------------------------------------------------------------------ #
# Stage A — SIE + shear + MGE source light  (NO lens light)
# ------------------------------------------------------------------ #
def build_stage_a_likelihood(image_data, noise_map, psf_kernel, circular_mask=None):
    sie = SIE(
        theta_E=ParamU("theta_E", 1.5, prior_type="uniform",
                       prior_settings=[0.5, 3.0], limits=[0.0, 5.0]),
        e1=ParamU("e1_mass", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        e2=ParamU("e2_mass", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        center_x=ParamU("center_x_mass", 0.0, prior_type="gaussian",
                        prior_settings=[0.0, 0.1], limits=[-1.0, 1.0]),
        center_y=ParamU("center_y_mass", 0.0, prior_type="gaussian",
                        prior_settings=[0.0, 0.1], limits=[-1.0, 1.0]),
    )
    shear = Shear(
        gamma1=ParamU("gamma1", 0.0, prior_type="uniform",
                      prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
        gamma2=ParamU("gamma2", 0.0, prior_type="uniform",
                      prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
    )
    for p in (sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y,
              shear.gamma1, shear.gamma2):
        p.to_dynamic()

    # MGE source
    cx_s = ParamU("center_x_src", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.5], limits=[-3.0, 3.0])
    cy_s = ParamU("center_y_src", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.5], limits=[-3.0, 3.0])
    e1_s = ParamU("e1_src", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_s = ParamU("e2_src", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    sigma_list_src = generate_radial_basis_knots(
        dpix=DPIX, n_sigmas=N_GAUSSIANS_SRC,
        log_rmin=-2.0, log_rmax=np.log10(1.0), mode="mge"
    )
    source_gaussians = []
    for i, sigma in enumerate(sigma_list_src):
        gauss = GaussianEllipse(
            sigma=ParamU(f"sigma_src_{i}", float(sigma)),
            center_x=cx_s,
            center_y=cy_s,
            e1=e1_s,
            e2=e2_s,
            flux=ParamU(f"flux_src_{i}", 1.0),
        )
        gauss.sigma.to_static(float(sigma))
        gauss.flux.to_static(1.0)
        source_gaussians.append(gauss)
    cx_s.to_dynamic(); cy_s.to_dynamic(); e1_s.to_dynamic(); e2_s.to_dynamic()

    # NO lens light — the image is already lens-light-subtracted
    phys = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=source_gaussians,
        lens_light=[],
    )
    return ImageProbModel(
        image_data=image_data, noise_map=noise_map,
        psf_kernel=psf_kernel, dpix=DPIX, nsub=NSUB,
        phys_model=phys, use_linear=True, solver_type="nnls",
        mask=circular_mask,
    )


def run_stage_a(image_data, noise_map, psf_kernel, circular_mask=None):
    OUT_DIR.mkdir(exist_ok=True)
    print("\n" + "=" * 60)
    print(" Stage A : SIE + shear + MGE source light  (no lens light)")
    print("=" * 60)
    t0 = time.time()
    likelihood = build_stage_a_likelihood(image_data, noise_map, psf_kernel,
                                          circular_mask=circular_mask)
    samples, weights, names, logz = _run_sampler(
        likelihood, n_live=300, n_eff=2000, tag="stage-A", vectorized=True,
    )
    t1 = time.time()
    _print_summary("stage-A", samples, weights, names)
    print(f"[stage-A] time taken: {t1 - t0:.2f} seconds")

    medians = _posterior_median(samples, weights, names)
    q50 = [medians[n] for n in names]

    _dump_stage(
        "a", samples, weights, names, logz,
        extra=dict(
            medians=medians,
            time_taken=t1 - t0,
        ),
    )

    try:
        plot_model_results(
            likelihood, jnp.asarray(q50),
            save_path=str(OUT_DIR / "stage_a_model.png"),
            title="Stage A : MGE source (no lens light)",
            show_critical_lines=True,
            show_caustics=True,
        )
    except Exception as err:
        print(f"[stage-A] plotting failed (non-fatal): {err}")
    return samples, weights, names, medians


# ------------------------------------------------------------------ #
# Stage B — build an arc mask from image_no_lens S/N
# ------------------------------------------------------------------ #
def run_stage_b(image_data, noise_map, circular_mask=None):
    """Build arc mask directly from the lens-subtracted image.

    Since the lens light is already removed, the arc features are directly
    visible as positive S/N excess in ``image_data``.
    """
    print("\n" + "=" * 60)
    print(" Stage B : build arc feature mask from lens-subtracted image")
    print("=" * 60)

    # Pre-masked pixels from the noise map (interfering sources, etc.)
    noise_masked = noise_map > NOISE_MASK_THRESHOLD
    n_noise_masked = int(noise_masked.sum())
    print(f"[stage-B] pre-masked (noise={1e8:.0e}) pixels = {n_noise_masked} / {noise_masked.size}")

    # S/N map (set S/N=0 for pre-masked pixels to avoid spurious detections)
    snr_map = image_data / noise_map

    # arc_mask_from returns True for EXCLUDED pixels (arc region)
    arc_mask = arc_mask_from(snr_map, threshold=2.0,
                             ignor_size=20, ext_size=5, close_size=3)

    # Combine: arc mask + pre-masked (noise=1e8) + circular mask
    feature_mask = arc_mask | noise_masked
    if circular_mask is not None:
        feature_mask = feature_mask | circular_mask

    n_in = int((~feature_mask).sum())
    print(f"[stage-B] lens-light pixels kept = {n_in} / {feature_mask.size}")

    DATA_DIR.mkdir(exist_ok=True)
    fits.writeto(DATA_DIR / "feature_mask.fits",
                 feature_mask.astype(np.uint8), overwrite=True)

    # Quick-look figure
    ny_img, nx_img = image_data.shape
    extent = [-nx_img * DPIX / 2, nx_img * DPIX / 2,
              -ny_img * DPIX / 2, ny_img * DPIX / 2]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im0 = axes[0].imshow(image_data, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("image_no_lens (lens-subtracted data)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(snr_map, origin="lower", extent=extent, cmap="viridis",
                         vmin=-3, vmax=np.nanpercentile(snr_map, 99.5))
    axes[1].set_title("S/N map + arc mask boundary")
    axes[1].contour(~feature_mask, levels=[0.5], origin="lower", extent=extent,
                    colors="red", linewidths=1.5)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUT_DIR / "stage_b_mask.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return feature_mask


# ------------------------------------------------------------------ #
# Position likelihood from stage-A medians
# ------------------------------------------------------------------ #
def _position_likelihood_from_stage_a(medians_a: dict) -> dict:
    """Build a position-likelihood config dict from stage-A posterior medians."""
    required = (
        "theta_E",
        "e1_mass",
        "e2_mass",
        "center_x_mass",
        "center_y_mass",
        "gamma1",
        "gamma2",
        "center_x_src",
        "center_y_src",
    )
    missing = [k for k in required if k not in medians_a]
    if missing:
        raise KeyError(
            "Stage-A medians missing required keys for position likelihood: "
            + ", ".join(missing)
        )

    mass = PhysicalModel(
        lens_mass=[
            SIE(
                theta_E=float(medians_a["theta_E"]),
                e1=float(medians_a["e1_mass"]),
                e2=float(medians_a["e2_mass"]),
                center_x=float(medians_a["center_x_mass"]),
                center_y=float(medians_a["center_y_mass"]),
            ),
            Shear(
                gamma1=float(medians_a["gamma1"]),
                gamma2=float(medians_a["gamma2"]),
            ),
        ],
        source_light=[],
        lens_light=[],
    )

    solver = PointSourceProbModel(
        phys_model=mass,
        observed_positions=[[0.0, 0.0]],
        position_sigma=[0.01],  # dummy value
        source_x=float(medians_a["center_x_src"]),
        source_y=float(medians_a["center_y_src"]),
        source_position_fixed=True,
        solver="optimization",
        solver_config={
            "initial_range": 4.0,
            "n_x": 400,
            "n_y": 400,
            "k_keep": 30,
            "num_iters": 20,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        },
    )
    img_positions, _ = solver.solve_image_positions()
    img_positions = np.asarray(img_positions, dtype=np.float64)
    if img_positions.ndim != 2 or img_positions.shape[1] != 2:
        raise RuntimeError(
            f"Lens-equation solver returned invalid image_positions shape={img_positions.shape}"
        )
    if img_positions.shape[0] < 2:
        raise RuntimeError(
            "Lens-equation solver returned fewer than 2 lensed image positions. "
            "This pipeline assumes a strong-lensing configuration; treat this as a solver failure."
        )

    threshold_arcsec = 0.1
    min_log_like = -1.0e10
    print(f"[pos-like] solved {img_positions.shape[0]} lensed image positions from stage-A medians")
    for p in img_positions:
        print(f"    ({p[0]:+.4f}, {p[1]:+.4f})")
    return dict(
        positions=img_positions.tolist(),
        threshold_arcsec=float(threshold_arcsec),
        min_log_like=float(min_log_like),
    )


# ------------------------------------------------------------------ #
# Helpers for Stage M1 / M2
# ------------------------------------------------------------------ #
def _sie_mass_from_stage_a(medians_a: dict):
    """SIE (+ shear) with all parameters FIXED at stage-A posterior medians."""
    sie = SIE(
        theta_E=ParamU("theta_E", float(medians_a["theta_E"])),
        e1=ParamU("e1_mass", float(medians_a["e1_mass"])),
        e2=ParamU("e2_mass", float(medians_a["e2_mass"])),
        center_x=ParamU("center_x_mass", float(medians_a["center_x_mass"])),
        center_y=ParamU("center_y_mass", float(medians_a["center_y_mass"])),
    )
    for p in (sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y):
        p.to_static()

    shear = Shear(
        gamma1=ParamU("gamma1", float(medians_a["gamma1"])),
        gamma2=ParamU("gamma2", float(medians_a["gamma2"])),
    )
    shear.gamma1.to_static()
    shear.gamma2.to_static()
    return sie, shear


def _epl_mass_from_stage_a(passer: GaussianPriorPasser):
    """EPL (+ shear) with Gaussian priors inherited from stage-A posterior."""
    theta_E = passer.gaussian(
        "theta_E", model="EPL", attr="theta_E", limits=[0.0, 5.0],
    )
    e1m = passer.gaussian(
        "e1_mass", model="EPL", attr="e1", limits=[-1.0, 1.0],
    )
    e2m = passer.gaussian(
        "e2_mass", model="EPL", attr="e2", limits=[-1.0, 1.0],
    )
    cx = passer.gaussian(
        "center_x_mass", model="EPL", attr="center_x", limits=[-1.0, 1.0],
    )
    cy = passer.gaussian(
        "center_y_mass", model="EPL", attr="center_y", limits=[-1.0, 1.0],
    )
    gamma = ParamU(
        "gamma",
        2.0,
        prior_type="truncated_gaussian",
        prior_settings=[2.0, 0.2],
        limits=[1.0, 3.0],
    )

    epl = EPL(theta_E=theta_E, gamma=gamma, e1=e1m, e2=e2m, center_x=cx, center_y=cy)
    for p in (epl.theta_E, epl.gamma, epl.e1, epl.e2, epl.center_x, epl.center_y):
        p.to_dynamic()

    shear = Shear(
        gamma1=passer.gaussian(
            "gamma1", model="Shear", attr="gamma1", limits=[-0.5, 0.5],
        ),
        gamma2=passer.gaussian(
            "gamma2", model="Shear", attr="gamma2", limits=[-0.5, 0.5],
        ),
    )
    shear.gamma1.to_dynamic()
    shear.gamma2.to_dynamic()
    return epl, shear


# ------------------------------------------------------------------ #
# Stage M1 — SIE + shear + pixelized source  (GPU grid search for λ)
# ------------------------------------------------------------------ #
def build_stage_m1_likelihood(
    image_data, noise_map, psf_kernel, feature_mask,
    medians_a, position_likelihood, circular_mask=None,
):
    """Build likelihood for M1: SIE+shear FIXED at stage-A medians, only λ free."""
    sie, shear = _sie_mass_from_stage_a(medians_a)

    log_lam = ParamU(
        "log_lambda_reg",
        0.0,
        prior_type="uniform",
        prior_settings=[-13.815510557964274, 13.815510557964274],
        limits=[-13.815510557964274, 13.815510557964274],
    )
    log_lam.to_dynamic()

    pix_src = PixelizedSourceModel(
        nx=NSRCX,
        ny=NSRCY,
        log_lambda_reg=log_lam,
        regularization_type="first-order",
    )
    phys = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=[pix_src],
        lens_light=[],
    )
    combined_mask = feature_mask
    if circular_mask is not None:
        combined_mask = combined_mask | circular_mask
    return PixelizedImageProbModelOperator(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
    )


def _plot_m1_grid(likelihood, log_lam_coarse, log_ev_coarse,
                   log_lam_fine, log_ev_fine, log_lam_best, log_ev_best):
    """2-panel diagnostic: log-evidence vs lambda_reg for both grid stages."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.semilogx(np.asarray(log_lam_coarse), np.asarray(log_ev_coarse), 'b.-', markersize=2)
    ax1.axvline(log_lam_best, color='r', linestyle='--', alpha=0.8,
                label=f'best λ = {log_lam_best:.3e}')
    ax1.set_xlabel('λ_reg'); ax1.set_ylabel('log-evidence')
    ax1.set_title('Stage M1: Coarse grid (200 pts, 1e-8 – 1e8)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.semilogx(np.asarray(log_lam_fine), np.asarray(log_ev_fine), 'b.-', markersize=2)
    ax2.axvline(log_lam_best, color='r', linestyle='--', alpha=0.8,
                label=f'best λ = {log_lam_best:.3e}')
    ax2.set_xlabel('λ_reg'); ax2.set_ylabel('log-evidence')
    ax2.set_title('Stage M1: Refinement (±0.5 dex, 200 pts)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Stage M1 — λ grid search  (best log-ev = {log_ev_best:.2f})', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "stage_m1_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage-M1] grid diagnostic saved to {OUT_DIR / 'stage_m1_grid.png'}")


def run_stage_m1(image_data, noise_map, psf_kernel, feature_mask,
                  medians_a, position_likelihood, circular_mask=None):
    """Stage M1: GPU-batch grid search for lambda_reg (coarse → refine).

    ``image_data`` is already lens-subtracted so we use it directly.
    """
    print("\n" + "=" * 60)
    print(" Stage M1 : SIE + shear + pix source  (GPU grid search for λ)")
    print("=" * 60)
    t0 = time.time()

    likelihood = build_stage_m1_likelihood(
        image_data, noise_map, psf_kernel, feature_mask,
        medians_a, position_likelihood, circular_mask=circular_mask,
    )

    # Batched log-evidence evaluator (jax.vmap + jit → GPU)
    loglike_batch = make_likelihood(likelihood, vectorized=True)

    n_grid = 200

    # --- Coarse grid: 200 linearly-spaced points in log-λ over [ln(1e-8), ln(1e8)] ---
    log_lam_min, log_lam_max = jnp.log(1e-8), jnp.log(1e8)
    print(f"[stage-M1] Running coarse grid (200 pts, λ in [{float(jnp.exp(log_lam_min)):.1e}, {float(jnp.exp(log_lam_max)):.1e}]) ...")
    log_lam_grid_coarse = jnp.linspace(log_lam_min, log_lam_max, n_grid)
    log_ev_coarse = jnp.asarray(loglike_batch(log_lam_grid_coarse.reshape(-1, 1)))

    if not jnp.any(jnp.isfinite(log_ev_coarse)):
        raise RuntimeError(
            "[stage-M1] All log-λ values in coarse grid produced non-finite log-evidence "
            "— PCG solver likely failed globally. Check data, noise map, and PSF."
        )

    best_idx_coarse = int(jnp.argmax(log_ev_coarse))
    log_lam_best_coarse = float(log_lam_grid_coarse[best_idx_coarse])
    log_ev_best_coarse = float(log_ev_coarse[best_idx_coarse])
    print(f"[stage-M1] Coarse best: λ = {float(jnp.exp(log_lam_best_coarse)):.4e}  (log-ev = {log_ev_best_coarse:.2f})")

    # --- Refinement grid: 200 points around coarse optimum (±0.5 dex in log10) ---
    half_width = 0.5 * jnp.log(10)  # 0.5 dex in natural log
    log_lam_grid_fine = jnp.linspace(log_lam_best_coarse - half_width,
                                     log_lam_best_coarse + half_width, n_grid)
    print(f"[stage-M1] Running refinement grid (200 pts, λ in [{float(jnp.exp(log_lam_grid_fine[0])):.4e}, {float(jnp.exp(log_lam_grid_fine[-1])):.4e}]) ...")
    log_ev_fine = jnp.asarray(loglike_batch(log_lam_grid_fine.reshape(-1, 1)))

    best_idx_fine = int(jnp.argmax(log_ev_fine))
    log_lam_best = float(log_lam_grid_fine[best_idx_fine])
    log_ev_best = float(log_ev_fine[best_idx_fine])
    print(f"[stage-M1] Refined best: λ = {float(jnp.exp(log_lam_best)):.4e}  (log-ev = {log_ev_best:.2f})")

    if best_idx_fine == 0 or best_idx_fine == n_grid - 1:
        print(
            f"[stage-M1] WARNING: refinement optimum at grid edge "
            f"(idx={best_idx_fine}, λ={float(jnp.exp(log_lam_best)):.4e}). "
            f"The true optimum may lie outside the ±0.5 dex refinement window. "
            f"Consider widening the refinement range or inspecting the coarse grid."
        )

    t1 = time.time()
    print(f"\n[stage-M1] Grid search summary:")
    print(f"    {'lambda_reg':25s} = {float(jnp.exp(log_lam_best)):+.4e}")
    print(f"[stage-M1] time taken: {t1 - t0:.2f} seconds")

    _dump_stage(
        "m1", None, None, ["log_lambda_reg"], log_ev_best,
        extra=dict(
            lambda_best=log_lam_best,  # log-space
            lambda_grid_coarse=np.asarray(log_lam_grid_coarse, dtype=np.float64),
            log_ev_coarse=np.asarray(log_ev_coarse, dtype=np.float64),
            lambda_grid_fine=np.asarray(log_lam_grid_fine, dtype=np.float64),
            log_ev_fine=np.asarray(log_ev_fine, dtype=np.float64),
            time_taken=t1 - t0,
        ),
    )

    try:
        _plot_m1_grid(likelihood,
                       jnp.exp(log_lam_grid_coarse), log_ev_coarse,
                       jnp.exp(log_lam_grid_fine), log_ev_fine,
                       float(jnp.exp(log_lam_best)), log_ev_best)
    except Exception as err:
        print(f"[stage-M1] grid plot failed (non-fatal): {err}")

    # Generate M1 pixelized source reconstruction plot
    try:
        medians_m1 = {**medians_a, "log_lambda_reg": log_lam_best}
        _plot_pix_stage("stage-M1", likelihood, medians_m1, ["log_lambda_reg"],
                        str(OUT_DIR / "stage_m1_model.png"))
    except Exception as err:
        print(f"[stage-M1] pix_stage plot failed (non-fatal): {err}")

    return log_lam_best


# ------------------------------------------------------------------ #
# Stage M2 — EPL + shear + pixelized source  (λ fixed from M1)
# ------------------------------------------------------------------ #
def _plot_pix_stage(tag, likelihood, medians, param_names, save_path):
    """4-panel diagnostic: data | model | norm-residual | source."""
    q50 = [medians[n] for n in param_names]
    image_data = np.asarray(likelihood.image_data)
    noise_map  = np.asarray(likelihood.noise_map)
    mask       = ~np.asarray(likelihood.unmask, dtype=bool)  # True = excluded

    has_pos_penalty = False
    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.array(q50))
        lam_val = jnp.exp(likelihood.phys_model.source_light[0].log_lambda_reg.value)
        lambda_j = jnp.asarray(lam_val)

        xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub = likelihood._get_bbox()
        reg_data, _, reg_matrix_dense = likelihood._regularization_data(xmin, xmax, ymin, ymax)
        op_data = likelihood.sim_obj.precompute_operator_data(
            xmin, xmax, ymin, ymax, _betas_sub=(beta_x_sub, beta_y_sub),
        )
        P, P_chol = likelihood.sim_obj.build_preconditioner(
            likelihood.noise_1d, xmin, xmax, ymin, ymax, lambda_j, reg_matrix_dense,
        )
        source_pixels, pcg_info = likelihood._solve_source(
            xmin, xmax, ymin, ymax, lambda_j, reg_data, P_chol, op_data=op_data,
        )
        model_1d_j = likelihood.sim_obj.forward_model(
            source_pixels, xmin, xmax, ymin, ymax, op_data=op_data,
        )

        inv_P_R = jsl.cho_solve((P_chol, True), reg_matrix_dense)
        N_eff = float(reg_matrix_dense.shape[0] - lambda_j * jnp.trace(inv_P_R))

        has_pos_penalty = likelihood._has_pos_penalty
        if has_pos_penalty:
            pos_penalty = float(likelihood._position_likelihood_penalty_jax())
            beta_x, beta_y = likelihood.phys_model.deflection(likelihood._pos_px, likelihood._pos_py)
            dx = beta_x[:, None] - beta_x[None, :]
            dy = beta_y[:, None] - beta_y[None, :]
            dist = jnp.sqrt(dx * dx + dy * dy)
            max_sep = float(jnp.max(dist))

            print(f"[{tag}] Position likelihood penalty: {pos_penalty:.4e}")
            print(f"[{tag}] Maximum source-plane separation of marked images: {max_sep:.4e} arcsec")

            beta_x_np = np.array(beta_x)
            beta_y_np = np.array(beta_y)
            pos_px_np = np.array(likelihood._pos_px)
            pos_py_np = np.array(likelihood._pos_py)

    model_1d = np.array(model_1d_j)
    model_image = np.zeros(image_data.shape)
    model_image[~mask] = model_1d
    resid_norm = (image_data - model_image) / noise_map

    chi2 = float(np.sum(resid_norm[~mask] ** 2))
    dof  = int((~mask).sum()) - N_eff
    chi2_nu = chi2 / dof if dof > 0 else 0.0

    nx = likelihood.phys_model.source_light[0].nx
    ny = likelihood.phys_model.source_light[0].ny
    src_img = np.array(source_pixels).reshape(ny, nx)

    npix = image_data.shape[0]
    ext_i = [-npix * DPIX / 2, npix * DPIX / 2, -npix * DPIX / 2, npix * DPIX / 2]
    ext_s = [float(xmin), float(xmax), float(ymin), float(ymax)]
    vmax  = np.nanpercentile(image_data[~mask], 99.5)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    # Compute bounding box of unmasked pixels for panels 1-3
    rows_unmasked, cols_unmasked = np.where(~mask)
    pad = 3  # pixels of padding
    row_min = max(rows_unmasked.min() - pad, 0)
    row_max = min(rows_unmasked.max() + pad, npix - 1)
    col_min = max(cols_unmasked.min() - pad, 0)
    col_max = min(cols_unmasked.max() + pad, npix - 1)
    xlim_unmasked = (-npix * DPIX / 2 + col_min * DPIX,
                     -npix * DPIX / 2 + (col_max + 1) * DPIX)
    ylim_unmasked = (-npix * DPIX / 2 + row_min * DPIX,
                     -npix * DPIX / 2 + (row_max + 1) * DPIX)
    for ax, img, title, kw in [
        (axes[0], image_data,  "Data (lens-subtracted)", dict(vmin=0, vmax=vmax, cmap="viridis")),
        (axes[1], model_image, "Model image",            dict(vmin=0, vmax=vmax, cmap="viridis")),
        (axes[2], np.where(mask, np.nan, resid_norm),
                               f"Norm. residual\nχ²/ν={chi2_nu:.3f}",
                               dict(vmin=-3, vmax=3, cmap="RdBu_r")),
    ]:
        im = ax.imshow(img, origin="lower", extent=ext_i, **kw)
        if ax == axes[0] and has_pos_penalty:
            ax.plot(pos_px_np, pos_py_np, 'rx', markersize=8, label='Marked pos')
            ax.legend(loc='upper right', fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("arcsec")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    im3 = axes[3].imshow(src_img, origin="lower", extent=ext_s, cmap="viridis")
    if has_pos_penalty:
        axes[3].plot(beta_x_np, beta_y_np, 'rx', markersize=8, label='Traced pos')
        axes[3].legend(loc='upper right', fontsize=8)
    axes[3].set_title(f"Source reconstruction\n(λ={float(lam_val):.2e})", fontsize=11)
    axes[3].set_xlabel("arcsec")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    axes[0].set_ylabel("arcsec")
    # Panels 1-3: zoom to unmasked pixel region
    for ax in axes[:3]:
        ax.set_xlim(*xlim_unmasked)
        ax.set_ylim(*ylim_unmasked)

    lbl = "  ".join(f"{n}={medians[n]:+.4f}" for n in
                    ("theta_E", "gamma", "e1_mass", "e2_mass") if n in medians)
    plt.suptitle(f"[{tag}]  {lbl}", fontsize=10)
    overlay_critical_and_caustics(
        image_axes=[axes[0], axes[1]],
        source_ax=axes[3],
        lens_mass=likelihood.phys_model,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] diagnostic plot saved to {save_path}")


def build_stage_m2_likelihood(
    image_data, noise_map, psf_kernel, feature_mask,
    passer: GaussianPriorPasser, position_likelihood, log_lambda_fixed: float,
    circular_mask=None,
):
    """Build likelihood for M2: EPL+shear with Gaussian priors, λ FIXED at M1 best."""
    epl, shear = _epl_mass_from_stage_a(passer)

    log_lam = ParamU("log_lambda_reg", float(log_lambda_fixed))
    log_lam.to_static()

    pix_src = PixelizedSourceModel(
        nx=NSRCX,
        ny=NSRCY,
        log_lambda_reg=log_lam,
        regularization_type="first-order",
    )
    phys = PhysicalModel(
        lens_mass=[epl, shear],
        source_light=[pix_src],
        lens_light=[],
    )
    combined_mask = feature_mask
    if circular_mask is not None:
        combined_mask = combined_mask | circular_mask
    return PixelizedImageProbModelOperator(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
    )


def run_stage_m2(image_data, noise_map, psf_kernel, feature_mask,
                  samples_a, weights_a, names_a,
                  position_likelihood, log_lambda_fixed: float, circular_mask=None):
    """Stage M2: Nautilus sampling of EPL+shear with λ fixed at M1 optimum.

    ``image_data`` is already lens-subtracted so we use it directly.
    """
    print("\n" + "=" * 60)
    print(" Stage M2 : EPL + shear + pix source (λ fixed from M1)")
    print("=" * 60)
    print(f"[stage-M2] lambda_reg fixed = {float(jnp.exp(log_lambda_fixed)):.4e}")
    t0 = time.time()
    passer = GaussianPriorPasser(samples_a, weights_a, names_a)
    likelihood = build_stage_m2_likelihood(
        image_data,
        noise_map,
        psf_kernel,
        feature_mask,
        passer,
        position_likelihood,
        log_lambda_fixed,
        circular_mask=circular_mask,
    )
    samples, weights, names, logz = _run_sampler(
        likelihood, n_live=300, n_eff=600, tag="stage-M2", vectorized=True,
    )
    t1 = time.time()
    _print_summary("stage-M2", samples, weights, names)
    print(f"[stage-M2] time taken: {t1 - t0:.2f} seconds")
    medians = _posterior_median(samples, weights, names)
    _dump_stage("m2", samples, weights, names, logz, extra=dict(
        medians=medians,
        lambda_fixed=log_lambda_fixed,
        time_taken=t1 - t0,
    ))
    try:
        _plot_pix_stage("stage-M2", likelihood, medians, names,
                        str(OUT_DIR / "stage_m2_model.png"))
    except Exception as err:
        print(f"[stage-M2] plotting failed (non-fatal): {err}")
    return samples, weights, names, medians


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def main(skip_done: bool = False):
    image_data, noise_map, psf_kernel, _ = load_lens_data(
        image_path=str(DATA_DIR / "image_no_lens.fits"),
        noise_path=str(DATA_DIR / "noise_map.fits"),
        psf_path=str(DATA_DIR / "psf_map.fits"),
    )

    # Circular mask applied to all stages: exclude pixels > MASK_RADIUS arcsec from centre
    circular_mask = _make_circular_mask(image_data.shape, DPIX, radius_arcsec=MASK_RADIUS)
    n_excl = int(circular_mask.sum())
    print(f"[circular mask] excluded {n_excl} / {circular_mask.size} pixels (> {MASK_RADIUS} arcsec)")

    # ---- stage A ---------------------------------------------------- #
    time_a = 0.0
    if skip_done and (OUT_DIR / "stage_a.pkl").exists():
        print("[stage-A] loading cached output/stage_a.pkl")
        d = _load_stage("a")
        samples_a, weights_a, names_a = d["samples"], d["weights"], d["param_names"]
        medians_a = d["extra"]["medians"]
        time_a = d["extra"].get("time_taken", 0.0)
    else:
        samples_a, weights_a, names_a, medians_a = run_stage_a(
            image_data, noise_map, psf_kernel, circular_mask=circular_mask,
        )
        time_a = _load_stage("a")["extra"].get("time_taken", 0.0)

    # ---- stage B ---------------------------------------------------- #
    # Build arc mask directly from lens-subtracted image (no lens_light_model needed)
    feature_mask = run_stage_b(image_data, noise_map, circular_mask=circular_mask)

    # ---- position likelihood ----------------------------------------- #
    position_likelihood = _position_likelihood_from_stage_a(medians_a)

    # ---- stage M1 --------------------------------------------------- #
    time_m1 = 0.0
    log_lambda_m1 = None
    if skip_done and (OUT_DIR / "stage_m1.pkl").exists():
        print("[stage-M1] loading cached output/stage_m1.pkl")
        d = _load_stage("m1")
        log_lambda_m1 = d["extra"]["lambda_best"]
        time_m1 = d["extra"].get("time_taken", 0.0)
    else:
        log_lambda_m1 = run_stage_m1(
            image_data, noise_map, psf_kernel, feature_mask,
            medians_a, position_likelihood,
            circular_mask=circular_mask,
        )
        time_m1 = _load_stage("m1")["extra"].get("time_taken", 0.0)

    if log_lambda_m1 is None:
        raise RuntimeError(
            "[stage-M1] Failed to determine lambda_reg — "
            "stage_m1.pkl may be corrupted (missing 'lambda_best' in extra dict)."
        )

    # If M1 posteriors exist but the plot is missing, re-plot now
    if (OUT_DIR / "stage_m1.pkl").exists() and not (OUT_DIR / "stage_m1_model.png").exists():
        print("[stage-M1] stage_m1_model.png missing — re-plotting")
        lkl_m1_replot = build_stage_m1_likelihood(
            image_data, noise_map, psf_kernel, feature_mask,
            medians_a, position_likelihood, circular_mask=circular_mask,
        )
        medians_m1_replot = {**medians_a, "log_lambda_reg": log_lambda_m1}
        try:
            _plot_pix_stage("stage-M1", lkl_m1_replot, medians_m1_replot, ["log_lambda_reg"],
                            str(OUT_DIR / "stage_m1_model.png"))
        except Exception as err:
            print(f"[stage-M1] re-plotting failed (non-fatal): {err}")

    # ---- stage M2 --------------------------------------------------- #
    time_m2 = 0.0
    if skip_done and (OUT_DIR / "stage_m2.pkl").exists():
        print("[stage-M2] loading cached output/stage_m2.pkl")
        d = _load_stage("m2")
        samples_m2, weights_m2, names_m2 = d["samples"], d["weights"], d["param_names"]
        medians_m2 = d["extra"]["medians"]
        time_m2 = d["extra"].get("time_taken", 0.0)
    else:
        samples_m2, weights_m2, names_m2, medians_m2 = run_stage_m2(
            image_data, noise_map, psf_kernel, feature_mask,
            samples_a, weights_a, names_a,
            position_likelihood, log_lambda_m1, circular_mask=circular_mask,
        )
        time_m2 = _load_stage("m2")["extra"].get("time_taken", 0.0)

    # Re-plot M2 if the png is missing but posteriors exist
    if not (OUT_DIR / "stage_m2_model.png").exists():
        passer_m2 = GaussianPriorPasser(samples_a, weights_a, names_a)
        lkl_m2 = build_stage_m2_likelihood(
            image_data,
            noise_map,
            psf_kernel,
            feature_mask,
            passer_m2,
            position_likelihood,
            log_lambda_m1,
            circular_mask=circular_mask,
        )
        try:
            _plot_pix_stage("stage-M2", lkl_m2, medians_m2, names_m2,
                            str(OUT_DIR / "stage_m2_model.png"))
        except Exception as err:
            print(f"[stage-M2] plotting failed (non-fatal): {err}")

    print("\n" + "=" * 60)
    print(" Pipeline complete")
    print("=" * 60)
    print(f" Time summary:")
    print(f"    Stage A:  {time_a/60:.2f} min")
    print(f"    Stage M1: {time_m1/60:.2f} min")
    print(f"    Stage M2: {time_m2/60:.2f} min")
    print(f"    Total:    {(time_a + time_m1 + time_m2)/60:.2f} min\n")
    print(f"    M1 best lambda_reg     = {float(jnp.exp(log_lambda_m1)):.4e}")

    for k in ("theta_E", "gamma", "e1_mass", "e2_mass",
              "center_x_mass", "center_y_mass", "gamma1", "gamma2"):
        if k in medians_m2:
            print(f"    final  {k:15s} = {medians_m2[k]:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-done", action="store_true",
                        help="Re-use cached posteriors in output/stage_*.pkl")
    args = parser.parse_args()
    main(skip_done=args.skip_done)
