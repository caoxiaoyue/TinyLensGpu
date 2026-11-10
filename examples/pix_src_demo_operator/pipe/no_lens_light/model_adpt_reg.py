"""
Four-stage inference pipeline for lensing data with pre-subtracted lens light.

The input ``image_no_lens.fits`` already has the lens light removed, so this
pipeline skips lens-light modelling entirely and focuses on the mass model and
source reconstruction.

Stage a  : SIE + shear + MGE source light (uniform priors, NO lens light)
Stage b  : build an arc feature mask from image_no_lens S/N
Stage m1 : SIE + shear + pixelized source — Nautilus sampling of
           log_lambda_reg and adaptive_reg_rho
           (SIE+shear fixed at stage-A medians)
Stage m2 : EPL + shear + pixelized source — Nautilus nested sampling
           (Gaussian priors from stage a on EPL+shear; source-reg
           hyperparameters fixed at M1 posterior medians)

Each stage pickles its posterior samples/weights to
``output_adpt_reg/stage_{a,m1,m2}.pkl`` and is re-runnable via ``--skip-done``.

Usage::

    # From no_lens_light/
    python model_adpt_reg.py
    python model_adpt_reg.py --skip-done
"""

from __future__ import annotations

import argparse
import os
from functools import partial
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Disable NVIDIA TF32 (10-bit mantissa) for matmul.  The λ grid search
# spans 16 decades [1e-8, 1e8]; at the extremes the preconditioner blocks
# contain entries separated by 10^20, which TF32 truncation turns into
# NaN inside XLA's GEMM fusion autotuner ("nan, expected 0").  Full
# float32 avoids this entirely at a modest matmul throughput cost.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

os.chdir(Path(__file__).parent)

import jax
jax.config.update("jax_default_matmul_precision", "float32")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from nautilus import Sampler

from TinyLensGpu.example_support import (
    adapt_image_kwargs,
    pixelized_objective_provenance,
    require_reference_observation,
    SamplingConfig,
    StageStorageConfig,
    draw_image_panel,
    export_source_package,
    load_stage,
    require_valid_reconstruction,
    run_sampling,
    save_stage,
    validate_source_package,
    resolve_source_interpolation,
    resolve_stage_source_interpolation,
)
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.example_support.pixelized_stage_cache import (
    load_pixelized_checkpoint,
    matching_pixelized_checkpoint,
    pixelized_stage_identity,
    plot_cached_pixelized_stage,
    recover_pixelized_producer,
)
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import ImageProbModel
from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
    InvalidReconstructionError,
)
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

from TinyLensGpu.Inference import StagePosterior

import caskade as ck


# ------------------------------------------------------------------ #
NSRC = 100
DPIX = 0.05
NSUB = 4
NSUB_PIX = 4
N_GAUSSIANS_SRC = 10
MASK_RADIUS = 2.5
NOISE_MASK_THRESHOLD = 1e7  # noise_map pixels above this are pre-masked
ADAPTIVE_REG_RHO = 2.0             # 0 = uniform, >0 strengthens faint-region reg
OUT_DIR = Path("output_adpt_reg")
DATA_DIR = Path("data")


# ------------------------------------------------------------------ #
def _run_sampler(likelihood, n_live: int, n_eff: int, tag: str, vectorized: bool = True):
    return run_sampling(
        likelihood,
        SamplingConfig(n_live=n_live, n_eff=n_eff, vectorized=vectorized,
                       n_batch=100 if vectorized else None),
        tag=tag, sampler_factory=Sampler,
    )


def _dump_stage(tag: str, samples, weights, param_names, log_z, extra=None, *, source_interpolation=None):
    save_stage(StageStorageConfig(OUT_DIR, include_schema=False, source_interpolation=source_interpolation),
               tag, samples, weights, param_names, log_z, extra=extra)


def _load_stage(tag: str, *, source_interpolation=None):
    return load_stage(
        StageStorageConfig(
            OUT_DIR, include_schema=False, source_interpolation=source_interpolation
        ),
        tag,
    )


def _posterior_median(samples, weights, param_names):
    """Return dict of weighted medians keyed by param name."""
    out = {}
    q = np.array([0.5])
    for i, name in enumerate(param_names):
        out[name] = float(weighted_quantile(
            np.asarray(samples[:, i]), weights, q)[0])
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


def _validate_source_package(source_package):
    return validate_source_package(
        source_package, expected_n=NSRC, rho=ADAPTIVE_REG_RHO,
        require_regularization=False, check_axes=True,
    )


def _fixed_source_kwargs(source_package):
    return adapt_image_kwargs(source_package, expected_n=NSRC)


def _reg_hyperparams_from_m1_payload(stage_payload):
    medians = stage_payload.get("extra", {}).get("medians")
    if not medians:
        raise KeyError("stage-M1 payload missing posterior medians.")
    required = ("log_lambda_reg", "adaptive_reg_rho")
    missing = [name for name in required if name not in medians]
    if missing:
        raise KeyError("stage-M1 medians missing required keys: " + ", ".join(missing))
    return {name: float(medians[name]) for name in required}


def _format_reg_hyperparams(reg_hyperparams):
    return (
        f"lambda={float(jnp.exp(reg_hyperparams['log_lambda_reg'])):.4e}, "
        f"rho={reg_hyperparams['adaptive_reg_rho']:.4f}"
    )


def _solve_pixel_source_for_package(likelihood, medians, param_names):
    return export_source_package(
        likelihood, medians, param_names, n=NSRC, record_adapt_image=True,
    )


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


def run_stage_a(image_data, noise_map, psf_kernel, circular_mask=None, *, source_interpolation=None):
    OUT_DIR.mkdir(exist_ok=True)
    print("\n" + "=" * 60)
    print(" Stage A : SIE + shear + MGE source light  (no lens light)")
    print("=" * 60)
    t0 = time.time()
    likelihood = build_stage_a_likelihood(image_data, noise_map, psf_kernel,
                                          circular_mask=circular_mask)
    stage = _run_sampler(
        likelihood, n_live=300, n_eff=2000, tag="stage-A", vectorized=True,
    )
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
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
        source_interpolation=resolve_source_interpolation(source_interpolation),
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
    im0 = draw_image_panel(
        axes[0], image_data,
        title='image_no_lens (lens-subtracted data)',
        title_size=plt.rcParams["axes.titlesize"], xlabel=None,
        extent=extent, cmap='viridis',
    )
    im1 = draw_image_panel(
        axes[1], snr_map,
        title='S/N map + arc mask boundary',
        title_size=plt.rcParams["axes.titlesize"], xlabel=None,
        extent=extent, cmap='viridis', vmin=-3, vmax=np.nanpercentile(snr_map, 99.5),
    )
    axes[1].contour(~feature_mask, levels=[0.5], origin="lower", extent=extent,
                    colors="red", linewidths=1.5)
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


def _epl_mass_from_stage_a(passer: StagePosterior):
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
# Stage M0 — SIE + shear + uniform pixelized source
# ------------------------------------------------------------------ #
def build_stage_m0_likelihood(
    image_data, noise_map, psf_kernel, feature_mask,
    medians_a, position_likelihood, circular_mask=None,
    *, source_interpolation=None,
    evidence_logdet="overlap", evidence_block_size=10,
):
    """Build M0 likelihood: fixed SIE+shear, uniform pixelized source."""
    sie, shear = _sie_mass_from_stage_a(medians_a)

    log_lam = ParamU(
        "log_lambda_reg",
        0.0,
        prior_type="uniform",
        prior_settings=[-13.815510557964274, 13.815510557964274],
        limits=[-13.815510557964274, 13.815510557964274],
    )
    log_lam.to_dynamic()

    pix_src = PixelizedSourceModel(n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type="first-order",
        interpolation=resolve_source_interpolation(source_interpolation),
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
        evidence_logdet=evidence_logdet,
        evidence_block_size=evidence_block_size,
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
    )


def run_stage_m0(image_data, noise_map, psf_kernel, feature_mask,
                  medians_a, position_likelihood, circular_mask=None, *, source_interpolation=None,
                  cached=None, upstream=None,
):
    """Stage M0: uniform-reg source inversion used to export the S0 reference arc."""
    print("\n" + "=" * 60)
    print(" Stage M0 : fixed SIE + shear + uniform pix source ")
    print("=" * 60)
    t0 = time.time()

    build = partial(
        build_stage_m0_likelihood,
        image_data, noise_map, psf_kernel, feature_mask,
        medians_a, position_likelihood, circular_mask=circular_mask,
        source_interpolation=source_interpolation,
    )
    likelihood = build()
    loglike_batch = make_likelihood(likelihood, vectorized=True)
    n_grid = 200

    objective = pixelized_objective_provenance(
        likelihood, upstream=upstream,
        sampling=dict(
            method="grid_search", vectorized=True, grid_size=n_grid,
            coarse_lambda_bounds=[1e-8, 1e8], fine_width_decades=1,
        ),
    )
    checkpoint = matching_pixelized_checkpoint(cached, objective, "m0", producer=True)
    if checkpoint is not None:
        try:
            log_lambda = checkpoint["extra"]["lambda_best"]
            medians = {**medians_a, "log_lambda_reg": log_lambda}
            package = recover_pixelized_producer(
                likelihood, checkpoint, "s0", medians, ["log_lambda_reg"],
            )
        except (KeyError, TypeError, ValueError) as err:
            print(f"[stage-M0] cannot verify cached source reference ({err}); recomputing.")
        else:
            print("[stage-M0] reusing verified source reference")
            extra = checkpoint["extra"]
            if package is not extra["s0"]:
                _dump_stage(
                    "m0", None, None, ["log_lambda_reg"], checkpoint["log_z"],
                    extra={**extra, "s0": package},
                    source_interpolation=likelihood.source_model.interpolation,
                )
            plot_cached_pixelized_stage(
                _plot_pix_stage, OUT_DIR, "m0", likelihood, medians, ["log_lambda_reg"],
                producer=(package, build),
            )
            return package, log_lambda

    log_lam_min, log_lam_max = jnp.log(1e-8), jnp.log(1e8)
    print(f"[stage-M0] Running coarse grid (200 pts, λ in [{float(jnp.exp(log_lam_min)):.1e}, {float(jnp.exp(log_lam_max)):.1e}]) ...")
    log_lam_grid_coarse = jnp.linspace(log_lam_min, log_lam_max, n_grid)
    log_ev_coarse = jnp.asarray(loglike_batch(log_lam_grid_coarse.reshape(-1, 1)))
    if not jnp.any(jnp.isfinite(log_ev_coarse)):
        raise RuntimeError(
            "[stage-M0] All log-λ values in coarse grid produced non-finite log-evidence."
        )
    best_idx_coarse = int(jnp.argmax(log_ev_coarse))
    log_lam_best_coarse = float(log_lam_grid_coarse[best_idx_coarse])
    log_ev_best_coarse = float(log_ev_coarse[best_idx_coarse])
    print(f"[stage-M0] Coarse best: λ = {float(jnp.exp(log_lam_best_coarse)):.4e}  (log-ev = {log_ev_best_coarse:.2f})")

    half_width = 0.5 * jnp.log(10)
    log_lam_grid_fine = jnp.linspace(
        log_lam_best_coarse - half_width,
        log_lam_best_coarse + half_width,
        n_grid,
    )
    print(f"[stage-M0] Running refinement grid (200 pts, λ in [{float(jnp.exp(log_lam_grid_fine[0])):.4e}, {float(jnp.exp(log_lam_grid_fine[-1])):.4e}]) ...")
    log_ev_fine = jnp.asarray(loglike_batch(log_lam_grid_fine.reshape(-1, 1)))
    best_idx_fine = int(jnp.argmax(log_ev_fine))
    log_lam_best = float(log_lam_grid_fine[best_idx_fine])
    log_ev_best = float(log_ev_fine[best_idx_fine])
    print(f"[stage-M0] Refined best: λ = {float(jnp.exp(log_lam_best)):.4e}  (log-ev = {log_ev_best:.2f})")

    medians_m0 = {**medians_a, "log_lambda_reg": log_lam_best}
    source_package = _solve_pixel_source_for_package(
        likelihood, medians_m0, ["log_lambda_reg"],
    )
    source_package.update(
        lambda_best=float(jnp.exp(log_lam_best)),
        log_lambda_best=log_lam_best,
        stage_a_medians=dict(medians_a),
    )
    _validate_source_package(source_package)

    t1 = time.time()
    print("\n[stage-M0] Grid search summary:")
    print(f"    {'lambda_reg_uniform':25s} = {float(jnp.exp(log_lam_best)):+.4e}")
    print(f"[stage-M0] time taken: {t1 - t0:.2f} seconds")

    _dump_stage(
        "m0", None, None, ["log_lambda_reg"], log_ev_best,
        extra=dict(
            lambda_best=log_lam_best,
            s0=source_package,
            lambda_grid_coarse=np.asarray(log_lam_grid_coarse, dtype=np.float64),
            log_ev_coarse=np.asarray(log_ev_coarse, dtype=np.float64),
            lambda_grid_fine=np.asarray(log_lam_grid_fine, dtype=np.float64),
            log_ev_fine=np.asarray(log_ev_fine, dtype=np.float64),
            objective=objective,
            time_taken=t1 - t0,
        ),
        source_interpolation=likelihood.source_model.interpolation,
    )

    try:
        _plot_pix_stage("stage-M0", likelihood, medians_m0, ["log_lambda_reg"],
                        str(OUT_DIR / "stage_m0_model.png"))
    except InvalidReconstructionError:
        raise
    except Exception as err:
        print(f"[stage-M0] plotting failed (non-fatal): {err}")

    return source_package, log_lam_best


# ------------------------------------------------------------------ #
# Stage M1 — SIE + shear + pixelized source  (fit adaptive-reg hyperparameters)
# ------------------------------------------------------------------ #
def build_stage_m1_likelihood(
    image_data, noise_map, psf_kernel, feature_mask,
    medians_a, position_likelihood, source_package, circular_mask=None,
    *, source_interpolation=None,
):
    """Build likelihood for M1: SIE+shear fixed, source-reg hyperparams free."""
    source_interpolation = resolve_source_interpolation(
        source_interpolation, metadata=source_package, context="S0 source package"
    )
    sie, shear = _sie_mass_from_stage_a(medians_a)

    log_lam = ParamU(
        "log_lambda_reg",
        0.0,
        prior_type="uniform",
        prior_settings=[-13.815510557964274, 13.815510557964274],
        limits=[-13.815510557964274, 13.815510557964274],
    )
    log_lam.to_dynamic()
    rho = ParamU(
        "adaptive_reg_rho",
        ADAPTIVE_REG_RHO,
        prior_type="uniform",
        prior_settings=[0.0, 3.0],
        limits=[0.0, 3.0],
    )
    rho.to_dynamic()

    pix_src = PixelizedSourceModel(n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type="first-order",
        adaptive_reg_rho=rho,
        interpolation=source_interpolation,
    )
    phys = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=[pix_src],
        lens_light=[],
    )
    combined_mask = feature_mask
    if circular_mask is not None:
        combined_mask = combined_mask | circular_mask
    likelihood = PixelizedImageProbModelOperator(
        evidence_logdet="overlap",
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        **_fixed_source_kwargs(source_package),
    )
    require_reference_observation(likelihood, source_package)
    return likelihood


def run_stage_m1(
    image_data,
    noise_map,
    psf_kernel,
    feature_mask,
    medians_a,
    position_likelihood,
    source_package,
    circular_mask=None,
    *,
    source_interpolation=None,
                  cached=None, upstream=None,
):
    """Stage M1: Nautilus fit for adaptive source-regularization hyperparams.

    ``image_data`` is already lens-subtracted so we use it directly.
    """
    print("\n" + "=" * 60)
    print(" Stage M1 : SIE + shear + pix source  (fit λ, rho)")
    print("=" * 60)
    t0 = time.time()

    likelihood = build_stage_m1_likelihood(
        image_data, noise_map, psf_kernel, feature_mask,
        medians_a, position_likelihood, source_package, circular_mask=circular_mask,
        source_interpolation=source_interpolation,
    )
    sampling = SamplingConfig(n_live=250, n_eff=600, vectorized=True, n_batch=100)
    objective = pixelized_objective_provenance(
        likelihood, reference=source_package, sampling=sampling, upstream=upstream,
    )
    checkpoint = matching_pixelized_checkpoint(cached, objective, "m1")
    if checkpoint is not None:
        medians = checkpoint["extra"]["medians"]
        plot_cached_pixelized_stage(
            _plot_pix_stage, OUT_DIR, "m1", likelihood, medians, checkpoint["param_names"],
        )
        return _reg_hyperparams_from_m1_payload(checkpoint)

    stage = run_sampling(likelihood, sampling, tag="stage-M1", sampler_factory=Sampler)
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    t1 = time.time()
    _print_summary("stage-M1", samples, weights, names)
    print(f"[stage-M1] time taken: {t1 - t0:.2f} seconds")
    medians = _posterior_median(samples, weights, names)
    reg_hyperparams = {
        "log_lambda_reg": float(medians["log_lambda_reg"]),
        "adaptive_reg_rho": float(medians["adaptive_reg_rho"]),
    }
    print(f"[stage-M1] median hyperparams: {_format_reg_hyperparams(reg_hyperparams)}")

    _dump_stage(
        "m1", samples, weights, names, logz,
        extra=dict(
            medians=medians,
            lambda_best=reg_hyperparams["log_lambda_reg"],
            reg_hyperparams=reg_hyperparams,
            objective=objective,
            time_taken=t1 - t0,
        ),
        source_interpolation=likelihood.source_model.interpolation,
    )

    # Generate M1 pixelized source reconstruction plot
    try:
        _plot_pix_stage("stage-M1", likelihood, medians, names,
                        str(OUT_DIR / "stage_m1_model.png"))
    except InvalidReconstructionError:
        raise
    except Exception as err:
        print(f"[stage-M1] pix_stage plot failed (non-fatal): {err}")

    return reg_hyperparams


# ------------------------------------------------------------------ #
# Stage M2 — EPL + shear + pixelized source  (λ fixed from M1)
# ------------------------------------------------------------------ #
def _plot_pix_stage(tag, likelihood, medians, param_names, save_path):
    """5-panel diagnostic: data | model | norm-residual | source | reg-scale."""
    q50 = [medians[n] for n in param_names]
    image_data = np.asarray(likelihood.image_data)
    noise_map  = np.asarray(likelihood.noise_map)
    mask       = ~np.asarray(likelihood.unmask, dtype=bool)  # True = excluded

    has_pos_penalty = False
    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.array(q50))
        result, diagnostics = likelihood.evaluate_with_diagnostics(
            include_effective_dof=True,
        )
        require_valid_reconstruction(
            result, diagnostics, context=f"plotting [{tag}]",
        )
        lam_val = diagnostics.lambda_reg
        rho_val = diagnostics.adaptive_reg_rho
        xmin, xmax, ymin, ymax = result.source_bbox
        source_pixels = result.source_pixels
        model_image = np.asarray(result.model_image)
        resid_norm = np.asarray(diagnostics.normalized_residuals)
        chi2 = diagnostics.chi2
        N_eff = diagnostics.effective_parameters
        dof = diagnostics.residual_dof
        chi2_nu = diagnostics.reduced_chi2

        has_pos_penalty = diagnostics.position is not None
        if has_pos_penalty:
            position = diagnostics.position
            print(f"[{tag}] Position likelihood penalty: {position.penalty:.4e}")
            print(f"[{tag}] Maximum source-plane separation of marked images: {position.max_separation:.4e} arcsec")

            beta_x_np = np.asarray(position.source_positions[:, 0])
            beta_y_np = np.asarray(position.source_positions[:, 1])
            pos_px_np = np.asarray(position.image_positions[:, 0])
            pos_py_np = np.asarray(position.image_positions[:, 1])

    n = likelihood.phys_model.source_light[0].n
    src_img = np.array(source_pixels).reshape(n, n)

    scale_img = np.asarray(diagnostics.regularization_scale)
    rho = rho_val

    npix = image_data.shape[0]
    ext_i = [-npix * DPIX / 2, npix * DPIX / 2, -npix * DPIX / 2, npix * DPIX / 2]
    ext_s = [float(xmin), float(xmax), float(ymin), float(ymax)]
    vmax  = np.nanpercentile(image_data[~mask], 99.5)

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))
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
        im = draw_image_panel(
            ax,
            img,
            title=title,
            xlabel=None,
            extent=ext_i,
            **kw,
        )
        if ax == axes[0] and has_pos_penalty:
            ax.plot(pos_px_np, pos_py_np, 'rx', markersize=8, label='Marked pos')
            ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel("arcsec")

    im3 = draw_image_panel(
        axes[3],
        src_img,
        title=f"Source reconstruction\n(λ={float(lam_val):.2e})",
        xlabel=None,
        extent=ext_s,
        cmap="viridis",
    )
    if has_pos_penalty:
        axes[3].plot(beta_x_np, beta_y_np, 'rx', markersize=8, label='Traced pos')
        axes[3].legend(loc='upper right', fontsize=8)
    axes[3].set_xlabel("arcsec")


    # Panel 5: adaptive reg scale map
    im4 = draw_image_panel(
        axes[4],
        scale_img,
        title=f"Reg precision scale\n(rho={rho:.2f})",
        xlabel=None,
        extent=ext_s,
        cmap="plasma",
        vmin=1.0,
        vmax=max(1.0, float(np.nanmax(scale_img))),
        colorbar_kwargs={"label": "$\\lambda_i / \\lambda_{\\rm global}$"},
    )
    axes[4].set_xlabel("arcsec")


    axes[0].set_ylabel("arcsec")
    # Panels 1-3: zoom to unmasked pixel region
    for ax in axes[:3]:
        ax.set_xlim(*xlim_unmasked)
        ax.set_ylim(*ylim_unmasked)

    lbl = "  ".join(f"{n}={medians[n]:+.4f}" for n in
                    ("theta_E", "gamma", "e1_mass", "e2_mass") if n in medians)
    plt.suptitle(f"[{tag}]  {lbl}", fontsize=10)
    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.asarray(q50))
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
    passer: StagePosterior, position_likelihood, reg_hyperparams_fixed: dict,
    source_package, circular_mask=None,
    *, source_interpolation=None,
):
    """Build likelihood for M2: EPL+shear with M1 source-reg medians fixed."""
    source_interpolation = resolve_source_interpolation(
        source_interpolation, metadata=source_package, context="S0 source package"
    )
    epl, shear = _epl_mass_from_stage_a(passer)

    log_lam = ParamU("log_lambda_reg", float(reg_hyperparams_fixed["log_lambda_reg"]))
    log_lam.to_static()
    rho = ParamU(
        "adaptive_reg_rho",
        float(reg_hyperparams_fixed["adaptive_reg_rho"]),
    )
    rho.to_static()

    pix_src = PixelizedSourceModel(n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type="first-order",
        adaptive_reg_rho=rho,
        interpolation=source_interpolation,
    )
    phys = PhysicalModel(
        lens_mass=[epl, shear],
        source_light=[pix_src],
        lens_light=[],
    )
    combined_mask = feature_mask
    if circular_mask is not None:
        combined_mask = combined_mask | circular_mask
    likelihood = PixelizedImageProbModelOperator(
        evidence_logdet="overlap",
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        **_fixed_source_kwargs(source_package),
    )
    require_reference_observation(likelihood, source_package)
    return likelihood


def run_stage_m2(image_data, noise_map, psf_kernel, feature_mask,
                  samples_a, weights_a, names_a,
                  position_likelihood, reg_hyperparams_fixed: dict, source_package,
                  circular_mask=None, *, source_interpolation=None,
                  cached=None, upstream=None,
):
    """Stage M2: Nautilus sampling of EPL+shear with M1 source-reg medians fixed.

    ``image_data`` is already lens-subtracted so we use it directly.
    """
    print("\n" + "=" * 60)
    print(" Stage M2 : EPL + shear + pix source (source reg fixed from M1)")
    print("=" * 60)
    print(f"[stage-M2] fixed hyperparams: {_format_reg_hyperparams(reg_hyperparams_fixed)}")
    t0 = time.time()
    passer = StagePosterior(samples_a, weights_a, names_a)
    likelihood = build_stage_m2_likelihood(
        image_data,
        noise_map,
        psf_kernel,
        feature_mask,
        passer,
        position_likelihood,
        reg_hyperparams_fixed,
        source_package,
        circular_mask=circular_mask,
        source_interpolation=source_interpolation,
    )
    sampling = SamplingConfig(n_live=300, n_eff=600, vectorized=True, n_batch=100)
    objective = pixelized_objective_provenance(
        likelihood, reference=source_package, sampling=sampling, upstream=upstream,
    )
    checkpoint = matching_pixelized_checkpoint(cached, objective, "m2")
    if checkpoint is not None:
        medians = checkpoint["extra"]["medians"]
        plot_cached_pixelized_stage(
            _plot_pix_stage, OUT_DIR, "m2", likelihood, medians, checkpoint["param_names"],
        )
        return checkpoint["samples"], checkpoint["weights"], checkpoint["param_names"], medians

    stage = run_sampling(likelihood, sampling, tag="stage-M2", sampler_factory=Sampler)
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    t1 = time.time()
    _print_summary("stage-M2", samples, weights, names)
    print(f"[stage-M2] time taken: {t1 - t0:.2f} seconds")
    medians = _posterior_median(samples, weights, names)
    _dump_stage("m2", samples, weights, names, logz, extra=dict(
        medians=medians,
        lambda_fixed=reg_hyperparams_fixed["log_lambda_reg"],
        reg_hyperparams_fixed=dict(reg_hyperparams_fixed),
        objective=objective,
        time_taken=t1 - t0,
    ), source_interpolation=likelihood.source_model.interpolation)
    try:
        _plot_pix_stage("stage-M2", likelihood, medians, names,
                        str(OUT_DIR / "stage_m2_model.png"))
    except InvalidReconstructionError:
        raise
    except Exception as err:
        print(f"[stage-M2] plotting failed (non-fatal): {err}")
    return samples, weights, names, medians


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def main(skip_done: bool = False, *, source_interpolation=None):
    source_interpolation = resolve_stage_source_interpolation(
        OUT_DIR, ("a", "m0", "m1", "m2"),
        requested=source_interpolation, resume=skip_done,
    )
    source_options = dict(source_interpolation=source_interpolation)
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
        print("[stage-A] loading cached output_adpt_reg/stage_a.pkl")
        d = _load_stage("a", **source_options)
        samples_a, weights_a, names_a = d["samples"], d["weights"], d["param_names"]
        medians_a = d["extra"]["medians"]
        time_a = d["extra"].get("time_taken", 0.0)
    else:
        samples_a, weights_a, names_a, medians_a = run_stage_a(
            image_data, noise_map, psf_kernel, circular_mask=circular_mask,
            **source_options,
        )
        time_a = _load_stage("a", **source_options)["extra"].get("time_taken", 0.0)

    # ---- stage B ---------------------------------------------------- #
    # Build arc mask directly from lens-subtracted image (no lens_light_model needed)
    feature_mask = run_stage_b(image_data, noise_map, circular_mask=circular_mask)

    # ---- position likelihood ----------------------------------------- #
    position_likelihood = _position_likelihood_from_stage_a(medians_a)

    def checkpoint(tag):
        return load_pixelized_checkpoint(_load_stage, OUT_DIR, tag, resume=skip_done)

    data = (image_data, noise_map, psf_kernel, feature_mask)
    options = dict(circular_mask=circular_mask, **source_options)
    stage_a_id = pixelized_stage_identity(_load_stage("a"))
    source_package, log_lambda_m0 = run_stage_m0(
        *data, medians_a, position_likelihood,
        cached=checkpoint("m0"), upstream={"a": stage_a_id}, **options,
    )
    m0_id = pixelized_stage_identity(_load_stage("m0"))
    reg_hyperparams_m1 = run_stage_m1(
        *data, medians_a, position_likelihood, source_package,
        cached=checkpoint("m1"), upstream={"m0": m0_id}, **options,
    )
    _, _, _, medians_m2 = run_stage_m2(
        *data, samples_a, weights_a, names_a, position_likelihood,
        reg_hyperparams_m1, source_package, cached=checkpoint("m2"),
        upstream={"a": stage_a_id, "m0": m0_id, "m1": pixelized_stage_identity(_load_stage("m1"))},
        **options,
    )
    time_m0, time_m1, time_m2 = (
        _load_stage(tag)["extra"].get("time_taken", 0.0) for tag in ("m0", "m1", "m2")
    )

    print("\n" + "=" * 60)
    print(" Pipeline complete")
    print("=" * 60)
    print(" Time summary:")
    print(f"    Stage A:  {time_a/60:.2f} min")
    print(f"    Stage M0: {time_m0/60:.2f} min")
    print(f"    Stage M1: {time_m1/60:.2f} min")
    print(f"    Stage M2: {time_m2/60:.2f} min")
    print(f"    Total:    {(time_a + time_m0 + time_m1 + time_m2)/60:.2f} min\n")
    print(f"    M0 best lambda_reg     = {float(jnp.exp(log_lambda_m0)):.4e}")
    print(f"    M1 median source reg   = {_format_reg_hyperparams(reg_hyperparams_m1)}")

    for k in ("theta_E", "gamma", "e1_mass", "e2_mass",
              "center_x_mass", "center_y_mass", "gamma1", "gamma2"):
        if k in medians_m2:
            print(f"    final  {k:15s} = {medians_m2[k]:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-done", action="store_true",
                        help="Re-use cached posteriors in output_adpt_reg/stage_*.pkl")
    parser.add_argument("--source-interpolation", choices=("bilinear", "cubic_hermite"))
    args = parser.parse_args()
    main(skip_done=args.skip_done, source_interpolation=args.source_interpolation)
