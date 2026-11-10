"""
Adaptive-regularization inference pipeline for the pix_src_pipe_sersic demo.

Stage a  : SIE + shear + single-Sersic lens light + single-Sersic source light
Stage b  : build an arc feature mask from stage-A residuals
Stage l  : arc-masked single-Sersic lens light refinement (Gaussian priors from stage a)
Stage m0 : SIE + shear + uniform pixelized source — GPU grid search for
            evidence-best lambda_reg on lens-subtracted image
Stage m1 : EPL + shear + non-adaptive pixelized source — Nautilus sampling
            of mass parameters on lens-subtracted image with lambda_reg constrained around stage-M0,
            then exports the S1 reference arc and fitted source grid
Stage m2 : fixed EPL + shear + adaptive pixelized source — Nautilus sampling
            of log_lambda_reg and adaptive_reg_rho only, on lens-subtracted image
Stage m3 : EPL + shear + adaptive pixelized source — Nautilus nested sampling
            with source-reg hyperparameters fixed from stage-M2 medians

Each stage pickles its posterior samples/weights to
``output/stage_{a,l,m0,m1,m2,m3}.pkl`` and is re-runnable via ``--skip-done``.
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

os.chdir(Path(__file__).parent)

import jax
jax.config.update("jax_default_matmul_precision", "float32")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from nautilus import Sampler

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.example_support import (
    SamplingConfig,
    StageStorageConfig,
    run_sampling,
    save_stage,
    load_stage,
    stage_from_payload,
    draw_image_panel,
    adapt_image_kwargs,
    export_source_package,
    pixelized_objective_provenance,
    require_reference_observation,
    require_valid_reconstruction,
    validate_source_package,
    resolve_source_interpolation,
    resolve_stage_source_interpolation,
)
from TinyLensGpu.example_support.pixelized_stage_cache import (
    load_pixelized_checkpoint,
    matching_pixelized_checkpoint,
    pixelized_stage_identity,
    plot_cached_pixelized_stage,
    recover_pixelized_producer,
)
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.ObservationModel.LensImage.inversion_result import InvalidReconstructionError
from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import ImageProbModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
from TinyLensGpu.PhysicalModel import (
    PhysicalModel,
    PixelizedSourceModel,
    SersicEllipse,
    Shear,
)
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE, EPL
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.utils.misc import arc_mask_from, weighted_quantile
from TinyLensGpu.visualizer import plot_model_results, overlay_critical_and_caustics

from TinyLensGpu.Inference import StagePosterior

import caskade as ck

# ------------------------------------------------------------------ #
NSRC = 30
DPIX = 0.05
NSUB = 8
NSUB_PIX = 4
MASK_RADIUS = 2.5
ADAPTIVE_REG_RHO = 2.0
ADAPTIVE_REG_RHO_PRIOR_MAX = 8.0
PIXEL_REGULARIZATION_TYPE = "first-order"
SOURCE_BBOX_PADDING = 0.30
PNPG_MAX_ITER = 1000
PNPG_RTOL = 2.0e-2
PNPG_POWER_ITER = 20
SOLVER_TYPE = "pcg"
OUT_DIR = Path("output")
DATA_DIR = Path("data")


# ------------------------------------------------------------------ #
def _run_sampler(likelihood, n_live: int, n_eff: int, tag: str, vectorized: bool = True):
    return run_sampling(
        likelihood,
        SamplingConfig(n_live=n_live, n_eff=n_eff, vectorized=vectorized,
                       n_batch=100 if vectorized else None),
        tag=tag, sampler_factory=Sampler,
    )


def _dump_stage(
    tag: str, samples, weights, param_names, log_z, extra=None, stage=None,
    *, source_interpolation=None,
):
    save_stage(StageStorageConfig(output_dir=OUT_DIR, include_schema=True, source_interpolation=source_interpolation),
               tag, samples, weights, param_names, log_z, extra=extra, stage=stage)


def _load_stage(tag: str, *, source_interpolation=None):
    return load_stage(StageStorageConfig(output_dir=OUT_DIR, include_schema=True, source_interpolation=source_interpolation), tag)


def _stage_from_payload(payload):
    return stage_from_payload(payload)


def _print_summary(tag: str, samples, weights, param_names):
    print(f"\n[{tag}] Posterior summary:")
    q = np.array([0.16, 0.5, 0.84])
    for i, name in enumerate(param_names):
        qs = weighted_quantile(np.asarray(samples[:, i]), weights, q)
        q16, q50, q84 = float(qs[0]), float(qs[1]), float(qs[2])
        print(f"    {name:25s} = {q50:+.4f} ({q16-q50:+.4f}, {q84-q50:+.4f})")


# ------------------------------------------------------------------ #
def _make_circular_mask(image_shape, dpix, radius_arcsec=3.5):
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


def _pnpg_kwargs():
    return dict(pnpg_max_iter=PNPG_MAX_ITER, pnpg_rtol=PNPG_RTOL,
                pnpg_power_iter=PNPG_POWER_ITER)


def _fixed_source_kwargs(source_package):
    return adapt_image_kwargs(source_package, expected_n=NSRC)


def _reg_hyperparams_from_payload(stage_payload, tag):
    medians = stage_payload.get("extra", {}).get("medians")
    if not medians:
        raise KeyError(f"stage-{tag.upper()} payload missing posterior medians.")
    required = ("log_lambda_reg", "adaptive_reg_rho")
    missing = [name for name in required if name not in medians]
    if missing:
        raise KeyError(f"stage-{tag.upper()} medians missing required keys: " + ", ".join(missing))
    return {name: float(medians[name]) for name in required}


def _reg_hyperparams_from_m2_payload(stage_payload):
    return _reg_hyperparams_from_payload(stage_payload, "m2")


def _format_reg_hyperparams(reg_hyperparams):
    return f"lambda={float(jnp.exp(reg_hyperparams['log_lambda_reg'])):.4e}, rho={reg_hyperparams['adaptive_reg_rho']:.4f}"


def _valid_log_evidence(values) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float64)
    return np.isfinite(values_np) & (values_np > -1.0e9)


def _solve_pixel_source_for_package(likelihood, medians, param_names):
    return export_source_package(
        likelihood, medians, param_names, n=NSRC,
        context="exporting the stage source reference",
        record_adapt_image=True,
    )


# ------------------------------------------------------------------ #
# Stage A — SIE + shear + single-Sersic lens light + single-Sersic source light
# ------------------------------------------------------------------ #
def build_stage_a_likelihood(image_data, noise_map, psf_kernel, mask=None):
    sie = SIE(
        theta_E=ParamU("theta_E", 1.5, prior_type="uniform", prior_settings=[0.5, 2.5], limits=[0.0, 5.0]),
        e1=ParamU("e1_mass", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        e2=ParamU("e2_mass", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        center_x=ParamU("center_x_mass", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.1], limits=[-1.0, 1.0]),
        center_y=ParamU("center_y_mass", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.1], limits=[-1.0, 1.0]),
    )
    shear = Shear(
        gamma1=ParamU("gamma1", 0.0, prior_type="uniform", prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
        gamma2=ParamU("gamma2", 0.0, prior_type="uniform", prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
    )
    for p in (sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y, shear.gamma1, shear.gamma2):
        p.to_dynamic()

    # Single-Sersic source
    R_s_src = ParamU("R_sersic_src", 0.3, prior_type="uniform", prior_settings=[0.05, 1.5], limits=[0.01, 3.0])
    n_s_src = ParamU("n_sersic_src", 2.0, prior_type="uniform", prior_settings=[0.5, 8.0], limits=[0.3, 10.0])
    e1_s = ParamU("e1_src", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_s = ParamU("e2_src", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    cx_s = ParamU("center_x_src", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.5], limits=[-3.0, 3.0])
    cy_s = ParamU("center_y_src", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.5], limits=[-3.0, 3.0])
    source_sersic = SersicEllipse(R_sersic=R_s_src, n_sersic=n_s_src, e1=e1_s, e2=e2_s, center_x=cx_s, center_y=cy_s, Ie=1.0)
    for p in (R_s_src, n_s_src, cx_s, cy_s, e1_s, e2_s):
        p.to_dynamic()

    # Single-Sersic lens light
    R_s_lens = ParamU("R_sersic_lens", 1.0, prior_type="uniform", prior_settings=[0.1, 3.0], limits=[0.05, 5.0])
    n_s_lens = ParamU("n_sersic_lens", 3.0, prior_type="uniform", prior_settings=[0.5, 8.0], limits=[0.3, 10.0])
    e1_l = ParamU("e1_lens", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_l = ParamU("e2_lens", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    cx_l = ParamU("center_x_lens", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    cy_l = ParamU("center_y_lens", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    lens_sersic = SersicEllipse(R_sersic=R_s_lens, n_sersic=n_s_lens, e1=e1_l, e2=e2_l, center_x=cx_l, center_y=cy_l, Ie=1.0)
    for p in (R_s_lens, n_s_lens, cx_l, cy_l, e1_l, e2_l):
        p.to_dynamic()

    phys = PhysicalModel(lens_mass=[sie, shear], source_light=[source_sersic], lens_light=[lens_sersic])
    return ImageProbModel(image_data=image_data, noise_map=noise_map, psf_kernel=psf_kernel,
                           dpix=DPIX, nsub=NSUB, phys_model=phys, use_linear=True, solver_type="nnls", mask=mask)


def run_stage_a(image_data, noise_map, psf_kernel, circular_mask=None, *, source_interpolation=None):
    OUT_DIR.mkdir(exist_ok=True)
    print("\n" + "=" * 60)
    print(" Stage A : SIE + shear + single-Sersic lens light + single-Sersic source light")
    print("=" * 60)
    t0 = time.time()
    likelihood = build_stage_a_likelihood(image_data, noise_map, psf_kernel, mask=circular_mask)
    stage = _run_sampler(likelihood, n_live=200, n_eff=2000, tag="stage-A", vectorized=True)
    t1 = time.time()
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    _print_summary("stage-A", samples, weights, names)
    print(f"[stage-A] time taken: {t1 - t0:.2f} seconds")
    medians = stage.medians(); q50 = [medians[n] for n in names]
    likelihood.set_values(q50)
    fwd = likelihood.forward_model(use_linear=True, return_intensity=True, ret_each_plane=True,
                                    image_map=likelihood.image_data, noise_map=likelihood.noise_map)
    lens_image_model = np.asarray(fwd[1])
    _dump_stage("a", samples, weights, names, logz,
                extra=dict(medians=medians, lens_light_model=lens_image_model, time_taken=t1 - t0), stage=stage, source_interpolation=resolve_source_interpolation(source_interpolation))
    try:
        plot_model_results(likelihood, jnp.asarray(q50), save_path=str(OUT_DIR / "stage_a_model.png"),
                          title="Stage A : Sersic lens+source", show_critical_lines=True, show_caustics=True)
    except Exception as err:
        print(f"[stage-A] plotting failed (non-fatal): {err}")
    return stage, medians, lens_image_model


# ------------------------------------------------------------------ #
# Stage B
# ------------------------------------------------------------------ #
def run_stage_b(image_data, noise_map, lens_light_model, circular_mask=None):
    print("\n" + "=" * 60)
    print(" Stage B : build arc feature mask from stage-A residuals")
    print("=" * 60)
    residual = image_data - lens_light_model; snr_map = residual / noise_map
    arc_mask = arc_mask_from(snr_map, threshold=2.0, ignor_size=25, ext_size=2, close_size=3)
    feature_mask = arc_mask
    if circular_mask is not None:
        feature_mask = feature_mask | circular_mask
    print(f"[stage-B] arc pixels kept = {int((~feature_mask).sum())} / {feature_mask.size}")
    DATA_DIR.mkdir(exist_ok=True)
    fits.writeto(DATA_DIR / "feature_mask.fits", feature_mask.astype(np.uint8), overwrite=True)
    ny_img, nx_img = image_data.shape
    extent = [-nx_img * DPIX / 2, nx_img * DPIX / 2, -ny_img * DPIX / 2, ny_img * DPIX / 2]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(residual, origin="lower", extent=extent, cmap="viridis"); axes[0].set_title("residual = image - stage-A lens light")
    axes[1].imshow(snr_map, origin="lower", extent=extent, cmap="viridis", vmin=-3, vmax=np.nanpercentile(snr_map, 99.5))
    axes[1].set_title("S/N map + arc mask boundary")
    axes[1].contour(~feature_mask, levels=[0.5], origin="lower", extent=extent, colors="red", linewidths=1.5)
    plt.tight_layout(); OUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUT_DIR / "stage_b_mask.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return feature_mask


# ------------------------------------------------------------------ #
# Stage L — arc-masked single-Sersic lens light refinement
# ------------------------------------------------------------------ #
def build_stage_l_likelihood(image_data, noise_map, psf_kernel, feature_mask,
                               stage_a: StagePosterior, circular_mask=None):
    R_s = stage_a.gaussian("R_sersic_lens", model="Sersic", attr="R_sersic", limits=[0.05, 5.0])
    n_s = stage_a.gaussian("n_sersic_lens", model="Sersic", attr="n_sersic", limits=[0.3, 10.0])
    e1_l = ParamU("e1_lens", 0.0, prior_type="uniform", prior_settings=[-0.3, 0.3], limits=[-1.0, 1.0])
    e2_l = ParamU("e2_lens", 0.0, prior_type="uniform", prior_settings=[-0.3, 0.3], limits=[-1.0, 1.0])
    cx_l = stage_a.gaussian("center_x_lens", model="Sersic", attr="center_x", limits=[-1.0, 1.0])
    cy_l = stage_a.gaussian("center_y_lens", model="Sersic", attr="center_y", limits=[-1.0, 1.0])
    lens_sersic = SersicEllipse(R_sersic=R_s, n_sersic=n_s, e1=e1_l, e2=e2_l, center_x=cx_l, center_y=cy_l, Ie=1.0)
    for p in (R_s, n_s, cx_l, cy_l, e1_l, e2_l):
        p.to_dynamic()
    noise_map_soft = np.array(noise_map, copy=True)
    noise_map_soft[~feature_mask] *= 1000.0
    phys = PhysicalModel(lens_mass=[], source_light=[], lens_light=[lens_sersic])
    return ImageProbModel(image_data=image_data, noise_map=noise_map_soft, psf_kernel=psf_kernel,
                           dpix=DPIX, nsub=NSUB, phys_model=phys, use_linear=True, solver_type="nnls", mask=circular_mask)


def run_stage_l(image_data, noise_map, psf_kernel, feature_mask,
                stage_a: StagePosterior, circular_mask=None, *, source_interpolation=None):
    print("\n" + "=" * 60)
    print(" Stage L : arc-masked single-Sersic lens light refinement")
    print("=" * 60)
    t0 = time.time()
    n_arc = int((~feature_mask).sum()); n_nonarc = feature_mask.size - n_arc
    print(f"[stage-L] arc pixels excluded     = {n_arc}    / {feature_mask.size}")
    print(f"[stage-L] lens-light pixels kept  = {n_nonarc} / {feature_mask.size}")
    likelihood = build_stage_l_likelihood(image_data, noise_map, psf_kernel, feature_mask, stage_a, circular_mask=circular_mask)
    stage = _run_sampler(likelihood, n_live=1000, n_eff=2000, tag="stage-L", vectorized=True)
    t1 = time.time()
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    _print_summary("stage-L", samples, weights, names)
    print(f"[stage-L] time taken: {t1 - t0:.2f} seconds")
    medians = stage.medians(); q50 = [medians[n] for n in names]
    likelihood.set_values(q50)
    fwd = likelihood.forward_model(use_linear=True, return_intensity=True, ret_each_plane=True,
                                    image_map=likelihood.image_data, noise_map=likelihood.noise_map)
    lens_image_model = np.asarray(fwd[1])
    _dump_stage("l", samples, weights, names, logz,
                extra=dict(medians=medians, lens_light_model=lens_image_model, time_taken=t1 - t0), stage=stage, source_interpolation=resolve_source_interpolation(source_interpolation))
    try:
        plot_model_results(likelihood, jnp.asarray(q50), save_path=str(OUT_DIR / "stage_l_model.png"),
                          title="Stage L : arc-masked Sersic lens light")
    except Exception as err:
        print(f"[stage-L] plotting failed (non-fatal): {err}")
    # True-vs-fit lens light diagnostic
    try:
        lens_light_true = np.asarray(fits.getdata(DATA_DIR / "lens_light_true.fits"))
        diff = lens_light_true - lens_image_model; diff_norm = diff / noise_map
        npix = image_data.shape[0]; ext = [-npix*DPIX/2, npix*DPIX/2, -npix*DPIX/2, npix*DPIX/2]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
        for ax, img, title, kw in [
            (axes[0], lens_light_true, "True lens light (X)", dict(vmin=0, cmap="viridis")),
            (axes[1], lens_image_model, "Fitted lens light (M)", dict(vmin=0, cmap="viridis")),
            (axes[2], diff, "Residual (X - M)", dict(cmap="RdBu_r")),
            (axes[3], diff_norm, "(X - M) / noise", dict(vmin=-2, vmax=2, cmap="RdBu_r")),
        ]:
            im = ax.imshow(img, origin="lower", extent=ext, **kw)
            ax.set_title(title, fontsize=11); ax.set_xlabel("arcsec"); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[0].set_ylabel("arcsec"); plt.suptitle("Stage L : lens light diagnostic", fontsize=11)
        plt.tight_layout(); plt.savefig(OUT_DIR / "stage_l_diagnostic.png", dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"[stage-L] diagnostic plot saved to {OUT_DIR / 'stage_l_diagnostic.png'}")
    except Exception as err:
        print(f"[stage-L] diagnostic plot failed (non-fatal): {err}")
    return stage, medians, lens_image_model


# ------------------------------------------------------------------ #
# Position likelihood
# ------------------------------------------------------------------ #
def _position_likelihood_from_stage_a(medians_a: dict) -> dict:
    required = ("theta_E","e1_mass","e2_mass","center_x_mass","center_y_mass","gamma1","gamma2","center_x_src","center_y_src")
    missing = [k for k in required if k not in medians_a]
    if missing:
        raise KeyError("Stage-A medians missing required keys: "+", ".join(missing))
    mass = PhysicalModel(
        lens_mass=[SIE(theta_E=float(medians_a["theta_E"]), e1=float(medians_a["e1_mass"]),
                        e2=float(medians_a["e2_mass"]), center_x=float(medians_a["center_x_mass"]),
                        center_y=float(medians_a["center_y_mass"])),
                    Shear(gamma1=float(medians_a["gamma1"]), gamma2=float(medians_a["gamma2"]))],
        source_light=[], lens_light=[])
    solver = PointSourceProbModel(
        phys_model=mass, observed_positions=[[0.0, 0.0]], position_sigma=[0.01],
        source_x=float(medians_a["center_x_src"]), source_y=float(medians_a["center_y_src"]),
        source_position_fixed=True, solver="optimization",
        solver_config={"initial_range":3.0,"n_x":200,"n_y":200,"k_keep":30,"num_iters":20,"tolerance":5.0e-4,"cluster_tol":0.08})
    img_positions, _ = solver.solve_image_positions()
    img_positions = np.asarray(img_positions, dtype=np.float64)
    if img_positions.ndim != 2 or img_positions.shape[1] != 2:
        raise RuntimeError(f"Lens-equation solver returned invalid image_positions shape={img_positions.shape}")
    if img_positions.shape[0] < 2:
        raise RuntimeError("Lens-equation solver returned fewer than 2 lensed image positions.")
    print(f"[pos-like] solved {img_positions.shape[0]} lensed image positions from stage-A medians")
    for p in img_positions: print(f"    ({p[0]:+.4f}, {p[1]:+.4f})")
    return dict(positions=img_positions.tolist(), threshold_arcsec=0.3, min_log_like=-1.0e10)


# ------------------------------------------------------------------ #
# Mass helpers
# ------------------------------------------------------------------ #
def _sie_mass_from_stage_a(stage_a: StagePosterior):
    sie = SIE(theta_E=stage_a.fixed("theta_E"), e1=stage_a.fixed("e1_mass"), e2=stage_a.fixed("e2_mass"),
              center_x=stage_a.fixed("center_x_mass"), center_y=stage_a.fixed("center_y_mass"))
    return sie, Shear(gamma1=stage_a.fixed("gamma1"), gamma2=stage_a.fixed("gamma2"))


def _epl_mass_from_stage(stage: StagePosterior):
    theta_E = stage.gaussian("theta_E", model="EPL", attr="theta_E", limits=[0.0, 5.0])
    e1m = stage.gaussian("e1_mass", model="EPL", attr="e1", limits=[-1.0, 1.0])
    e2m = stage.gaussian("e2_mass", model="EPL", attr="e2", limits=[-1.0, 1.0])
    cx = stage.gaussian("center_x_mass", model="EPL", attr="center_x", limits=[-1.0, 1.0])
    cy = stage.gaussian("center_y_mass", model="EPL", attr="center_y", limits=[-1.0, 1.0])
    gamma = ParamU("gamma", 2.0, prior_type="uniform", prior_settings=[1.0, 3.0], limits=[1.0, 3.0])
    epl = EPL(theta_E=theta_E, gamma=gamma, e1=e1m, e2=e2m, center_x=cx, center_y=cy); epl.gamma.to_dynamic()
    shear = Shear(gamma1=stage.gaussian("gamma1", model="Shear", attr="gamma1", limits=[-0.5, 0.5]),
                  gamma2=stage.gaussian("gamma2", model="Shear", attr="gamma2", limits=[-0.5, 0.5]))
    return epl, shear


def _epl_mass_from_medians(medians: dict):
    required = ("theta_E","gamma","e1_mass","e2_mass","center_x_mass","center_y_mass","gamma1","gamma2")
    missing = [n for n in required if n not in medians]
    if missing: raise KeyError("EPL medians missing: "+", ".join(missing))
    epl = EPL(theta_E=ParamU("theta_E",float(medians["theta_E"])), gamma=ParamU("gamma",float(medians["gamma"])),
              e1=ParamU("e1_mass",float(medians["e1_mass"])), e2=ParamU("e2_mass",float(medians["e2_mass"])),
              center_x=ParamU("center_x_mass",float(medians["center_x_mass"])), center_y=ParamU("center_y_mass",float(medians["center_y_mass"])))
    for p in (epl.theta_E, epl.gamma, epl.e1, epl.e2, epl.center_x, epl.center_y): p.to_static()
    shear = Shear(gamma1=ParamU("gamma1",float(medians["gamma1"])), gamma2=ParamU("gamma2",float(medians["gamma2"])))
    shear.gamma1.to_static(); shear.gamma2.to_static()
    return epl, shear


# ------------------------------------------------------------------ #
# _plot_pix_stage
# ------------------------------------------------------------------ #
def _plot_pix_stage(tag, likelihood, medians, param_names, save_path):
    q50 = [medians[n] for n in param_names]
    result, diagnostics = likelihood.evaluate_with_diagnostics(
        jnp.asarray(q50), include_effective_dof=True,
    )
    require_valid_reconstruction(result, diagnostics, context=f"plotting {tag}")
    image_data = np.asarray(likelihood.image_data)
    mask = ~np.asarray(likelihood.unmask, dtype=bool)
    xmin, xmax, ymin, ymax = result.source_bbox
    model_image = np.asarray(result.model_image)
    resid_norm = np.asarray(diagnostics.normalized_residuals)
    chi2 = diagnostics.chi2
    N_eff = diagnostics.effective_parameters
    dof = diagnostics.residual_dof
    chi2_nu = diagnostics.reduced_chi2
    lam_val = diagnostics.lambda_reg
    rho_val = diagnostics.adaptive_reg_rho
    n = likelihood.phys_model.source_light[0].n
    src_img = np.asarray(result.source_pixels).reshape(n, n)
    scale_img = np.asarray(diagnostics.regularization_scale)
    has_pos_penalty = diagnostics.position is not None
    if has_pos_penalty:
        position = diagnostics.position
        print(f"[{tag}] Position likelihood penalty: {position.penalty:.4e}")
        print(f"[{tag}] Maximum source-plane separation of marked images: {position.max_separation:.4e} arcsec")
        beta_x_np, beta_y_np = np.asarray(position.source_positions).T
        pos_px_np, pos_py_np = np.asarray(position.image_positions).T
    npix = image_data.shape[0]
    ext_i = [-npix*DPIX/2, npix*DPIX/2, -npix*DPIX/2, npix*DPIX/2]
    ext_s = [float(xmin), float(xmax), float(ymin), float(ymax)]
    vmax = np.nanpercentile(image_data[~mask], 99.5)
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))
    rows_unmasked, cols_unmasked = np.where(~mask); pad=3
    row_min=max(rows_unmasked.min()-pad,0); row_max=min(rows_unmasked.max()+pad,npix-1)
    col_min=max(cols_unmasked.min()-pad,0); col_max=min(cols_unmasked.max()+pad,npix-1)
    xlim=(-npix*DPIX/2+col_min*DPIX, -npix*DPIX/2+(col_max+1)*DPIX)
    ylim=(-npix*DPIX/2+row_min*DPIX, -npix*DPIX/2+(row_max+1)*DPIX)
    for ax, img, title, kw in [
        (axes[0], image_data, "Data (lens-subtracted)", dict(vmin=0, vmax=vmax, cmap="viridis")),
        (axes[1], model_image, "Model image", dict(vmin=0, vmax=vmax, cmap="viridis")),
        (axes[2], np.where(mask, np.nan, resid_norm), f"Norm. residual\nχ²/ν={chi2_nu:.3f}", dict(vmin=-3, vmax=3, cmap="RdBu_r")),
    ]:
        draw_image_panel(ax, img, title=title, extent=ext_i, **kw)
        if ax==axes[0] and has_pos_penalty: ax.plot(pos_px_np, pos_py_np, 'rx', markersize=8, label='Marked pos'); ax.legend(loc='upper right', fontsize=8)
    draw_image_panel(axes[3], src_img, title=f'Source reconstruction\n(λ={float(lam_val):.2e})', extent=ext_s, cmap='viridis')
    if has_pos_penalty: axes[3].plot(beta_x_np, beta_y_np, 'rx', markersize=8, label='Traced pos'); axes[3].legend(loc='upper right', fontsize=8)
    draw_image_panel(axes[4], scale_img, title=f'Reg precision scale\n(rho={rho_val:.2f})', extent=ext_s, cmap='plasma', vmin=1.0, vmax=max(1.0, float(np.nanmax(scale_img))), colorbar_kwargs={"label": '$\\lambda_i / \\lambda_{\\rm global}$'})
    axes[0].set_ylabel("arcsec")
    for ax in axes[:3]: ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    lbl = "  ".join(f"{n}={medians[n]:+.4f}" for n in ("theta_E","gamma","e1_mass","e2_mass") if n in medians)
    plt.suptitle(f"[{tag}]  {lbl}", fontsize=10)
    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.asarray(q50))
        overlay_critical_and_caustics(image_axes=[axes[0], axes[1]], source_ax=axes[3], lens_mass=likelihood.phys_model)
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[{tag}] diagnostic plot saved to {save_path}")


# ------------------------------------------------------------------ #
# Stage M0-M3 (identical structure to slacs, adapted for Sersic pipeline)
# ------------------------------------------------------------------ #
def build_stage_m0_likelihood(
    lens_subtracted_image, noise_map, psf_kernel, feature_mask,
    stage_a, position_likelihood, circular_mask=None,
    *, source_interpolation=None, evidence_logdet="overlap", evidence_block_size=10,
):
    sie, shear = _sie_mass_from_stage_a(stage_a)
    log_lam = ParamU("log_lambda_reg", 0.0, prior_type="uniform",
                     prior_settings=[-13.815510557964274, 13.815510557964274], limits=[-13.815510557964274, 13.815510557964274])
    log_lam.to_dynamic()
    pix_src = PixelizedSourceModel(
        n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        interpolation=resolve_source_interpolation(source_interpolation),
    )
    phys = PhysicalModel(lens_mass=[sie, shear], source_light=[pix_src], lens_light=[])
    cm = feature_mask
    if circular_mask is not None:
        cm = cm | circular_mask
    return PixelizedImageProbModelOperator(
        image_data=lens_subtracted_image, noise_map=noise_map, psf_kernel=psf_kernel,
        dpix=DPIX, nsub=NSUB_PIX, phys_model=phys, mask=cm,
        position_likelihood=position_likelihood, solver_type=SOLVER_TYPE,
        evidence_logdet=evidence_logdet, evidence_block_size=evidence_block_size,
        source_bbox_padding=SOURCE_BBOX_PADDING, **_pnpg_kwargs(),
    )


def run_stage_m0(
    image_data, noise_map, psf_kernel, feature_mask, lens_light_model,
    stage_a, position_likelihood, circular_mask=None,
    *, source_interpolation=None, cached=None, upstream=None,
    evidence_logdet="overlap", evidence_block_size=10,
):
    print("\n" + "=" * 60); print(" Stage M0 : fixed SIE + shear + uniform pix source "); print("=" * 60)
    t0 = time.time(); ls = image_data - lens_light_model
    likelihood = build_stage_m0_likelihood(
        ls, noise_map, psf_kernel, feature_mask,
        stage_a, position_likelihood, circular_mask=circular_mask,
        source_interpolation=source_interpolation,
        evidence_logdet=evidence_logdet, evidence_block_size=evidence_block_size,
    )
    n_grid = 200
    objective = pixelized_objective_provenance(
        likelihood, upstream=upstream,
        sampling=dict(
            method="grid_search", vectorized=True, grid_size=n_grid,
            coarse_lambda_bounds=[1e-8, 1e8], fine_width_decades=1,
        ),
    )
    checkpoint = matching_pixelized_checkpoint(cached, objective, "m0")
    if checkpoint is not None:
        log_lambda = checkpoint["extra"]["log_lambda_best"]
        medians = {**stage_a.medians(), "log_lambda_reg": log_lambda}
        plot_cached_pixelized_stage(
            _plot_pix_stage, OUT_DIR, "m0", likelihood, medians, ["log_lambda_reg"],
        )
        return log_lambda

    loglike_batch = make_likelihood(likelihood, vectorized=True)
    log_lam_min, log_lam_max = jnp.log(1e-8), jnp.log(1e8)
    log_lam_grid_coarse = jnp.linspace(log_lam_min, log_lam_max, n_grid)
    log_ev_coarse = jnp.asarray(loglike_batch(log_lam_grid_coarse.reshape(-1, 1)))
    vc = _valid_log_evidence(log_ev_coarse)
    if not np.any(vc): raise RuntimeError("[stage-M0] All coarse grid values failed.")
    best_idx = int(np.argmax(np.where(vc, np.asarray(log_ev_coarse), -np.inf)))
    log_lam_bc = float(log_lam_grid_coarse[best_idx])
    print(f"[stage-M0] Coarse best: λ = {float(jnp.exp(log_lam_bc)):.4e}")
    hw = 0.5*jnp.log(10)
    log_lam_grid_fine = jnp.linspace(log_lam_bc-hw, log_lam_bc+hw, n_grid)
    log_ev_fine = jnp.asarray(loglike_batch(log_lam_grid_fine.reshape(-1, 1)))
    vf = _valid_log_evidence(log_ev_fine)
    if not np.any(vf): raise RuntimeError("[stage-M0] All refinement grid values failed.")
    best_idx_f = int(np.argmax(np.where(vf, np.asarray(log_ev_fine), -np.inf)))
    log_lam_best = float(log_lam_grid_fine[best_idx_f]); log_ev_best = float(log_ev_fine[best_idx_f])
    print(f"[stage-M0] Refined best: λ = {float(jnp.exp(log_lam_best)):.4e}")
    medians_a = stage_a.medians()
    t1 = time.time(); print(f"[stage-M0] time taken: {t1-t0:.2f} seconds")
    _dump_stage("m0", None, None, ["log_lambda_reg"], log_ev_best,
                extra=dict(objective=objective, log_lambda_best=log_lam_best, evidence_lambda_best=float(jnp.exp(log_lam_best)),
                           evidence_log_lambda_best=log_lam_best, time_taken=t1-t0,
                           lambda_grid_coarse=np.asarray(log_lam_grid_coarse, dtype=np.float64),
                           log_ev_coarse=np.asarray(log_ev_coarse, dtype=np.float64),
                           lambda_grid_fine=np.asarray(log_lam_grid_fine, dtype=np.float64),
                           log_ev_fine=np.asarray(log_ev_fine, dtype=np.float64)), source_interpolation=likelihood.source_model.interpolation)
    try: _plot_pix_stage("stage-M0", likelihood, {**medians_a, "log_lambda_reg": log_lam_best}, ["log_lambda_reg"], str(OUT_DIR / "stage_m0_model.png"))
    except Exception as err:
        if isinstance(err, InvalidReconstructionError):
            raise
        print(f"[stage-M0] plotting failed (non-fatal): {err}")
    return log_lam_best

def build_stage_m1_likelihood(
    ls_img, noise_map, psf_kernel, feature_mask,
    stage_a, position_likelihood, log_lam_fixed, circular_mask=None,
    *, source_interpolation=None, evidence_logdet="overlap", evidence_block_size=10,
):
    epl, shear = _epl_mass_from_stage(stage_a)
    log_lam = ParamU("log_lambda_reg", float(log_lam_fixed), prior_type="truncated_gaussian",
                     prior_settings=[float(log_lam_fixed), 0.15], limits=[float(log_lam_fixed)-0.5, float(log_lam_fixed)+0.5])
    log_lam.to_dynamic()
    pix_src = PixelizedSourceModel(
        n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        interpolation=resolve_source_interpolation(source_interpolation),
    )
    phys = PhysicalModel(lens_mass=[epl, shear], source_light=[pix_src], lens_light=[])
    cm = feature_mask
    if circular_mask is not None:
        cm = cm | circular_mask
    return PixelizedImageProbModelOperator(
        image_data=ls_img, noise_map=noise_map, psf_kernel=psf_kernel,
        dpix=DPIX, nsub=NSUB_PIX, phys_model=phys, mask=cm,
        position_likelihood=position_likelihood, solver_type=SOLVER_TYPE,
        evidence_logdet=evidence_logdet, evidence_block_size=evidence_block_size,
        source_bbox_padding=SOURCE_BBOX_PADDING, **_pnpg_kwargs(),
    )


def run_stage_m1(
    image_data, noise_map, psf_kernel, feature_mask, lens_light_model,
    stage_a, position_likelihood, log_lam_fixed, circular_mask=None,
    *, source_interpolation=None, cached=None, upstream=None,
    evidence_logdet="overlap", evidence_block_size=10,
):
    print("\n" + "=" * 60); print(" Stage M1 : EPL + shear + non-adaptive pix source (mass fit)"); print("=" * 60)
    t0 = time.time(); ls = image_data - lens_light_model
    build = partial(
        build_stage_m1_likelihood,
        ls, noise_map, psf_kernel, feature_mask,
        stage_a, position_likelihood, log_lam_fixed, circular_mask=circular_mask,
        source_interpolation=source_interpolation,
        evidence_logdet=evidence_logdet, evidence_block_size=evidence_block_size,
    )
    likelihood = build()
    sampling = SamplingConfig(n_live=300, n_eff=600, vectorized=True, n_batch=100)
    objective = pixelized_objective_provenance(
        likelihood, sampling=sampling, upstream=upstream,
    )
    checkpoint = matching_pixelized_checkpoint(cached, objective, "m1", producer=True)
    if checkpoint is not None:
        try:
            stage = _stage_from_payload(checkpoint)
            medians = checkpoint["extra"]["medians"]
            package = recover_pixelized_producer(
                likelihood, checkpoint, "s1", medians, stage.param_names,
            )
        except (KeyError, TypeError, ValueError) as err:
            print(f"[stage-M1] cannot verify cached source reference ({err}); recomputing.")
        else:
            print("[stage-M1] reusing verified source reference")
            extra = checkpoint["extra"]
            if package is not extra["s1"]:
                _dump_stage(
                    "m1", stage.samples, stage.weights, stage.param_names, stage.log_z,
                    stage=stage, extra={**extra, "s1": package},
                    source_interpolation=likelihood.source_model.interpolation,
                )
            plot_cached_pixelized_stage(
                _plot_pix_stage, OUT_DIR, "m1", likelihood, medians, stage.param_names,
                producer=(package, build),
            )
            return stage, medians, package

    stage = run_sampling(likelihood, sampling, tag="stage-M1", sampler_factory=Sampler)
    t1 = time.time(); samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    _print_summary("stage-M1", samples, weights, names); print(f"[stage-M1] time taken: {t1-t0:.2f} seconds")
    medians = stage.medians(); s1_medians = dict(medians); log_lam_m1 = float(medians["log_lambda_reg"])
    s1_package = _solve_pixel_source_for_package(likelihood, s1_medians, names)
    s1_package.update(lambda_best=float(jnp.exp(log_lam_m1)), log_lambda_best=log_lam_m1, stage_m1_medians=dict(medians))
    _validate_source_package(s1_package)
    _dump_stage("m1", samples, weights, names, logz,
                extra=dict(objective=objective, medians=medians, m1_mass_model="EPL+Shear", lambda_prior_center=float(log_lam_fixed),
                           lambda_prior_sigma=0.15,
                           lambda_prior_limits=[float(log_lam_fixed - 0.5), float(log_lam_fixed + 0.5)],
                           s1=s1_package, time_taken=t1-t0), stage=stage, source_interpolation=likelihood.source_model.interpolation)
    try: _plot_pix_stage("stage-M1", likelihood, medians, names, str(OUT_DIR / "stage_m1_model.png"))
    except Exception as err:
        if isinstance(err, InvalidReconstructionError):
            raise
        print(f"[stage-M1] plot failed (non-fatal): {err}")
    return stage, medians, s1_package

def build_stage_m2_likelihood(
    ls_img, noise_map, psf_kernel, feature_mask,
    medians_m1, position_likelihood, s1_package, circular_mask=None,
    *, source_interpolation=None, evidence_logdet="overlap", evidence_block_size=10,
):
    source_interpolation = resolve_source_interpolation(
        source_interpolation, metadata=s1_package, context="S1 source package"
    )
    epl, shear = _epl_mass_from_medians(medians_m1)
    log_lam = ParamU("log_lambda_reg", float(s1_package["log_lambda_best"]), prior_type="uniform",
                     prior_settings=[-13.815510557964274, 13.815510557964274], limits=[-13.815510557964274, 13.815510557964274]); log_lam.to_dynamic()
    rho = ParamU("adaptive_reg_rho", ADAPTIVE_REG_RHO, prior_type="uniform", prior_settings=[0.0, ADAPTIVE_REG_RHO_PRIOR_MAX], limits=[0.0, ADAPTIVE_REG_RHO_PRIOR_MAX]); rho.to_dynamic()
    pix_src = PixelizedSourceModel(
        n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        adaptive_reg_rho=rho,
        interpolation=source_interpolation,
    )
    phys = PhysicalModel(lens_mass=[epl, shear], source_light=[pix_src], lens_light=[])
    cm = feature_mask
    if circular_mask is not None:
        cm = cm | circular_mask
    likelihood = PixelizedImageProbModelOperator(
        image_data=ls_img, noise_map=noise_map, psf_kernel=psf_kernel,
        dpix=DPIX, nsub=NSUB_PIX, phys_model=phys, mask=cm,
        position_likelihood=position_likelihood, solver_type=SOLVER_TYPE,
        evidence_logdet=evidence_logdet, evidence_block_size=evidence_block_size,
        source_bbox_padding=SOURCE_BBOX_PADDING,
        **_pnpg_kwargs(), **_fixed_source_kwargs(s1_package),
    )
    require_reference_observation(likelihood, s1_package)
    return likelihood


def run_stage_m2(
    image_data, noise_map, psf_kernel, feature_mask, lens_light_model,
    medians_m1, position_likelihood, s1_package, circular_mask=None,
    *, source_interpolation=None, cached=None, upstream=None,
    evidence_logdet="overlap", evidence_block_size=10,
):
    print("\n" + "=" * 60); print(" Stage M2 : fixed EPL + shear + adaptive pix source (fit λ, rho)"); print("=" * 60)
    t0 = time.time(); ls = image_data - lens_light_model
    likelihood = build_stage_m2_likelihood(
        ls, noise_map, psf_kernel, feature_mask,
        medians_m1, position_likelihood, s1_package, circular_mask=circular_mask,
        source_interpolation=source_interpolation,
        evidence_logdet=evidence_logdet, evidence_block_size=evidence_block_size,
    )
    sampling = SamplingConfig(n_live=250, n_eff=600, vectorized=True, n_batch=100)
    objective = pixelized_objective_provenance(
        likelihood, reference=s1_package, sampling=sampling, upstream=upstream,
    )
    checkpoint = matching_pixelized_checkpoint(cached, objective, "m2")
    if checkpoint is not None:
        stage = _stage_from_payload(checkpoint)
        medians = checkpoint["extra"]["medians"]
        plot_cached_pixelized_stage(
            _plot_pix_stage, OUT_DIR, "m2", likelihood, medians, stage.param_names,
        )
        return stage, medians, _reg_hyperparams_from_m2_payload(checkpoint)

    stage = run_sampling(likelihood, sampling, tag="stage-M2", sampler_factory=Sampler)
    t1 = time.time(); samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    _print_summary("stage-M2", samples, weights, names); print(f"[stage-M2] time taken: {t1-t0:.2f} seconds")
    medians = stage.medians()
    rhm2 = {"log_lambda_reg": float(medians["log_lambda_reg"]), "adaptive_reg_rho": float(medians["adaptive_reg_rho"])}
    print(f"[stage-M2] median source reg: {_format_reg_hyperparams(rhm2)}")
    _dump_stage("m2", samples, weights, names, logz, extra=dict(objective=objective, medians=medians, m2_mass_model="fixed-EPL+Shear", mass_medians_fixed=dict(medians_m1), reg_hyperparams=rhm2, time_taken=t1-t0), stage=stage, source_interpolation=likelihood.source_model.interpolation)
    try: _plot_pix_stage("stage-M2", likelihood, medians, names, str(OUT_DIR / "stage_m2_model.png"))
    except Exception as err:
        if isinstance(err, InvalidReconstructionError):
            raise
        print(f"[stage-M2] plot failed (non-fatal): {err}")
    return stage, medians, rhm2

def build_stage_m3_likelihood(
    ls_img, noise_map, psf_kernel, feature_mask,
    stage_m1, position_likelihood, rh_fixed, s1_package, circular_mask=None,
    *, source_interpolation=None, evidence_logdet="overlap", evidence_block_size=10,
):
    source_interpolation = resolve_source_interpolation(
        source_interpolation, metadata=s1_package, context="S1 source package"
    )
    epl, shear = _epl_mass_from_stage(stage_m1)
    log_lam = ParamU("log_lambda_reg", float(rh_fixed["log_lambda_reg"])); log_lam.to_static()
    rho = ParamU("adaptive_reg_rho", float(rh_fixed["adaptive_reg_rho"])); rho.to_static()
    pix_src = PixelizedSourceModel(
        n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        adaptive_reg_rho=rho,
        interpolation=source_interpolation,
    )
    phys = PhysicalModel(lens_mass=[epl, shear], source_light=[pix_src], lens_light=[])
    cm = feature_mask
    if circular_mask is not None:
        cm = cm | circular_mask
    likelihood = PixelizedImageProbModelOperator(
        image_data=ls_img, noise_map=noise_map, psf_kernel=psf_kernel,
        dpix=DPIX, nsub=NSUB_PIX, phys_model=phys, mask=cm,
        position_likelihood=position_likelihood, solver_type=SOLVER_TYPE,
        evidence_logdet=evidence_logdet, evidence_block_size=evidence_block_size,
        source_bbox_padding=SOURCE_BBOX_PADDING,
        **_pnpg_kwargs(), **_fixed_source_kwargs(s1_package),
    )
    require_reference_observation(likelihood, s1_package)
    return likelihood


def run_stage_m3(
    image_data, noise_map, psf_kernel, feature_mask, lens_light_model,
    stage_m1, position_likelihood, rh_fixed, s1_package, circular_mask=None,
    *, source_interpolation=None, cached=None, upstream=None,
    evidence_logdet="overlap", evidence_block_size=10,
):
    print("\n" + "=" * 60); print(" Stage M3 : EPL + shear + adaptive pix source (final mass fit)"); print("=" * 60)
    print(f"[stage-M3] fixed source reg: {_format_reg_hyperparams(rh_fixed)}")
    t0 = time.time(); ls = image_data - lens_light_model
    likelihood = build_stage_m3_likelihood(
        ls, noise_map, psf_kernel, feature_mask,
        stage_m1, position_likelihood, rh_fixed, s1_package, circular_mask=circular_mask,
        source_interpolation=source_interpolation,
        evidence_logdet=evidence_logdet, evidence_block_size=evidence_block_size,
    )
    sampling = SamplingConfig(n_live=300, n_eff=600, vectorized=True, n_batch=100)
    objective = pixelized_objective_provenance(
        likelihood, reference=s1_package, sampling=sampling, upstream=upstream,
    )
    checkpoint = matching_pixelized_checkpoint(cached, objective, "m3")
    if checkpoint is not None:
        stage = _stage_from_payload(checkpoint)
        medians = checkpoint["extra"]["medians"]
        plot_cached_pixelized_stage(
            _plot_pix_stage, OUT_DIR, "m3", likelihood, medians, stage.param_names,
        )
        return stage, medians

    stage = run_sampling(likelihood, sampling, tag="stage-M3", sampler_factory=Sampler)
    t1 = time.time(); samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    _print_summary("stage-M3", samples, weights, names); print(f"[stage-M3] time taken: {t1-t0:.2f} seconds")
    medians = stage.medians()
    _dump_stage("m3", samples, weights, names, logz, extra=dict(objective=objective, medians=medians, m3_mass_model="EPL+Shear", reg_hyperparams_fixed=dict(rh_fixed), time_taken=t1-t0), stage=stage, source_interpolation=likelihood.source_model.interpolation)
    try: _plot_pix_stage("stage-M3", likelihood, medians, names, str(OUT_DIR / "stage_m3_model.png"))
    except Exception as err:
        if isinstance(err, InvalidReconstructionError):
            raise
        print(f"[stage-M3] plot failed (non-fatal): {err}")
    return stage, medians


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main(
    skip_done=False, out_dir=None, *,
    source_interpolation=None, evidence_logdet="overlap", evidence_block_size=10,
):
    global OUT_DIR
    if out_dir is not None: OUT_DIR = Path(out_dir)
    source_interpolation = resolve_stage_source_interpolation(
        OUT_DIR, ("a", "l", "m0", "m1", "m2", "m3"),
        requested=source_interpolation, resume=skip_done,
    )
    source_options = dict(source_interpolation=source_interpolation)
    image_data, noise_map, psf_kernel, _ = load_lens_data(image_path=str(DATA_DIR/"image.fits"), noise_path=str(DATA_DIR/"noise.fits"), psf_path=str(DATA_DIR/"psf.fits"))
    circular_mask = _make_circular_mask(image_data.shape, DPIX, radius_arcsec=MASK_RADIUS)
    print(f"[circular mask] excluded {int(circular_mask.sum())} / {circular_mask.size} pixels (> {MASK_RADIUS} arcsec)")

    ta=0.0
    if skip_done and (OUT_DIR/"stage_a.pkl").exists():
        print(f"[stage-A] loading cached {OUT_DIR}/stage_a.pkl"); d=_load_stage("a", **source_options); stage_a=_stage_from_payload(d); medians_a=d["extra"]["medians"]; lens_light_model=d["extra"]["lens_light_model"]; ta=d["extra"].get("time_taken",0.0)
    else: stage_a, medians_a, lens_light_model = run_stage_a(image_data, noise_map, psf_kernel, circular_mask=circular_mask, **source_options); ta=_load_stage("a", **source_options)["extra"].get("time_taken",0.0)

    feature_mask = run_stage_b(image_data, noise_map, lens_light_model, circular_mask=circular_mask)

    tl=0.0
    if skip_done and (OUT_DIR/"stage_l.pkl").exists():
        print(f"[stage-L] loading cached {OUT_DIR}/stage_l.pkl"); d=_load_stage("l", **source_options); stage_l=_stage_from_payload(d); medians_l=d["extra"]["medians"]; lens_light_model=d["extra"]["lens_light_model"]; tl=d["extra"].get("time_taken",0.0)
    else: stage_l, medians_l, lens_light_model = run_stage_l(image_data, noise_map, psf_kernel, feature_mask, stage_a, circular_mask=circular_mask, **source_options); tl=_load_stage("l", **source_options)["extra"].get("time_taken",0.0)

    position_likelihood = _position_likelihood_from_stage_a(medians_a)

    def checkpoint(tag):
        return load_pixelized_checkpoint(_load_stage, OUT_DIR, tag, resume=skip_done)

    data = (image_data, noise_map, psf_kernel, feature_mask, lens_light_model)
    options = dict(
        **source_options, evidence_logdet=evidence_logdet, evidence_block_size=evidence_block_size,
    )
    upstream = {tag: pixelized_stage_identity(_load_stage(tag)) for tag in ("a", "l")}
    log_lam_m0 = run_stage_m0(
        *data, stage_a, position_likelihood, circular_mask,
        cached=checkpoint("m0"), upstream=upstream, **options,
    )
    stage_m1, medians_m1, s1 = run_stage_m1(
        *data, stage_a, position_likelihood, log_lam_m0, circular_mask,
        cached=checkpoint("m1"),
        upstream={**upstream, "m0": pixelized_stage_identity(_load_stage("m0"))},
        **options,
    )
    m1_id = pixelized_stage_identity(_load_stage("m1"))
    _, _, rhm2 = run_stage_m2(
        *data, medians_m1, position_likelihood, s1, circular_mask,
        cached=checkpoint("m2"), upstream={"m1": m1_id}, **options,
    )
    _, medians_m3 = run_stage_m3(
        *data, stage_m1, position_likelihood, rhm2, s1, circular_mask,
        cached=checkpoint("m3"),
        upstream={"m1": m1_id, "m2": pixelized_stage_identity(_load_stage("m2"))},
        **options,
    )
    tm0, tm1, tm2, tm3 = (
        _load_stage(tag)["extra"].get("time_taken", 0.0)
        for tag in ("m0", "m1", "m2", "m3")
    )

    print("\n"+"="*60); print(" Pipeline complete"); print("="*60)
    print(" Time summary:"); print(f"    Stage A:  {ta/60:.2f} min"); print(f"    Stage L:  {tl/60:.2f} min")
    print(f"    Stage M0: {tm0/60:.2f} min"); print(f"    Stage M1: {tm1/60:.2f} min")
    print(f"    Stage M2: {tm2/60:.2f} min"); print(f"    Stage M3: {tm3/60:.2f} min")
    print(f"    Total:    {(ta+tl+tm0+tm1+tm2+tm3)/60:.2f} min\n")
    print(f"    M0 best lambda_reg     = {float(jnp.exp(log_lam_m0)):.4e}")
    print(f"    M1 median gamma        = {medians_m1.get('gamma', np.nan):+.4f}")
    print(f"    M2 median source reg   = {_format_reg_hyperparams(rhm2)}")
    for k in ("theta_E","gamma","e1_mass","e2_mass","center_x_mass","center_y_mass","gamma1","gamma2"):
        if k in medians_m3: print(f"    final  {k:15s} = {medians_m3[k]:+.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-done", action="store_true"); parser.add_argument("--out-dir", default=None)
    parser.add_argument("--source-interpolation", choices=("bilinear", "cubic_hermite"))
    parser.add_argument("--evidence-logdet", choices=("overlap", "block"), default="overlap")
    parser.add_argument("--evidence-block-size", type=int, default=10, help="Overlap tile width.")
    main(**vars(parser.parse_args()))
