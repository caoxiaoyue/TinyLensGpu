"""
Five-stage inference pipeline for the pix_src_pipe demo.

Stage a : SIE + shear + MGE lens light + MGE source light (uniform priors)
Stage b : build an arc feature mask from stage-a residuals
Stage l : arc-masked MGE lens light refinement (Gaussian priors from stage a)
Stage c : SIE + shear + pixelized source (Gaussian priors from stage a,
          + log-uniform on lambda_reg)
Stage d : EPL + shear + pixelized source, with lambda_reg fixed to the stage-c
          median; all shared mass/shear params use Gaussian priors from stage c,
          and EPL.gamma is sampled with an explicit prior.

Each stage pickles its posterior samples/weights to
``output/stage_{a,l,c,d}.pkl`` and is re-runnable via ``--skip-done``.
"""

from __future__ import annotations

import argparse
import os
import pickle
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
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import PixelizedImageProbModel
from TinyLensGpu.PhysicalModel import (
    GaussianEllipse,
    PhysicalModel,
    PixelizedSourceModel,
    SersicEllipse,
    Shear,
)
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE, EPL
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.visualizer import plot_model_results

from mask_tool import arc_mask_from
from prior_passing import GaussianPriorPasser

import caskade as ck
import jax.scipy.linalg as jsl

# ------------------------------------------------------------------ #
DPIX = 0.05
NSUB = 4
NSUB_PIX = 2  # lower oversampling for pixelized stages (memory)
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
        sampler_kwargs["n_batch"] = n_live
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
    order = np.argsort(samples, axis=0)
    w = weights / weights.sum()
    for i, name in enumerate(param_names):
        s = samples[order[:, i], i]
        cw = np.cumsum(w[order[:, i]])
        cw /= cw[-1]
        out[name] = float(np.interp(0.5, cw, s))
    return out


def _print_summary(tag: str, samples, weights, param_names):
    print(f"\n[{tag}] Posterior summary:")
    w = weights / weights.sum()
    for i, name in enumerate(param_names):
        order = np.argsort(samples[:, i])
        s = samples[order, i]
        cw = np.cumsum(w[order])
        cw /= cw[-1]
        q16, q50, q84 = [float(np.interp(q, cw, s)) for q in (0.16, 0.5, 0.84)]
        print(f"    {name:25s} = {q50:+.4f} ({q16-q50:+.4f}, {q84-q50:+.4f})")


# ------------------------------------------------------------------ #
# Stage A — SIE + shear + MGE lens light + MGE source light
# ------------------------------------------------------------------ #
def build_stage_a_likelihood(image_data, noise_map, psf_kernel):
    sie = SIE(
        theta_E=ParamU("theta_E", 1.5, prior_type="uniform",
                       prior_settings=[0.5, 2.5], limits=[0.0, 5.0]),
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
    N_src = 10
    sigmas_src = 10 ** np.linspace(-2.0, 0.0, N_src)
    cx_s = ParamU("center_x_src", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.5], limits=[-3.0, 3.0])
    cy_s = ParamU("center_y_src", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.5], limits=[-3.0, 3.0])
    e1_s = ParamU("e1_src", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_s = ParamU("e2_src", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    source_gaussians = []
    for i, s in enumerate(sigmas_src):
        g = GaussianEllipse(
            sigma=ParamU(f"sigma_src_{i}", float(s)),
            center_x=cx_s, center_y=cy_s, e1=e1_s, e2=e2_s,
            flux=ParamU(f"flux_src_{i}", 1.0),
        )
        g.sigma.to_static(float(s))
        g.flux.to_static(1.0)
        source_gaussians.append(g)
    cx_s.to_dynamic(); cy_s.to_dynamic(); e1_s.to_dynamic(); e2_s.to_dynamic()

    # MGE lens light
    N_ll = 20
    sigmas_ll = 10 ** np.linspace(-2.0, np.log10(3.0), N_ll)
    cx_l = ParamU("center_x_lens", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    cy_l = ParamU("center_y_lens", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    e1_l = ParamU("e1_lens", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_l = ParamU("e2_lens", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    lens_gaussians = []
    for i, s in enumerate(sigmas_ll):
        g = GaussianEllipse(
            sigma=ParamU(f"sigma_lens_{i}", float(s)),
            center_x=cx_l, center_y=cy_l, e1=e1_l, e2=e2_l,
            flux=ParamU(f"flux_lens_{i}", 1.0),
        )
        g.sigma.to_static(float(s))
        g.flux.to_static(1.0)
        lens_gaussians.append(g)
    cx_l.to_dynamic(); cy_l.to_dynamic(); e1_l.to_dynamic(); e2_l.to_dynamic()

    phys = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=source_gaussians,
        lens_light=lens_gaussians,
    )
    return ImageProbModel(
        image_data=image_data, noise_map=noise_map,
        psf_kernel=psf_kernel, dpix=DPIX, nsub=NSUB,
        phys_model=phys, use_linear=True, solver_type="nnls",
    )


def run_stage_a(image_data, noise_map, psf_kernel):
    OUT_DIR.mkdir(exist_ok=True)
    print("\n" + "=" * 60)
    print(" Stage A : SIE + shear + MGE lens light + MGE source light")
    print("=" * 60)
    likelihood = build_stage_a_likelihood(image_data, noise_map, psf_kernel)
    samples, weights, names, logz = _run_sampler(
        likelihood, n_live=400, n_eff=2000, tag="stage-A",
    )
    _print_summary("stage-A", samples, weights, names)

    medians = _posterior_median(samples, weights, names)
    q50 = [medians[n] for n in names]

    # Evaluate lens-light model image at median (needed for stage B).
    likelihood.set_values(q50)
    fwd = likelihood.forward_model(
        use_linear=True, return_intensity=True, ret_each_plane=True,
        image_map=likelihood.image_data, noise_map=likelihood.noise_map,
    )
    lens_image_model = np.asarray(fwd[1])

    # Persist posterior before plotting so a plot failure can't lose results.
    _dump_stage(
        "a", samples, weights, names, logz,
        extra=dict(
            medians=medians,
            lens_light_model=lens_image_model,
        ),
    )

    try:
        plot_model_results(
            likelihood, jnp.asarray(q50),
            save_path=str(OUT_DIR / "stage_a_model.png"),
            title="Stage A : MGE lens+source",
        )
    except Exception as err:
        print(f"[stage-A] plotting failed (non-fatal): {err}")
    return samples, weights, names, medians, lens_image_model


# ------------------------------------------------------------------ #
# Stage B — build an arc mask from stage-A residuals
# ------------------------------------------------------------------ #
def run_stage_b(image_data, noise_map, lens_light_model):
    print("\n" + "=" * 60)
    print(" Stage B : build arc feature mask from stage-A residuals")
    print("=" * 60)
    residual = image_data - lens_light_model
    snr_map = residual / noise_map
    # arc_mask_from returns True for EXCLUDED pixels (library convention)
    arc_mask = arc_mask_from(snr_map, threshold=3.0,
                             ignor_size=25, ext_size=5, close_size=3)
    feature_mask = arc_mask
    n_in = int((~arc_mask).sum())
    print(f"[stage-B] arc pixels kept = {n_in} / {arc_mask.size}")

    DATA_DIR.mkdir(exist_ok=True)
    fits.writeto(DATA_DIR / "feature_mask.fits",
                 feature_mask.astype(np.uint8), overwrite=True)

    # Quick-look figure
    extent = [-image_data.shape[0] * DPIX / 2,
               image_data.shape[0] * DPIX / 2] * 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    im0 = axes[0].imshow(residual, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("residual = image - stage-A lens light")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(snr_map, origin="lower", extent=extent, cmap="viridis",
                         vmin=-3, vmax=np.nanpercentile(snr_map, 99.5))
    axes[1].set_title("S/N map")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    # Plot the kept region (the arc itself) for visualization
    axes[2].imshow(~arc_mask, origin="lower", extent=extent, cmap="gray")
    axes[2].set_title("arc region (True = kept)")
    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUT_DIR / "stage_b_mask.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return feature_mask


# ------------------------------------------------------------------ #
# Stage L — arc-masked MGE lens light refinement
# ------------------------------------------------------------------ #
def build_stage_new_likelihood(
    image_data,
    noise_map,
    psf_kernel,
    feature_mask,
    samples_a,
    weights_a,
    names_a,
):
    """
    Build a lens-light-only likelihood using Gaussian priors from stage a.

    The arc region (``feature_mask == False``) is excluded so the fit is
    driven only by the clean lens-light pixels, yielding a better
    lens-light subtraction for the subsequent pixelized-source stages.
    """
    passer = GaussianPriorPasser(samples_a, weights_a, names_a)

    # Lens light geometry — Gaussian priors from stage a posterior
    cx_l = passer.gaussian(
        "center_x_lens", model="Gaussian", attr="center_x", limits=[-1.0, 1.0]
    )
    cy_l = passer.gaussian(
        "center_y_lens", model="Gaussian", attr="center_y", limits=[-1.0, 1.0]
    )
    e1_l = passer.gaussian(
        "e1_lens", model="Gaussian", attr="e1", limits=[-1.0, 1.0]
    )
    e2_l = passer.gaussian(
        "e2_lens", model="Gaussian", attr="e2", limits=[-1.0, 1.0]
    )

    # MGE lens light — same structure as stage a
    N_ll = 20
    sigmas_ll = 10 ** np.linspace(-2.0, np.log10(3.0), N_ll)
    lens_gaussians = []
    for i, s in enumerate(sigmas_ll):
        g = GaussianEllipse(
            sigma=ParamU(f"sigma_lens_{i}", float(s)),
            center_x=cx_l,
            center_y=cy_l,
            e1=e1_l,
            e2=e2_l,
            flux=ParamU(f"flux_lens_{i}", 1.0),
        )
        g.sigma.to_static(float(s))
        g.flux.to_static(1.0)
        lens_gaussians.append(g)
    cx_l.to_dynamic()
    cy_l.to_dynamic()
    e1_l.to_dynamic()
    e2_l.to_dynamic()

    # Invert mask: feature_mask has True = EXCLUDED; stage L wants to
    # exclude the arc (where feature_mask is False) and keep everything else.
    lens_light_mask = ~feature_mask

    phys = PhysicalModel(
        lens_mass=[],
        source_light=[],
        lens_light=lens_gaussians,
    )
    return ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB,
        phys_model=phys,
        use_linear=True,
        solver_type="nnls",
        mask=lens_light_mask,
    )


def run_stage_new(image_data, noise_map, psf_kernel, feature_mask,
                  samples_a, weights_a, names_a):
    print("\n" + "=" * 60)
    print(" Stage L : arc-masked MGE lens light refinement")
    print("=" * 60)
    n_excluded = int(feature_mask.sum())
    n_kept = feature_mask.size - n_excluded
    print(f"[stage-L] arc pixels excluded = {n_excluded} / {feature_mask.size}")
    print(f"[stage-L] lens-light pixels kept  = {n_kept} / {feature_mask.size}")

    likelihood = build_stage_new_likelihood(
        image_data,
        noise_map,
        psf_kernel,
        feature_mask,
        samples_a,
        weights_a,
        names_a,
    )
    samples, weights, names, logz = _run_sampler(
        likelihood,
        n_live=200,
        n_eff=800,
        tag="stage-L",
        vectorized=False,
    )
    _print_summary("stage-L", samples, weights, names)

    medians = _posterior_median(samples, weights, names)
    q50 = [medians[n] for n in names]

    # Evaluate lens-light model image at median (needed for stage C/D).
    likelihood.set_values(q50)
    fwd = likelihood.forward_model(
        use_linear=True,
        return_intensity=True,
        ret_each_plane=True,
        image_map=likelihood.image_data,
        noise_map=likelihood.noise_map,
    )
    lens_image_model = np.asarray(fwd[1])

    _dump_stage(
        "l",
        samples,
        weights,
        names,
        logz,
        extra=dict(
            medians=medians,
            lens_light_model=lens_image_model,
        ),
    )

    try:
        plot_model_results(
            likelihood,
            jnp.asarray(q50),
            save_path=str(OUT_DIR / "stage_l_model.png"),
            title="Stage L : arc-masked MGE lens light",
        )
    except Exception as err:
        print(f"[stage-L] plotting failed (non-fatal): {err}")

    # 1×4 diagnostic: true lens light vs. fit
    try:
        lens_light_true = np.asarray(fits.getdata(DATA_DIR / "lens_light_true.fits"))
        diff = lens_light_true - lens_image_model
        diff_norm = diff / noise_map
        npix = image_data.shape[0]
        ext = [-npix * DPIX / 2, npix * DPIX / 2] * 2

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
        for ax, img, title, kw in [
            (axes[0], lens_light_true, "True lens light (X)", dict(vmin=0, cmap="viridis")),
            (axes[1], lens_image_model,  "Fitted lens light (M)", dict(vmin=0, cmap="viridis")),
            (axes[2], diff,               "Residual (X - M)", dict(cmap="RdBu_r")),
            (axes[3], diff_norm,          "(X - M) / noise", dict(vmin=-5, vmax=5, cmap="RdBu_r")),
        ]:
            im = ax.imshow(img, origin="lower", extent=ext, **kw)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("arcsec")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[0].set_ylabel("arcsec")
        plt.suptitle("Stage L : lens light diagnostic", fontsize=11)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "stage_l_diagnostic.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[stage-L] diagnostic plot saved to {OUT_DIR / 'stage_l_diagnostic.png'}")
    except Exception as err:
        print(f"[stage-L] diagnostic plot failed (non-fatal): {err}")

    return samples, weights, names, medians, lens_image_model


# ------------------------------------------------------------------ #
# Stage C / D — pixelized source stages
# ------------------------------------------------------------------ #
def _sie_mass_from_stage_a(passer: GaussianPriorPasser):
    """SIE (+ shear) with Gaussian priors inherited from stage-A posterior."""
    theta_E = passer.gaussian(
        "theta_E", model="SIE", attr="theta_E", limits=[0.0, 5.0],
    )
    e1m = passer.gaussian(
        "e1_mass", model="SIE", attr="e1", limits=[-1.0, 1.0],
    )
    e2m = passer.gaussian(
        "e2_mass", model="SIE", attr="e2", limits=[-1.0, 1.0],
    )
    cx = passer.gaussian(
        "center_x_mass", model="SIE", attr="center_x", limits=[-1.0, 1.0],
    )
    cy = passer.gaussian(
        "center_y_mass", model="SIE", attr="center_y", limits=[-1.0, 1.0],
    )
    sie = SIE(theta_E=theta_E, e1=e1m, e2=e2m, center_x=cx, center_y=cy)
    for p in (sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y):
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
    return sie, shear


def _epl_mass_for_stage_d(passer: GaussianPriorPasser):
    """EPL (+ shear) with Gaussian priors inherited from stage-C posterior.

    Stage C uses SIE, so it has no slope parameter. EPL.gamma therefore uses an
    explicit prior in stage D.
    """
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
        prior_type="uniform",
        prior_settings=[1.0, 3.0],
        limits=[1.0, 3.0],
    )
    gamma.to_dynamic()

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
def _position_likelihood_from_stage_a(medians_a: dict) -> dict:
    """
    Build a position-likelihood config dict from stage-A posterior medians.

    This follows the pipeline assumption that the lensed image positions are
    not known a priori. Instead, we estimate them by:

    1) taking the stage-A median mass model (SIE + shear),
    2) taking the stage-A median source center (center_x_src, center_y_src),
    3) solving the lens equation to get the corresponding multiple image
       positions in the image plane.

    The returned dict matches `PixelizedImageProbModel(position_likelihood=...)`:
    - positions: list[[x, y], ...] in arcsec
    - threshold_arcsec: fixed to 0.3 (task requirement)
    - min_log_like: likelihood floor used when the constraint is violated
    """
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
        position_sigma=[0.01], #dummy value
        source_x=float(medians_a["center_x_src"]),
        source_y=float(medians_a["center_y_src"]),
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

    threshold_arcsec = 0.3
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
def _plot_pix_stage(tag, likelihood, medians, param_names, save_path):
    """4-panel diagnostic: data | model | norm-residual | source."""
    q50 = [medians[n] for n in param_names]
    image_data = np.asarray(likelihood.image_data)
    noise_map  = np.asarray(likelihood.noise_map)
    mask       = ~np.asarray(likelihood.unmask, dtype=bool)  # True = excluded

    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.array(q50))
        design_matrix, src_half_size = likelihood.sim_obj.design_matrix()
        lam_val = likelihood.phys_model.source_light[0].lambda_reg.value
        reg_matrix, _ = likelihood._regularization_matrix(src_half_size)
        source_pixels, chol, curvature = likelihood._solve_source(
            design_matrix, reg_matrix, jnp.asarray(lam_val)
        )
        inv_F = jsl.cho_solve((chol, True), jnp.eye(curvature.shape[0]))
        N_eff = float(jnp.trace(inv_F @ (curvature - jnp.asarray(lam_val) * reg_matrix)))

    model_1d = np.array(design_matrix @ source_pixels)
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
    ext_i = [-npix * DPIX / 2, npix * DPIX / 2] * 2
    ext_s = [-float(src_half_size), float(src_half_size)] * 2
    vmax  = np.nanpercentile(image_data[~mask], 99.5)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, img, title, kw in [
        (axes[0], image_data,  "Data (lens-subtracted)", dict(vmin=0, vmax=vmax, cmap="viridis")),
        (axes[1], model_image, "Model image",            dict(vmin=0, vmax=vmax, cmap="viridis")),
        (axes[2], np.where(mask, np.nan, resid_norm),
                               f"Norm. residual\nχ²/ν={chi2_nu:.3f}",
                               dict(vmin=-5, vmax=5, cmap="RdBu_r")),
    ]:
        im = ax.imshow(img, origin="lower", extent=ext_i, **kw)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("arcsec")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    im3 = axes[3].imshow(src_img, origin="lower", extent=ext_s, cmap="viridis")
    axes[3].set_title(f"Source reconstruction\n(λ={float(lam_val):.2e})", fontsize=11)
    axes[3].set_xlabel("arcsec")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    axes[0].set_ylabel("arcsec")

    lbl = "  ".join(f"{n}={medians[n]:+.4f}" for n in
                    ("theta_E", "gamma", "e1_mass", "e2_mass") if n in medians)
    plt.suptitle(f"[{tag}]  {lbl}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] diagnostic plot saved to {save_path}")


# ------------------------------------------------------------------ #
# Stage C — SIE + shear + pixelized source (on lens-subtracted image)
# ------------------------------------------------------------------ #
def build_stage_c_likelihood(
    lens_subtracted_image, noise_map, psf_kernel, feature_mask,
    passer: GaussianPriorPasser, position_likelihood,
):
    sie, shear = _sie_mass_from_stage_a(passer)
    lam = ParamU(
        "lambda_reg",
        1.0,
        prior_type="log_uniform",
        prior_settings=[1e-3, 1e3],
        limits=[1e-6, 1e6],
    )
    lam.to_dynamic()

    pix_src = PixelizedSourceModel(
        nx=30,
        ny=30,
        lambda_reg=lam,
        regularization_type="first-order",
    )
    phys = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=[pix_src],
        lens_light=[],
    )
    return PixelizedImageProbModel(
        image_data=lens_subtracted_image,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=feature_mask,
        position_likelihood=position_likelihood,
    )


# ------------------------------------------------------------------ #
# Stage D — EPL + shear + pixelized source (lambda_reg fixed)
# ------------------------------------------------------------------ #
def build_stage_d_likelihood(
    lens_subtracted_image, noise_map, psf_kernel, feature_mask,
    passer: GaussianPriorPasser, lambda_reg_value, position_likelihood,
):
    epl, shear = _epl_mass_for_stage_d(passer)
    lam = ParamU("lambda_reg", float(lambda_reg_value))
    lam.to_static(float(lambda_reg_value))

    pix_src = PixelizedSourceModel(
        nx=30,
        ny=30,
        lambda_reg=lam,
        regularization_type="first-order",
    )
    phys = PhysicalModel(
        lens_mass=[epl, shear],
        source_light=[pix_src],
        lens_light=[],
    )
    return PixelizedImageProbModel(
        image_data=lens_subtracted_image,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=feature_mask,
        position_likelihood=position_likelihood,
    )


def run_stage_c(image_data, noise_map, psf_kernel, feature_mask,
                lens_light_model, samples_a, weights_a, names_a, position_likelihood):
    print("\n" + "=" * 60)
    print(" Stage C : SIE + shear + pix source (lambda_reg free)")
    print("=" * 60)
    lens_subtracted = image_data - lens_light_model
    passer = GaussianPriorPasser(samples_a, weights_a, names_a)
    likelihood = build_stage_c_likelihood(
        lens_subtracted,
        noise_map,
        psf_kernel,
        feature_mask,
        passer,
        position_likelihood,
    )
    samples, weights, names, logz = _run_sampler(
        likelihood, n_live=200, n_eff=800, tag="stage-C", vectorized=False,
    )
    _print_summary("stage-C", samples, weights, names)
    medians = _posterior_median(samples, weights, names)
    _dump_stage("c", samples, weights, names, logz, extra=dict(medians=medians))
    try:
        _plot_pix_stage("stage-C", likelihood, medians, names,
                        str(OUT_DIR / "stage_c_model.png"))
    except Exception as err:
        print(f"[stage-C] plotting failed (non-fatal): {err}")
    return samples, weights, names, medians


def run_stage_d(image_data, noise_map, psf_kernel, feature_mask,
                lens_light_model, samples_c, weights_c, names_c, medians_c, position_likelihood):
    print("\n" + "=" * 60)
    print(" Stage D : EPL + shear + pix source (lambda_reg fixed at stage-C median)")
    print("=" * 60)
    lambda_reg_fixed = medians_c["lambda_reg"]
    lens_subtracted = image_data - lens_light_model
    passer = GaussianPriorPasser(samples_c, weights_c, names_c)
    likelihood = build_stage_d_likelihood(
        lens_subtracted,
        noise_map,
        psf_kernel,
        feature_mask,
        passer,
        lambda_reg_fixed,
        position_likelihood,
    )
    samples, weights, names, logz = _run_sampler(
        likelihood, n_live=200, n_eff=800, tag="stage-D", vectorized=False,
    )
    _print_summary("stage-D", samples, weights, names)
    medians = _posterior_median(samples, weights, names)
    _dump_stage("d", samples, weights, names, logz,
                extra=dict(medians=medians,
                           lambda_reg_fixed=lambda_reg_fixed))
    try:
        _plot_pix_stage("stage-D", likelihood, medians, names,
                        str(OUT_DIR / "stage_d_model.png"))
    except Exception as err:
        print(f"[stage-D] plotting failed (non-fatal): {err}")
    return samples, weights, names, medians


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def main(skip_done: bool = False):
    image_data, noise_map, psf_kernel, _ = load_lens_data(
        image_path=str(DATA_DIR / "image.fits"),
        noise_path=str(DATA_DIR / "noise.fits"),
        psf_path=str(DATA_DIR / "psf.fits"),
    )

    # ---- stage A ---------------------------------------------------- #
    if skip_done and (OUT_DIR / "stage_a.pkl").exists():
        print("[stage-A] loading cached output/stage_a.pkl")
        d = _load_stage("a")
        samples_a, weights_a, names_a = d["samples"], d["weights"], d["param_names"]
        medians_a = d["extra"]["medians"]
        lens_light_model = d["extra"]["lens_light_model"]
    else:
        samples_a, weights_a, names_a, medians_a, lens_light_model = run_stage_a(
            image_data, noise_map, psf_kernel,
        )

    # ---- stage B ---------------------------------------------------- #
    feature_mask = run_stage_b(image_data, noise_map, lens_light_model)

    # ---- stage L ---------------------------------------------------- #
    if skip_done and (OUT_DIR / "stage_l.pkl").exists():
        print("[stage-L] loading cached output/stage_l.pkl")
        d = _load_stage("l")
        samples_l, weights_l, names_l = d["samples"], d["weights"], d["param_names"]
        medians_l = d["extra"]["medians"]
        lens_light_model = d["extra"]["lens_light_model"]
    else:
        samples_l, weights_l, names_l, medians_l, lens_light_model = run_stage_new(
            image_data, noise_map, psf_kernel, feature_mask,
            samples_a, weights_a, names_a,
        )

    lens_subtracted = image_data - lens_light_model
    position_likelihood = _position_likelihood_from_stage_a(medians_a)

    # ---- stage C ---------------------------------------------------- #
    if skip_done and (OUT_DIR / "stage_c.pkl").exists():
        print("[stage-C] loading cached output/stage_c.pkl")
        d = _load_stage("c")
        samples_c, weights_c, names_c = d["samples"], d["weights"], d["param_names"]
        medians_c = d["extra"]["medians"]
    else:
        samples_c, weights_c, names_c, medians_c = run_stage_c(
            image_data, noise_map, psf_kernel, feature_mask,
            lens_light_model, samples_a, weights_a, names_a,
            position_likelihood,
        )

    if not (OUT_DIR / "stage_c_model.png").exists():
        passer_c = GaussianPriorPasser(samples_a, weights_a, names_a)
        lkl_c = build_stage_c_likelihood(
            lens_subtracted,
            noise_map,
            psf_kernel,
            feature_mask,
            passer_c,
            position_likelihood,
        )
        try:
            _plot_pix_stage("stage-C", lkl_c, medians_c, names_c,
                            str(OUT_DIR / "stage_c_model.png"))
        except Exception as err:
            print(f"[stage-C] plotting failed (non-fatal): {err}")

    # ---- stage D ---------------------------------------------------- #
    if skip_done and (OUT_DIR / "stage_d.pkl").exists():
        print("[stage-D] loading cached output/stage_d.pkl")
        d = _load_stage("d")
        samples_d = d["samples"]; weights_d = d["weights"]
        names_d = d["param_names"]; medians_d = d["extra"]["medians"]
    else:
        samples_d, weights_d, names_d, medians_d = run_stage_d(
            image_data, noise_map, psf_kernel, feature_mask,
            lens_light_model, samples_c, weights_c, names_c, medians_c,
            position_likelihood,
        )

    if not (OUT_DIR / "stage_d_model.png").exists():
        lambda_reg_fixed = medians_c["lambda_reg"]
        passer_d = GaussianPriorPasser(samples_c, weights_c, names_c)
        lkl_d = build_stage_d_likelihood(
            lens_subtracted,
            noise_map,
            psf_kernel,
            feature_mask,
            passer_d,
            lambda_reg_fixed,
            position_likelihood,
        )
        try:
            _plot_pix_stage("stage-D", lkl_d, medians_d, names_d,
                            str(OUT_DIR / "stage_d_model.png"))
        except Exception as err:
            print(f"[stage-D] plotting failed (non-fatal): {err}")

    print("\n" + "=" * 60)
    print(" Pipeline complete")
    print("=" * 60)
    for k in ("theta_E", "gamma", "e1_mass", "e2_mass",
              "center_x_mass", "center_y_mass", "gamma1", "gamma2"):
        if k in medians_d:
            print(f"    final  {k:15s} = {medians_d[k]:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-done", action="store_true",
                        help="Re-use cached posteriors in output/stage_*.pkl")
    args = parser.parse_args()
    main(skip_done=args.skip_done)
