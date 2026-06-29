"""
Shared four-stage inference pipeline for SLACS lens modeling.

Stage a : SIE + shear + two-MGE lens light + MGE source light (uniform priors)
Stage b : build an arc feature mask from stage-a residuals
Stage l : arc-masked MGE lens light refinement (Gaussian priors from stage a)
Stage m : EPL + shear + pixelized source (Gaussian priors from stage a,
          + uniform prior on gamma, + log-uniform on lambda_reg)

Each stage pickles its posterior samples/weights to
``output/stage_{a,l,m}.pkl`` and is re-runnable via ``--skip-done``.

Usage::

    # In a lens subdirectory (e.g. J0737+3216/)
    python model.py          # full pipeline
    python model.py --skip-done   # resume from cached posteriors
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

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
NSRC = 40
NSRC = 40
DPIX = 0.05
NSUB = 4
NSUB_PIX = 4  # lower oversampling for pixelized stages (memory)
N_GAUSSIANS_SRC = 10
N_GAUSSIANS_LENS = 10
MASK_RADIUS = 2.5
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


# ------------------------------------------------------------------ #
# Stage A -- SIE + shear + MGE lens light + MGE source light
# ------------------------------------------------------------------ #
def build_stage_a_likelihood(image_data, noise_map, psf_kernel, mask=None):
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

    # Two-group MGE lens light (mirrors stage-l configuration)
    # Group 1 geometry
    cx_l1 = ParamU("center_x_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    cy_l1 = ParamU("center_y_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    e1_l1 = ParamU("e1_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_l1 = ParamU("e2_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])

    # Group 2 geometry
    cx_l2 = ParamU("center_x_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    cy_l2 = ParamU("center_y_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    e1_l2 = ParamU("e1_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_l2 = ParamU("e2_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])

    sigma_list_lens = generate_radial_basis_knots(
        dpix=DPIX, n_sigmas=N_GAUSSIANS_LENS,
        log_rmin=-2.0, log_rmax=np.log10(MASK_RADIUS), mode="mge"
    )

    lens_gaussians = []
    for group_id, center_x, center_y, e1, e2 in (
        (1, cx_l1, cy_l1, e1_l1, e2_l1),
        (2, cx_l2, cy_l2, e1_l2, e2_l2),
    ):
        for i, sigma in enumerate(sigma_list_lens):
            gauss = GaussianEllipse(
                sigma=ParamU(f"sigma_lens_{group_id}_{i}", float(sigma)),
                center_x=center_x,
                center_y=center_y,
                e1=e1,
                e2=e2,
                flux=ParamU(f"flux_lens_{group_id}_{i}", 1.0),
            )
            gauss.sigma.to_static(float(sigma))
            gauss.flux.to_static(1.0)
            lens_gaussians.append(gauss)

    for p in (cx_l1, cy_l1, e1_l1, e2_l1, cx_l2, cy_l2, e1_l2, e2_l2):
        p.to_dynamic()

    phys = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=source_gaussians,
        lens_light=lens_gaussians,
    )
    return ImageProbModel(
        image_data=image_data, noise_map=noise_map,
        psf_kernel=psf_kernel, dpix=DPIX, nsub=NSUB,
        phys_model=phys, use_linear=True, solver_type="nnls",
        mask=mask,
    )


def run_stage_a(image_data, noise_map, psf_kernel, circular_mask=None):
    OUT_DIR.mkdir(exist_ok=True)
    print("\n" + "=" * 60)
    print(" Stage A : SIE + shear + two-MGE lens light + MGE source light")
    print("=" * 60)
    t0 = time.time()
    likelihood = build_stage_a_likelihood(image_data, noise_map, psf_kernel, mask=circular_mask)
    samples, weights, names, logz = _run_sampler(
        likelihood, n_live=300, n_eff=2000, tag="stage-A", vectorized=True,
    )
    t1 = time.time()
    _print_summary("stage-A", samples, weights, names)
    print(f"[stage-A] time taken: {t1 - t0:.2f} seconds")

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
            time_taken=t1 - t0,
        ),
    )

    try:
        plot_model_results(
            likelihood, jnp.asarray(q50),
            save_path=str(OUT_DIR / "stage_a_model.png"),
            title="Stage A : MGE lens+source",
            show_critical_lines=True,
            show_caustics=True,
        )
    except Exception as err:
        print(f"[stage-A] plotting failed (non-fatal): {err}")
    return samples, weights, names, medians, lens_image_model


# ------------------------------------------------------------------ #
# Stage B -- build an arc mask from stage-A residuals
# ------------------------------------------------------------------ #
def run_stage_b(image_data, noise_map, lens_light_model, circular_mask=None):
    print("\n" + "=" * 60)
    print(" Stage B : build arc feature mask from stage-A residuals")
    print("=" * 60)
    residual = image_data - lens_light_model
    snr_map = residual / noise_map
    # arc_mask_from returns True for EXCLUDED pixels (library convention)
    arc_mask = arc_mask_from(snr_map, threshold=3.0,
                             ignor_size=25, ext_size=5, close_size=3)
    feature_mask = arc_mask
    if circular_mask is not None:
        feature_mask = feature_mask | circular_mask
    n_in = int((~feature_mask).sum())
    print(f"[stage-B] arc pixels kept = {n_in} / {feature_mask.size}")

    DATA_DIR.mkdir(exist_ok=True)
    fits.writeto(DATA_DIR / "feature_mask.fits",
                 feature_mask.astype(np.uint8), overwrite=True)

    # Quick-look figure
    ny_img, nx_img = image_data.shape
    extent = [-nx_img * DPIX / 2, nx_img * DPIX / 2, -ny_img * DPIX / 2, ny_img * DPIX / 2]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im0 = axes[0].imshow(residual, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("residual = image - stage-A lens light")
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
# Stage L -- arc-masked MGE lens light refinement
# ------------------------------------------------------------------ #
def build_stage_new_likelihood(
    image_data,
    noise_map,
    psf_kernel,
    feature_mask,
    circular_mask=None,
):
    """
    Build a lens-light-only likelihood with two independent MGE groups.

    Each group contains 20 GaussianEllipse components that share the same
    centre and ellipticity within the group, but the two groups are
    independent.  All geometric priors mirror stage-a lens-light settings.

    The arc region (``feature_mask == False``) is down-weighted so the fit
    is driven mostly by clean lens-light pixels while preserving weak
    geometric information from the arc direction.
    """
    # Group 1 geometry
    cx_l1 = ParamU("center_x_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    cy_l1 = ParamU("center_y_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    e1_l1 = ParamU("e1_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_l1 = ParamU("e2_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])

    # Group 2 geometry
    cx_l2 = ParamU("center_x_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    cy_l2 = ParamU("center_y_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    e1_l2 = ParamU("e1_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_l2 = ParamU("e2_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])

    sigma_list_lens = generate_radial_basis_knots(
        dpix=DPIX, n_sigmas=N_GAUSSIANS_LENS,
        log_rmin=-2.0, log_rmax=np.log10(MASK_RADIUS), mode="mge"
    )

    lens_gaussians = []
    for group_id, center_x, center_y, e1, e2 in (
        (1, cx_l1, cy_l1, e1_l1, e2_l1),
        (2, cx_l2, cy_l2, e1_l2, e2_l2),
    ):
        for i, sigma in enumerate(sigma_list_lens):
            gauss = GaussianEllipse(
                sigma=ParamU(f"sigma_lens_{group_id}_{i}", float(sigma)),
                center_x=center_x,
                center_y=center_y,
                e1=e1,
                e2=e2,
                flux=ParamU(f"flux_lens_{group_id}_{i}", 1.0),
            )
            gauss.sigma.to_static(float(sigma))
            gauss.flux.to_static(1.0)
            lens_gaussians.append(gauss)

    for p in (cx_l1, cy_l1, e1_l1, e2_l1, cx_l2, cy_l2, e1_l2, e2_l2):
        p.to_dynamic()

    # Soft mask: downweight arc pixels by inflating their noise instead of
    # hard-excluding them.  This preserves ellipticity information from the
    # arc direction while suppressing contamination.
    noise_map_soft = np.array(noise_map, copy=True)
    noise_map_soft[~feature_mask] *= 10000.0   # True in ~feature_mask = arc

    # Combine with the circular mask if provided
    combined_mask = circular_mask

    phys = PhysicalModel(
        lens_mass=[],
        source_light=[],
        lens_light=lens_gaussians,
    )
    return ImageProbModel(
        image_data=image_data,
        noise_map=noise_map_soft,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB,
        phys_model=phys,
        use_linear=True,
        solver_type="nnls",
        mask=combined_mask,
    )


def run_stage_l(image_data, noise_map, psf_kernel, feature_mask,
                circular_mask=None):
    print("\n" + "=" * 60)
    print(" Stage L : arc-masked MGE lens light refinement")
    print("=" * 60)
    t0 = time.time()
    n_arc    = int((~feature_mask).sum())
    n_nonarc = feature_mask.size - n_arc
    print(f"[stage-L] arc pixels excluded     = {n_arc}    / {feature_mask.size}")
    print(f"[stage-L] lens-light pixels kept  = {n_nonarc} / {feature_mask.size}")

    likelihood = build_stage_new_likelihood(
        image_data,
        noise_map,
        psf_kernel,
        feature_mask,
        circular_mask=circular_mask,
    )
    samples, weights, names, logz = _run_sampler(
        likelihood,
        n_live=200,
        n_eff=800,
        tag="stage-L",
        vectorized=True,
    )
    t1 = time.time()
    _print_summary("stage-L", samples, weights, names)
    print(f"[stage-L] time taken: {t1 - t0:.2f} seconds")

    medians = _posterior_median(samples, weights, names)
    q50 = [medians[n] for n in names]

    # Evaluate lens-light model image at median (needed for stage M).
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
            time_taken=t1 - t0,
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

    return samples, weights, names, medians, lens_image_model


# ------------------------------------------------------------------ #
# Stage M -- merged EPL + shear + pixelized source (replaces C + D)
# ------------------------------------------------------------------ #
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
        prior_type="uniform",
        prior_settings=[1.0, 3.0],
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
def _position_likelihood_from_stage_a(medians_a: dict) -> dict:
    """
    Build a position-likelihood config dict from stage-A posterior medians.

    This follows the pipeline assumption that the lensed image positions are
    not known a priori. Instead, we estimate them by:

    1) taking the stage-A median mass model (SIE + shear),
    2) taking the stage-A median source center (center_x_src, center_y_src),
    3) solving the lens equation to get the corresponding multiple image
       positions in the image plane.

    The returned dict matches ``PixelizedImageProbModel(position_likelihood=...)``:
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
def _plot_pix_stage(tag, likelihood, medians, param_names, save_path):
    """4-panel diagnostic: data | model | norm-residual | source."""
    q50 = [medians[n] for n in param_names]
    image_data = np.asarray(likelihood.image_data)
    noise_map  = np.asarray(likelihood.noise_map)
    mask       = ~np.asarray(likelihood.unmask, dtype=bool)  # True = excluded

    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.array(q50))
        design_matrix, source_bbox = likelihood.sim_obj.design_matrix()
        lam_val = jnp.exp(likelihood.phys_model.source_light[0].log_lambda_reg.value)
        reg_matrix, _ = likelihood._regularization_matrix(source_bbox)
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

    nx = likelihood.phys_model.source_light[0].n
    ny = likelihood.phys_model.source_light[0].n
    src_img = np.array(source_pixels).reshape(ny, nx)

    npix = image_data.shape[0]
    ext_i = [-npix * DPIX / 2, npix * DPIX / 2, -npix * DPIX / 2, npix * DPIX / 2]
    ext_s = [float(source_bbox[0]), float(source_bbox[1]), float(source_bbox[2]), float(source_bbox[3])]
    vmax  = np.nanpercentile(image_data[~mask], 99.5)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, img, title, kw in [
        (axes[0], image_data,  "Data (lens-subtracted)", dict(vmin=0, vmax=vmax, cmap="viridis")),
        (axes[1], model_image, "Model image",            dict(vmin=0, vmax=vmax, cmap="viridis")),
        (axes[2], np.where(mask, np.nan, resid_norm),
                               f"Norm. residual\nχ²/ν={chi2_nu:.3f}",
                               dict(vmin=-3, vmax=3, cmap="RdBu_r")),
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
    overlay_critical_and_caustics(
        image_axes=[axes[0], axes[1], axes[2]],
        source_ax=axes[3],
        lens_mass=likelihood.phys_model,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] diagnostic plot saved to {save_path}")


# ------------------------------------------------------------------ #
# Stage M -- EPL + shear + pixelized source (merged C + D)
# ------------------------------------------------------------------ #
def build_stage_m_likelihood(
    lens_subtracted_image, noise_map, psf_kernel, feature_mask,
    passer: GaussianPriorPasser, position_likelihood, circular_mask=None,
):
    epl, shear = _epl_mass_from_stage_a(passer)
    lam = ParamU(
        "log_lambda_reg",
        1.0,
        prior_type="uniform",
        prior_settings=[-9.210340371976182, 9.210340371976184],
        limits=[-13.815510557964274, 13.815510557964274],
    )
    lam.to_dynamic()

    pix_src = PixelizedSourceModel(n=NSRC,
        log_lambda_reg=lam,
        regularization_type="second-order",
    )
    phys = PhysicalModel(
        lens_mass=[epl, shear],
        source_light=[pix_src],
        lens_light=[],
    )
    combined_mask = feature_mask
    if circular_mask is not None:
        combined_mask = combined_mask | circular_mask
    return PixelizedImageProbModel(
        image_data=lens_subtracted_image,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
    )


def run_stage_m(image_data, noise_map, psf_kernel, feature_mask,
                lens_light_model, samples_a, weights_a, names_a,
                position_likelihood, circular_mask=None):
    print("\n" + "=" * 60)
    print(" Stage M : EPL + shear + pix source (lambda_reg free)")
    print("=" * 60)
    t0 = time.time()
    lens_subtracted = image_data - lens_light_model
    passer = GaussianPriorPasser(samples_a, weights_a, names_a)
    likelihood = build_stage_m_likelihood(
        lens_subtracted,
        noise_map,
        psf_kernel,
        feature_mask,
        passer,
        position_likelihood,
        circular_mask=circular_mask,
    )
    samples, weights, names, logz = _run_sampler(
        likelihood, n_live=300, n_eff=800, tag="stage-M", vectorized=False,
    )
    t1 = time.time()
    _print_summary("stage-M", samples, weights, names)
    print(f"[stage-M] time taken: {t1 - t0:.2f} seconds")
    medians = _posterior_median(samples, weights, names)
    _dump_stage("m", samples, weights, names, logz, extra=dict(medians=medians, time_taken=t1 - t0))
    try:
        _plot_pix_stage("stage-M", likelihood, medians, names,
                        str(OUT_DIR / "stage_m_model.png"))
    except Exception as err:
        print(f"[stage-M] plotting failed (non-fatal): {err}")
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
        lens_light_model = d["extra"]["lens_light_model"]
        time_a = d["extra"].get("time_taken", 0.0)
    else:
        samples_a, weights_a, names_a, medians_a, lens_light_model = run_stage_a(
            image_data, noise_map, psf_kernel, circular_mask=circular_mask,
        )
        time_a = _load_stage("a")["extra"].get("time_taken", 0.0)

    # ---- stage B ---------------------------------------------------- #
    feature_mask = run_stage_b(image_data, noise_map, lens_light_model, circular_mask=circular_mask)

    # ---- stage L ---------------------------------------------------- #
    time_l = 0.0
    if skip_done and (OUT_DIR / "stage_l.pkl").exists():
        print("[stage-L] loading cached output/stage_l.pkl")
        d = _load_stage("l")
        samples_l, weights_l, names_l = d["samples"], d["weights"], d["param_names"]
        medians_l = d["extra"]["medians"]
        lens_light_model = d["extra"]["lens_light_model"]
        time_l = d["extra"].get("time_taken", 0.0)
    else:
        samples_l, weights_l, names_l, medians_l, lens_light_model = run_stage_l(
            image_data, noise_map, psf_kernel, feature_mask,
            circular_mask=circular_mask,
        )
        time_l = _load_stage("l")["extra"].get("time_taken", 0.0)

    lens_subtracted = image_data - lens_light_model
    position_likelihood = _position_likelihood_from_stage_a(medians_a)

    # ---- stage M ---------------------------------------------------- #
    time_m = 0.0
    if skip_done and (OUT_DIR / "stage_m.pkl").exists():
        print("[stage-M] loading cached output/stage_m.pkl")
        d = _load_stage("m")
        samples_m, weights_m, names_m = d["samples"], d["weights"], d["param_names"]
        medians_m = d["extra"]["medians"]
        time_m = d["extra"].get("time_taken", 0.0)
    else:
        samples_m, weights_m, names_m, medians_m = run_stage_m(
            image_data, noise_map, psf_kernel, feature_mask,
            lens_light_model, samples_a, weights_a, names_a,
            position_likelihood, circular_mask=circular_mask,
        )
        time_m = _load_stage("m")["extra"].get("time_taken", 0.0)

    if not (OUT_DIR / "stage_m_model.png").exists():
        passer_m = GaussianPriorPasser(samples_a, weights_a, names_a)
        lkl_m = build_stage_m_likelihood(
            lens_subtracted,
            noise_map,
            psf_kernel,
            feature_mask,
            passer_m,
            position_likelihood,
            circular_mask=circular_mask,
        )
        try:
            _plot_pix_stage("stage-M", lkl_m, medians_m, names_m,
                            str(OUT_DIR / "stage_m_model.png"))
        except Exception as err:
            print(f"[stage-M] plotting failed (non-fatal): {err}")

    print("\n" + "=" * 60)
    print(" Pipeline complete")
    print("=" * 60)
    print(f" Time summary:")
    print(f"    Stage A: {time_a/60:.2f} min")
    print(f"    Stage L: {time_l/60:.2f} min")
    print(f"    Stage M: {time_m/60:.2f} min")
    print(f"    Total:   {(time_a + time_l + time_m)/60:.2f} min\n")

    for k in ("theta_E", "gamma", "e1_mass", "e2_mass",
              "center_x_mass", "center_y_mass", "gamma1", "gamma2"):
        if k in medians_m:
            print(f"    final  {k:15s} = {medians_m[k]:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-done", action="store_true",
                        help="Re-use cached posteriors in output/stage_*.pkl")
    args = parser.parse_args()
    main(skip_done=args.skip_done)
