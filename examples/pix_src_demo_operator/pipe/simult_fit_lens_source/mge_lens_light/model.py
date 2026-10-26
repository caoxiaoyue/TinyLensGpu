"""
Two-stage inference pipeline for the pix_src_pipe demo with MGE light models.

Stage a : SIE + shear + MGE lens light + MGE source light
Stage b : EPL + shear + MGE lens light + pixelized source (joint fit)
          Mass priors inherited from stage-a posterior.
          MGE lens-light non-linear params inherited from stage-a posterior.
          Position likelihood built from stage-a median mass + source center.

Each stage pickles its posterior samples/weights to
``output/stage_{a,b}.pkl`` and is re-runnable via ``--skip-done``.
"""

from __future__ import annotations

import argparse
import os
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
from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import ImageProbModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
from TinyLensGpu.ObservationModel.LensImage.inversion_result import InvalidReconstructionError
from TinyLensGpu.PhysicalModel import (
    PhysicalModel,
    PixelizedSourceModel,
    GaussianEllipse,
    Shear,
)
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE, EPL
from TinyLensGpu.utils import load_lens_data, generate_radial_basis_knots
from TinyLensGpu.utils.misc import arc_mask_from, weighted_quantile
from TinyLensGpu.visualizer import plot_model_results, overlay_critical_and_caustics

from TinyLensGpu.Inference import StagePosterior
from TinyLensGpu.example_support import (
    SamplingConfig, StageStorageConfig, run_sampling, save_stage, load_stage,
    require_valid_reconstruction,
    draw_image_panel,
)

import caskade as ck

# ------------------------------------------------------------------ #
DPIX = 0.05
NSUB = 4
NSUB_PIX = 4  # lower oversampling for pixelized stage (memory)
OUT_DIR = Path("output")
DATA_DIR = Path("data")

N_GAUSSIANS_LENS = 10
N_GAUSSIANS_SRC = 10


# ------------------------------------------------------------------ #
def _run_sampler(likelihood, n_live: int, n_eff: int, tag: str, vectorized: bool = True):
    start_time = time.time()
    stage = run_sampling(
        likelihood,
        SamplingConfig(
            n_live=n_live, n_eff=n_eff, vectorized=vectorized,
            n_batch=n_live if vectorized else None,
            vectorized_chunk_size=None,
        ),
        tag=tag, sampler_factory=Sampler,
    )
    print(f"[{tag}] Sampling finished in {time.time() - start_time:.2f} seconds.")
    return stage


def _dump_stage(tag: str, samples, weights, param_names, log_z, extra=None):
    save_stage(StageStorageConfig(OUT_DIR, include_schema=False),
               tag, samples, weights, param_names, log_z, extra=extra)


def _load_stage(tag: str):
    return load_stage(StageStorageConfig(OUT_DIR, include_schema=False), tag)





def _print_summary(tag: str, samples, weights, param_names):
    print(f"\n[{tag}] Posterior summary:")
    q = np.array([0.16, 0.5, 0.84])
    for i, name in enumerate(param_names):
        qs = weighted_quantile(np.asarray(samples[:, i]), weights, q)
        q16, q50, q84 = float(qs[0]), float(qs[1]), float(qs[2])
        print(f"    {name:25s} = {q50:+.4f} ({q16-q50:+.4f}, {q84-q50:+.4f})")


# ------------------------------------------------------------------ #
# Stage A — SIE + shear + MGE lens light + MGE source light
# ------------------------------------------------------------------ #
def _build_mge_light(
    prefix: str,
    n_gaussians: int,
    log_rmin: float,
    log_rmax: float,
    center_init: tuple[float, float],
    e_init: tuple[float, float],
    center_prior_settings: list,
    e_prior_settings: list,
    center_limits: list,
    e_limits: list,
):
    """Build a list of GaussianEllipse components for MGE light."""
    sigma_list = generate_radial_basis_knots(
        dpix=DPIX, n_sigmas=n_gaussians,
        log_rmin=log_rmin, log_rmax=log_rmax, mode='mge'
    )

    center_x = ParamU(
        f"center_x_{prefix}", center_init[0],
        prior_type="gaussian",
        prior_settings=center_prior_settings,
        limits=center_limits,
    )
    center_y = ParamU(
        f"center_y_{prefix}", center_init[1],
        prior_type="gaussian",
        prior_settings=center_prior_settings,
        limits=center_limits,
    )
    e1 = ParamU(
        f"e1_{prefix}", e_init[0],
        prior_type="gaussian",
        prior_settings=e_prior_settings,
        limits=e_limits,
    )
    e2 = ParamU(
        f"e2_{prefix}", e_init[1],
        prior_type="gaussian",
        prior_settings=e_prior_settings,
        limits=e_limits,
    )

    components = []
    for i, sigma in enumerate(sigma_list):
        gauss = GaussianEllipse(
            sigma=ParamU(f"sigma_{prefix}_{i}", float(sigma)),
            center_x=center_x,
            center_y=center_y,
            e1=e1,
            e2=e2,
            flux=ParamU(f"flux_{prefix}_{i}", 1.0),
        )
        gauss.sigma.to_static(float(sigma))
        gauss.flux.to_static(1.0)
        components.append(gauss)

    # Mark non-linear geometric params as dynamic
    for p in (center_x, center_y, e1, e2):
        p.to_dynamic()

    return components, center_x, center_y, e1, e2


def build_stage_a_likelihood(image_data, noise_map, psf_kernel):
    # --- mass: SIE + shear -----------------------------------------
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

    # --- source light: MGE -----------------------------------------
    source_light, _, _, _, _ = _build_mge_light(
        prefix="src",
        n_gaussians=N_GAUSSIANS_SRC,
        log_rmin=-2,
        log_rmax=np.log10(1.0),
        center_init=(0.0, 0.0),
        e_init=(0.0, 0.0),
        center_prior_settings=[0.0, 0.5],
        e_prior_settings=[0.0, 0.3],
        center_limits=[-3.0, 3.0],
        e_limits=[-1.0, 1.0],
    )

    # --- lens light: MGE -------------------------------------------
    lens_light, _, _, _, _ = _build_mge_light(
        prefix="lens",
        n_gaussians=N_GAUSSIANS_LENS,
        log_rmin=-2.0,
        log_rmax=np.log10(3.0),
        center_init=(0.0, 0.0),
        e_init=(0.0, 0.0),
        center_prior_settings=[0.0, 0.1],
        e_prior_settings=[0.0, 0.3],
        center_limits=[-1.0, 1.0],
        e_limits=[-1.0, 1.0],
    )

    phys = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=source_light,
        lens_light=lens_light,
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
    stage = _run_sampler(
        likelihood, n_live=200, n_eff=800, tag="stage-A", vectorized=True,
    )
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    _print_summary("stage-A", samples, weights, names)

    medians = stage.medians()
    q50 = [medians[n] for n in names]

    # Evaluate lens-light model image at median (needed for diagnostics).
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
            show_critical_lines=True,
            show_caustics=True,
        )
    except Exception as err:
        print(f"[stage-A] plotting failed (non-fatal): {err}")
    return samples, weights, names, medians, lens_image_model


# ------------------------------------------------------------------ #
# Position likelihood helper (reused from original pipeline)
# ------------------------------------------------------------------ #
def _position_likelihood_from_stage_a(medians_a: dict) -> dict:
    """
    Build a position-likelihood config dict from stage-A posterior medians.

    Uses the stage-A median mass model (SIE + shear) and source center
    to solve for the corresponding multiple image positions in the image
    plane.  These positions are then used as a fixed constraint in stage-b.
    """
    required = (
        "theta_E", "e1_mass", "e2_mass",
        "center_x_mass", "center_y_mass",
        "gamma1", "gamma2",
        "center_x_src", "center_y_src",
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
            "This pipeline assumes a strong-lensing configuration."
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
# Arc mask from stage-A residuals
# ------------------------------------------------------------------ #
def _build_arc_mask(image_data, noise_map, lens_light_model):
    """Build arc feature mask from stage-A residuals (True = excluded)."""
    residual = image_data - lens_light_model
    snr_map = residual / noise_map
    arc_mask = arc_mask_from(snr_map, threshold=2.0,
                             ignor_size=25, ext_size=2, close_size=3)
    n_in = int((~arc_mask).sum())
    print(f"[arc-mask] arc pixels kept = {n_in} / {arc_mask.size}")

    DATA_DIR.mkdir(exist_ok=True)
    fits.writeto(DATA_DIR / "feature_mask.fits",
                 arc_mask.astype(np.uint8), overwrite=True)

    # Quick-look figure
    ny_img, nx_img = image_data.shape
    extent = [-nx_img * DPIX / 2, nx_img * DPIX / 2, -ny_img * DPIX / 2, ny_img * DPIX / 2]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im0 = draw_image_panel(
        axes[0],
        residual,
        extent=extent,
        title='residual = image - stage-A lens light',
        cmap='viridis',
        xlabel=None,
    )
    im1 = draw_image_panel(
        axes[1],
        snr_map,
        extent=extent,
        title='S/N map + arc mask boundary',
        cmap='viridis',
        vmin=-3,
        vmax=np.nanpercentile(snr_map, 99.5),
        xlabel=None,
    )
    axes[1].contour(~arc_mask, levels=[0.5], origin="lower", extent=extent,
                    colors="red", linewidths=1.5)
    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUT_DIR / "stage_arc_mask.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[arc-mask] diagnostic plot saved to {OUT_DIR / 'stage_arc_mask.png'}")

    return arc_mask


# ------------------------------------------------------------------ #
# Stage B — joint fit: EPL + shear + MGE lens light + pixelized source
# ------------------------------------------------------------------ #
def _epl_mass_from_stage_a(passer: StagePosterior):
    """EPL (+ shear) with Gaussian priors inherited from stage-A posterior."""
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
    gamma = ParamU(
        "gamma", 2.0,
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


def _mge_lens_light_from_stage_a(passer: StagePosterior):
    """MGE lens light with non-linear geometric params inherited from stage-A."""
    sigma_list = generate_radial_basis_knots(
        dpix=DPIX, n_sigmas=N_GAUSSIANS_LENS,
        log_rmin=-2.0, log_rmax=np.log10(3.0), mode='mge'
    )

    center_x = passer.gaussian(
        "center_x_lens", model="Gaussian", attr="center_x", limits=[-1.0, 1.0],
    )
    center_y = passer.gaussian(
        "center_y_lens", model="Gaussian", attr="center_y", limits=[-1.0, 1.0],
    )
    e1 = passer.gaussian(
        "e1_lens", model="Gaussian", attr="e1", limits=[-1.0, 1.0],
    )
    e2 = passer.gaussian(
        "e2_lens", model="Gaussian", attr="e2", limits=[-1.0, 1.0],
    )

    lens_light = []
    for i, sigma in enumerate(sigma_list):
        gauss = GaussianEllipse(
            sigma=ParamU(f"sigma_lens_{i}", float(sigma)),
            center_x=center_x,
            center_y=center_y,
            e1=e1,
            e2=e2,
            flux=ParamU(f"flux_lens_{i}", 1.0),
        )
        gauss.sigma.to_static(float(sigma))
        gauss.flux.to_static(1.0)
        lens_light.append(gauss)

    for p in (center_x, center_y, e1, e2):
        p.to_dynamic()

    return lens_light


def build_stage_b_likelihood(
    image_data,
    noise_map,
    psf_kernel,
    passer: StagePosterior,
    position_likelihood,
    feature_mask,
):
    epl, shear = _epl_mass_from_stage_a(passer)
    lens_light = _mge_lens_light_from_stage_a(passer)

    lam = ParamU(
        "log_lambda_reg", 0.0,
        prior_type="uniform",
        prior_settings=[-6.907755278982137, 6.907755278982137],
        limits=[-13.815510557964274, 13.815510557964274],
    )
    lam.to_dynamic()

    pix_src = PixelizedSourceModel(n=40,
        log_lambda_reg=lam,
        regularization_type="first-order",
    )
    phys = PhysicalModel(
        lens_mass=[epl, shear],
        source_light=[pix_src],
        lens_light=lens_light,
    )
    return PixelizedImageProbModelOperator(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=feature_mask,
        position_likelihood=position_likelihood,
    )


def _plot_stage_b(tag, likelihood, medians, param_names, save_path, positions=None):
    """7-panel diagnostic: data | model | resid | source | lensed src | lens light | mask."""
    # Keep the legacy positions argument callable; the reconstruction's public
    # diagnostics are now the source of the plotted constraint coordinates.
    q50 = [medians[n] for n in param_names]
    theta = jnp.array(q50)
    result, diagnostics = likelihood.evaluate_with_diagnostics(
        theta, include_components=True, include_effective_dof=True,
    )
    require_valid_reconstruction(result, diagnostics, context=f"plotting {tag}")
    image_data = np.asarray(likelihood.image_data)
    mask = ~np.asarray(likelihood.unmask, dtype=bool)  # True = excluded

    xmin, xmax, ymin, ymax = result.source_bbox
    src_1d = np.asarray(result.source_pixels)
    ll_amps = result.lens_amplitudes
    npix = image_data.shape[0]
    model_image = np.asarray(result.model_image)
    lensed_src_image = np.asarray(diagnostics.source_image)
    lens_image = np.asarray(diagnostics.lens_light_image)
    resid_norm = np.asarray(diagnostics.normalized_residuals)
    chi2 = diagnostics.chi2
    if diagnostics.reduced_chi2 is None:
        statistic_label = (
            "Effective parameters, residual DOF,\n"
            "reduced chi^2 unavailable"
        )
        print(f"[{tag}] {diagnostics.effective_dof_reason}")
    else:
        statistic_label = f"chi^2/ν = {diagnostics.reduced_chi2:.3f}"

    n = likelihood.phys_model.source_light[0].n
    src_img = src_1d.reshape(n, n)

    ext_i = [-npix * DPIX / 2, npix * DPIX / 2, -npix * DPIX / 2, npix * DPIX / 2]
    ext_s = [float(xmin), float(xmax), float(ymin), float(ymax)]
    vmax = np.nanpercentile(image_data[~mask], 99.5)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Row 0
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
        title='Joint model',
        cmap='viridis',
        vmin=0,
        vmax=vmax,
    )

    resid_display = np.where(mask, np.nan, resid_norm)
    im2 = draw_image_panel(
        axes[0, 2],
        resid_display,
        extent=ext_i,
        title=f'Norm. residual\nchi^2 = {chi2:.3f}\n{statistic_label}',
        cmap='RdBu_r',
        vmin=-3,
        vmax=3,
    )

    im3 = draw_image_panel(
        axes[0, 3],
        src_img,
        extent=ext_s,
        title=f'Source reconstruction\n(lambda={diagnostics.lambda_reg:.2e})',
        cmap='viridis',
        ylabel='arcsec',
    )

    # Row 1
    im4 = draw_image_panel(
        axes[1, 0],
        lensed_src_image,
        extent=ext_i,
        title='Lensed source only',
        cmap='viridis',
        vmin=0,
        vmax=vmax,
        ylabel='arcsec',
    )

    total_ll_flux = float(np.sum(ll_amps))
    im5 = draw_image_panel(
        axes[1, 1],
        lens_image,
        extent=ext_i,
        title=f'Lens light only (MGE, N={N_GAUSSIANS_LENS})\n(total flux={total_ll_flux:.3f})',
        cmap='viridis',
        vmin=0,
        vmax=vmax,
    )

    im6 = draw_image_panel(
        axes[1, 2],
        image_data - lens_image,
        extent=ext_i,
        title='Data - lens light',
        cmap='viridis',
        vmin=0,
        vmax=vmax,
    )

    # Mask overlay
    mask_overlay = np.zeros((*mask.shape, 3))
    mask_overlay[mask] = [0.5, 0.5, 0.5]
    im7 = axes[1, 3].imshow(mask_overlay, origin="lower", extent=ext_i)
    axes[1, 3].set_title("Masks (grey=masked)", fontsize=11)
    axes[1, 3].set_xlabel("arcsec")

    if diagnostics.position is not None:
        pos = np.asarray(diagnostics.position.image_positions)
        for ax in (axes[0, 0], axes[0, 1]):
            ax.scatter(pos[:, 0], pos[:, 1], s=40, fc='none', ec='red',
                       linewidth=1.5, marker='o')

    lbl = "  ".join(f"{n}={medians[n]:+.4f}" for n in
                    ("theta_E", "gamma", "e1_mass", "e2_mass") if n in medians)
    plt.suptitle(f"[{tag}]  {lbl}", fontsize=10)
    with ck.ActiveContext(likelihood):
        likelihood.fill_params(theta)
        overlay_critical_and_caustics(
            image_axes=[axes[0, 0], axes[0, 1], axes[1, 0]],
            source_ax=axes[0, 3],
            lens_mass=likelihood.phys_model,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] diagnostic plot saved to {save_path}")


def run_stage_b(image_data, noise_map, psf_kernel,
                samples_a, weights_a, names_a, position_likelihood, feature_mask):
    print("\n" + "=" * 60)
    print(" Stage B : EPL + shear + MGE lens light + pix source (joint fit)")
    print("=" * 60)
    passer = StagePosterior(samples_a, weights_a, names_a)
    likelihood = build_stage_b_likelihood(
        image_data, noise_map, psf_kernel, passer, position_likelihood, feature_mask,
    )
    stage = _run_sampler(
        likelihood, n_live=300, n_eff=800, tag="stage-B", vectorized=False,
    )
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    _print_summary("stage-B", samples, weights, names)
    medians = stage.medians()
    _dump_stage("b", samples, weights, names, logz, extra=dict(medians=medians))
    try:
        _plot_stage_b("stage-B", likelihood, medians, names,
                      str(OUT_DIR / "stage_b_model.png"))
    except InvalidReconstructionError:
        raise
    except Exception as err:
        print(f"[stage-B] plotting failed (non-fatal): {err}")
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

    # ---- position likelihood ---------------------------------------- #
    position_likelihood = _position_likelihood_from_stage_a(medians_a)

    # ---- build arc mask --------------------------------------------- #
    feature_mask = _build_arc_mask(image_data, noise_map, lens_light_model)

    # ---- stage B ---------------------------------------------------- #
    if skip_done and (OUT_DIR / "stage_b.pkl").exists():
        print("[stage-B] loading cached output/stage_b.pkl")
        d = _load_stage("b")
        samples_b, weights_b, names_b = d["samples"], d["weights"], d["param_names"]
        medians_b = d["extra"]["medians"]
    else:
        samples_b, weights_b, names_b, medians_b = run_stage_b(
            image_data, noise_map, psf_kernel,
            samples_a, weights_a, names_a, position_likelihood, feature_mask,
        )

    print("\n" + "=" * 60)
    print(" Pipeline complete")
    print("=" * 60)
    for k in ("theta_E", "gamma", "e1_mass", "e2_mass",
              "center_x_mass", "center_y_mass", "gamma1", "gamma2"):
        if k in medians_b:
            print(f"    final  {k:15s} = {medians_b[k]:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-done", action="store_true",
                        help="Re-use cached posteriors in output/stage_*.pkl")
    args = parser.parse_args()
    main(skip_done=args.skip_done)
