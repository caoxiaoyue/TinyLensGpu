"""Fresh J0946 macro fit: A, B, M0, M1, M2 and M3, with no subhalo stages.

Derived from the original subhalo_0946/model.py, preserving its scientific
configuration (including the macro operator likelihood). Potential correction
is a separate Dense Float32 workflow in pt.py. Raw inputs are copied locally;
no previous fit, arc mask or posterior is imported.
"""

from __future__ import annotations

# ruff: noqa: E402  -- configure JAX and the local checkout before importing it.

import hashlib

import json

import os

import time

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

os.environ.setdefault("OMP_NUM_THREADS", "1")

os.environ.setdefault("MKL_NUM_THREADS", "1")

os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import jax

jax.config.update("jax_default_matmul_precision", "float32")

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

from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)

from TinyLensGpu.PhysicalModel import (
    PhysicalModel,
    PixelizedSourceModel,
    Shear,
)

from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import GaussianEllipse

from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import (
    EPL,
    SIE,
)

from TinyLensGpu.example_support import (
    StageStorageConfig,
    adapt_image_kwargs,
    export_source_package,
    load_stage,
    pixelized_objective_provenance,
    resolve_source_interpolation,
    save_stage,
    source_interpolation_metadata,
    stage_from_payload as _stage_from_payload,
)

from TinyLensGpu.example_support.pixelized_stage_cache import (
    load_pixelized_checkpoint,
    matching_pixelized_checkpoint,
    pixelized_stage_identity,
    plot_cached_pixelized_stage,
    recover_pixelized_producer,
)

from TinyLensGpu.utils import generate_radial_basis_knots

from TinyLensGpu.utils import load_lens_data

from TinyLensGpu.utils.misc import arc_mask_from, weighted_quantile

from TinyLensGpu.visualizer import plot_model_results, overlay_critical_and_caustics

from TinyLensGpu.Inference import StagePosterior

from workspace._j0946_macro_provenance import (
    require_current_handoff_schema,
    require_sampler_objective,
    resolve_run_source_interpolation,
    restore_adaptive_stage,
    source_settings_match,
    write_macro_handoff,
)

import caskade as ck

NSRC = 100

DPIX = 0.05

NSUB = 4

NSUB_PIX = 4

N_GAUSSIANS_SRC = 20

N_GAUSSIANS_LENS = 20

MASK_RADIUS = 2.5

ADAPTIVE_REG_RHO = 2.0

ADAPTIVE_REG_RHO_PRIOR_MAX = 8.0

PIXEL_REGULARIZATION_TYPE = "first-order"

SOURCE_BBOX_PADDING = 0.0

SOURCE_BBOX_HALF_WIDTH = 0.5  # arcsec about the stage-A source center

SOLVER_TYPE = "pcg"
EVIDENCE_LOGDET = "overlap"
EVIDENCE_BLOCK_SIZE = 10

VECTORIZED_CHUNK_SIZE = 8  # Bound peak GPU memory during batched FFTs/solves.

SCRIPT_DIR = Path(__file__).resolve().parent

OUT_DIR = SCRIPT_DIR / "macro"

DATA_DIR = SCRIPT_DIR / "data"


RESUME = False
SEED = 946
SAMPLER_BATCH_SIZE = 32


SOURCE_INTERPOLATION = None

PIXELIZED_SAMPLING = {"m1": (300, 600), "m2": (250, 600), "m3": (300, 600)}


def _sampler_seed(tag):
    return SEED + {"a": 0, "m1": 1, "m2": 2, "m3": 3}[tag]


def _pixelized_sampling(tag):
    n_live, n_eff = PIXELIZED_SAMPLING[tag]
    return dict(
        method="nautilus",
        n_live=n_live,
        n_eff=n_eff,
        seed=_sampler_seed(tag),
        n_batch=SAMPLER_BATCH_SIZE,
        vectorized=True,
        vectorized_chunk_size=VECTORIZED_CHUNK_SIZE,
    )


def _run_sampler(likelihood, n_live, n_eff, tag, vectorized=True, *, objective=None):
    """Persist both unfinished Nautilus state and sample likelihoods per stage."""
    prior, specs = make_prior_transformation(likelihood)
    names = [spec.name for spec in specs]
    loglike = make_likelihood(
        likelihood,
        vectorized=vectorized,
        vectorized_chunk_size=VECTORIZED_CHUNK_SIZE if vectorized else None,
    )
    stage_id = tag.lower().replace("stage-", "")
    checkpoint = OUT_DIR / f"sampler_{stage_id}.h5"
    mode = getattr(getattr(likelihood, "source_model", None), "interpolation", None)
    if mode is None:
        mode = resolve_source_interpolation(SOURCE_INTERPOLATION)
    settings = dict(
        **source_interpolation_metadata(mode),
        n_live=n_live,
        n_eff=n_eff,
        seed=_sampler_seed(stage_id),
        n_batch=SAMPLER_BATCH_SIZE,
        param_names=names,
        prior_specs=[spec.describe() for spec in specs],
    )
    if objective is not None:
        settings["objective"] = objective["fingerprint"]
    settings_path = OUT_DIR / f"sampler_{stage_id}.json"
    if settings_path.exists():
        if not RESUME or not source_settings_match(
            json.loads(settings_path.read_text()), settings, settings_path,
        ):
            raise RuntimeError(f"Incompatible existing sampler: {settings_path}. Start a fresh run.")
    else:
        if checkpoint.exists():
            raise RuntimeError(f"Sampler checkpoint has no matching settings: {settings_path}")
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    sampler = Sampler(
        prior,
        loglike,
        n_dim=len(names),
        n_live=n_live,
        vectorized=vectorized,
        n_batch=SAMPLER_BATCH_SIZE,
        seed=settings["seed"],
        filepath=checkpoint,
        resume=RESUME and checkpoint.exists(),
    )
    complete = False
    while not complete:
        complete = bool(sampler.run(n_eff=n_eff, timeout=30, verbose=False))
        state = dict(
            stage=stage_id,
            complete=complete,
            n_like=int(sampler.n_like),
            n_eff=float(sampler.n_eff),
            log_z=float(sampler.log_z),
        )
        print(json.dumps(state), flush=True)
        (OUT_DIR / "progress.json").write_text(json.dumps(state, indent=2) + "\n")
    samples, log_weights, log_likelihood = sampler.posterior()
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()
    np.savez_compressed(
        OUT_DIR / f"posterior_{stage_id}.npz",
        samples=samples,
        weights=weights,
        log_likelihood=log_likelihood,
        param_names=names,
    )
    return StagePosterior.from_likelihood(
        likelihood,
        samples,
        weights,
        log_z=float(sampler.log_z),
    )


def _dump_stage(
    tag, samples, weights, param_names, log_z, extra=None, stage=None, *,
    source_interpolation=None,
):
    mode = resolve_source_interpolation(
        SOURCE_INTERPOLATION if source_interpolation is None else source_interpolation
    )
    save_stage(
        StageStorageConfig(OUT_DIR, include_schema=True, source_interpolation=mode),
        tag, samples, weights, param_names, log_z, extra=extra, stage=stage,
    )


def _load_stage(tag):
    return load_stage(
        StageStorageConfig(OUT_DIR, include_schema=True, source_interpolation=SOURCE_INTERPOLATION),
        tag,
    )


def _print_summary(tag: str, samples, weights, param_names):
    print(f"\n[{tag}] Posterior summary:")
    q = np.array([0.16, 0.5, 0.84])
    for i, name in enumerate(param_names):
        qs = weighted_quantile(np.asarray(samples[:, i]), weights, q)
        q16, q50, q84 = float(qs[0]), float(qs[1]), float(qs[2])
        print(f"    {name:25s} = {q50:+.4f} ({q16-q50:+.4f}, {q84-q50:+.4f})")


def _make_circular_mask(image_shape, dpix, radius_arcsec=3.5):
    """Return a boolean mask where True = pixels outside ``radius_arcsec``."""
    ny, nx = image_shape
    y = (np.arange(ny) - (ny - 1) / 2) * dpix
    x = (np.arange(nx) - (nx - 1) / 2) * dpix
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return (xx**2 + yy**2) > radius_arcsec**2


def _stage_a_source_bbox(stage_a: StagePosterior):
    """Fixed square source grid centered on the stage-A source posterior."""
    cx = stage_a.median("center_x_src")
    cy = stage_a.median("center_y_src")
    half_width = SOURCE_BBOX_HALF_WIDTH
    return (cx - half_width, cx + half_width, cy - half_width, cy + half_width)


def _fixed_source_kwargs(source_package):
    return adapt_image_kwargs(
        source_package, expected_n=NSRC, source_interpolation=SOURCE_INTERPOLATION,
    )


def _source_param_value(value):
    """Return a scalar value from a static scalar or ParamU-like object."""
    return value.value if hasattr(value, "value") else value


def _reg_hyperparams_from_payload(stage_payload, tag):
    medians = stage_payload.get("extra", {}).get("medians")
    if not medians:
        raise KeyError(f"stage-{tag.upper()} payload missing posterior medians.")
    required = ("log_lambda_reg", "adaptive_reg_rho")
    missing = [name for name in required if name not in medians]
    if missing:
        raise KeyError(
            f"stage-{tag.upper()} medians missing required keys: " + ", ".join(missing)
        )
    return {name: float(medians[name]) for name in required}


def _reg_hyperparams_from_m2_payload(stage_payload):
    return _reg_hyperparams_from_payload(stage_payload, "m2")


def _format_reg_hyperparams(reg_hyperparams):
    return (
        f"lambda={float(jnp.exp(reg_hyperparams['log_lambda_reg'])):.4e}, "
        f"rho={reg_hyperparams['adaptive_reg_rho']:.4f}"
    )


def _valid_log_evidence(values) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float64)
    return np.isfinite(values_np) & (values_np > -1.0e9)


def _solve_pixel_source_for_package(likelihood, medians, param_names):
    """Export the M1 source and its PSF-convolved reference arc."""
    return export_source_package(
        likelihood, medians, param_names, n=NSRC,
        context="exporting the M1 reference", record_adapt_image=True,
    )


def build_stage_a_likelihood(image_data, noise_map, psf_kernel, mask=None):
    sie = SIE(
        theta_E=ParamU(
            "theta_E",
            1.5,
            prior_type="uniform",
            prior_settings=[0.5, 3.0],
            limits=[0.0, 5.0],
        ),
        e1=ParamU(
            "e1_mass",
            0.0,
            prior_type="gaussian",
            prior_settings=[0.0, 0.3],
            limits=[-1.0, 1.0],
        ),
        e2=ParamU(
            "e2_mass",
            0.0,
            prior_type="gaussian",
            prior_settings=[0.0, 0.3],
            limits=[-1.0, 1.0],
        ),
        center_x=ParamU(
            "center_x_mass",
            0.0,
            prior_type="gaussian",
            prior_settings=[0.0, 0.1],
            limits=[-1.0, 1.0],
        ),
        center_y=ParamU(
            "center_y_mass",
            0.0,
            prior_type="gaussian",
            prior_settings=[0.0, 0.1],
            limits=[-1.0, 1.0],
        ),
    )
    shear = Shear(
        gamma1=ParamU(
            "gamma1",
            0.0,
            prior_type="uniform",
            prior_settings=[-0.2, 0.2],
            limits=[-0.5, 0.5],
        ),
        gamma2=ParamU(
            "gamma2",
            0.0,
            prior_type="uniform",
            prior_settings=[-0.2, 0.2],
            limits=[-0.5, 0.5],
        ),
    )
    for p in (
        sie.theta_E,
        sie.e1,
        sie.e2,
        sie.center_x,
        sie.center_y,
        shear.gamma1,
        shear.gamma2,
    ):
        p.to_dynamic()

    # MGE source
    cx_s = ParamU(
        "center_x_src",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.5],
        limits=[-3.0, 3.0],
    )
    cy_s = ParamU(
        "center_y_src",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.5],
        limits=[-3.0, 3.0],
    )
    e1_s = ParamU(
        "e1_src",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.3],
        limits=[-1.0, 1.0],
    )
    e2_s = ParamU(
        "e2_src",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.3],
        limits=[-1.0, 1.0],
    )
    sigma_list_src = generate_radial_basis_knots(
        dpix=DPIX,
        n_sigmas=N_GAUSSIANS_SRC,
        log_rmin=-2.0,
        log_rmax=np.log10(1.0),
        mode="mge",
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
    cx_s.to_dynamic()
    cy_s.to_dynamic()
    e1_s.to_dynamic()
    e2_s.to_dynamic()

    # Two-group MGE lens light
    cx_l1 = ParamU(
        "center_x_lens_1",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.1],
        limits=[-1.0, 1.0],
    )
    cy_l1 = ParamU(
        "center_y_lens_1",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.1],
        limits=[-1.0, 1.0],
    )
    e1_l1 = ParamU(
        "e1_lens_1",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.3],
        limits=[-1.0, 1.0],
    )
    e2_l1 = ParamU(
        "e2_lens_1",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.3],
        limits=[-1.0, 1.0],
    )

    cx_l2 = ParamU(
        "center_x_lens_2",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.1],
        limits=[-1.0, 1.0],
    )
    cy_l2 = ParamU(
        "center_y_lens_2",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.1],
        limits=[-1.0, 1.0],
    )
    e1_l2 = ParamU(
        "e1_lens_2",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.3],
        limits=[-1.0, 1.0],
    )
    e2_l2 = ParamU(
        "e2_lens_2",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, 0.3],
        limits=[-1.0, 1.0],
    )

    sigma_list_lens = generate_radial_basis_knots(
        dpix=DPIX,
        n_sigmas=N_GAUSSIANS_LENS,
        log_rmin=-2.0,
        log_rmax=np.log10(MASK_RADIUS),
        mode="mge",
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
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB,
        phys_model=phys,
        use_linear=True,
        solver_type="nnls",
        mask=mask,
    )


def run_stage_a(image_data, noise_map, psf_kernel, circular_mask=None):
    OUT_DIR.mkdir(exist_ok=True)
    print("\n" + "=" * 60)
    print(" Stage A : SIE + shear + two-MGE lens light + MGE source light")
    print("=" * 60)
    t0 = time.time()
    likelihood = build_stage_a_likelihood(
        image_data, noise_map, psf_kernel, mask=circular_mask
    )
    stage = _run_sampler(
        likelihood,
        n_live=300,
        n_eff=2000,
        tag="stage-A",
        vectorized=True,
    )
    t1 = time.time()
    samples, weights, names, logz = (
        stage.samples,
        stage.weights,
        stage.param_names,
        stage.log_z,
    )
    _print_summary("stage-A", samples, weights, names)
    print(f"[stage-A] time taken: {t1 - t0:.2f} seconds")

    medians = stage.medians()
    q50 = [medians[n] for n in names]

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
        "a",
        samples,
        weights,
        names,
        logz,
        extra=dict(
            medians=medians,
            lens_light_model=lens_image_model,
            time_taken=t1 - t0,
        ),
        stage=stage,
    )

    try:
        plot_model_results(
            likelihood,
            jnp.asarray(q50),
            save_path=str(OUT_DIR / "stage_a_model.png"),
            title="Stage A : MGE lens+source",
            show_critical_lines=True,
            show_caustics=True,
        )
    except Exception as err:
        print(f"[stage-A] plotting failed (non-fatal): {err}")
    return stage, medians, lens_image_model


def run_stage_b(image_data, noise_map, lens_light_model, circular_mask=None):
    print("\n" + "=" * 60)
    print(" Stage B : build arc feature mask from stage-A residuals")
    print("=" * 60)
    residual = image_data - lens_light_model
    snr_map = residual / noise_map
    arc_mask = arc_mask_from(
        snr_map, threshold=3.0, ignor_size=25, ext_size=5, close_size=3
    )
    feature_mask = arc_mask
    if circular_mask is not None:
        feature_mask = feature_mask | circular_mask
    n_in = int((~feature_mask).sum())
    print(f"[stage-B] arc pixels kept = {n_in} / {feature_mask.size}")

    OUT_DIR.mkdir(exist_ok=True)
    fits.writeto(
        OUT_DIR / "feature_mask.fits", feature_mask.astype(np.uint8), overwrite=True
    )

    ny_img, nx_img = image_data.shape
    extent = [
        -nx_img * DPIX / 2,
        nx_img * DPIX / 2,
        -ny_img * DPIX / 2,
        ny_img * DPIX / 2,
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im0 = axes[0].imshow(residual, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("residual = image - stage-A lens light")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(
        snr_map,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=-3,
        vmax=np.nanpercentile(snr_map, 99.5),
    )
    axes[1].set_title("S/N map + arc mask boundary")
    axes[1].contour(
        ~feature_mask,
        levels=[0.5],
        origin="lower",
        extent=extent,
        colors="red",
        linewidths=1.5,
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "stage_b_mask.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return feature_mask


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
        position_sigma=[0.01],
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
    print(
        f"[pos-like] solved {img_positions.shape[0]} lensed image positions from stage-A medians"
    )
    for p in img_positions:
        print(f"    ({p[0]:+.4f}, {p[1]:+.4f})")
    return dict(
        positions=img_positions.tolist(),
        threshold_arcsec=float(threshold_arcsec),
        min_log_like=float(min_log_like),
    )


def _sie_mass_from_stage_a(stage_a: StagePosterior):
    """SIE (+ shear) with all parameters FIXED at stage-A posterior medians."""
    sie = SIE(
        theta_E=stage_a.fixed("theta_E"),
        e1=stage_a.fixed("e1_mass"),
        e2=stage_a.fixed("e2_mass"),
        center_x=stage_a.fixed("center_x_mass"),
        center_y=stage_a.fixed("center_y_mass"),
    )

    shear = Shear(
        gamma1=stage_a.fixed("gamma1"),
        gamma2=stage_a.fixed("gamma2"),
    )
    return sie, shear


def _sie_mass_from_stage(stage: StagePosterior):
    """SIE (+ shear) with Gaussian priors inherited from a stage posterior."""
    sie = SIE(
        theta_E=stage.gaussian(
            "theta_E", model="SIE", attr="theta_E", limits=[0.0, 5.0]
        ),
        e1=stage.gaussian("e1_mass", model="SIE", attr="e1", limits=[-1.0, 1.0]),
        e2=stage.gaussian("e2_mass", model="SIE", attr="e2", limits=[-1.0, 1.0]),
        center_x=stage.gaussian(
            "center_x_mass", model="SIE", attr="center_x", limits=[-1.0, 1.0]
        ),
        center_y=stage.gaussian(
            "center_y_mass", model="SIE", attr="center_y", limits=[-1.0, 1.0]
        ),
    )
    shear = Shear(
        gamma1=stage.gaussian(
            "gamma1", model="Shear", attr="gamma1", limits=[-0.5, 0.5]
        ),
        gamma2=stage.gaussian(
            "gamma2", model="Shear", attr="gamma2", limits=[-0.5, 0.5]
        ),
    )
    return sie, shear


def _epl_mass_from_stage(stage: StagePosterior):
    """EPL (+ shear) with Gaussian priors inherited from a stage posterior."""
    theta_E = stage.gaussian(
        "theta_E",
        model="EPL",
        attr="theta_E",
        limits=[0.0, 5.0],
    )
    e1m = stage.gaussian(
        "e1_mass",
        model="EPL",
        attr="e1",
        limits=[-1.0, 1.0],
    )
    e2m = stage.gaussian(
        "e2_mass",
        model="EPL",
        attr="e2",
        limits=[-1.0, 1.0],
    )
    cx = stage.gaussian(
        "center_x_mass",
        model="EPL",
        attr="center_x",
        limits=[-1.0, 1.0],
    )
    cy = stage.gaussian(
        "center_y_mass",
        model="EPL",
        attr="center_y",
        limits=[-1.0, 1.0],
    )
    gamma = ParamU(
        "gamma",
        2.0,
        prior_type="uniform",
        prior_settings=[1.0, 3.0],
        limits=[1.0, 3.0],
    )

    epl = EPL(theta_E=theta_E, gamma=gamma, e1=e1m, e2=e2m, center_x=cx, center_y=cy)
    epl.gamma.to_dynamic()

    shear = Shear(
        gamma1=stage.gaussian(
            "gamma1",
            model="Shear",
            attr="gamma1",
            limits=[-0.5, 0.5],
        ),
        gamma2=stage.gaussian(
            "gamma2",
            model="Shear",
            attr="gamma2",
            limits=[-0.5, 0.5],
        ),
    )
    return epl, shear


def _sie_mass_from_medians(medians: dict):
    """SIE (+ shear) with all parameters fixed at posterior medians."""
    required = (
        "theta_E",
        "e1_mass",
        "e2_mass",
        "center_x_mass",
        "center_y_mass",
        "gamma1",
        "gamma2",
    )
    missing = [name for name in required if name not in medians]
    if missing:
        raise KeyError("SIE medians missing required keys: " + ", ".join(missing))

    sie = SIE(
        theta_E=ParamU("theta_E", float(medians["theta_E"])),
        e1=ParamU("e1_mass", float(medians["e1_mass"])),
        e2=ParamU("e2_mass", float(medians["e2_mass"])),
        center_x=ParamU("center_x_mass", float(medians["center_x_mass"])),
        center_y=ParamU("center_y_mass", float(medians["center_y_mass"])),
    )
    for p in (sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y):
        p.to_static()

    shear = Shear(
        gamma1=ParamU("gamma1", float(medians["gamma1"])),
        gamma2=ParamU("gamma2", float(medians["gamma2"])),
    )
    shear.gamma1.to_static()
    shear.gamma2.to_static()
    return sie, shear


def build_stage_m0_likelihood(
    lens_subtracted_image,
    noise_map,
    psf_kernel,
    feature_mask,
    stage_a: StagePosterior,
    position_likelihood,
    circular_mask=None,
):
    """Build M0 on the fixed stage-A-centered grid with SIE+shear fixed."""
    sie, shear = _sie_mass_from_stage_a(stage_a)
    source_bbox = _stage_a_source_bbox(stage_a)

    log_lam = ParamU(
        "log_lambda_reg",
        0.0,
        prior_type="uniform",
        prior_settings=[-13.815510557964274, 13.815510557964274],
        limits=[-13.815510557964274, 13.815510557964274],
    )
    log_lam.to_dynamic()

    pix_src = PixelizedSourceModel(
        n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        interpolation=resolve_source_interpolation(SOURCE_INTERPOLATION),
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
        image_data=lens_subtracted_image,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        solver_type=SOLVER_TYPE,
        evidence_logdet=EVIDENCE_LOGDET,
        evidence_block_size=EVIDENCE_BLOCK_SIZE,
        source_bbox_padding=SOURCE_BBOX_PADDING,
        fixed_source_bbox=source_bbox,
    )


def run_stage_m0(
    image_data,
    noise_map,
    psf_kernel,
    feature_mask,
    lens_light_model,
    stage_a: StagePosterior,
    position_likelihood,
    circular_mask=None,
    *,
    cached=None,
    upstream=None,
):
    """Stage M0 calibrates the uniform source regularization."""
    print("\n" + "=" * 60)
    print(" Stage M0 : fixed SIE + shear + uniform pix source ")
    print("=" * 60)
    t0 = time.time()

    lens_subtracted = image_data - lens_light_model
    likelihood = build_stage_m0_likelihood(
        lens_subtracted,
        noise_map,
        psf_kernel,
        feature_mask,
        stage_a,
        position_likelihood,
        circular_mask=circular_mask,
    )
    n_grid = 200
    objective = pixelized_objective_provenance(
        likelihood, upstream=upstream,
        sampling=dict(
            method="grid", n_grid=n_grid, lambda_range=[1e-8, 1e8],
            refinement_half_width=float(0.5 * np.log(10)), vectorized=True,
            vectorized_chunk_size=VECTORIZED_CHUNK_SIZE,
        ),
    )
    cached = matching_pixelized_checkpoint(cached, objective, "m0")
    if cached is not None:
        log_lambda = cached["extra"]["log_lambda_best"]
        plot_cached_pixelized_stage(
            _plot_pix_stage, OUT_DIR, "m0", likelihood,
            {**stage_a.medians(), "log_lambda_reg": log_lambda}, ["log_lambda_reg"],
        )
        return log_lambda
    loglike_batch = make_likelihood(
        likelihood,
        vectorized=True,
        vectorized_chunk_size=VECTORIZED_CHUNK_SIZE,
    )

    log_lam_min, log_lam_max = jnp.log(1e-8), jnp.log(1e8)
    print(
        f"[stage-M0] Running coarse grid (200 pts, λ in [{float(jnp.exp(log_lam_min)):.1e}, {float(jnp.exp(log_lam_max)):.1e}]) ..."
    )
    log_lam_grid_coarse = jnp.linspace(log_lam_min, log_lam_max, n_grid)
    log_ev_coarse = jnp.asarray(loglike_batch(log_lam_grid_coarse.reshape(-1, 1)))
    valid_coarse = _valid_log_evidence(log_ev_coarse)
    if not np.any(valid_coarse):
        raise RuntimeError(
            "[stage-M0] All log-λ values in coarse grid failed or produced "
            "non-finite log-evidence."
        )
    log_ev_coarse_select = np.where(valid_coarse, np.asarray(log_ev_coarse), -np.inf)
    best_idx_coarse = int(np.argmax(log_ev_coarse_select))
    log_lam_best_coarse = float(log_lam_grid_coarse[best_idx_coarse])
    log_ev_best_coarse = float(log_ev_coarse[best_idx_coarse])
    print(
        f"[stage-M0] Coarse best: λ = {float(jnp.exp(log_lam_best_coarse)):.4e}  (log-ev = {log_ev_best_coarse:.2f})"
    )

    half_width = 0.5 * jnp.log(10)
    log_lam_grid_fine = jnp.linspace(
        log_lam_best_coarse - half_width,
        log_lam_best_coarse + half_width,
        n_grid,
    )
    print(
        f"[stage-M0] Running refinement grid (200 pts, λ in [{float(jnp.exp(log_lam_grid_fine[0])):.4e}, {float(jnp.exp(log_lam_grid_fine[-1])):.4e}]) ..."
    )
    log_ev_fine = jnp.asarray(loglike_batch(log_lam_grid_fine.reshape(-1, 1)))
    valid_fine = _valid_log_evidence(log_ev_fine)
    if not np.any(valid_fine):
        raise RuntimeError(
            "[stage-M0] All log-λ values in refinement grid failed or produced "
            "non-finite log-evidence."
        )
    log_ev_fine_select = np.where(valid_fine, np.asarray(log_ev_fine), -np.inf)
    best_idx_fine = int(np.argmax(log_ev_fine_select))
    log_lam_best = float(log_lam_grid_fine[best_idx_fine])
    log_ev_best = float(log_ev_fine[best_idx_fine])
    print(
        f"[stage-M0] Refined best: λ = {float(jnp.exp(log_lam_best)):.4e}  (log-ev = {log_ev_best:.2f})"
    )

    medians_a = stage_a.medians()
    medians_m0 = {**medians_a, "log_lambda_reg": log_lam_best}

    t1 = time.time()
    print("\n[stage-M0] Grid search summary:")
    print(f"    {'lambda_reg_uniform':25s} = {float(jnp.exp(log_lam_best)):+.4e}")
    print(f"[stage-M0] time taken: {t1 - t0:.2f} seconds")

    _dump_stage(
        "m0",
        None,
        None,
        ["log_lambda_reg"],
        log_ev_best,
        extra=dict(
            source_bbox=_stage_a_source_bbox(stage_a),
            log_lambda_best=log_lam_best,
            evidence_lambda_best=float(jnp.exp(log_lam_best)),
            evidence_log_lambda_best=log_lam_best,
            lambda_grid_coarse=np.asarray(log_lam_grid_coarse, dtype=np.float64),
            log_ev_coarse=np.asarray(log_ev_coarse, dtype=np.float64),
            lambda_grid_fine=np.asarray(log_lam_grid_fine, dtype=np.float64),
            log_ev_fine=np.asarray(log_ev_fine, dtype=np.float64),
            objective=objective,
            time_taken=t1 - t0,
        ),
    )

    try:
        _plot_pix_stage(
            "stage-M0",
            likelihood,
            medians_m0,
            ["log_lambda_reg"],
            str(OUT_DIR / "stage_m0_model.png"),
        )
    except Exception as err:
        print(f"[stage-M0] plotting failed (non-fatal): {err}")

    return log_lam_best


def build_stage_m1_likelihood(
    lens_subtracted_image,
    noise_map,
    psf_kernel,
    feature_mask,
    stage_a: StagePosterior,
    position_likelihood,
    log_lambda_fixed: float,
    circular_mask=None,
    *,
    evidence_logdet=None,
    evidence_block_size=None,
):
    """Build M1: SIE+shear free, source lambda constrained around M0."""
    sie, shear = _sie_mass_from_stage(stage_a)
    source_bbox = _stage_a_source_bbox(stage_a)
    log_lam = ParamU(
        "log_lambda_reg",
        float(log_lambda_fixed),
        prior_type="truncated_gaussian",
        prior_settings=[float(log_lambda_fixed), 0.15],
        limits=[float(log_lambda_fixed) - 0.5, float(log_lambda_fixed) + 0.5],
    )
    log_lam.to_dynamic()

    pix_src = PixelizedSourceModel(
        n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        interpolation=resolve_source_interpolation(SOURCE_INTERPOLATION),
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
        image_data=lens_subtracted_image,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        solver_type=SOLVER_TYPE,
        evidence_logdet=EVIDENCE_LOGDET if evidence_logdet is None else evidence_logdet,
        evidence_block_size=EVIDENCE_BLOCK_SIZE if evidence_block_size is None else evidence_block_size,
        source_bbox_padding=SOURCE_BBOX_PADDING,
        fixed_source_bbox=source_bbox,
    )


def run_stage_m1(
    image_data,
    noise_map,
    psf_kernel,
    feature_mask,
    lens_light_model,
    stage_a: StagePosterior,
    position_likelihood,
    log_lambda_fixed: float,
    circular_mask=None,
    *,
    cached=None,
    upstream=None,
):
    """Stage M1: fit SIE+shear with source lambda constrained around M0."""
    print("\n" + "=" * 60)
    print(" Stage M1 : SIE + shear + non-adaptive pix source (mass fit)")
    print("=" * 60)
    print(
        f"[stage-M1] lambda_reg prior from M0: "
        f"truncated Gaussian centered at {float(jnp.exp(log_lambda_fixed)):.4e} "
        f"(sigma=0.15, limits=[{float(jnp.exp(log_lambda_fixed - 0.5)):.4e}, "
        f"{float(jnp.exp(log_lambda_fixed + 0.5)):.4e}])"
    )
    t0 = time.time()

    lens_subtracted = image_data - lens_light_model
    likelihood = build_stage_m1_likelihood(
        lens_subtracted,
        noise_map,
        psf_kernel,
        feature_mask,
        stage_a,
        position_likelihood,
        log_lambda_fixed,
        circular_mask=circular_mask,
    )
    objective = pixelized_objective_provenance(
        likelihood, sampling=_pixelized_sampling("m1"), upstream=upstream,
    )
    cached = matching_pixelized_checkpoint(cached, objective, "m1", producer=True)
    if cached is not None:
        try:
            stage = _stage_from_payload(cached)
            medians = cached["extra"]["medians"]
            package = recover_pixelized_producer(
                likelihood, cached, "s1", medians, stage.param_names,
            )
        except (KeyError, TypeError, ValueError) as err:
            print(f"[stage-M1] source reference cannot be verified ({err}). Recomputing.")
        else:
            if cached["extra"].get("objective") is not None:
                require_sampler_objective(OUT_DIR, "m1", objective)
            if package is not cached["extra"]["s1"]:
                _dump_stage(
                    "m1", stage.samples, stage.weights, stage.param_names, stage.log_z,
                    extra={**cached["extra"], "s1": package}, stage=stage,
                )

            def build_replot(**score_options):
                return build_stage_m1_likelihood(
                    lens_subtracted, noise_map, psf_kernel, feature_mask, stage_a,
                    position_likelihood, log_lambda_fixed, circular_mask=circular_mask,
                    **score_options,
                )

            plot_cached_pixelized_stage(
                _plot_pix_stage, OUT_DIR, "m1", likelihood, medians, stage.param_names,
                producer=(package, build_replot),
            )
            return stage, medians, package
    stage = _run_sampler(
        likelihood, *PIXELIZED_SAMPLING["m1"], tag="stage-M1", objective=objective,
    )
    t1 = time.time()
    samples, weights, names, logz = (
        stage.samples,
        stage.weights,
        stage.param_names,
        stage.log_z,
    )
    _print_summary("stage-M1", samples, weights, names)
    print(f"[stage-M1] time taken: {t1 - t0:.2f} seconds")
    medians = stage.medians()

    s1_medians = dict(medians)
    log_lambda_m1 = float(medians["log_lambda_reg"])
    s1_package = _solve_pixel_source_for_package(likelihood, s1_medians, names)
    s1_package.update(
        lambda_best=float(jnp.exp(log_lambda_m1)),
        log_lambda_best=log_lambda_m1,
        stage_m1_medians=dict(medians),
    )
    _dump_stage(
        "m1",
        samples,
        weights,
        names,
        logz,
        extra=dict(
            medians=medians,
            m1_mass_model="SIE+Shear",
            lambda_prior_center=float(log_lambda_fixed),
            lambda_prior_sigma=0.15,
            lambda_prior_limits=[
                float(log_lambda_fixed - 0.5),
                float(log_lambda_fixed + 0.5),
            ],
            s1=s1_package,
            objective=objective,
            time_taken=t1 - t0,
        ),
        stage=stage,
    )

    try:
        _plot_pix_stage(
            "stage-M1", likelihood, medians, names, str(OUT_DIR / "stage_m1_model.png")
        )
    except Exception as err:
        print(f"[stage-M1] pix_stage plot failed (non-fatal): {err}")

    return stage, medians, s1_package


def _plot_pix_stage(tag, likelihood, medians, param_names, save_path):
    """5-panel diagnostic: data | model | norm-residual | source | reg-scale."""
    q50 = [medians[n] for n in param_names]
    image_data = np.asarray(likelihood.image_data)
    noise_map = np.asarray(likelihood.noise_map)
    mask = ~np.asarray(likelihood.unmask, dtype=bool)  # True = excluded

    has_pos_penalty = False
    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.array(q50))
        lam_val = jnp.exp(likelihood.phys_model.source_light[0].log_lambda_reg.value)

        rho_val = float(
            np.asarray(
                _source_param_value(
                    likelihood.phys_model.source_light[0].adaptive_reg_rho
                )
            )
        )
        result = likelihood.evaluate()
        source_pixels = result.source_pixels
        xmin, xmax, ymin, ymax = result.source_bbox
        scale = likelihood._get_reg_scale()

        has_pos_penalty = likelihood._has_pos_penalty
        if has_pos_penalty:
            pos_penalty = float(likelihood._position_likelihood_penalty_jax())
            beta_x, beta_y = likelihood.phys_model.deflection(
                likelihood._pos_px, likelihood._pos_py
            )
            dx = beta_x[:, None] - beta_x[None, :]
            dy = beta_y[:, None] - beta_y[None, :]
            dist = jnp.sqrt(dx * dx + dy * dy)
            max_sep = float(jnp.max(dist))

            print(f"[{tag}] Position likelihood penalty: {pos_penalty:.4e}")
            print(
                f"[{tag}] Maximum source-plane separation of marked images: {max_sep:.4e} arcsec"
            )

            beta_x_np = np.array(beta_x)
            beta_y_np = np.array(beta_y)
            pos_px_np = np.array(likelihood._pos_px)
            pos_py_np = np.array(likelihood._pos_py)

    model_image = np.asarray(result.model_image)
    resid_norm = (image_data - model_image) / noise_map
    chi2_map = np.where(mask, np.nan, resid_norm**2)
    chi2 = float(np.nansum(chi2_map))

    n = likelihood.phys_model.source_light[0].n
    src_img = np.array(source_pixels).reshape(n, n)

    scale_img = np.array(scale).reshape(n, n) if scale is not None else np.ones((n, n))
    rho = rho_val

    npix = image_data.shape[0]
    ext_i = [-npix * DPIX / 2, npix * DPIX / 2, -npix * DPIX / 2, npix * DPIX / 2]
    source_dx = (float(xmax) - float(xmin)) / (n - 1)
    source_dy = (float(ymax) - float(ymin)) / (n - 1)
    # The bbox bounds locate source nodes; imshow needs their outer pixel edges.
    ext_s = [
        float(xmin) - source_dx / 2,
        float(xmax) + source_dx / 2,
        float(ymin) - source_dy / 2,
        float(ymax) + source_dy / 2,
    ]
    vmax = np.nanpercentile(image_data[~mask], 99.5)

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))
    rows_unmasked, cols_unmasked = np.where(~mask)
    pad = 3
    row_min = max(rows_unmasked.min() - pad, 0)
    row_max = min(rows_unmasked.max() + pad, npix - 1)
    col_min = max(cols_unmasked.min() - pad, 0)
    col_max = min(cols_unmasked.max() + pad, npix - 1)
    xlim_unmasked = (
        -npix * DPIX / 2 + col_min * DPIX,
        -npix * DPIX / 2 + (col_max + 1) * DPIX,
    )
    ylim_unmasked = (
        -npix * DPIX / 2 + row_min * DPIX,
        -npix * DPIX / 2 + (row_max + 1) * DPIX,
    )
    for ax, img, title, kw in [
        (
            axes[0],
            image_data,
            "Data (lens-subtracted)",
            dict(vmin=0, vmax=vmax, cmap="viridis"),
        ),
        (axes[1], model_image, "Model image", dict(vmin=0, vmax=vmax, cmap="viridis")),
        (
            axes[2],
            np.where(mask, np.nan, resid_norm),
            f"Norm. residual\nχ²={chi2:.1f}",
            dict(vmin=-3, vmax=3, cmap="RdBu_r"),
        ),
    ]:
        im = ax.imshow(img, origin="lower", extent=ext_i, **kw)
        if ax == axes[0] and has_pos_penalty:
            ax.plot(pos_px_np, pos_py_np, "rx", markersize=8, label="Marked pos")
            ax.legend(loc="upper right", fontsize=8)
        if ax == axes[1] and tag.lower() == "stage-m5":
            ax.plot(
                medians["center_x_subhalo"],
                medians["center_y_subhalo"],
                marker="*",
                color="black",
                markersize=14,
                linestyle="none",
                label="Pseudo-Jaffe subhalo",
            )
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("arcsec")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    im3 = axes[3].imshow(src_img, origin="lower", extent=ext_s, cmap="viridis")
    if has_pos_penalty:
        axes[3].plot(beta_x_np, beta_y_np, "rx", markersize=8, label="Traced pos")
        axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_title(f"Source reconstruction\n(λ={float(lam_val):.2e})", fontsize=11)
    axes[3].set_xlabel("arcsec")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    im4 = axes[4].imshow(
        scale_img,
        origin="lower",
        extent=ext_s,
        cmap="plasma",
        vmin=1.0,
        vmax=max(1.0, float(np.nanmax(scale_img))),
    )
    axes[4].set_title(f"Reg precision scale\n(rho={rho:.2f})", fontsize=11)
    axes[4].set_xlabel("arcsec")
    plt.colorbar(
        im4,
        ax=axes[4],
        fraction=0.046,
        pad=0.04,
        label=r"$\lambda_i / \lambda_{\rm global}$",
    )

    axes[0].set_ylabel("arcsec")
    for ax in axes[:3]:
        ax.set_xlim(*xlim_unmasked)
        ax.set_ylim(*ylim_unmasked)

    lbl = "  ".join(
        f"{n}={medians[n]:+.4f}"
        for n in ("theta_E", "gamma", "e1_mass", "e2_mass")
        if n in medians
    )
    plt.suptitle(f"[{tag}]  {lbl}", fontsize=10)
    # ActiveContext restores the model's initial parameters on exit. Evaluate
    # the overlays while the same posterior medians used for the image are live.
    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.array(q50))
        overlay_critical_and_caustics(
            image_axes=[axes[0], axes[1]],
            source_ax=axes[3],
            lens_mass=likelihood.phys_model,
        )
    # Caustic paths may extend beyond the reconstructed pixel source grid.
    axes[3].set_xlim(ext_s[0], ext_s[1])
    axes[3].set_ylim(ext_s[2], ext_s[3])
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] diagnostic plot saved to {save_path}")


def build_stage_m2_likelihood(
    lens_subtracted_image,
    noise_map,
    psf_kernel,
    feature_mask,
    medians_m1: dict,
    position_likelihood,
    s1_package,
    circular_mask=None,
):
    """Build M2: mass fixed from M1, source-reg hyperparameters free."""
    sie, shear = _sie_mass_from_medians(medians_m1)
    log_lam = ParamU(
        "log_lambda_reg",
        float(s1_package["log_lambda_best"]),
        prior_type="uniform",
        prior_settings=[-13.815510557964274, 13.815510557964274],
        limits=[-13.815510557964274, 13.815510557964274],
    )
    log_lam.to_dynamic()
    rho = ParamU(
        "adaptive_reg_rho",
        ADAPTIVE_REG_RHO,
        prior_type="uniform",
        prior_settings=[0.0, ADAPTIVE_REG_RHO_PRIOR_MAX],
        limits=[0.0, ADAPTIVE_REG_RHO_PRIOR_MAX],
    )
    rho.to_dynamic()

    pix_src = PixelizedSourceModel(
        n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        adaptive_reg_rho=rho,
        interpolation=resolve_source_interpolation(
            SOURCE_INTERPOLATION, metadata=s1_package, context="M1 source reference",
        ),
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
        image_data=lens_subtracted_image,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        solver_type=SOLVER_TYPE,
        evidence_logdet=EVIDENCE_LOGDET,
        evidence_block_size=EVIDENCE_BLOCK_SIZE,
        source_bbox_padding=SOURCE_BBOX_PADDING,
        **_fixed_source_kwargs(s1_package),
    )


def run_stage_m2(
    image_data,
    noise_map,
    psf_kernel,
    feature_mask,
    lens_light_model,
    medians_m1,
    position_likelihood,
    s1_package,
    circular_mask=None,
    *,
    cached=None,
    upstream=None,
):
    """Stage M2: fit adaptive source-regularization hyperparameters."""
    print("\n" + "=" * 60)
    print(" Stage M2 : fixed SIE + shear + adaptive pix source (fit λ, rho)")
    print("=" * 60)
    t0 = time.time()
    lens_subtracted = image_data - lens_light_model
    likelihood = build_stage_m2_likelihood(
        lens_subtracted,
        noise_map,
        psf_kernel,
        feature_mask,
        medians_m1,
        position_likelihood,
        s1_package,
        circular_mask=circular_mask,
    )
    objective = pixelized_objective_provenance(
        likelihood, reference=s1_package, sampling=_pixelized_sampling("m2"), upstream=upstream,
    )
    restored = restore_adaptive_stage(OUT_DIR, "m2", cached, objective)
    if restored is not None:
        stage, medians = restored
        plot_cached_pixelized_stage(
            _plot_pix_stage, OUT_DIR, "m2", likelihood, medians, stage.param_names,
        )
        return stage, medians, _reg_hyperparams_from_m2_payload(cached)
    stage = _run_sampler(
        likelihood, *PIXELIZED_SAMPLING["m2"], tag="stage-M2", objective=objective,
    )
    t1 = time.time()
    samples, weights, names, logz = (
        stage.samples,
        stage.weights,
        stage.param_names,
        stage.log_z,
    )
    _print_summary("stage-M2", samples, weights, names)
    print(f"[stage-M2] time taken: {t1 - t0:.2f} seconds")
    medians = stage.medians()
    reg_hyperparams_m2 = {
        "log_lambda_reg": float(medians["log_lambda_reg"]),
        "adaptive_reg_rho": float(medians["adaptive_reg_rho"]),
    }
    print(
        f"[stage-M2] median source reg: {_format_reg_hyperparams(reg_hyperparams_m2)}"
    )
    extra = dict(
        medians=medians,
        m2_mass_model="fixed-SIE+Shear",
        mass_medians_fixed=dict(medians_m1),
        reg_hyperparams=reg_hyperparams_m2,
        objective=objective,
        time_taken=t1 - t0,
    )
    _dump_stage(
        "m2", samples, weights, names, logz, extra=extra, stage=stage,
        source_interpolation=likelihood.source_model.interpolation,
    )
    try:
        _plot_pix_stage(
            "stage-M2", likelihood, medians, names, str(OUT_DIR / "stage_m2_model.png")
        )
    except Exception as err:
        print(f"[stage-M2] plotting failed (non-fatal): {err}")
    return stage, medians, reg_hyperparams_m2


def build_stage_m3_likelihood(
    lens_subtracted_image,
    noise_map,
    psf_kernel,
    feature_mask,
    stage_m1: StagePosterior,
    position_likelihood,
    reg_hyperparams_fixed: dict,
    s1_package,
    circular_mask=None,
):
    """Build M3: EPL+shear free, adaptive source-reg fixed from M2."""
    epl, shear = _epl_mass_from_stage(stage_m1)
    log_lam = ParamU("log_lambda_reg", float(reg_hyperparams_fixed["log_lambda_reg"]))
    log_lam.to_static()
    rho = ParamU("adaptive_reg_rho", float(reg_hyperparams_fixed["adaptive_reg_rho"]))
    rho.to_static()

    pix_src = PixelizedSourceModel(
        n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        adaptive_reg_rho=rho,
        interpolation=resolve_source_interpolation(
            SOURCE_INTERPOLATION, metadata=s1_package, context="M1 source reference",
        ),
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
        image_data=lens_subtracted_image,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        solver_type=SOLVER_TYPE,
        evidence_logdet=EVIDENCE_LOGDET,
        evidence_block_size=EVIDENCE_BLOCK_SIZE,
        source_bbox_padding=SOURCE_BBOX_PADDING,
        **_fixed_source_kwargs(s1_package),
    )


def run_stage_m3(
    image_data,
    noise_map,
    psf_kernel,
    feature_mask,
    lens_light_model,
    stage_m1: StagePosterior,
    position_likelihood,
    reg_hyperparams_fixed: dict,
    s1_package,
    circular_mask=None,
    *,
    cached=None,
    upstream=None,
):
    """Stage M3: final EPL+shear sampling with source-reg fixed from M2."""
    print("\n" + "=" * 60)
    print(" Stage M3 : EPL + shear + adaptive pix source (final mass fit)")
    print("=" * 60)
    print(
        f"[stage-M3] fixed source reg: {_format_reg_hyperparams(reg_hyperparams_fixed)}"
    )
    t0 = time.time()
    lens_subtracted = image_data - lens_light_model
    likelihood = build_stage_m3_likelihood(
        lens_subtracted,
        noise_map,
        psf_kernel,
        feature_mask,
        stage_m1,
        position_likelihood,
        reg_hyperparams_fixed,
        s1_package,
        circular_mask=circular_mask,
    )
    objective = pixelized_objective_provenance(
        likelihood, reference=s1_package, sampling=_pixelized_sampling("m3"), upstream=upstream,
    )
    restored = restore_adaptive_stage(OUT_DIR, "m3", cached, objective)
    if restored is not None:
        stage, medians = restored
        plot_cached_pixelized_stage(
            _plot_pix_stage, OUT_DIR, "m3", likelihood, medians, stage.param_names,
        )
        return stage, medians
    stage = _run_sampler(
        likelihood, *PIXELIZED_SAMPLING["m3"], tag="stage-M3", objective=objective,
    )
    t1 = time.time()
    samples, weights, names, logz = (
        stage.samples,
        stage.weights,
        stage.param_names,
        stage.log_z,
    )
    _print_summary("stage-M3", samples, weights, names)
    print(f"[stage-M3] time taken: {t1 - t0:.2f} seconds")
    medians = stage.medians()
    _dump_stage(
        "m3",
        samples,
        weights,
        names,
        logz,
        extra=dict(
            medians=medians,
            m3_mass_model="EPL+Shear",
            reg_hyperparams_fixed=dict(reg_hyperparams_fixed),
            objective=objective,
            time_taken=t1 - t0,
        ),
        stage=stage,
        source_interpolation=likelihood.source_model.interpolation,
    )
    try:
        _plot_pix_stage(
            "stage-M3", likelihood, medians, names, str(OUT_DIR / "stage_m3_model.png")
        )
    except Exception as err:
        print(f"[stage-M3] plotting failed (non-fatal): {err}")
    return stage, medians


def main(skip_done: bool = False, out_dir: str | None = None, *, source_interpolation=None):
    global OUT_DIR, RESUME, SOURCE_INTERPOLATION
    if out_dir is not None:
        OUT_DIR = SCRIPT_DIR / Path(out_dir).expanduser()
    RESUME = skip_done
    SOURCE_INTERPOLATION = resolve_run_source_interpolation(
        OUT_DIR, source_interpolation, resume=RESUME,
    )
    _run_macro(skip_done)


def _run_macro(skip_done):
    if skip_done:
        require_current_handoff_schema(OUT_DIR)
    image_data, noise_map, psf_kernel, _ = load_lens_data(
        image_path=str(DATA_DIR / "image.fits"),
        noise_path=str(DATA_DIR / "noise.fits"),
        psf_path=str(DATA_DIR / "psf.fits"),
    )

    circular_mask = _make_circular_mask(
        image_data.shape, DPIX, radius_arcsec=MASK_RADIUS
    )
    n_excl = int(circular_mask.sum())
    print(
        f"[circular mask] excluded {n_excl} / {circular_mask.size} pixels (> {MASK_RADIUS} arcsec)"
    )

    # ---- stage A ---------------------------------------------------- #
    time_a = 0.0
    if skip_done and (OUT_DIR / "stage_a.pkl").exists():
        print(f"[stage-A] loading cached {OUT_DIR}/stage_a.pkl")
        d = _load_stage("a")
        stage_a = _stage_from_payload(d)
        medians_a = d["extra"]["medians"]
        lens_light_model = d["extra"]["lens_light_model"]
        time_a = d["extra"].get("time_taken", 0.0)
    else:
        stage_a, medians_a, lens_light_model = run_stage_a(
            image_data,
            noise_map,
            psf_kernel,
            circular_mask=circular_mask,
        )
        time_a = _load_stage("a")["extra"].get("time_taken", 0.0)

    # ---- stage B ---------------------------------------------------- #
    feature_mask = run_stage_b(
        image_data, noise_map, lens_light_model, circular_mask=circular_mask
    )

    lens_subtracted = image_data - lens_light_model
    position_likelihood = _position_likelihood_from_stage_a(medians_a)
    source_bbox = _stage_a_source_bbox(stage_a)
    print(f"[stage-A] pixelized stages will use fixed source bbox {source_bbox}")

    data = (image_data, noise_map, psf_kernel, feature_mask, lens_light_model)

    def checkpoint(tag):
        return load_pixelized_checkpoint(_load_stage, OUT_DIR, tag, resume=skip_done)

    def identity(tag):
        return pixelized_stage_identity(_load_stage(tag))

    log_lambda_m0 = run_stage_m0(
        *data, stage_a, position_likelihood, circular_mask,
        cached=checkpoint("m0"), upstream={"a": identity("a")},
    )
    stage_m1, medians_m1, s1_package = run_stage_m1(
        *data, stage_a, position_likelihood, log_lambda_m0, circular_mask,
        cached=checkpoint("m1"), upstream={"a": identity("a"), "m0": identity("m0")},
    )
    m1_id = identity("m1")
    _, _, reg_hyperparams_m2 = run_stage_m2(
        *data, medians_m1, position_likelihood, s1_package, circular_mask,
        cached=checkpoint("m2"), upstream={"m1": m1_id},
    )
    stage_m3, _ = run_stage_m3(
        *data, stage_m1, position_likelihood, reg_hyperparams_m2, s1_package, circular_mask,
        cached=checkpoint("m3"), upstream={"m1": m1_id, "m2": identity("m2")},
    )
    stage_m3_extra = _load_stage("m3")["extra"]
    time_m0, time_m1, time_m2, time_m3 = (
        _load_stage(tag)["extra"].get("time_taken", 0.0)
        for tag in ("m0", "m1", "m2", "m3")
    )

    (OUT_DIR / "stage_timings.json").write_text(
        json.dumps(
            dict(a=time_a, m0=time_m0, m1=time_m1, m2=time_m2, m3=time_m3), indent=2
        )
        + "\n"
    )
    export_m3_bundle(
        lens_subtracted,
        image_data,
        noise_map,
        psf_kernel,
        feature_mask,
        lens_light_model,
        stage_m1,
        stage_m3,
        position_likelihood,
        reg_hyperparams_m2,
        s1_package,
        circular_mask,
        fitted_objective=stage_m3_extra["objective"],
    )


def export_m3_bundle(
    lens_subtracted,
    image,
    noise,
    psf,
    feature_mask,
    lens_light,
    stage_m1,
    stage_m3,
    position_likelihood,
    reg_hyperparams,
    s1_package,
    circular_mask,
    *,
    fitted_objective=None,
):
    """Export a self-contained Float32 handoff at the best M3 sampled point."""
    likelihood = build_stage_m3_likelihood(
        lens_subtracted,
        noise,
        psf,
        feature_mask,
        stage_m1,
        position_likelihood,
        reg_hyperparams,
        s1_package,
        circular_mask=circular_mask,
    )
    written = write_macro_handoff(
        OUT_DIR, likelihood, stage_m3, s1_package, image, lens_light,
        fitted_objective=fitted_objective,
    )
    if written:
        print("[M3] self-contained handoff saved", flush=True)
    else:
        print("[M3] reusing the verified handoff", flush=True)


def cli():
    import argparse
    import shutil
    from datetime import datetime, timezone

    global OUT_DIR, RESUME, SEED, SAMPLER_BATCH_SIZE, SOURCE_INTERPOLATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=946)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path,
        help="Macro output directory, relative to this script (default: macro)",
    )
    parser.add_argument(
        "--source-interpolation", choices=("bilinear", "cubic_hermite"),
        help="Pixelized source interpolation (fresh fit: bilinear by default; "
        "resume: use the saved mode unless explicitly specified)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Raw FITS directory for the first run; subsequent runs use local copies",
    )
    args = parser.parse_args()
    RESUME, SEED, SAMPLER_BATCH_SIZE = args.resume, args.seed, args.batch_size
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.out_dir is not None:
        OUT_DIR = SCRIPT_DIR / args.out_dir.expanduser()
    SOURCE_INTERPOLATION = resolve_run_source_interpolation(
        OUT_DIR, args.source_interpolation, resume=RESUME,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_directory = args.input_dir or Path(
        "/home/xycao/workspace/tinylens_dev/TinyLensGpu/examples/pix_src_demo_operator/pipe/subhalo_0946/data"
    )
    inputs = {}
    for name in ("image.fits", "noise.fits", "psf.fits"):
        destination = DATA_DIR / name
        original = input_directory / name
        if destination.exists():
            if (
                args.input_dir is not None
                and destination.read_bytes() != original.read_bytes()
            ):
                raise RuntimeError(f"Input changed: {name}")
        else:
            shutil.copy2(original, destination)
        inputs[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    settings = dict(
        **source_interpolation_metadata(SOURCE_INTERPOLATION),
        seed=SEED,
        batch_size=SAMPLER_BATCH_SIZE,
        source_n=NSRC,
        dpix=DPIX,
        nsub=NSUB_PIX,
        source_regularization=PIXEL_REGULARIZATION_TYPE,
        source_bbox_half_width=SOURCE_BBOX_HALF_WIDTH,
        mask_radius=MASK_RADIUS,
        macro_backend="operator",
        source_solver=SOLVER_TYPE,
        evidence_logdet=EVIDENCE_LOGDET,
        evidence_block_size=EVIDENCE_BLOCK_SIZE if EVIDENCE_LOGDET == "overlap" else None,
        vectorized_chunk_size=VECTORIZED_CHUNK_SIZE,
        source_adaptation="native_source_arc_v1",
        stages=["a", "b", "m0", "m1", "m2", "m3"],
        inputs_sha256=inputs,
    )
    manifest_path = OUT_DIR / "manifest.json"
    if manifest_path.exists():
        if not RESUME or not source_settings_match(
            json.loads(manifest_path.read_text())["settings"], settings, manifest_path,
        ):
            raise RuntimeError(
                "Existing macro run. Use --resume with matching settings or a fresh output directory."
            )
    else:
        manifest_path.write_text(
            json.dumps(
                dict(
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    settings=settings,
                    source_script_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    original_pipeline="TinyLensGpu/examples/pix_src_demo_operator/pipe/subhalo_0946/model.py",
                    initialization="fresh; no previous fitted products imported",
                ),
                indent=2,
            )
            + "\n"
        )
    if args.prepare_only:
        print(json.dumps(settings, indent=2))
        return
    print(
        f"[runtime] package={__import__('TinyLensGpu').__file__}; devices={jax.devices()}",
        flush=True,
    )
    _run_macro(skip_done=RESUME)


if __name__ == "__main__":
    cli()
