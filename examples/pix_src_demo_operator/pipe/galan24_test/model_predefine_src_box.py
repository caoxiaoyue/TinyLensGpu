"""
Adaptive-regularization inference pipeline for the galan24_test lensing data.

The input ``obs.fits`` is treated as lens-light-subtracted, so this pipeline
skips lens-light modelling entirely and focuses on the mass model and source
reconstruction.

Stage a  : SIE + shear + MGE source light; infer a fixed source bbox from the
            posterior-median source center, half-light radius, and ellipticity
Stage b  : build an arc feature mask from obs S/N
Stage m0 : SIE + shear + uniform pixelized source on the fixed stage-A bbox —
            GPU grid search for evidence-best lambda_reg
Stage m1 : EPL + shear + non-adaptive pixelized source on the same fixed bbox —
            Nautilus sampling of mass parameters with lambda_reg from stage-M0,
            then builds the fixed S1 source template
Stage m2 : fixed EPL + shear + adaptive pixelized source — Nautilus sampling
            of log_lambda_reg and adaptive_reg_rho only
Stage m3 : EPL + shear + adaptive pixelized source — Nautilus nested sampling
            with source-reg hyperparameters fixed from stage-M2 medians

Each stage pickles its posterior samples/weights to
``output_adpt_reg_m1epl/stage_{a,m0,m1,m2,m3}.pkl`` and is re-runnable via ``--skip-done``.

Usage::

    # From galan24_test/
    python model_predefine_src_box.py
    python model_predefine_src_box.py --skip-done
    python model_predefine_src_box.py --skip-done --out-dir output_adpt_reg_m1epl_fixed_bbox
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
from matplotlib.colors import LogNorm
from nautilus import Sampler

from TinyLensGpu.example_support import (
    SamplingConfig,
    StageStorageConfig,
    draw_image_panel,
    export_source_package,
    is_square_bbox,
    load_stage,
    normalize_source_bbox,
    require_valid_reconstruction,
    run_sampling,
    save_stage,
    source_bboxes_match,
    stage_from_payload,
    validate_source_package,
    resolve_source_interpolation,
    resolve_stage_source_interpolation,
)
from TinyLensGpu.Inference import ParamU
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
from TinyLensGpu.utils import source_bbox_from_center_reff
from TinyLensGpu.utils.geometry import ellipticity2phi_q
from TinyLensGpu.utils.inversion.regularization import source_template_scale_map
from TinyLensGpu.utils.misc import arc_mask_from, weighted_quantile
from TinyLensGpu.visualizer import plot_model_results, overlay_critical_and_caustics

from TinyLensGpu.Inference import StagePosterior

import caskade as ck


# ------------------------------------------------------------------ #
NSRC = 100
DPIX = 0.1
NSUB = 4
NSUB_PIX = 4
N_GAUSSIANS_SRC = 10
MASK_RADIUS = 3.0
NOISE_MASK_THRESHOLD = 1e7  # noise_map pixels above this are pre-masked
ADAPTIVE_REG_RHO = 2.0             # 0 = uniform, >0 strengthens faint-region reg
ADAPTIVE_REG_RHO_PRIOR_MAX = 8.0
PIXEL_REGULARIZATION_TYPE = "first-order"
SOURCE_BBOX_RADIUS_FACTOR = 2.5
PNPG_MAX_ITER = 1000
PNPG_RTOL = 2.0e-2
PNPG_POWER_ITER = 20
# Source solver for all pixelized stages.
#   "pcg"  : preconditioned CG, unconstrained source, faster.
#   "pnpg": non-negative source and lens-light amplitudes.
SOLVER_TYPE = "pcg"
OUT_DIR = Path("output_predefine_src_box")
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
    tag: str,
    samples,
    weights,
    param_names,
    log_z,
    extra=None,
    stage=None,
    *,
    source_interpolation=None,
):
    save_stage(StageStorageConfig(OUT_DIR, include_schema=True, source_interpolation=source_interpolation),
               tag, samples, weights, param_names, log_z, extra=extra, stage=stage)


def _load_stage(tag: str, *, source_interpolation=None):
    return load_stage(
        StageStorageConfig(
            OUT_DIR, include_schema=True, source_interpolation=source_interpolation
        ),
        tag,
    )


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
    """Return a boolean mask where True = pixels outside ``radius_arcsec``."""
    ny, nx = image_shape
    y = (np.arange(ny) - (ny - 1) / 2) * dpix
    x = (np.arange(nx) - (nx - 1) / 2) * dpix
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return (xx ** 2 + yy ** 2) > radius_arcsec ** 2


def _is_square_bbox(source_bbox, *, rtol=1.0e-6, atol=1.0e-7):
    return is_square_bbox(source_bbox, rtol=rtol, atol=atol)


def _normalize_source_bbox(source_bbox, *, label="Source bbox"):
    return normalize_source_bbox(source_bbox, label=label)


def _source_bboxes_match(left, right):
    return source_bboxes_match(left, right)


def _make_source_scale(
    source_package,
    rho: float = ADAPTIVE_REG_RHO,
):
    """Build the fixed adaptive scale map from the stage-M0 source template."""
    return source_template_scale_map(
        source_package["source_pixels"],
        int(source_package["n"]),
        rho=rho,
    )


def _validate_source_package(source_package, expected_source_bbox=None):
    return validate_source_package(
        source_package, expected_n=NSRC, rho=ADAPTIVE_REG_RHO,
        expected_source_bbox=expected_source_bbox, normalize_bbox=True, check_axes=True,
    )


def _pnpg_kwargs():
    return dict(
        pnpg_max_iter=PNPG_MAX_ITER,
        pnpg_rtol=PNPG_RTOL,
        pnpg_power_iter=PNPG_POWER_ITER,
    )


def _fixed_source_kwargs(source_package):
    source_package = _validate_source_package(source_package)
    return dict(
        fixed_source_bbox=tuple(float(v) for v in source_package["source_bbox"]),
        fixed_reg_template=jnp.asarray(source_package["source_pixels"], dtype=jnp.float32),
    )


def _source_param_value(value):
    """Return a scalar value from a static scalar or ParamU-like object."""
    return value.value if hasattr(value, "value") else value


def _mge_half_light_radius(sigmas, fluxes):
    """Return the circularized half-light radius of a co-centered MGE."""
    sigmas = np.asarray(sigmas, dtype=np.float64)
    fluxes = np.asarray(fluxes, dtype=np.float64)
    if sigmas.ndim != 1 or fluxes.ndim != 1 or sigmas.shape != fluxes.shape:
        raise ValueError("MGE sigmas and fluxes must be same-length 1D arrays.")
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("MGE sigmas must be finite and positive.")
    if not np.all(np.isfinite(fluxes)):
        raise ValueError("MGE fluxes must be finite.")

    flux_tolerance = 100.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.sum(np.abs(fluxes))),
    )
    if np.any(fluxes < -flux_tolerance):
        raise ValueError("MGE fluxes must be non-negative.")
    fluxes = np.maximum(fluxes, 0.0)
    total_flux = float(np.sum(fluxes))
    if not np.isfinite(total_flux) or total_flux <= 0.0:
        raise ValueError("MGE total flux must be finite and positive.")

    target_flux = 0.5 * total_flux

    def enclosed_flux(radius):
        enclosed_fraction = -np.expm1(-0.5 * (radius / sigmas) ** 2)
        return float(np.dot(fluxes, enclosed_fraction))

    lower = 0.0
    upper = float(np.max(sigmas))
    while enclosed_flux(upper) < target_flux:
        upper *= 2.0
        if not np.isfinite(upper):
            raise ValueError("Could not bracket the MGE half-light radius.")

    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        if enclosed_flux(midpoint) < target_flux:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _stage_a_source_bbox_info(likelihood, medians, param_names):
    """Derive a fixed square source bbox from the median stage-A MGE."""
    required = ("center_x_src", "center_y_src", "e1_src", "e2_src")
    missing = [name for name in required if name not in medians]
    if missing:
        raise KeyError("Stage-A medians missing required keys: " + ", ".join(missing))

    q50 = [medians[name] for name in param_names]
    solved_params = likelihood.get_linear_solved_params(q50)
    sigmas = []
    fluxes = []
    for source_component in likelihood.phys_model.source_light:
        flux_name = source_component.flux.name
        if flux_name in solved_params:
            solved_flux = solved_params[flux_name]
        else:
            suffix = f"_{flux_name}"
            matches = [
                value for name, value in solved_params.items()
                if name.endswith(suffix)
            ]
            if len(matches) != 1:
                raise KeyError(
                    f"Could not uniquely identify solved MGE flux {flux_name!r}."
                )
            solved_flux = matches[0]
        sigmas.append(float(np.asarray(_source_param_value(source_component.sigma))))
        fluxes.append(float(np.asarray(solved_flux)))

    effective_radius = _mge_half_light_radius(sigmas, fluxes)
    _, axis_ratio = ellipticity2phi_q(
        jnp.asarray(medians["e1_src"]),
        jnp.asarray(medians["e2_src"]),
    )
    axis_ratio = float(np.asarray(axis_ratio))
    if not np.isfinite(axis_ratio) or axis_ratio <= 0.0:
        raise ValueError("Stage-A source axis ratio must be finite and positive.")

    center_x = float(medians["center_x_src"])
    center_y = float(medians["center_y_src"])
    major_axis_effective_radius = effective_radius / np.sqrt(axis_ratio)
    source_bbox = source_bbox_from_center_reff(
        center_x,
        center_y,
        major_axis_effective_radius,
        radius_factor=SOURCE_BBOX_RADIUS_FACTOR,
    )
    if not _is_square_bbox(source_bbox):
        raise ValueError("Stage-A source bbox must be square.")

    return _validate_source_bbox_info(dict(
        center_x=center_x,
        center_y=center_y,
        effective_radius=effective_radius,
        axis_ratio=axis_ratio,
        major_axis_effective_radius=major_axis_effective_radius,
        radius_factor=SOURCE_BBOX_RADIUS_FACTOR,
        source_bbox=source_bbox,
    ))


def _validate_source_bbox_info(source_bbox_info):
    """Validate and normalize cached stage-A source-bbox metadata."""
    required = (
        "center_x",
        "center_y",
        "effective_radius",
        "axis_ratio",
        "major_axis_effective_radius",
        "radius_factor",
        "source_bbox",
    )
    if not isinstance(source_bbox_info, dict):
        raise TypeError("Stage-A source bbox metadata must be a dictionary.")
    missing = [name for name in required if name not in source_bbox_info]
    if missing:
        raise KeyError(
            "Stage-A source bbox metadata missing: " + ", ".join(missing)
        )

    normalized = {
        name: float(source_bbox_info[name])
        for name in required
        if name != "source_bbox"
    }
    if not np.all(np.isfinite(tuple(normalized.values()))):
        raise ValueError("Stage-A source bbox metadata must be finite.")
    if normalized["effective_radius"] <= 0.0:
        raise ValueError("Stage-A effective radius must be positive.")
    if normalized["axis_ratio"] <= 0.0:
        raise ValueError("Stage-A source axis ratio must be positive.")
    if normalized["major_axis_effective_radius"] <= 0.0:
        raise ValueError("Stage-A major-axis effective radius must be positive.")
    expected_major_radius = (
        normalized["effective_radius"] / np.sqrt(normalized["axis_ratio"])
    )
    if not np.isclose(
        normalized["major_axis_effective_radius"],
        expected_major_radius,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Stage-A major-axis effective radius is inconsistent with R_e and q."
        )
    if not np.isclose(
        normalized["radius_factor"],
        SOURCE_BBOX_RADIUS_FACTOR,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Cached source bbox uses a different radius factor.")

    source_bbox = _normalize_source_bbox(
        source_bbox_info["source_bbox"], label="Stage-A source bbox",
    )
    expected_bbox = source_bbox_from_center_reff(
        normalized["center_x"],
        normalized["center_y"],
        normalized["major_axis_effective_radius"],
        radius_factor=normalized["radius_factor"],
    )
    if not _source_bboxes_match(source_bbox, expected_bbox):
        raise ValueError("Stage-A source bbox metadata is internally inconsistent.")
    normalized["source_bbox"] = source_bbox
    return normalized


def _reg_hyperparams_from_payload(stage_payload, tag):
    medians = stage_payload.get("extra", {}).get("medians")
    if not medians:
        raise KeyError(f"stage-{tag.upper()} payload missing posterior medians.")
    required = ("log_lambda_reg", "adaptive_reg_rho")
    missing = [name for name in required if name not in medians]
    if missing:
        raise KeyError(
            f"stage-{tag.upper()} medians missing required keys: "
            + ", ".join(missing)
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
    return export_source_package(likelihood, medians, param_names, n=NSRC)


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
    cx_s.to_dynamic()
    cy_s.to_dynamic()
    e1_s.to_dynamic()
    e2_s.to_dynamic()

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
    t1 = time.time()
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    _print_summary("stage-A", samples, weights, names)
    print(f"[stage-A] time taken: {t1 - t0:.2f} seconds")

    medians = stage.medians()
    q50 = [medians[n] for n in names]
    source_bbox_info = _stage_a_source_bbox_info(likelihood, medians, names)
    print(
        "[stage-A] fixed source bbox: "
        f"center=({source_bbox_info['center_x']:+.4f}, "
        f"{source_bbox_info['center_y']:+.4f}), "
        f"R_e={source_bbox_info['effective_radius']:.4f}, "
        f"q={source_bbox_info['axis_ratio']:.4f}, "
        f"bounds={source_bbox_info['source_bbox']}"
    )

    _dump_stage(
        "a", samples, weights, names, logz,
        extra=dict(
            medians=medians,
            time_taken=t1 - t0,
            source_bbox_info=source_bbox_info,
        ),
        stage=stage,
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
    return stage, medians, source_bbox_info


# ------------------------------------------------------------------ #
# Stage B — build an arc mask from obs S/N
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

    # arc_mask_from returns True for EXCLUDED pixels (non-arc/background).
    arc_mask = arc_mask_from(snr_map, threshold=2.0,
                             ignor_size=20, ext_size=5, close_size=3)

    # Combine: non-arc/background mask + pre-masked pixels + circular mask.
    # The pixelized source stages fit the remaining unmasked arc pixels.
    feature_mask = arc_mask | noise_masked
    if circular_mask is not None:
        feature_mask = feature_mask | circular_mask

    n_in = int((~feature_mask).sum())
    print(f"[stage-B] arc/source pixels kept = {n_in} / {feature_mask.size}")

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
        title='obs (treated as lens-subtracted)',
        title_size=plt.rcParams["axes.titlesize"], xlabel=None,
        extent=extent, cmap='magma',
    )
    im1 = draw_image_panel(
        axes[1], snr_map,
        title='S/N map + arc mask boundary',
        title_size=plt.rcParams["axes.titlesize"], xlabel=None,
        extent=extent, cmap='magma', vmin=-3, vmax=np.nanpercentile(snr_map, 99.5),
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


def _epl_mass_from_stage(stage: StagePosterior):
    """EPL (+ shear) with Gaussian priors inherited from a stage posterior."""
    theta_E = stage.gaussian(
        "theta_E", model="EPL", attr="theta_E", limits=[0.0, 5.0],
    )
    e1m = stage.gaussian(
        "e1_mass", model="EPL", attr="e1", limits=[-1.0, 1.0],
    )
    e2m = stage.gaussian(
        "e2_mass", model="EPL", attr="e2", limits=[-1.0, 1.0],
    )
    cx = stage.gaussian(
        "center_x_mass", model="EPL", attr="center_x", limits=[-1.0, 1.0],
    )
    cy = stage.gaussian(
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
    epl.gamma.to_dynamic()

    shear = Shear(
        gamma1=stage.gaussian(
            "gamma1", model="Shear", attr="gamma1", limits=[-0.5, 0.5],
        ),
        gamma2=stage.gaussian(
            "gamma2", model="Shear", attr="gamma2", limits=[-0.5, 0.5],
        ),
    )
    return epl, shear


def _epl_mass_from_medians(medians: dict):
    """EPL (+ shear) with all parameters fixed at posterior medians."""
    required = (
        "theta_E",
        "gamma",
        "e1_mass",
        "e2_mass",
        "center_x_mass",
        "center_y_mass",
        "gamma1",
        "gamma2",
    )
    missing = [name for name in required if name not in medians]
    if missing:
        raise KeyError("EPL medians missing required keys: " + ", ".join(missing))

    epl = EPL(
        theta_E=ParamU("theta_E", float(medians["theta_E"])),
        gamma=ParamU("gamma", float(medians["gamma"])),
        e1=ParamU("e1_mass", float(medians["e1_mass"])),
        e2=ParamU("e2_mass", float(medians["e2_mass"])),
        center_x=ParamU("center_x_mass", float(medians["center_x_mass"])),
        center_y=ParamU("center_y_mass", float(medians["center_y_mass"])),
    )
    for p in (epl.theta_E, epl.gamma, epl.e1, epl.e2, epl.center_x, epl.center_y):
        p.to_static()

    shear = Shear(
        gamma1=ParamU("gamma1", float(medians["gamma1"])),
        gamma2=ParamU("gamma2", float(medians["gamma2"])),
    )
    shear.gamma1.to_static()
    shear.gamma2.to_static()
    return epl, shear


# ------------------------------------------------------------------ #
# Stage M0 — SIE + shear + uniform pixelized source
# ------------------------------------------------------------------ #
def build_stage_m0_likelihood(
    image_data, noise_map, psf_kernel, feature_mask,
    stage_a: StagePosterior, position_likelihood, fixed_source_bbox,
    circular_mask=None,
    *, source_interpolation=None,
):
    """Build M0 on the fixed stage-A source grid with uniform regularization."""
    sie, shear = _sie_mass_from_stage_a(stage_a)

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
        regularization_type=PIXEL_REGULARIZATION_TYPE,
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
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        solver_type=SOLVER_TYPE,
        fixed_source_bbox=fixed_source_bbox,
        **_pnpg_kwargs(),
    )


def run_stage_m0(image_data, noise_map, psf_kernel, feature_mask,
                  stage_a: StagePosterior, position_likelihood,
                  fixed_source_bbox, circular_mask=None, *, source_interpolation=None):
    """Find the uniform-reg strength on the fixed stage-A source grid."""
    print("\n" + "=" * 60)
    print(" Stage M0 : fixed SIE + shear + uniform pix source ")
    print("=" * 60)
    t0 = time.time()

    likelihood = build_stage_m0_likelihood(
        image_data, noise_map, psf_kernel, feature_mask,
        stage_a, position_likelihood, fixed_source_bbox,
        circular_mask=circular_mask,
        source_interpolation=source_interpolation,
    )
    loglike_batch = make_likelihood(likelihood, vectorized=True)
    n_grid = 200

    log_lam_min, log_lam_max = jnp.log(1e-8), jnp.log(1e8)
    print(f"[stage-M0] Running coarse grid (200 pts, λ in [{float(jnp.exp(log_lam_min)):.1e}, {float(jnp.exp(log_lam_max)):.1e}]) ...")
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
    print(f"[stage-M0] Coarse best: λ = {float(jnp.exp(log_lam_best_coarse)):.4e}  (log-ev = {log_ev_best_coarse:.2f})")

    half_width = 0.5 * jnp.log(10)
    log_lam_grid_fine = jnp.linspace(
        log_lam_best_coarse - half_width,
        log_lam_best_coarse + half_width,
        n_grid,
    )
    print(f"[stage-M0] Running refinement grid (200 pts, λ in [{float(jnp.exp(log_lam_grid_fine[0])):.4e}, {float(jnp.exp(log_lam_grid_fine[-1])):.4e}]) ...")
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
    print(f"[stage-M0] Refined best: λ = {float(jnp.exp(log_lam_best)):.4e}  (log-ev = {log_ev_best:.2f})")

    medians_a = stage_a.medians()
    medians_m0 = {**medians_a, "log_lambda_reg": log_lam_best}

    t1 = time.time()
    print("\n[stage-M0] Grid search summary:")
    print(f"    {'lambda_reg_uniform':25s} = {float(jnp.exp(log_lam_best)):+.4e}")
    print(f"[stage-M0] time taken: {t1 - t0:.2f} seconds")

    _dump_stage(
        "m0", None, None, ["log_lambda_reg"], log_ev_best,
        extra=dict(
            lambda_best=log_lam_best,
            evidence_lambda_best=float(jnp.exp(log_lam_best)),
            evidence_log_lambda_best=log_lam_best,
            lambda_grid_coarse=np.asarray(log_lam_grid_coarse, dtype=np.float64),
            log_ev_coarse=np.asarray(log_ev_coarse, dtype=np.float64),
            lambda_grid_fine=np.asarray(log_lam_grid_fine, dtype=np.float64),
            log_ev_fine=np.asarray(log_ev_fine, dtype=np.float64),
            time_taken=t1 - t0,
            fixed_source_bbox=tuple(float(v) for v in fixed_source_bbox),
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

    return log_lam_best


# ------------------------------------------------------------------ #
# Stage M1 — EPL + shear + non-adaptive pixelized source  (fit mass)
# ------------------------------------------------------------------ #
def build_stage_m1_likelihood(
    image_data, noise_map, psf_kernel, feature_mask,
    stage_a: StagePosterior, position_likelihood, log_lambda_fixed: float,
    fixed_source_bbox, circular_mask=None,
    *, source_interpolation=None,
):
    """Build M1 mass inference on the fixed stage-A source grid."""
    epl, shear = _epl_mass_from_stage(stage_a)
    log_lam = ParamU(
        "log_lambda_reg",
        float(log_lambda_fixed),
        prior_type="truncated_gaussian",
        prior_settings=[float(log_lambda_fixed), 0.15],
        limits=[float(log_lambda_fixed) - 0.5, float(log_lambda_fixed) + 0.5],
    )
    log_lam.to_dynamic()

    pix_src = PixelizedSourceModel(n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        interpolation=resolve_source_interpolation(source_interpolation),
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
        solver_type=SOLVER_TYPE,
        fixed_source_bbox=fixed_source_bbox,
        **_pnpg_kwargs(),
    )


def run_stage_m1(image_data, noise_map, psf_kernel, feature_mask,
                  stage_a: StagePosterior, position_likelihood,
                  log_lambda_fixed: float, fixed_source_bbox,
                  circular_mask=None, *, source_interpolation=None):
    """Fit EPL+shear while keeping the stage-A source grid fixed.

    ``image_data`` is already lens-subtracted so we use it directly.
    """
    print("\n" + "=" * 60)
    print(" Stage M1 : EPL + shear + non-adaptive pix source (mass fit)")
    print("=" * 60)
    print(
        f"[stage-M1] lambda_reg prior from M0: "
        f"truncated Gaussian centered at {float(jnp.exp(log_lambda_fixed)):.4e} "
        f"(sigma=0.15, limits=[{float(jnp.exp(log_lambda_fixed - 0.5)):.4e}, "
        f"{float(jnp.exp(log_lambda_fixed + 0.5)):.4e}])"
    )
    t0 = time.time()

    likelihood = build_stage_m1_likelihood(
        image_data, noise_map, psf_kernel, feature_mask,
        stage_a, position_likelihood, log_lambda_fixed,
        fixed_source_bbox, circular_mask=circular_mask,
        source_interpolation=source_interpolation,
    )
    stage = _run_sampler(
        likelihood, n_live=300, n_eff=600, tag="stage-M1", vectorized=True,
    )
    t1 = time.time()
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
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
    s1_package["scale_map"] = np.asarray(_make_source_scale(s1_package), dtype=np.float32)
    _validate_source_package(
        s1_package, expected_source_bbox=fixed_source_bbox,
    )

    _dump_stage(
        "m1", samples, weights, names, logz,
        extra=dict(
            medians=medians,
            m1_mass_model="EPL+Shear",
            lambda_prior_center=float(log_lambda_fixed),
            lambda_prior_sigma=0.15,
            lambda_prior_limits=[float(log_lambda_fixed - 0.5), float(log_lambda_fixed + 0.5)],
            s1=s1_package,
            time_taken=t1 - t0,
        ),
        stage=stage,
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

    return stage, medians, s1_package


# ------------------------------------------------------------------ #
# Stage M2 — fixed EPL + shear + adaptive pixelized source  (fit source reg)
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
    # Log-scale display range for the lensing/model/source images.
    # 0.1% of vmax keeps faint features visible while staying strictly positive
    # (LogNorm is undefined for non-positive values).
    log_vmin_img = max(float(vmax) * 1e-3, 1e-3)
    src_vmax = float(np.nanpercentile(src_img, 99.5))
    log_vmin_src = max(src_vmax * 1e-3, 1e-3)

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
        (axes[0], image_data,  "Data (lens-subtracted, log)",
         dict(norm=LogNorm(vmin=log_vmin_img, vmax=vmax), cmap="magma")),
        (axes[1], model_image, "Model image (log)",
         dict(norm=LogNorm(vmin=log_vmin_img, vmax=vmax), cmap="magma")),
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
        cmap="magma",
        norm=LogNorm(vmin=log_vmin_src, vmax=src_vmax),
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
    medians_m1: dict, position_likelihood, s1_package, circular_mask=None,
    *, source_interpolation=None,
):
    """Build M2: mass fixed from M1, source-reg hyperparameters free."""
    source_interpolation = resolve_source_interpolation(
        source_interpolation, metadata=s1_package, context="S1 source package"
    )
    epl, shear = _epl_mass_from_medians(medians_m1)
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

    pix_src = PixelizedSourceModel(n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
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
    return PixelizedImageProbModelOperator(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        solver_type=SOLVER_TYPE,
        **_pnpg_kwargs(),
        **_fixed_source_kwargs(s1_package),
    )


def run_stage_m2(image_data, noise_map, psf_kernel, feature_mask,
                  medians_m1, position_likelihood, s1_package, circular_mask=None, *, source_interpolation=None):
    """Stage M2: fit adaptive source-regularization hyperparameters.

    ``image_data`` is already lens-subtracted so we use it directly.
    """
    print("\n" + "=" * 60)
    print(" Stage M2 : fixed EPL + shear + adaptive pix source (fit λ, rho)")
    print("=" * 60)
    t0 = time.time()
    likelihood = build_stage_m2_likelihood(
        image_data,
        noise_map,
        psf_kernel,
        feature_mask,
        medians_m1,
        position_likelihood,
        s1_package,
        circular_mask=circular_mask,
        source_interpolation=source_interpolation,
    )
    stage = _run_sampler(
        likelihood, n_live=250, n_eff=600, tag="stage-M2", vectorized=True,
    )
    t1 = time.time()
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    _print_summary("stage-M2", samples, weights, names)
    print(f"[stage-M2] time taken: {t1 - t0:.2f} seconds")
    medians = stage.medians()
    reg_hyperparams_m2 = {
        "log_lambda_reg": float(medians["log_lambda_reg"]),
        "adaptive_reg_rho": float(medians["adaptive_reg_rho"]),
    }
    print(f"[stage-M2] median source reg: {_format_reg_hyperparams(reg_hyperparams_m2)}")
    extra = dict(
        medians=medians,
        m2_mass_model="fixed-EPL+Shear",
        mass_medians_fixed=dict(medians_m1),
        reg_hyperparams=reg_hyperparams_m2,
        time_taken=t1 - t0,
    )
    _dump_stage(
        "m2",
        samples,
        weights,
        names,
        logz,
        extra=extra,
        stage=stage,
        source_interpolation=likelihood.source_model.interpolation,
    )
    try:
        _plot_pix_stage("stage-M2", likelihood, medians, names,
                        str(OUT_DIR / "stage_m2_model.png"))
    except InvalidReconstructionError:
        raise
    except Exception as err:
        print(f"[stage-M2] plotting failed (non-fatal): {err}")
    return stage, medians, reg_hyperparams_m2


# ------------------------------------------------------------------ #
# Stage M3 — EPL + shear + adaptive pixelized source  (final mass fit)
# ------------------------------------------------------------------ #
def build_stage_m3_likelihood(
    image_data, noise_map, psf_kernel, feature_mask,
    stage_m1: StagePosterior, position_likelihood, reg_hyperparams_fixed: dict,
    s1_package, circular_mask=None,
    *, source_interpolation=None,
):
    """Build M3: EPL+shear free, adaptive source-reg fixed from M2."""
    source_interpolation = resolve_source_interpolation(
        source_interpolation, metadata=s1_package, context="S1 source package"
    )
    epl, shear = _epl_mass_from_stage(stage_m1)
    log_lam = ParamU("log_lambda_reg", float(reg_hyperparams_fixed["log_lambda_reg"]))
    log_lam.to_static()
    rho = ParamU("adaptive_reg_rho", float(reg_hyperparams_fixed["adaptive_reg_rho"]))
    rho.to_static()

    pix_src = PixelizedSourceModel(n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
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
    return PixelizedImageProbModelOperator(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        solver_type=SOLVER_TYPE,
        **_pnpg_kwargs(),
        **_fixed_source_kwargs(s1_package),
    )


def run_stage_m3(image_data, noise_map, psf_kernel, feature_mask,
                  stage_m1: StagePosterior, position_likelihood,
                  reg_hyperparams_fixed: dict, s1_package, circular_mask=None, *, source_interpolation=None):
    """Stage M3: final EPL+shear sampling with source-reg fixed from M2.

    Mass priors are inherited from the stage-M1 EPL+shear posterior.
    """
    print("\n" + "=" * 60)
    print(" Stage M3 : EPL + shear + adaptive pix source (final mass fit)")
    print("=" * 60)
    print(f"[stage-M3] fixed source reg: {_format_reg_hyperparams(reg_hyperparams_fixed)}")
    t0 = time.time()
    likelihood = build_stage_m3_likelihood(
        image_data,
        noise_map,
        psf_kernel,
        feature_mask,
        stage_m1,
        position_likelihood,
        reg_hyperparams_fixed,
        s1_package,
        circular_mask=circular_mask,
        source_interpolation=source_interpolation,
    )
    stage = _run_sampler(
        likelihood, n_live=300, n_eff=600, tag="stage-M3", vectorized=True,
    )
    t1 = time.time()
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    _print_summary("stage-M3", samples, weights, names)
    print(f"[stage-M3] time taken: {t1 - t0:.2f} seconds")
    medians = stage.medians()
    _dump_stage(
        "m3", samples, weights, names, logz,
        extra=dict(
            medians=medians,
            m3_mass_model="EPL+Shear",
            reg_hyperparams_fixed=dict(reg_hyperparams_fixed),
            time_taken=t1 - t0,
        ),
        stage=stage,
        source_interpolation=likelihood.source_model.interpolation,
    )
    try:
        _plot_pix_stage("stage-M3", likelihood, medians, names,
                        str(OUT_DIR / "stage_m3_model.png"))
    except InvalidReconstructionError:
        raise
    except Exception as err:
        print(f"[stage-M3] plotting failed (non-fatal): {err}")
    return stage, medians


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def main(skip_done: bool = False, out_dir: str | None = None, *, source_interpolation=None):
    global OUT_DIR
    if out_dir is not None:
        OUT_DIR = Path(out_dir)

    source_interpolation = resolve_stage_source_interpolation(
        OUT_DIR, ("a", "m0", "m1", "m2", "m3"),
        requested=source_interpolation, resume=skip_done,
    )
    source_options = dict(source_interpolation=source_interpolation)
    image_data, noise_map, psf_kernel, _ = load_lens_data(
        image_path=str(DATA_DIR / "obs.fits"),
        noise_path=str(DATA_DIR / "noise_map.fits"),
        psf_path=str(DATA_DIR / "psf.fits"),
    )

    # Circular mask applied to all stages: exclude pixels > MASK_RADIUS arcsec from centre
    circular_mask = _make_circular_mask(image_data.shape, DPIX, radius_arcsec=MASK_RADIUS)
    n_excl = int(circular_mask.sum())
    print(f"[circular mask] excluded {n_excl} / {circular_mask.size} pixels (> {MASK_RADIUS} arcsec)")

    # ---- stage A ---------------------------------------------------- #
    time_a = 0.0
    if skip_done and (OUT_DIR / "stage_a.pkl").exists():
        print(f"[stage-A] loading cached {OUT_DIR}/stage_a.pkl")
        d = _load_stage("a", **source_options)
        stage_a = _stage_from_payload(d)
        medians_a = d["extra"]["medians"]
        time_a = d["extra"].get("time_taken", 0.0)
        try:
            source_bbox_info = _validate_source_bbox_info(
                d["extra"]["source_bbox_info"]
            )
        except (KeyError, TypeError, ValueError) as err:
            print(
                "[stage-A] cached source bbox metadata is unavailable or stale "
                f"({err}); deriving it from the cached posterior."
            )
            stage_a_likelihood = build_stage_a_likelihood(
                image_data,
                noise_map,
                psf_kernel,
                circular_mask=circular_mask,
            )
            source_bbox_info = _stage_a_source_bbox_info(
                stage_a_likelihood, medians_a, stage_a.param_names,
            )
            d["extra"]["source_bbox_info"] = source_bbox_info
            with open(OUT_DIR / "stage_a.pkl", "wb") as f:
                pickle.dump(d, f)
            print("[stage-A] cached source bbox metadata updated.")
    else:
        stage_a, medians_a, source_bbox_info = run_stage_a(
            image_data, noise_map, psf_kernel, circular_mask=circular_mask,
            **source_options,
        )
        time_a = _load_stage("a", **source_options)["extra"].get("time_taken", 0.0)
    source_bbox_info = _validate_source_bbox_info(source_bbox_info)
    fixed_source_bbox = source_bbox_info["source_bbox"]
    print(f"[stage-A] pixelized stages will use fixed bbox {fixed_source_bbox}")

    # ---- stage B ---------------------------------------------------- #
    # Build arc mask directly from lens-subtracted image (no lens_light_model needed)
    feature_mask = run_stage_b(image_data, noise_map, circular_mask=circular_mask)

    # ---- position likelihood ----------------------------------------- #
    position_likelihood = _position_likelihood_from_stage_a(medians_a)

    # ---- stage M0 --------------------------------------------------- #
    time_m0 = 0.0
    reuse_m0 = False
    if skip_done and (OUT_DIR / "stage_m0.pkl").exists():
        d = _load_stage("m0", **source_options)
        if _source_bboxes_match(
            d.get("extra", {}).get("fixed_source_bbox"),
            fixed_source_bbox,
        ):
            print(f"[stage-M0] loading cached {OUT_DIR}/stage_m0.pkl")
            log_lambda_m0 = d["extra"]["lambda_best"]
            time_m0 = d["extra"].get("time_taken", 0.0)
            reuse_m0 = True
        else:
            print("[stage-M0] cached bbox differs from stage A; recomputing.")
    if not reuse_m0:
        log_lambda_m0 = run_stage_m0(
            image_data, noise_map, psf_kernel, feature_mask,
            stage_a, position_likelihood, fixed_source_bbox,
            circular_mask=circular_mask,
            **source_options,
        )
        time_m0 = _load_stage("m0", **source_options)["extra"].get("time_taken", 0.0)
    m0_recomputed = not reuse_m0

    # If M0 output exists but the plot is missing, re-plot now.
    if (OUT_DIR / "stage_m0.pkl").exists() and not (OUT_DIR / "stage_m0_model.png").exists():
        print("[stage-M0] stage_m0_model.png missing — re-plotting")
        lkl_m0_replot = build_stage_m0_likelihood(
            image_data, noise_map, psf_kernel, feature_mask,
            stage_a, position_likelihood, fixed_source_bbox,
            circular_mask=circular_mask,
            **source_options,
        )
        medians_m0_replot = {
            **medians_a,
            "log_lambda_reg": float(log_lambda_m0),
        }
        try:
            _plot_pix_stage("stage-M0", lkl_m0_replot, medians_m0_replot, ["log_lambda_reg"],
                            str(OUT_DIR / "stage_m0_model.png"))
        except InvalidReconstructionError:
            raise
        except Exception as err:
            print(f"[stage-M0] re-plotting failed (non-fatal): {err}")

    # ---- stage M1 --------------------------------------------------- #
    time_m1 = 0.0
    s1_package = None
    reuse_m1 = False
    if (
        skip_done
        and not m0_recomputed
        and (OUT_DIR / "stage_m1.pkl").exists()
    ):
        print(f"[stage-M1] loading cached {OUT_DIR}/stage_m1.pkl")
        d = _load_stage("m1", **source_options)
        try:
            names_m1 = d["param_names"]
            stage_m1 = _stage_from_payload(d)
            medians_m1 = d["extra"]["medians"]
            s1_package = _validate_source_package(
                d["extra"]["s1"],
                expected_source_bbox=fixed_source_bbox,
            )
            time_m1 = d["extra"].get("time_taken", 0.0)
            reuse_m1 = True
        except (KeyError, TypeError, ValueError) as err:
            print(
                f"[stage-M1] cached output is incompatible ({err}); recomputing."
            )
    if not reuse_m1:
        stage_m1, medians_m1, s1_package = run_stage_m1(
            image_data, noise_map, psf_kernel, feature_mask,
            stage_a, position_likelihood,
            log_lambda_m0, fixed_source_bbox,
            circular_mask=circular_mask,
            **source_options,
        )
        names_m1 = stage_m1.param_names
        time_m1 = _load_stage("m1", **source_options)["extra"].get("time_taken", 0.0)
    m1_recomputed = not reuse_m1

    if s1_package is None:
        raise RuntimeError(
            "[stage-M1] Failed to determine S1 source package — "
            "stage_m1.pkl may be corrupted."
        )

    # If M1 posteriors exist but the plot is missing, re-plot now
    if (OUT_DIR / "stage_m1.pkl").exists() and not (OUT_DIR / "stage_m1_model.png").exists():
        print("[stage-M1] stage_m1_model.png missing — re-plotting")
        lkl_m1_replot = build_stage_m1_likelihood(
            image_data, noise_map, psf_kernel, feature_mask,
            stage_a, position_likelihood, log_lambda_m0,
            fixed_source_bbox, circular_mask=circular_mask,
            **source_options,
        )
        try:
            _plot_pix_stage(
                "stage-M1",
                lkl_m1_replot,
                medians_m1,
                names_m1,
                str(OUT_DIR / "stage_m1_model.png"),
            )
        except InvalidReconstructionError:
            raise
        except Exception as err:
            print(f"[stage-M1] re-plotting failed (non-fatal): {err}")

    # ---- stage M2 --------------------------------------------------- #
    time_m2 = 0.0
    reg_hyperparams_m2 = None
    reuse_m2 = False
    if (
        skip_done
        and not m1_recomputed
        and (OUT_DIR / "stage_m2.pkl").exists()
    ):
        print(f"[stage-M2] loading cached {OUT_DIR}/stage_m2.pkl")
        d = _load_stage("m2", **source_options)
        try:
            stage_m2 = _stage_from_payload(d)
            names_m2 = stage_m2.param_names
            medians_m2 = d["extra"]["medians"]
            reg_hyperparams_m2 = _reg_hyperparams_from_m2_payload(d)
            time_m2 = d["extra"].get("time_taken", 0.0)
            reuse_m2 = True
        except (KeyError, TypeError, ValueError) as err:
            print(
                f"[stage-M2] cached output is incompatible ({err}); recomputing."
            )
    if not reuse_m2:
        stage_m2, medians_m2, reg_hyperparams_m2 = run_stage_m2(
            image_data, noise_map, psf_kernel, feature_mask,
            medians_m1, position_likelihood, s1_package,
            circular_mask=circular_mask,
            **source_options,
        )
        names_m2 = stage_m2.param_names
        time_m2 = _load_stage("m2", **source_options)["extra"].get("time_taken", 0.0)
    m2_recomputed = not reuse_m2

    # Re-plot M2 if the png is missing but posteriors exist
    if not (OUT_DIR / "stage_m2_model.png").exists():
        lkl_m2 = build_stage_m2_likelihood(
            image_data,
            noise_map,
            psf_kernel,
            feature_mask,
            medians_m1,
            position_likelihood,
            s1_package,
            circular_mask=circular_mask,
            **source_options,
        )
        try:
            _plot_pix_stage("stage-M2", lkl_m2, medians_m2, names_m2,
                            str(OUT_DIR / "stage_m2_model.png"))
        except InvalidReconstructionError:
            raise
        except Exception as err:
            print(f"[stage-M2] plotting failed (non-fatal): {err}")

    if reg_hyperparams_m2 is None:
        raise RuntimeError(
            "[stage-M2] Failed to determine source regularization hyperparameters."
        )

    # ---- stage M3 --------------------------------------------------- #
    time_m3 = 0.0
    reuse_m3 = False
    if (
        skip_done
        and not m2_recomputed
        and (OUT_DIR / "stage_m3.pkl").exists()
    ):
        print(f"[stage-M3] loading cached {OUT_DIR}/stage_m3.pkl")
        d = _load_stage("m3", **source_options)
        try:
            stage_m3 = _stage_from_payload(d)
            names_m3 = stage_m3.param_names
            medians_m3 = d["extra"]["medians"]
            time_m3 = d["extra"].get("time_taken", 0.0)
            reuse_m3 = True
        except (KeyError, TypeError, ValueError) as err:
            print(
                f"[stage-M3] cached output is incompatible ({err}); recomputing."
            )
    if not reuse_m3:
        stage_m3, medians_m3 = run_stage_m3(
            image_data, noise_map, psf_kernel, feature_mask,
            stage_m1,
            position_likelihood, reg_hyperparams_m2, s1_package,
            circular_mask=circular_mask,
            **source_options,
        )
        names_m3 = stage_m3.param_names
        time_m3 = _load_stage("m3", **source_options)["extra"].get("time_taken", 0.0)

    if not (OUT_DIR / "stage_m3_model.png").exists():
        lkl_m3 = build_stage_m3_likelihood(
            image_data,
            noise_map,
            psf_kernel,
            feature_mask,
            stage_m1,
            position_likelihood,
            reg_hyperparams_m2,
            s1_package,
            circular_mask=circular_mask,
            **source_options,
        )
        try:
            _plot_pix_stage("stage-M3", lkl_m3, medians_m3, names_m3,
                            str(OUT_DIR / "stage_m3_model.png"))
        except InvalidReconstructionError:
            raise
        except Exception as err:
            print(f"[stage-M3] plotting failed (non-fatal): {err}")

    print("\n" + "=" * 60)
    print(" Pipeline complete")
    print("=" * 60)
    print(" Time summary:")
    print(f"    Stage A:  {time_a/60:.2f} min")
    print(f"    Stage M0: {time_m0/60:.2f} min")
    print(f"    Stage M1: {time_m1/60:.2f} min")
    print(f"    Stage M2: {time_m2/60:.2f} min")
    print(f"    Stage M3: {time_m3/60:.2f} min")
    print(f"    Total:    {(time_a + time_m0 + time_m1 + time_m2 + time_m3)/60:.2f} min\n")
    print(f"    M0 best lambda_reg     = {float(jnp.exp(log_lambda_m0)):.4e}")
    print(f"    M1 median gamma        = {medians_m1.get('gamma', np.nan):+.4f}")
    print(f"    M2 median source reg   = {_format_reg_hyperparams(reg_hyperparams_m2)}")

    for k in ("theta_E", "gamma", "e1_mass", "e2_mass",
              "center_x_mass", "center_y_mass", "gamma1", "gamma2"):
        if k in medians_m3:
            print(f"    final  {k:15s} = {medians_m3[k]:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-done", action="store_true",
                        help=f"Re-use cached posteriors in {OUT_DIR}/stage_*.pkl")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory relative to this script.")
    parser.add_argument("--source-interpolation", choices=("bilinear", "cubic_hermite"))
    args = parser.parse_args()
    main(skip_done=args.skip_done, out_dir=args.out_dir, source_interpolation=args.source_interpolation)
