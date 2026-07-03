"""
Adaptive-regularization inference pipeline for the galan24_test lensing data.

The input ``obs.fits`` is treated as lens-light-subtracted, so this pipeline
skips lens-light modelling entirely and focuses on the mass model and source
reconstruction.

Stage a  : SIE + shear + MGE source light (uniform priors, NO lens light)
Stage b  : build an arc feature mask from obs S/N
Stage m0 : SIE + shear + uniform pixelized source — GPU grid search for
            evidence-best lambda_reg, builds the fixed S0 source template
Stage m1 : EPL + shear + non-adaptive pixelized source — Nautilus sampling
            of mass parameters with lambda_reg fixed from stage-M0, then
            builds the fixed S1 source template
Stage m2 : fixed EPL + shear + adaptive pixelized source — Nautilus sampling
            of log_lambda_reg and adaptive_reg_rho only
Stage m3 : EPL + shear + adaptive pixelized source — Nautilus nested sampling
            with source-reg hyperparameters fixed from stage-M2 medians

Each stage pickles its posterior samples/weights to
``output_adpt_reg_m1epl/stage_{a,m0,m1,m2,m3}.pkl`` and is re-runnable via ``--skip-done``.

Usage::

    # From galan24_test/
    python model.py
    python model.py --skip-done
    python model.py --skip-done --out-dir output_adpt_reg_m1epl_fista1000
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
from TinyLensGpu.utils.inversion.regularization import source_template_scale_map
from TinyLensGpu.utils.misc import arc_mask_from, weighted_quantile
from TinyLensGpu.visualizer import plot_model_results, overlay_critical_and_caustics

from TinyLensGpu.Inference import StagePosterior

import caskade as ck
import jax.scipy.linalg as jsl

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
SOURCE_BBOX_PADDING = 0.30
FISTA_MAX_ITER = 1000
FISTA_RTOL = 1.0e-5
FISTA_POWER_ITER = 10
FISTA_STEP_SAFETY = 1.2
# Source solver for all pixelized stages.
#   "pcg"  : preconditioned CG, unconstrained source, faster.
#   "fista": non-negative constrained source, slower.
SOLVER_TYPE = "pcg"
OUT_DIR = Path("output_adpt_reg_m1epl")
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
    return StagePosterior.from_likelihood(likelihood, samples, weights, log_z=log_z)


def _dump_stage(tag: str, samples, weights, param_names, log_z, extra=None, stage=None):
    OUT_DIR.mkdir(exist_ok=True)
    if stage is not None:
        stage_payload = stage.cache_payload()
        payload = dict(
            samples=stage_payload["samples"],
            weights=stage_payload["weights"],
            param_names=stage_payload["param_names"],
            log_z=stage_payload["log_z"],
            stage_schema={
                key: stage_payload[key]
                for key in ("param_names", "prior_specs")
                if key in stage_payload
            },
            extra=extra or {},
        )
    else:
        payload = dict(
            samples=samples, weights=weights,
            param_names=param_names, log_z=log_z,
            stage_schema={"param_names": list(param_names)},
            extra=extra or {},
        )
    with open(OUT_DIR / f"stage_{tag}.pkl", "wb") as f:
        pickle.dump(payload, f)
    print(f"[{tag}] posterior saved to {OUT_DIR}/stage_{tag}.pkl")


def _load_stage(tag: str):
    with open(OUT_DIR / f"stage_{tag}.pkl", "rb") as f:
        return pickle.load(f)


def _stage_from_payload(payload):
    """Rehydrate a stage posterior from a cached stage payload."""
    schema = payload.get("stage_schema", {})
    prior_specs = payload.get("prior_specs") or schema.get("prior_specs")
    param_names = payload.get("param_names") or schema.get("param_names")
    return StagePosterior.from_schema(
        payload["samples"],
        payload["weights"],
        prior_specs=prior_specs,
        param_names=None if prior_specs is not None else param_names,
        log_z=payload.get("log_z"),
    )


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


def _source_axes_from_bbox(source_bbox, n=NSRC):
    xmin, xmax, ymin, ymax = [float(v) for v in source_bbox]
    return (
        np.linspace(xmin, xmax, int(n), dtype=np.float64),
        np.linspace(ymin, ymax, int(n), dtype=np.float64),
    )


def _is_square_bbox(source_bbox, *, rtol=1.0e-6, atol=1.0e-7):
    xmin, xmax, ymin, ymax = [float(v) for v in source_bbox]
    return np.isclose(xmax - xmin, ymax - ymin, rtol=rtol, atol=atol)


def _make_s0_scale(
    s0_package,
    rho: float = ADAPTIVE_REG_RHO,
):
    """Build the fixed adaptive scale map from the stage-M0 source template."""
    return source_template_scale_map(
        s0_package["source_pixels"],
        int(s0_package["n"]),
        rho=rho,
    )


def _validate_s0_package(s0_package):
    legacy_keys = [k for k in ("nx", "ny") if k in s0_package]
    if legacy_keys:
        raise KeyError(
            "S0 package uses legacy source-grid keys "
            f"{', '.join(legacy_keys)}; regenerate S0 with the single-n "
            "source-grid schema."
        )

    required = (
        "source_pixels",
        "source_bbox",
        "source_x_axis",
        "source_y_axis",
        "n",
        "lambda_best",
        "log_lambda_best",
    )
    missing = [k for k in required if k not in s0_package]
    if missing:
        raise KeyError("S0 package missing required keys: " + ", ".join(missing))

    n = int(s0_package["n"])
    if n != NSRC:
        raise ValueError(
            f"S0 grid dimension n={n} does not match configured n={NSRC}."
        )
    source_pixels = np.asarray(s0_package["source_pixels"])
    if source_pixels.shape != (n * n,):
        raise ValueError(
            f"S0 source_pixels must have shape ({n * n},), "
            f"got {source_pixels.shape}."
        )
    for axis_name in ("source_x_axis", "source_y_axis"):
        axis = np.asarray(s0_package[axis_name])
        if axis.shape != (n,):
            raise ValueError(
                f"S0 {axis_name} must have shape ({n},), got {axis.shape}."
            )
    bbox = tuple(float(v) for v in s0_package["source_bbox"])
    if len(bbox) != 4 or not np.all(np.isfinite(bbox)):
        raise ValueError("S0 source_bbox must contain four finite values.")
    if not (bbox[0] < bbox[1] and bbox[2] < bbox[3]):
        raise ValueError("S0 source_bbox must satisfy xmin < xmax and ymin < ymax.")
    if not _is_square_bbox(bbox):
        raise ValueError(
            "S0 source_bbox is rectangular; regenerate S0 with a square "
            "pixelized source bbox."
        )

    scale_map = s0_package.get("scale_map")
    if scale_map is None:
        scale_map = np.asarray(_make_s0_scale(s0_package), dtype=np.float32)
        s0_package["scale_map"] = scale_map
    else:
        scale_map = np.asarray(scale_map, dtype=np.float32)
        if scale_map.shape != (n * n,):
            raise ValueError(
                f"S0 scale_map must have shape ({n * n},), got {scale_map.shape}."
            )
        if not np.all(np.isfinite(scale_map) & (scale_map > 0.0)):
            raise ValueError("S0 scale_map values must be finite and positive.")
        s0_package["scale_map"] = scale_map
    return s0_package


def _fista_kwargs():
    return dict(
        fista_max_iter=FISTA_MAX_ITER,
        fista_rtol=FISTA_RTOL,
        fista_power_iter=FISTA_POWER_ITER,
        fista_step_safety=FISTA_STEP_SAFETY,
    )


def _s0_fixed_kwargs(s0_package):
    s0_package = _validate_s0_package(s0_package)
    return dict(
        fixed_source_bbox=tuple(float(v) for v in s0_package["source_bbox"]),
        fixed_reg_template=jnp.asarray(s0_package["source_pixels"], dtype=jnp.float32),
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
    """Solve source pixels for the current pixelized likelihood configuration."""
    q50 = [medians[n] for n in param_names]
    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.array(q50))
        lambda_j = jnp.exp(likelihood.phys_model.source_light[0].log_lambda_reg.value)
        (xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub,
         beta_x_seed, beta_y_seed) = likelihood._get_bbox()
        scale = likelihood._get_reg_scale()
        reg_data = likelihood._regularization_data(xmin, xmax, ymin, ymax, scale=scale)
        op_data = likelihood.sim_obj.precompute_operator_data(
            xmin, xmax, ymin, ymax, _betas_sub=(beta_x_sub, beta_y_sub),
        )
        block_chols, block_masks = likelihood.sim_obj.build_block_diag_preconditioner(
            likelihood.noise_1d, xmin, xmax, ymin, ymax, lambda_j,
            likelihood.reg_builder, block_size=likelihood.block_size,
            scale=scale,
        )
        source_pixels, solver_info = likelihood._solve_source(
            xmin, xmax, ymin, ymax, lambda_j, reg_data, (block_chols, block_masks),
            op_data=op_data,
        )
        if not bool(np.asarray(solver_info.converged)):
            if hasattr(solver_info, "residual_norm"):
                metric_label = "residual"
                metric_value = float(solver_info.residual_norm)
            else:
                metric_label = "convergence_metric"
                metric_value = float(solver_info.convergence_metric)
            raise RuntimeError(
                f"{likelihood.solver_type.upper()} failed while solving the "
                "stage source template "
                f"({metric_label}={metric_value:.4e}, "
                f"n_iter={int(solver_info.n_iter)})."
            )
    source_bbox = (float(xmin), float(xmax), float(ymin), float(ymax))
    x_axis, y_axis = _source_axes_from_bbox(source_bbox, NSRC)
    return dict(
        source_pixels=np.asarray(source_pixels, dtype=np.float64),
        source_image=np.asarray(source_pixels, dtype=np.float64).reshape(NSRC, NSRC),
        source_bbox=source_bbox,
        source_x_axis=x_axis,
        source_y_axis=y_axis,
        n=NSRC,
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


def run_stage_a(image_data, noise_map, psf_kernel, circular_mask=None):
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

    _dump_stage(
        "a", samples, weights, names, logz,
        extra=dict(
            medians=medians,
            time_taken=t1 - t0,
        ),
        stage=stage,
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
    return stage, medians


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
    im0 = axes[0].imshow(image_data, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("obs (treated as lens-subtracted)")
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
# Stage M0 — SIE + shear + uniform pixelized source  (build S0)
# ------------------------------------------------------------------ #
def build_stage_m0_likelihood(
    image_data, noise_map, psf_kernel, feature_mask,
    stage_a: StagePosterior, position_likelihood, circular_mask=None,
):
    """Build M0 likelihood: fixed SIE+shear, uniform pixelized source."""
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
        source_bbox_padding=SOURCE_BBOX_PADDING,
        **_fista_kwargs(),
    )


def run_stage_m0(image_data, noise_map, psf_kernel, feature_mask,
                  stage_a: StagePosterior, position_likelihood, circular_mask=None):
    """Stage M0: uniform-reg source inversion used as the fixed S0 template."""
    print("\n" + "=" * 60)
    print(" Stage M0 : fixed SIE + shear + uniform pix source (build S0)")
    print("=" * 60)
    t0 = time.time()

    likelihood = build_stage_m0_likelihood(
        image_data, noise_map, psf_kernel, feature_mask,
        stage_a, position_likelihood, circular_mask=circular_mask,
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
    s0_package = _solve_pixel_source_for_package(
        likelihood, medians_m0, ["log_lambda_reg"],
    )
    s0_package.update(
        lambda_best=float(jnp.exp(log_lam_best)),
        log_lambda_best=log_lam_best,
        evidence_lambda_best=float(jnp.exp(log_lam_best)),
        evidence_log_lambda_best=log_lam_best,
        stage_a_medians=dict(medians_a),
    )
    s0_package["scale_map"] = np.asarray(_make_s0_scale(s0_package), dtype=np.float32)
    _validate_s0_package(s0_package)

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
            s0=s0_package,
            time_taken=t1 - t0,
        ),
    )

    try:
        _plot_pix_stage("stage-M0", likelihood, medians_m0, ["log_lambda_reg"],
                        str(OUT_DIR / "stage_m0_model.png"))
    except Exception as err:
        print(f"[stage-M0] plotting failed (non-fatal): {err}")

    return s0_package, log_lam_best


# ------------------------------------------------------------------ #
# Stage M1 — EPL + shear + non-adaptive pixelized source  (fit mass)
# ------------------------------------------------------------------ #
def build_stage_m1_likelihood(
    image_data, noise_map, psf_kernel, feature_mask,
    stage_a: StagePosterior, position_likelihood, log_lambda_fixed: float,
    circular_mask=None,
):
    """Build M1: EPL+shear free, source lambda as truncated Gaussian around M0."""
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
        source_bbox_padding=SOURCE_BBOX_PADDING,
        **_fista_kwargs(),
    )


def run_stage_m1(image_data, noise_map, psf_kernel, feature_mask,
                  stage_a: StagePosterior, position_likelihood,
                  log_lambda_fixed: float, circular_mask=None):
    """Stage M1: fit EPL+shear with non-adaptive source lambda fixed from M0.

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
        circular_mask=circular_mask,
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
    s1_package["scale_map"] = np.asarray(_make_s0_scale(s1_package), dtype=np.float32)
    _validate_s0_package(s1_package)

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
    )

    # Generate M1 pixelized source reconstruction plot
    try:
        _plot_pix_stage("stage-M1", likelihood, medians, names,
                        str(OUT_DIR / "stage_m1_model.png"))
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
        lam_val = jnp.exp(likelihood.phys_model.source_light[0].log_lambda_reg.value)
        lambda_j = jnp.asarray(lam_val)

        rho_val = float(np.asarray(_source_param_value(
            likelihood.phys_model.source_light[0].adaptive_reg_rho
        )))

        # Unpack 8-value _get_bbox (includes seed betas for adaptive reg)
        (xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub,
         beta_x_seed, beta_y_seed) = likelihood._get_bbox()

        # Compute adaptive regularization scale map
        scale = likelihood._get_reg_scale()

        reg_data = likelihood._regularization_data(xmin, xmax, ymin, ymax, scale=scale)
        op_data = likelihood.sim_obj.precompute_operator_data(
            xmin, xmax, ymin, ymax, _betas_sub=(beta_x_sub, beta_y_sub),
        )
        block_chols, block_masks = likelihood.sim_obj.build_block_diag_preconditioner(
            likelihood.noise_1d, xmin, xmax, ymin, ymax, lambda_j,
            likelihood.reg_builder, block_size=likelihood.block_size,
            scale=scale,
        )
        preconditioner = (block_chols, block_masks)
        source_pixels, solver_info = likelihood._solve_source(
            xmin, xmax, ymin, ymax, lambda_j, reg_data, preconditioner,
            op_data=op_data,
        )
        model_1d_j = likelihood.sim_obj.forward_model(
            source_pixels, xmin, xmax, ymin, ymax, op_data=op_data,
        )

        # N_eff = Ns - λ Tr(P⁻¹ R)  via block-diagonal preconditioner
        n_s = likelihood.sim_obj.source_n
        bs = likelihood.block_size
        n_blocks = (n_s + bs - 1) // bs
        trace_invPR = jnp.array(0.0, dtype=lambda_j.dtype)
        for by in range(n_blocks):
            for bx in range(n_blocks):
                bid = bx + by * n_blocks
                x_s, x_e = bx * bs, min((bx + 1) * bs, n_s)
                y_s, y_e = by * bs, min((by + 1) * bs, n_s)
                if bid >= len(block_chols):
                    break
                R_block = likelihood.reg_builder.block_diag_R(
                    x_s, x_e, y_s, y_e, xmin, xmax, ymin, ymax,
                    scale=scale,
                )
                chol = block_chols[bid]
                inv_block = jsl.cho_solve((chol, True), R_block)
                trace_invPR = trace_invPR + jnp.trace(inv_block)
        N_eff = float(likelihood.sim_obj.n_source_pixels - lambda_j * trace_invPR)

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

    n = likelihood.phys_model.source_light[0].n
    src_img = np.array(source_pixels).reshape(n, n)

    # Scale map (2D source grid)
    scale_img = np.array(scale).reshape(n, n) if scale is not None else np.ones((n, n))
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

    # Panel 5: adaptive reg scale map
    im4 = axes[4].imshow(scale_img, origin="lower", extent=ext_s,
                         cmap="plasma", vmin=1.0, vmax=max(1.0, float(np.nanmax(scale_img))))
    axes[4].set_title(f"Reg precision scale\n(rho={rho:.2f})", fontsize=11)
    axes[4].set_xlabel("arcsec")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04,
                 label=r"$\lambda_i / \lambda_{\rm global}$")

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
    medians_m1: dict, position_likelihood, s1_package, circular_mask=None,
):
    """Build M2: mass fixed from M1, source-reg hyperparameters free."""
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
        source_bbox_padding=SOURCE_BBOX_PADDING,
        **_fista_kwargs(),
        **_s0_fixed_kwargs(s1_package),
    )


def run_stage_m2(image_data, noise_map, psf_kernel, feature_mask,
                  medians_m1, position_likelihood, s1_package, circular_mask=None):
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
    _dump_stage("m2", samples, weights, names, logz, extra=extra, stage=stage)
    try:
        _plot_pix_stage("stage-M2", likelihood, medians, names,
                        str(OUT_DIR / "stage_m2_model.png"))
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
):
    """Build M3: EPL+shear free, adaptive source-reg fixed from M2."""
    epl, shear = _epl_mass_from_stage(stage_m1)
    log_lam = ParamU("log_lambda_reg", float(reg_hyperparams_fixed["log_lambda_reg"]))
    log_lam.to_static()
    rho = ParamU("adaptive_reg_rho", float(reg_hyperparams_fixed["adaptive_reg_rho"]))
    rho.to_static()

    pix_src = PixelizedSourceModel(n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        adaptive_reg_rho=rho,
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
        source_bbox_padding=SOURCE_BBOX_PADDING,
        **_fista_kwargs(),
        **_s0_fixed_kwargs(s1_package),
    )


def run_stage_m3(image_data, noise_map, psf_kernel, feature_mask,
                  stage_m1: StagePosterior, position_likelihood,
                  reg_hyperparams_fixed: dict, s1_package, circular_mask=None):
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
    )
    try:
        _plot_pix_stage("stage-M3", likelihood, medians, names,
                        str(OUT_DIR / "stage_m3_model.png"))
    except Exception as err:
        print(f"[stage-M3] plotting failed (non-fatal): {err}")
    return stage, medians


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def main(skip_done: bool = False, out_dir: str | None = None):
    global OUT_DIR
    if out_dir is not None:
        OUT_DIR = Path(out_dir)

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
        d = _load_stage("a")
        stage_a = _stage_from_payload(d)
        medians_a = d["extra"]["medians"]
        time_a = d["extra"].get("time_taken", 0.0)
    else:
        stage_a, medians_a = run_stage_a(
            image_data, noise_map, psf_kernel, circular_mask=circular_mask,
        )
        time_a = _load_stage("a")["extra"].get("time_taken", 0.0)

    # ---- stage B ---------------------------------------------------- #
    # Build arc mask directly from lens-subtracted image (no lens_light_model needed)
    feature_mask = run_stage_b(image_data, noise_map, circular_mask=circular_mask)

    # ---- position likelihood ----------------------------------------- #
    position_likelihood = _position_likelihood_from_stage_a(medians_a)

    # ---- stage M0 --------------------------------------------------- #
    time_m0 = 0.0
    if skip_done and (OUT_DIR / "stage_m0.pkl").exists():
        print(f"[stage-M0] loading cached {OUT_DIR}/stage_m0.pkl")
        d = _load_stage("m0")
        s0_package = _validate_s0_package(d["extra"]["s0"])
        log_lambda_m0 = d["extra"]["lambda_best"]
        time_m0 = d["extra"].get("time_taken", 0.0)
    else:
        s0_package, log_lambda_m0 = run_stage_m0(
            image_data, noise_map, psf_kernel, feature_mask,
            stage_a, position_likelihood,
            circular_mask=circular_mask,
        )
        time_m0 = _load_stage("m0")["extra"].get("time_taken", 0.0)

    # If M0 output exists but the plot is missing, re-plot now.
    if (OUT_DIR / "stage_m0.pkl").exists() and not (OUT_DIR / "stage_m0_model.png").exists():
        print("[stage-M0] stage_m0_model.png missing — re-plotting")
        lkl_m0_replot = build_stage_m0_likelihood(
            image_data, noise_map, psf_kernel, feature_mask,
            stage_a, position_likelihood, circular_mask=circular_mask,
        )
        medians_m0_replot = {
            **medians_a,
            "log_lambda_reg": float(s0_package["log_lambda_best"]),
        }
        try:
            _plot_pix_stage("stage-M0", lkl_m0_replot, medians_m0_replot, ["log_lambda_reg"],
                            str(OUT_DIR / "stage_m0_model.png"))
        except Exception as err:
            print(f"[stage-M0] re-plotting failed (non-fatal): {err}")

    # ---- stage M1 --------------------------------------------------- #
    time_m1 = 0.0
    s1_package = None
    if skip_done and (OUT_DIR / "stage_m1.pkl").exists():
        print(f"[stage-M1] loading cached {OUT_DIR}/stage_m1.pkl")
        d = _load_stage("m1")
        try:
            names_m1 = d["param_names"]
            stage_m1 = _stage_from_payload(d)
            medians_m1 = d["extra"]["medians"]
            s1_package = _validate_s0_package(d["extra"]["s1"])
            time_m1 = d["extra"].get("time_taken", 0.0)
        except KeyError as err:
            print(f"[stage-M1] cached output has old/incomplete format ({err}); recomputing.")
            stage_m1, medians_m1, s1_package = run_stage_m1(
                image_data, noise_map, psf_kernel, feature_mask,
                stage_a, position_likelihood,
                log_lambda_m0,
                circular_mask=circular_mask,
            )
            names_m1 = stage_m1.param_names
            time_m1 = _load_stage("m1")["extra"].get("time_taken", 0.0)
    else:
        stage_m1, medians_m1, s1_package = run_stage_m1(
            image_data, noise_map, psf_kernel, feature_mask,
            stage_a, position_likelihood,
            log_lambda_m0,
            circular_mask=circular_mask,
        )
        names_m1 = stage_m1.param_names
        time_m1 = _load_stage("m1")["extra"].get("time_taken", 0.0)

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
            circular_mask=circular_mask,
        )
        try:
            _plot_pix_stage(
                "stage-M1",
                lkl_m1_replot,
                medians_m1,
                names_m1,
                str(OUT_DIR / "stage_m1_model.png"),
            )
        except Exception as err:
            print(f"[stage-M1] re-plotting failed (non-fatal): {err}")

    # ---- stage M2 --------------------------------------------------- #
    time_m2 = 0.0
    reg_hyperparams_m2 = None
    if skip_done and (OUT_DIR / "stage_m2.pkl").exists():
        print(f"[stage-M2] loading cached {OUT_DIR}/stage_m2.pkl")
        d = _load_stage("m2")
        stage_m2 = _stage_from_payload(d)
        names_m2 = stage_m2.param_names
        medians_m2 = d["extra"]["medians"]
        reg_hyperparams_m2 = _reg_hyperparams_from_m2_payload(d)
        time_m2 = d["extra"].get("time_taken", 0.0)
    else:
        stage_m2, medians_m2, reg_hyperparams_m2 = run_stage_m2(
            image_data, noise_map, psf_kernel, feature_mask,
            medians_m1, position_likelihood, s1_package,
            circular_mask=circular_mask,
        )
        names_m2 = stage_m2.param_names
        time_m2 = _load_stage("m2")["extra"].get("time_taken", 0.0)

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
        )
        try:
            _plot_pix_stage("stage-M2", lkl_m2, medians_m2, names_m2,
                            str(OUT_DIR / "stage_m2_model.png"))
        except Exception as err:
            print(f"[stage-M2] plotting failed (non-fatal): {err}")

    if reg_hyperparams_m2 is None:
        raise RuntimeError(
            "[stage-M2] Failed to determine source regularization hyperparameters."
        )

    # ---- stage M3 --------------------------------------------------- #
    time_m3 = 0.0
    if skip_done and (OUT_DIR / "stage_m3.pkl").exists():
        print(f"[stage-M3] loading cached {OUT_DIR}/stage_m3.pkl")
        d = _load_stage("m3")
        stage_m3 = _stage_from_payload(d)
        names_m3 = stage_m3.param_names
        medians_m3 = d["extra"]["medians"]
        time_m3 = d["extra"].get("time_taken", 0.0)
    else:
        stage_m3, medians_m3 = run_stage_m3(
            image_data, noise_map, psf_kernel, feature_mask,
            stage_m1,
            position_likelihood, reg_hyperparams_m2, s1_package,
            circular_mask=circular_mask,
        )
        names_m3 = stage_m3.param_names
        time_m3 = _load_stage("m3")["extra"].get("time_taken", 0.0)

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
        )
        try:
            _plot_pix_stage("stage-M3", lkl_m3, medians_m3, names_m3,
                            str(OUT_DIR / "stage_m3_model.png"))
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
    args = parser.parse_args()
    main(skip_done=args.skip_done, out_dir=args.out_dir)
