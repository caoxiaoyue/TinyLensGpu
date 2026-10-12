"""B1938+666 F115W recentered modeling, adapted from subhalo_0946/model.py.

Stage A fits SIE + shear + three MGE lens-light groups + an MGE source,
with nonnegative linear fluxes. The supplied binary contamination mask is
mandatory: 1 means excluded. It is combined with the circular fitting aperture.
Coordinates are relative to the recentered array's geometric center; +x is
array-right (west), +y is array-up (north). The PSF uses its central array pixel.

Run from this directory in the tinylens_gpu environment::

    python model.py
    python model.py --skip-done
    python model.py --out-dir output_trial

Runs the complete single-source-plane pipeline: A, B, then M1 through M5.
M4/M5 compare a smooth EPL+shear model against the same model plus a
Pseudo-Jaffe subhalo. Sampling checkpoints and posterior caches are saved
per stage. Use --skip-done to resume a matching run.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path

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
from astropy import constants as astro_constants
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
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
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import (
    EPL,
    PseudoJaffeSubhalo,
    SIE,
)
from TinyLensGpu.utils import generate_radial_basis_knots
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.utils.inversion.regularization import source_template_scale_map
from TinyLensGpu.utils.misc import arc_mask_from, weighted_quantile
from TinyLensGpu.visualizer import plot_model_results, overlay_critical_and_caustics

from TinyLensGpu.Inference import StagePosterior

import caskade as ck

# ------------------------------------------------------------------ #
NSRC = 100
DPIX = 0.03  # F115W recentered FITS pixel scale, arcsec/pixel
NSUB = 4
NSUB_PIX = 4
N_GAUSSIANS_SRC = 10
N_GAUSSIANS_LENS = 10
# Shared image-plane scale (arcsec): fitting aperture radius, lens MGE maximum
# Gaussian width, subhalo search-square half-width, and diagnostic plot extent.
MASK_RADIUS = 0.7
# Exclude central lens-light subtraction residuals from Stage B and M1-M5.
# Stage A keeps the center to fit the lens light; set 0 to disable this mask.
INNER_MASK_RADIUS = 0.2  # arcsec about the recentered array's geometric center
ADAPTIVE_REG_RHO = 2.0
ADAPTIVE_REG_RHO_PRIOR_MAX = 8.0
PIXEL_REGULARIZATION_TYPE = "first-order"
SOURCE_BBOX_PADDING = 0.0
SOURCE_BBOX_HALF_WIDTH = 0.2  # arcsec about the stage-A source center
M1_LAMBDA_REG_PRIOR_RANGE = (1e-6, 1e6)  # Uniform prior on ln(lambda).
SOLVER_TYPE = "pcg"
PCG_MAX_ITER = 800
VECTORIZED_CHUNK_SIZE = 1  # Bound peak GPU memory during batched FFTs/solves.
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "output"
DATA_DIR = SCRIPT_DIR / "data"
RESUME_SAMPLERS = False

# ------------------------------------------------------------------ #
# M4 / M5 comparison configuration
# ------------------------------------------------------------------ #
# Trial half-widths for the shared M4/M5 uniform priors. M3 supplies only the
# center, not the width. These widths are not observationally calibrated.
SMOOTH_PRIOR_THETA_E_RELATIVE_HALF_WIDTH = 0.10
SMOOTH_PRIOR_ABSOLUTE_HALF_WIDTHS = {
    "gamma": 0.2,
    "e1_mass": 0.1,
    "e2_mass": 0.1,
    "center_x_mass": 0.05,  # arcsec
    "center_y_mass": 0.05,  # arcsec
    "gamma1": 0.05,
    "gamma2": 0.05,
}
SMOOTH_MASS_PRIOR_LIMITS = {
    "theta_E": (0.0, 5.0),
    "gamma": (1.0, 3.0),
    "e1_mass": (-1.0, 1.0),
    "e2_mass": (-1.0, 1.0),
    "center_x_mass": (-1.0, 1.0),
    "center_y_mass": (-1.0, 1.0),
    "gamma1": (-0.5, 0.5),
    "gamma2": (-0.5, 0.5),
}

# Source regularization strength: uniform in ln(lambda) == log-uniform in lambda.
# NOTE: ``log_lambda_reg`` stores ln(lambda), so the half-width must be given in
# natural-log units: 0.5 dex = 0.5 * ln(10) ~= 1.15129.
LAMBDA_PRIOR_HALF_WIDTH_DEX = 0.5

SUBHALO_B_RANGE = (0.0, 0.1)
# B1938+666 lens/source redshifts. Retain the reference cosmology (H0=73,
# Omega_m=0.25) for the Pseudo-Jaffe total projected mass conversion, not a
# mass inside a sharp tidal aperture or a pre-accretion virial mass.
SUBHALO_MASS_H0 = 73.0  # km / s / Mpc
SUBHALO_MASS_OMEGA_M = 0.25
SUBHALO_MASS_Z_LENS = 0.881
SUBHALO_MASS_Z_SOURCE = 2.059
# Subhalo position prior region:
#   "lensing_square" -> independent uniform priors on x and y in
#                     [-MASK_RADIUS, MASK_RADIUS] arcsec. This is a square
#                     position prior; the fitting aperture remains circular.
#   "gaussian_local" -> truncated Gaussian centred on SUBHALO_POSITION_CENTER with
#                     width SUBHALO_POSITION_SIGMA, truncated at
#                     SUBHALO_POSITION_TRUNCATE_SIGMA sigma per axis. Use this for
#                     the localised control experiment / per-cell search: with the
#                     position constrained, this tests position-dependent
#                     smooth-parameter drift.
SUBHALO_POSITION_MODE = "lensing_square"
# Centre of the localised position prior, in arcsec. Default is the position of
# the correlated residual on the upper-left arc (the location recovered by the
# original single-cell search on this dataset).
SUBHALO_POSITION_CENTER = (-0.70744, 1.00661)
SUBHALO_POSITION_SIGMA = 0.2
SUBHALO_POSITION_TRUNCATE_SIGMA = 3.0
# The stage-A image positions are themselves solved from the smooth model; keep
# them in both models for a controlled comparison, or set False to drop the
# penalty in both as a robustness test.
COMPARISON_USE_POSITION_LIKELIHOOD = True

M4_N_LIVE = 1000
M4_N_EFF = 1400
M5_N_LIVE = 1000
M5_N_EFF = 1500

# Prior-edge diagnostic: compare each posterior with its own prior. The flag is
# scale-free (posterior median vs prior edge in posterior sigmas); the weight in
# the outer fraction of the prior is reported alongside as context only.
PRIOR_EDGE_MARGIN_FRACTION = 0.02
PRIOR_EDGE_WARN_WEIGHT = 0.02
# Scale-free truncation flag: the posterior median must sit at least this many
# posterior sigmas from every prior edge. A fixed fraction of the prior width is
# NOT a safe threshold -- when the prior is far wider than the posterior, its
# outer few percent is narrower than the posterior itself and every healthy
# posterior looks edge-hugging.
PRIOR_EDGE_SIGMA = 3.0
# Smooth-solution drift flag: report when an arm's posterior median for a shared
# smooth parameter sits this many stage-M3 sigmas from the M3 solution. A large
# drift flags a possible trade with the main lens; compare multiple prior widths
# to determine whether the Bayes factor depends on the shared prior width.
SOLUTION_DRIFT_WARN_SIGMA = 5.0


# ------------------------------------------------------------------ #
def _make_pixel_likelihood(**kwargs):
    """Use the same PCG budget in all pixelized stages."""
    likelihood = PixelizedImageProbModelOperator(**kwargs)
    likelihood.pcg_max_iter = PCG_MAX_ITER
    return likelihood


def _run_sampler(likelihood, n_live: int, n_eff: int, tag: str, vectorized: bool = True):
    prior, prior_specs = make_prior_transformation(likelihood)
    param_names = [spec.name for spec in prior_specs]
    print(f"\n[{tag}] {len(param_names)} dynamic params:")
    for spec in prior_specs:
        print(f"    {spec.name:25s} {spec.describe()}")

    loglike = make_likelihood(
        likelihood,
        vectorized=vectorized,
        vectorized_chunk_size=VECTORIZED_CHUNK_SIZE if vectorized else None,
    )
    stage_id = tag.lower().replace("stage-", "")
    sampler_kwargs = dict(n_live=n_live, vectorized=vectorized,
                          seed=1938 + sum(ord(c) for c in stage_id) - ord("a"),
                          filepath=str(OUT_DIR / f"stage_{stage_id}_sampler.h5"),
                          resume=RESUME_SAMPLERS)
    if tag == "stage-A":
        # Warm up the same batch shape used by Nautilus before sampling.
        warm_theta = np.asarray(prior(np.full(len(param_names), 0.5)))
        t_warm = time.time()
        warm_like = np.asarray(loglike(np.tile(warm_theta, (100, 1))))
        if not np.all(np.isfinite(warm_like)):
            raise ValueError("Stage-A warm-up likelihood is not finite")
        print(f"[{tag}] GPU warm-up: {time.time() - t_warm:.1f}s; "
              f"loglike={warm_like[0]:.3f}", flush=True)
    if vectorized:
        sampler_kwargs["n_batch"] = 100
    sampler = Sampler(prior, loglike, n_dim=len(param_names), **sampler_kwargs)
    sampler.run(verbose=True, n_eff=n_eff)

    samples, log_w, _ = sampler.posterior()
    samples = np.asarray(samples, dtype=np.float64)
    weights = np.exp(np.asarray(log_w, dtype=np.float64))
    weights /= weights.sum()

    log_z = float(sampler.log_z)
    print(f"[{tag}] log_z = {log_z:.3f}; n_like={sampler.n_like}; n_eff={sampler.n_eff:.1f}")
    with (OUT_DIR / f"stage_{stage_id}_sampling.json").open("w") as handle:
        json.dump(dict(n_live=n_live, requested_n_eff=n_eff,
                       actual_n_eff=float(sampler.n_eff), n_like=int(sampler.n_like),
                       log_z=log_z), handle, indent=2)
    return StagePosterior.from_likelihood(likelihood, samples, weights, log_z=log_z)


def _comparison_cache_schema(
    stage_m2, stage_m3, lens_subtracted_image, noise_map, psf_kernel,
    feature_mask, position_likelihood, s1_package, circular_mask=None,
) -> str:
    """Identify the M4/M5 likelihood, its priors, and sampling settings."""
    position_config = None
    if COMPARISON_USE_POSITION_LIKELIHOOD and position_likelihood is not None:
        positions = position_likelihood.get("positions")
        if positions is not None and len(positions) >= 2:
            position_config = (
                np.asarray(positions, dtype=np.float32).tolist(),
                float(position_likelihood.get(
                    "threshold_arcsec", position_likelihood.get("position_threshold", 0.0)
                )),
                float(position_likelihood.get(
                    "min_log_like", position_likelihood.get("min_position_likelihood", 0.0)
                )),
            )

    values = (
        [(name, _smooth_mass_prior_bounds(stage_m3, name))
         for name in SMOOTH_MASS_PRIOR_LIMITS],
        float(stage_m2.median("log_lambda_reg")),
        LAMBDA_PRIOR_HALF_WIDTH_DEX,
        float(stage_m2.median("adaptive_reg_rho")),
        SUBHALO_B_RANGE,
        _subhalo_position_bounds(lens_subtracted_image.shape, DPIX),
        position_config,
        NSRC, DPIX, NSUB_PIX, PIXEL_REGULARIZATION_TYPE,
        SOLVER_TYPE, PCG_MAX_ITER, SOURCE_BBOX_PADDING,
        tuple(float(v) for v in s1_package["source_bbox"]),
        (M4_N_LIVE, M4_N_EFF, M5_N_LIVE, M5_N_EFF),
    )
    digest = hashlib.sha256(
        json.dumps(values, allow_nan=False, sort_keys=True).encode("utf-8")
    )
    combined_mask = np.asarray(feature_mask, dtype=bool)
    if circular_mask is not None:
        combined_mask = combined_mask | circular_mask
    for name, value in (
        ("image", lens_subtracted_image),
        ("noise", noise_map),
        ("psf", psf_kernel),
        ("mask", combined_mask),
        ("source_template", np.asarray(s1_package["source_pixels"], dtype=np.float32)),
    ):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(json.dumps((name, array.dtype.str, array.shape)).encode("utf-8"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _comparison_cache_matches(
    payload: dict,
    schema: str,
    expected_role: str,
) -> bool:
    """Return whether a stage cache matches the current settings and model role."""
    extra = payload.get("extra", {})
    return (
        extra.get("pipeline_schema") == schema
        and extra.get("comparison_role") == expected_role
    )


def _dump_stage(tag: str, samples, weights, param_names, log_z, extra=None, stage=None):
    OUT_DIR.mkdir(exist_ok=True)
    extra = dict(extra or {})
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
def _load_b1938_data():
    """Load the native recentered arrays and require a valid exclusion mask."""
    image_data, noise_map, psf_kernel, _ = load_lens_data(
        image_path=str(DATA_DIR / "B1938_666_F115W_recentered_science.fits"),
        noise_path=str(DATA_DIR / "B1938_666_F115W_recentered_sigma.fits"),
        psf_path=str(DATA_DIR / "B1938_666_F115W_psf_crop21.fits"),
    )
    mask_data = fits.getdata(DATA_DIR / "B1938_666_F115W_mask.fits")
    if image_data.ndim != 2 or image_data.shape[0] != image_data.shape[1]:
        raise ValueError("Science image must be a square 2D array")
    if noise_map.shape != image_data.shape or mask_data.shape != image_data.shape:
        raise ValueError("Science, sigma, and contamination mask shapes must agree")
    if not np.all(np.isin(mask_data, (0, 1))):
        raise ValueError("Contamination mask must contain only 0 and 1")
    if not np.all(np.isfinite(image_data)):
        raise ValueError("Science image contains nonfinite values")
    if not np.all(np.isfinite(noise_map) & (noise_map > 0)):
        raise ValueError("Sigma image must be finite and positive")
    if not np.all(np.isfinite(psf_kernel)) or psf_kernel.sum() <= 0:
        raise ValueError("PSF must be finite with a positive sum")
    pixel_scale = fits.getheader(DATA_DIR / "B1938_666_F115W_recentered_science.fits")["PIXSCALE"]
    if not np.isclose(pixel_scale, DPIX):
        raise ValueError(f"FITS pixel scale {pixel_scale} differs from DPIX={DPIX}")
    return image_data, noise_map, psf_kernel, mask_data == 1


def _make_circular_mask(image_shape, dpix, radius_arcsec=None):
    """Return a boolean mask where True = pixels outside ``radius_arcsec``."""
    if radius_arcsec is None:
        radius_arcsec = MASK_RADIUS
    ny, nx = image_shape
    y = (np.arange(ny) - (ny - 1) / 2) * dpix
    x = (np.arange(nx) - (nx - 1) / 2) * dpix
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return (xx ** 2 + yy ** 2) > radius_arcsec ** 2


def _make_feature_mask(snr_map, circular_mask=None):
    """Combine the arc selection, central exclusion, and supplied exclusions."""
    if not np.isfinite(INNER_MASK_RADIUS) or INNER_MASK_RADIUS < 0:
        raise ValueError("INNER_MASK_RADIUS must be finite and nonnegative.")
    feature_mask = arc_mask_from(
        snr_map, threshold=3.0, ignor_size=25, ext_size=5, close_size=3,
    )
    if INNER_MASK_RADIUS > 0:
        inner_mask = ~_make_circular_mask(snr_map.shape, DPIX, INNER_MASK_RADIUS)
        feature_mask = feature_mask | inner_mask
    if circular_mask is not None:
        feature_mask = feature_mask | circular_mask
    return feature_mask


def _source_axes_from_bbox(source_bbox, n=NSRC):
    xmin, xmax, ymin, ymax = [float(v) for v in source_bbox]
    return (
        np.linspace(xmin, xmax, int(n), dtype=np.float64),
        np.linspace(ymin, ymax, int(n), dtype=np.float64),
    )


def _stage_a_source_bbox(stage_a: StagePosterior):
    """Fixed square source grid centered on the stage-A source posterior."""
    cx = stage_a.median("center_x_src")
    cy = stage_a.median("center_y_src")
    half_width = SOURCE_BBOX_HALF_WIDTH
    return (cx - half_width, cx + half_width,
            cy - half_width, cy + half_width)


def _is_square_bbox(source_bbox, *, rtol=1.0e-6, atol=1.0e-7):
    xmin, xmax, ymin, ymax = [float(v) for v in source_bbox]
    return np.isclose(xmax - xmin, ymax - ymin, rtol=rtol, atol=atol)


def _make_source_scale(source_package, rho: float = ADAPTIVE_REG_RHO):
    """Build the fixed adaptive scale map from the stage-M1 source template."""
    return source_template_scale_map(
        source_package["source_pixels"],
        int(source_package["n"]),
        rho=rho,
    )


def _validate_source_package(source_package):
    legacy_keys = [k for k in ("nx", "ny") if k in source_package]
    if legacy_keys:
        raise KeyError(
            "Source package uses legacy source-grid keys "
            f"{', '.join(legacy_keys)}; regenerate S1 with the single-n "
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
    missing = [k for k in required if k not in source_package]
    if missing:
        raise KeyError("Source package missing required keys: " + ", ".join(missing))

    n = int(source_package["n"])
    if n != NSRC:
        raise ValueError(
            f"Source grid dimension n={n} does not match configured n={NSRC}."
        )
    source_pixels = np.asarray(source_package["source_pixels"])
    if source_pixels.shape != (n * n,):
        raise ValueError(
            f"Source source_pixels must have shape ({n * n},), "
            f"got {source_pixels.shape}."
        )
    for axis_name in ("source_x_axis", "source_y_axis"):
        axis = np.asarray(source_package[axis_name])
        if axis.shape != (n,):
            raise ValueError(
                f"S1 {axis_name} must have shape ({n},), got {axis.shape}."
            )
    bbox = tuple(float(v) for v in source_package["source_bbox"])
    if len(bbox) != 4 or not np.all(np.isfinite(bbox)):
        raise ValueError("Source source_bbox must contain four finite values.")
    if not (bbox[0] < bbox[1] and bbox[2] < bbox[3]):
        raise ValueError("Source source_bbox must satisfy xmin < xmax and ymin < ymax.")
    if not _is_square_bbox(bbox):
        raise ValueError(
            "Source source_bbox is rectangular; regenerate S1 with a square "
            "pixelized source bbox."
        )

    scale_map = source_package.get("scale_map")
    if scale_map is None:
        scale_map = np.asarray(_make_source_scale(source_package), dtype=np.float32)
        source_package["scale_map"] = scale_map
    else:
        scale_map = np.asarray(scale_map, dtype=np.float32)
        if scale_map.shape != (n * n,):
            raise ValueError(
                f"Source scale_map must have shape ({n * n},), got {scale_map.shape}."
            )
        if not np.all(np.isfinite(scale_map) & (scale_map > 0.0)):
            raise ValueError("Source scale_map values must be finite and positive.")
        source_package["scale_map"] = scale_map
    return source_package


def _fixed_source_kwargs(source_package):
    source_package = _validate_source_package(source_package)
    return dict(
        fixed_source_bbox=tuple(float(v) for v in source_package["source_bbox"]),
        fixed_reg_template=jnp.asarray(source_package["source_pixels"], dtype=jnp.float32),
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


def _require_converged_source(likelihood, result, context):
    """Reject failed reconstructions before exporting a template or figure."""
    info = result.solver_info
    if (bool(np.asarray(info.converged))
            and np.all(np.isfinite(np.asarray(result.source_pixels)))
            and np.all(np.isfinite(np.asarray(result.model_image)))):
        return
    metric = (float(info.residual_norm) if hasattr(info, "residual_norm")
              else float(info.convergence_metric))
    raise RuntimeError(
        f"{likelihood.solver_type.upper()} failed while {context} "
        f"(convergence metric={metric:.4e}, n_iter={int(info.n_iter)})."
    )


def _solve_pixel_source_for_package(likelihood, medians, param_names):
    """Solve source pixels for the current pixelized likelihood configuration."""
    q50 = [medians[n] for n in param_names]
    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.array(q50))
        result = likelihood.evaluate()
        _require_converged_source(likelihood, result, "solving the stage source template")
        source_pixels = result.source_pixels
        xmin, xmax, ymin, ymax = result.source_bbox
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
# Stage A -- SIE + shear + three-MGE lens light + MGE source light
# ------------------------------------------------------------------ #
def build_stage_a_likelihood(image_data, noise_map, psf_kernel, mask=None):
    sie = SIE(
        theta_E=ParamU("theta_E", 0.45, prior_type="uniform",
                       prior_settings=[0.2, 1.0], limits=[0.0, 5.0]),
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

    # Three-group MGE lens light, with independent geometry for each group.
    cx_l1 = ParamU("center_x_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.06], limits=[-1.0, 1.0])
    cy_l1 = ParamU("center_y_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.06], limits=[-1.0, 1.0])
    e1_l1 = ParamU("e1_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_l1 = ParamU("e2_lens_1", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])

    cx_l2 = ParamU("center_x_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.06], limits=[-1.0, 1.0])
    cy_l2 = ParamU("center_y_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.06], limits=[-1.0, 1.0])
    e1_l2 = ParamU("e1_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_l2 = ParamU("e2_lens_2", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])

    cx_l3 = ParamU("center_x_lens_3", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.06], limits=[-1.0, 1.0])
    cy_l3 = ParamU("center_y_lens_3", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.06], limits=[-1.0, 1.0])
    e1_l3 = ParamU("e1_lens_3", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    e2_l3 = ParamU("e2_lens_3", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])

    sigma_list_lens = generate_radial_basis_knots(
        dpix=DPIX, n_sigmas=N_GAUSSIANS_LENS,
        log_rmin=-2.0, log_rmax=np.log10(MASK_RADIUS), mode="mge"
    )

    lens_gaussians = []
    for group_id, center_x, center_y, e1, e2 in (
        (1, cx_l1, cy_l1, e1_l1, e2_l1),
        (2, cx_l2, cy_l2, e1_l2, e2_l2),
        (3, cx_l3, cy_l3, e1_l3, e2_l3),
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

    for p in (cx_l1, cy_l1, e1_l1, e2_l1, cx_l2, cy_l2, e1_l2, e2_l2,
              cx_l3, cy_l3, e1_l3, e2_l3):
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
    print(" Stage A : SIE + shear + three-MGE lens light + MGE source light")
    print("=" * 60)
    t0 = time.time()
    likelihood = build_stage_a_likelihood(image_data, noise_map, psf_kernel, mask=circular_mask)
    stage = _run_sampler(
        likelihood, n_live=300, n_eff=2000, tag="stage-A", vectorized=True,
    )
    t1 = time.time()
    samples, weights, names, logz = stage.samples, stage.weights, stage.param_names, stage.log_z
    _print_summary("stage-A", samples, weights, names)
    print(f"[stage-A] time taken: {t1 - t0:.2f} seconds")

    medians = stage.medians()
    q50 = [medians[n] for n in names]

    likelihood.set_values(q50)
    fwd = likelihood.forward_model(
        use_linear=True, return_intensity=True, ret_each_plane=True,
        image_map=likelihood.image_data, noise_map=likelihood.noise_map,
    )
    lens_image_model = np.asarray(fwd[1])
    source_image_model = np.asarray(fwd[0])
    total_model = source_image_model + lens_image_model
    residual = (np.asarray(image_data) - total_model) / np.asarray(noise_map)
    unmask = np.asarray(likelihood.unmask, dtype=bool)
    chi2 = float(np.sum(np.square(residual[unmask]), dtype=np.float64))
    summary = dict(
        target="B1938+666", band="F115W", data="recentered",
        model="SIE + shear + three MGE lens-light groups + MGE source; NNLS",
        lens_mge_groups=3, gaussians_per_lens_group=N_GAUSSIANS_LENS,
        lens_mge_sigma_range_arcsec=[0.01, MASK_RADIUS],
        source_mge_groups=1, gaussians_per_source_group=N_GAUSSIANS_SRC,
        source_mge_sigma_range_arcsec=[0.01, 1.0],
        dpix_arcsec=DPIX, fitting_radius_arcsec=MASK_RADIUS, nsub=NSUB,
        n_live=300, requested_n_eff=2000, seed=1938,
        masked_pixels=int((~unmask).sum()), fitted_pixels=int(unmask.sum()),
        mask_convention="1 = excluded; contaminants OR outside circular aperture",
        log_z=logz, time_seconds=t1 - t0, medians=medians,
        quantiles={name: weighted_quantile(samples[:, i], weights,
                                           np.array([0.16, 0.5, 0.84])).tolist()
                   for i, name in enumerate(names)},
        diagnostic_parameters="marginal posterior medians",
        chi2=chi2, chi2_per_fitted_pixel=chi2 / int(unmask.sum()),
        normalized_residual_quantiles=np.quantile(residual[unmask], [0.16, 0.5, 0.84]).tolist(),
    )
    with (OUT_DIR / "stage_a_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    header = fits.getheader(DATA_DIR / "B1938_666_F115W_recentered_science.fits")
    fits.HDUList([
        fits.PrimaryHDU(total_model, header=header),
        fits.ImageHDU(lens_image_model, name="LENS_LIGHT"),
        fits.ImageHDU(source_image_model, name="LENSED_SOURCE"),
        fits.ImageHDU(residual, name="NORM_RESIDUAL"),
        fits.ImageHDU((~unmask).astype(np.uint8), name="FIT_MASK"),
    ]).writeto(OUT_DIR / "stage_a_model.fits", overwrite=True, checksum=True)
    print(f"[stage-A] median-model chi2 / fitted pixel = {summary['chi2_per_fitted_pixel']:.3f}")

    _dump_stage(
        "a", samples, weights, names, logz,
        extra=dict(
            medians=medians,
            lens_light_model=lens_image_model,
            time_taken=t1 - t0,
            fit_mask=np.asarray(~unmask),
            diagnostic_summary=summary,
        ),
        stage=stage,
    )

    _plot_stage_a_diagnostics(likelihood, q50)
    return stage, medians, lens_image_model


def _plot_stage_a_diagnostics(likelihood, theta):
    """Save clipped diagnostics and a zoom with the complete residual range."""
    fig, axes = plot_model_results(
        likelihood, jnp.asarray(theta),
        save_path=str(OUT_DIR / "stage_a_model_residual_clipped_5sigma.png"),
        title="B1938+666 F115W — Stage A : MGE lens+source",
        show_critical_lines=True, show_caustics=True,
    )
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.set_xlim(-MASK_RADIUS, MASK_RADIUS)
        ax.set_ylim(-MASK_RADIUS, MASK_RADIUS)
    grid = likelihood.sim_obj.sim_config
    central = ((np.abs(np.asarray(grid.xgrid)) < MASK_RADIUS)
               & (np.abs(np.asarray(grid.ygrid)) < MASK_RADIUS))
    central &= np.asarray(likelihood.unmask)
    vmax = float(np.quantile(np.asarray(likelihood.image_data)[central], 0.99))
    for ax in (axes[0, 0], axes[0, 1]):
        ax.images[0].set_clim(vmin=0.0, vmax=vmax)
    axes[1, 2].set_xlim(-0.15, 0.15)
    axes[1, 2].set_ylim(-0.15, 0.15)
    fig.suptitle("B1938+666 F115W — Stage A : central zoom (data/lens display clipped)")
    fig.savefig(OUT_DIR / "stage_a_model_zoom_residual_clipped_5sigma.png",
                bbox_inches="tight", dpi=150)

    # Use every finite, unmasked residual, with zero at the neutral color.
    from matplotlib.colors import TwoSlopeNorm
    residual_image = axes[1, 1].images[0]
    residuals = np.ma.masked_invalid(residual_image.get_array())
    residual_min, residual_max = float(residuals.min()), float(residuals.max())
    if residual_min < 0.0 < residual_max:
        residual_image.set_norm(TwoSlopeNorm(vmin=residual_min, vcenter=0.0,
                                             vmax=residual_max))
    else:
        residual_image.set_clim(vmin=residual_min, vmax=residual_max)
    residual_image.colorbar.set_ticks(np.unique(np.concatenate(
        ([residual_min, 0.0, residual_max],
         np.arange(np.ceil(residual_min / 20) * 20, residual_max, 20)),
    )))
    axes[1, 1].set_title("Normalised Residuals (full range)")
    fig.suptitle("B1938+666 F115W — Stage A : central zoom (full residual range)")
    fig.savefig(OUT_DIR / "stage_a_model_zoom_residual_unclipped.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[stage-A] residual colorbar full range: {residual_min:.3f} to {residual_max:.3f}")


# ------------------------------------------------------------------ #
# Stage B -- build an arc mask from stage-A residuals
# ------------------------------------------------------------------ #
def run_stage_b(image_data, noise_map, lens_light_model, circular_mask=None):
    print("\n" + "=" * 60)
    print(" Stage B : build arc feature mask from stage-A residuals")
    print("=" * 60)
    residual = image_data - lens_light_model
    snr_map = residual / noise_map
    feature_mask = _make_feature_mask(snr_map, circular_mask=circular_mask)
    n_in = int((~feature_mask).sum())
    print(f"[stage-B] arc pixels kept = {n_in} / {feature_mask.size}")
    print(f"[stage-B] central exclusion radius = {INNER_MASK_RADIUS:g} arcsec")

    fits.writeto(OUT_DIR / "feature_mask.fits",
                 feature_mask.astype(np.uint8), overwrite=True, checksum=True)

    ny_img, nx_img = image_data.shape
    extent = [-nx_img * DPIX / 2, nx_img * DPIX / 2, -ny_img * DPIX / 2, ny_img * DPIX / 2]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im0 = axes[0].imshow(residual, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("residual = image - stage-A lens light")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(snr_map, origin="lower", extent=extent, cmap="viridis",
                         vmin=-3, vmax=np.nanpercentile(snr_map, 99.5))
    axes[1].set_title("S/N map + fitting mask boundary")
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
    print(f"[pos-like] solved {img_positions.shape[0]} lensed image positions from stage-A medians")
    for p in img_positions:
        print(f"    ({p[0]:+.4f}, {p[1]:+.4f})")
    return dict(
        positions=img_positions.tolist(),
        threshold_arcsec=float(threshold_arcsec),
        min_log_like=float(min_log_like),
    )


# ------------------------------------------------------------------ #
# Helpers for Stage M1 / M2 / M3
# ------------------------------------------------------------------ #
def _sie_mass_from_stage(stage: StagePosterior):
    """SIE (+ shear) with Gaussian priors inherited from a stage posterior."""
    sie = SIE(
        theta_E=stage.gaussian("theta_E", model="SIE", attr="theta_E", limits=[0.0, 5.0]),
        e1=stage.gaussian("e1_mass", model="SIE", attr="e1", limits=[-1.0, 1.0]),
        e2=stage.gaussian("e2_mass", model="SIE", attr="e2", limits=[-1.0, 1.0]),
        center_x=stage.gaussian("center_x_mass", model="SIE", attr="center_x", limits=[-1.0, 1.0]),
        center_y=stage.gaussian("center_y_mass", model="SIE", attr="center_y", limits=[-1.0, 1.0]),
    )
    shear = Shear(
        gamma1=stage.gaussian("gamma1", model="Shear", attr="gamma1", limits=[-0.5, 0.5]),
        gamma2=stage.gaussian("gamma2", model="Shear", attr="gamma2", limits=[-0.5, 0.5]),
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


def _sie_mass_from_medians(medians: dict):
    """SIE (+ shear) with all parameters fixed at posterior medians."""
    required = (
        "theta_E", "e1_mass", "e2_mass",
        "center_x_mass", "center_y_mass", "gamma1", "gamma2",
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


# ------------------------------------------------------------------ #
# Shared comparison priors for stages M4 / M5
# ------------------------------------------------------------------ #
def _posterior_quantiles(stage: StagePosterior, name: str, quantiles):
    """Return weighted posterior quantiles of ``name`` in ``stage``."""
    if name not in stage.param_names:
        raise KeyError(
            f"'{name}' not found in the stage posterior. "
            f"Available: {stage.param_names}"
        )
    col = stage.param_names.index(name)
    values = np.asarray(stage.samples[:, col], dtype=np.float64)
    weights = np.asarray(stage.weights, dtype=np.float64)
    return [
        float(v) for v in weighted_quantile(values, weights, np.asarray(list(quantiles)))
    ]


def _subhalo_position_bounds(_image_shape, _dpix):
    """Return the image-plane specification of the subhalo position prior.

    ``prior_type`` is either ``"uniform"`` over ``[x_lo, x_hi] x [y_lo, y_hi]``
    or ``"truncated_gaussian"`` with the same bounds as truncation limits and
    ``(center, sigma)`` settings per axis. ``area_arcsec2`` is the *effective*
    search area used as the look-elsewhere bookkeeping: the box area for a
    uniform prior, ``2 pi sigma^2`` for a 2D Gaussian (the area of exact
    equivalence, i.e. the integral of the density divided by its peak).
    """
    mode = SUBHALO_POSITION_MODE
    if mode == "lensing_square":
        half = float(MASK_RADIUS)
        if not np.isfinite(half) or half <= 0:
            raise ValueError("MASK_RADIUS must be positive and finite.")
        return dict(
            mode=mode, prior_type="uniform",
            x_lo=-half, x_hi=half,
            y_lo=-half, y_hi=half,
            area_arcsec2=(2.0 * half) ** 2,
        )
    if mode == "gaussian_local":
        centre_x, centre_y = (float(v) for v in SUBHALO_POSITION_CENTER)
        sigma = float(SUBHALO_POSITION_SIGMA)
        if sigma <= 0:
            raise ValueError("SUBHALO_POSITION_SIGMA must be positive.")
        trunc = float(SUBHALO_POSITION_TRUNCATE_SIGMA)
        return dict(
            mode=mode, prior_type="truncated_gaussian",
            center_x=centre_x, center_y=centre_y, sigma=sigma,
            truncate_sigma=trunc,
            x_lo=centre_x - trunc * sigma, x_hi=centre_x + trunc * sigma,
            y_lo=centre_y - trunc * sigma, y_hi=centre_y + trunc * sigma,
            area_arcsec2=float(2.0 * np.pi * sigma * sigma),
        )
    raise ValueError(
        f"Unknown SUBHALO_POSITION_MODE {mode!r} "
        "(use 'lensing_square' or 'gaussian_local')."
    )


def _smooth_mass_prior_bounds(stage_m3: StagePosterior, name: str):
    """Calculate one shared top-hat from M3 and the current global settings."""
    centre = float(stage_m3.median(name))
    half = (
        abs(centre) * SMOOTH_PRIOR_THETA_E_RELATIVE_HALF_WIDTH
        if name == "theta_E" else SMOOTH_PRIOR_ABSOLUTE_HALF_WIDTHS[name]
    )
    if not np.isfinite(half) or half <= 0:
        raise ValueError(f"Shared smooth prior for '{name}' has invalid half-width {half}.")
    hard_lo, hard_hi = SMOOTH_MASS_PRIOR_LIMITS[name]
    lo, hi = max(hard_lo, centre - half), min(hard_hi, centre + half)
    if not np.all(np.isfinite([lo, hi])) or hi <= lo:
        raise ValueError(f"Shared smooth prior for '{name}' collapsed to [{lo}, {hi}].")
    return float(lo), float(hi)


def _dynamic_param(name, value, prior_type, prior_settings, limits) -> ParamU:
    """Return a dynamic ``ParamU`` with the given prior type and settings."""
    param = ParamU(
        name,
        float(value),
        prior_type=prior_type,
        prior_settings=[float(v) for v in prior_settings],
        limits=[float(v) for v in limits],
    )
    param.to_dynamic()
    return param


def _uniform_dynamic_param(name, value, lo, hi) -> ParamU:
    """Return a dynamic ``ParamU`` with a uniform prior on ``[lo, hi]``."""
    return _dynamic_param(name, value, "uniform", [lo, hi], [lo, hi])


def _smooth_mass_for_comparison(stage_m3: StagePosterior):
    """Build EPL and shear using the current shared-prior settings."""
    def mass_param(name):
        lo, hi = _smooth_mass_prior_bounds(stage_m3, name)
        return _uniform_dynamic_param(name, stage_m3.median(name), lo, hi)

    epl = EPL(
        theta_E=mass_param("theta_E"),
        gamma=mass_param("gamma"),
        e1=mass_param("e1_mass"),
        e2=mass_param("e2_mass"),
        center_x=mass_param("center_x_mass"),
        center_y=mass_param("center_y_mass"),
    )
    shear = Shear(gamma1=mass_param("gamma1"), gamma2=mass_param("gamma2"))
    return epl, shear


def _source_reg_for_comparison(stage_m2: StagePosterior):
    """Build the dynamic log-uniform strength and fixed regularization shape."""
    center = float(stage_m2.median("log_lambda_reg"))
    half = LAMBDA_PRIOR_HALF_WIDTH_DEX * float(np.log(10.0))
    log_lam = _uniform_dynamic_param(
        "log_lambda_reg", center, center - half, center + half,
    )
    rho = ParamU("adaptive_reg_rho", float(stage_m2.median("adaptive_reg_rho")))
    rho.to_static()
    return log_lam, rho


def _pseudo_jaffe_subhalo(theta_E_host: ParamU, image_shape):
    """Vegetti-style Pseudo-Jaffe with a live Einstein-radius tidal approximation."""
    b_lo, b_hi = SUBHALO_B_RANGE
    pos = _subhalo_position_bounds(image_shape, DPIX)
    b_subhalo = _uniform_dynamic_param(
        "b_subhalo", 0.5 * (b_lo + b_hi), b_lo, b_hi,
    )
    if pos.get("prior_type") == "truncated_gaussian":
        def pos_param(name, centre, lo, hi):
            return _dynamic_param(
                name, centre, "truncated_gaussian",
                [centre, pos["sigma"]], [lo, hi],
            )
    else:
        def pos_param(name, centre, lo, hi):
            return _uniform_dynamic_param(name, 0.5 * (lo + hi), lo, hi)

    center_x_subhalo = pos_param(
        "center_x_subhalo", pos.get("center_x", 0.0), pos["x_lo"], pos["x_hi"],
    )
    center_y_subhalo = pos_param(
        "center_y_subhalo", pos.get("center_y", 0.0), pos["y_lo"], pos["y_hi"],
    )
    return PseudoJaffeSubhalo(
        b_sub=b_subhalo,
        theta_E_host=theta_E_host,
        center_x=center_x_subhalo,
        center_y=center_y_subhalo,
    )


def _prior_edge_report(tag: str, stage: StagePosterior,
                       margin_fraction: float = PRIOR_EDGE_MARGIN_FRACTION) -> dict:
    """Report whether the posterior is truncated by the shared top-hats.

    Two diagnostics per shared smooth-mass parameter:

    * ``edge_weight`` -- posterior weight inside the outer ``margin_fraction``
      of the prior. A posterior concentrated well inside the prior has weight
      near zero in these edge windows; weight grows when samples accumulate
      near either bound. This is context, not a flag on its own. (Observed: a
      run whose posterior sat 0.7 posterior-sigma from the edge reported 79%.)
    * ``distance in posterior sigmas`` -- how far the posterior median sits from
      each prior edge, in units of the posterior's own 16-84% half-width. This
      is the scale-free flag: fewer than ``PRIOR_EDGE_SIGMA`` sigmas means the
      posterior is genuinely cut by the prior.

    An edge-hugging posterior means the shared prior is narrower than the data
    require. M4 and M5 must then be re-run together with a wider *shared* prior;
    widening only the subhalo model would break the controlled comparison.
    """
    specs = stage.prior_specs
    if not specs:
        print(f"[{tag}] prior-edge diagnostic skipped (no prior specs available).")
        return {}

    weights = np.asarray(stage.weights, dtype=np.float64)
    weights = weights / weights.sum()
    report: dict = {}
    print(f"\n[{tag}] shared smooth-mass prior-edge diagnostic "
          f"(flag = posterior median within {PRIOR_EDGE_SIGMA:g} posterior sigma "
          f"of a prior edge):")
    for spec in specs:
        if spec.name not in SMOOTH_MASS_PRIOR_LIMITS or spec.prior_type != "uniform":
            continue
        col = stage.param_names.index(spec.name)
        values = np.asarray(stage.samples[:, col], dtype=np.float64)
        lo, hi = float(spec.settings[0]), float(spec.settings[1])
        q16, q50, q84 = (
            float(v) for v in weighted_quantile(values, weights, [0.16, 0.5, 0.84])
        )
        sigma_post = 0.5 * (q84 - q16)
        margin = margin_fraction * (hi - lo)
        w_lo = float(weights[values <= lo + margin].sum())
        w_hi = float(weights[values >= hi - margin].sum())
        edge_weight = max(w_lo, w_hi)
        if sigma_post > 0:
            n_sigma_lo = (q50 - lo) / sigma_post
            n_sigma_hi = (hi - q50) / sigma_post
            n_sigma = min(n_sigma_lo, n_sigma_hi)
            flagged = n_sigma < PRIOR_EDGE_SIGMA
        else:
            # Degenerate posterior: no internal scale to compare against, so fall
            # back to the weight-based rule.
            n_sigma = float("inf") if lo < q50 < hi else 0.0
            flagged = edge_weight > PRIOR_EDGE_WARN_WEIGHT
        print(f"    {spec.name:18s} prior=[{lo:+.4f}, {hi:+.4f}]  "
              f"q16/q50/q84={q16:+.4f}/{q50:+.4f}/{q84:+.4f}  "
              f"edge weight={edge_weight:6.2%}  "
              f"median={n_sigma:6.2f} sigma_post from an edge"
              f"{'  <-- PRIOR EDGE' if flagged else ''}")
        report[spec.name] = dict(
            lo=lo, hi=hi, q16=q16, q50=q50, q84=q84,
            edge_weight=edge_weight,
            sigma_post=float(sigma_post),
            n_sigma_to_edge=float(n_sigma),
            flagged=bool(flagged),
        )
    return report


def _subhalo_strength_report(tag: str, stage: StagePosterior) -> dict:
    """Summarize the physical zero-strength limit separately from prior edges."""
    if "b_subhalo" not in stage.param_names or not stage.prior_specs:
        return {}
    spec = next(spec for spec in stage.prior_specs if spec.name == "b_subhalo")
    lo, hi = (float(v) for v in spec.settings[:2])
    values = np.asarray(stage.samples[:, stage.param_names.index("b_subhalo")], dtype=np.float64)
    weights = np.asarray(stage.weights, dtype=np.float64)
    weights = weights / weights.sum()
    q16, q50, q84 = (
        float(v) for v in weighted_quantile(values, weights, [0.16, 0.5, 0.84])
    )
    near_zero_limit = lo + 0.1 * (hi - lo)
    near_zero_weight = float(weights[values <= near_zero_limit].sum())
    print(f"\n[{tag}] subhalo strength (physical zero limit):")
    print(f"    b_subhalo q16/q50/q84={q16:+.4f}/{q50:+.4f}/{q84:+.4f}")
    print(f"    posterior weight in lowest 10% of its prior "
          f"(b_subhalo <= {near_zero_limit:.4f}) = {near_zero_weight:.2%}")
    if lo == 0.0:
        print("    b_subhalo=0 means no subhalo; proximity to zero is not a shared-prior edge.")
    else:
        print("    This prior excludes b_subhalo=0; its lower end is not the no-subhalo limit.")
    return dict(q16=q16, q50=q50, q84=q84,
                near_zero_limit=near_zero_limit, near_zero_weight=near_zero_weight)


def _subhalo_mass_report(stage: StagePosterior) -> dict:
    """Convert every joint M5 sample to a Vegetti-style Pseudo-Jaffe mass.

    The sampled strengths are angles in arcsec. Both b_subhalo and the derived
    tidal radius must be multiplied by D_l before using the physical critical
    surface density in Msun/kpc^2. The tidal relation here uses the host
    Einstein radius, as implemented by PseudoJaffeSubhalo.
    """
    b_sub = np.asarray(
        stage.samples[:, stage.param_names.index("b_subhalo")], dtype=np.float64
    )
    theta_e = np.asarray(
        stage.samples[:, stage.param_names.index("theta_E")], dtype=np.float64
    )
    if np.any(b_sub < 0) or np.any(theta_e <= 0):
        raise ValueError("Pseudo-Jaffe mass requires b_subhalo >= 0 and theta_E > 0.")

    cosmology = FlatLambdaCDM(
        H0=SUBHALO_MASS_H0 * u.km / u.s / u.Mpc,
        Om0=SUBHALO_MASS_OMEGA_M,
        Tcmb0=0 * u.K,
    )
    d_l = cosmology.angular_diameter_distance(SUBHALO_MASS_Z_LENS)
    d_s = cosmology.angular_diameter_distance(SUBHALO_MASS_Z_SOURCE)
    d_ls = cosmology.angular_diameter_distance_z1z2(
        SUBHALO_MASS_Z_LENS, SUBHALO_MASS_Z_SOURCE
    )
    sigma_crit = (
        astro_constants.c**2 / (4 * np.pi * astro_constants.G)
        * d_s / (d_l * d_ls)
    ).to_value(u.Msun / u.kpc**2)
    kpc_per_arcsec = d_l.to_value(u.kpc) * u.arcsec.to(u.rad)

    tidal_radius_arcsec = np.sqrt(b_sub * theta_e)
    mass_msun = (
        np.pi * sigma_crit * kpc_per_arcsec**2 * b_sub * tidal_radius_arcsec
    )

    def quantiles(values):
        q16, q50, q84 = weighted_quantile(values, stage.weights, [0.16, 0.5, 0.84])
        return dict(q16=float(q16), q50=float(q50), q84=float(q84))

    return dict(
        mass_msun=quantiles(mass_msun),
        tidal_radius_arcsec=quantiles(tidal_radius_arcsec),
        tidal_radius_kpc=quantiles(tidal_radius_arcsec * kpc_per_arcsec),
        cosmology=dict(
            H0_km_s_Mpc=SUBHALO_MASS_H0,
            omega_m=SUBHALO_MASS_OMEGA_M,
            z_lens=SUBHALO_MASS_Z_LENS,
            z_source=SUBHALO_MASS_Z_SOURCE,
        ),
        sigma_crit_msun_kpc2=float(sigma_crit),
        kpc_per_arcsec=float(kpc_per_arcsec),
        tidal_relation="r_t = sqrt(b_subhalo * theta_E_host)",
        mass_definition="M_sub = pi * (D_l * r_t) * (D_l * b_subhalo) * Sigma_crit",
    )


def _print_subhalo_mass_report(tag: str, report: dict) -> None:
    mass = report["mass_msun"]
    radius = report["tidal_radius_kpc"]
    print(f"[{tag}] Pseudo-Jaffe M_sub = "
          f"{mass['q50']/1e9:.3f} (-{(mass['q50']-mass['q16'])/1e9:.3f}, "
          f"+{(mass['q84']-mass['q50'])/1e9:.3f}) x 10^9 Msun (68% posterior)")
    print(f"[{tag}] tidal radius = "
          f"{radius['q50']:.3f} (-{radius['q50']-radius['q16']:.3f}, "
          f"+{radius['q84']-radius['q50']:.3f}) kpc")


def _refresh_cached_boundary_reports(stage_id: str, payload: dict,
                                     stage: StagePosterior) -> None:
    """Update stale boundary summaries in a matching M4/M5 posterior cache."""
    if not stage.prior_specs:
        return
    tag = f"stage-{stage_id.upper()}"
    reports = dict(
        prior_edge_report=_prior_edge_report(tag, stage),
        subhalo_strength_report=(
            _subhalo_strength_report(tag, stage) if stage_id == "m5" else None
        ),
    )
    if stage_id == "m5":
        reports["subhalo_mass_report"] = _subhalo_mass_report(stage)
    extra = payload["extra"]
    if any(extra.get(name) != value for name, value in reports.items()):
        extra.update(reports)
        _dump_stage(stage_id, None, None, None, None, extra=extra, stage=stage)


def _solution_drift_report(tag: str, stage: StagePosterior, stage_m3: StagePosterior,
                           warn_sigma=None) -> dict:
    """Report how far this arm's smooth solution sits from the stage-M3 solution.

    A large drift may indicate a trade between the subhalo and the main lens.
    Historical SIS experiments on J0946 found a ladder of local optima as the
    shared smooth prior widened; those results do not establish the behavior of
    the current Pseudo-Jaffe model. Read this alongside the prior-edge report,
    then test the Pseudo-Jaffe evidence at multiple shared prior widths before
    drawing a conclusion about prior-width dependence.
    """
    if warn_sigma is None:
        warn_sigma = SOLUTION_DRIFT_WARN_SIGMA
    print(f"\n[{tag}] smooth-solution drift relative to the stage-M3 solution "
          f"(flag = |drift| > {warn_sigma:g} sigma_M3):")
    report: dict = {}
    for name in SMOOTH_MASS_PRIOR_LIMITS:
        if name not in stage.param_names:
            continue
        value = float(stage.median(name))
        sigma = float(stage_m3.std(name))
        center = float(stage_m3.median(name))
        drift = (value - center) / sigma if sigma > 0 else float("inf")
        flagged = abs(drift) > warn_sigma
        print(f"    {name:18s} M3 = {center:+.5f}  "
              f"this arm = {value:+.5f}  drift = {drift:+7.2f} sigma_M3"
              f"{'  <-- SOLUTION DRIFT' if flagged else ''}")
        report[name] = dict(m3=center, value=value,
                            drift_sigma=float(drift), flagged=bool(flagged))
    return report


# ------------------------------------------------------------------ #
# Stage M1 — SIE + shear + non-adaptive pixelized source  (fit mass)
# ------------------------------------------------------------------ #
def build_stage_m1_likelihood(
    lens_subtracted_image, noise_map, psf_kernel, feature_mask,
    stage_a: StagePosterior, position_likelihood,
    circular_mask=None,
):
    """Build M1: SIE+shear free, with a broad log-uniform source lambda prior."""
    sie, shear = _sie_mass_from_stage(stage_a)
    source_bbox = _stage_a_source_bbox(stage_a)
    lambda_min, lambda_max = M1_LAMBDA_REG_PRIOR_RANGE
    if not (0 < lambda_min < lambda_max < np.inf):
        raise ValueError("M1_LAMBDA_REG_PRIOR_RANGE must be finite, positive and ordered.")
    log_bounds = np.log([lambda_min, lambda_max]).tolist()
    log_lam = ParamU(
        "log_lambda_reg",
        float(np.mean(log_bounds)),
        prior_type="uniform",
        prior_settings=log_bounds,
        limits=log_bounds,
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
    return _make_pixel_likelihood(
        image_data=lens_subtracted_image,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        solver_type=SOLVER_TYPE,
        source_bbox_padding=SOURCE_BBOX_PADDING,
        fixed_source_bbox=source_bbox,
    )


def run_stage_m1(image_data, noise_map, psf_kernel, feature_mask,
                  lens_light_model, stage_a: StagePosterior,
                  position_likelihood, circular_mask=None):
    """Stage M1: jointly fit SIE+shear and the source regularization strength."""
    print("\n" + "=" * 60)
    print(" Stage M1 : SIE + shear + non-adaptive pix source (mass fit)")
    print("=" * 60)
    print(
        f"[stage-M1] lambda_reg log-uniform prior: "
        f"[{M1_LAMBDA_REG_PRIOR_RANGE[0]:.1e}, {M1_LAMBDA_REG_PRIOR_RANGE[1]:.1e}]"
    )
    t0 = time.time()

    lens_subtracted = image_data - lens_light_model
    likelihood = build_stage_m1_likelihood(
        lens_subtracted, noise_map, psf_kernel, feature_mask,
        stage_a, position_likelihood,
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
    s1_package["scale_map"] = np.asarray(_make_source_scale(s1_package), dtype=np.float32)
    _validate_source_package(s1_package)

    _dump_stage(
        "m1", samples, weights, names, logz,
        extra=dict(
            medians=medians,
            m1_mass_model="SIE+Shear",
            lambda_prior_type="log_uniform",
            lambda_prior_range=list(M1_LAMBDA_REG_PRIOR_RANGE),
            log_lambda_prior_limits=np.log(M1_LAMBDA_REG_PRIOR_RANGE).tolist(),
            s1=s1_package,
            time_taken=t1 - t0,
        ),
        stage=stage,
    )

    try:
        _plot_pix_stage("stage-M1", likelihood, medians, names,
                        str(OUT_DIR / "stage_m1_model.png"))
    except Exception as err:
        print(f"[stage-M1] pix_stage plot failed (non-fatal): {err}")

    return stage, medians, s1_package


# ------------------------------------------------------------------ #
# Stage M2 — fixed SIE + shear + adaptive pixelized source  (fit source reg)
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

        rho_val = float(np.asarray(_source_param_value(
            likelihood.phys_model.source_light[0].adaptive_reg_rho
        )))
        result = likelihood.evaluate()
        _require_converged_source(likelihood, result, f"plotting {tag}")
        source_pixels = result.source_pixels
        xmin, xmax, ymin, ymax = result.source_bbox
        scale = likelihood._get_reg_scale()

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

    model_image = np.asarray(result.model_image)
    resid_norm = (image_data - model_image) / noise_map
    chi2_map = np.where(mask, np.nan, resid_norm ** 2)
    chi2 = float(np.nansum(chi2_map))

    n = likelihood.phys_model.source_light[0].n
    src_img = np.array(source_pixels).reshape(n, n)

    scale_img = np.array(scale).reshape(n, n) if scale is not None else np.ones((n, n))
    rho = rho_val

    npix = image_data.shape[0]
    ext_i = [-npix * DPIX / 2, npix * DPIX / 2, -npix * DPIX / 2, npix * DPIX / 2]
    ext_s = [float(xmin), float(xmax), float(ymin), float(ymax)]
    vmax  = np.nanpercentile(image_data[~mask], 99.5)

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))
    rows_unmasked, cols_unmasked = np.where(~mask)
    pad = 3
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
                               f"Norm. residual\nχ²={chi2:.1f}",
                               dict(vmin=-3, vmax=3, cmap="RdBu_r")),
    ]:
        im = ax.imshow(img, origin="lower", extent=ext_i, **kw)
        if ax == axes[0] and has_pos_penalty:
            ax.plot(pos_px_np, pos_py_np, 'rx', markersize=8, label='Marked pos')
            ax.legend(loc='upper right', fontsize=8)
        if ax == axes[1] and tag.lower() == "stage-m5":
            ax.plot(
                medians["center_x_subhalo"], medians["center_y_subhalo"],
                marker="*", color="black", markersize=14,
                linestyle="none", label="Pseudo-Jaffe subhalo",
            )
            ax.legend(loc="upper right", fontsize=8)
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

    im4 = axes[4].imshow(scale_img, origin="lower", extent=ext_s,
                         cmap="plasma", vmin=1.0, vmax=max(1.0, float(np.nanmax(scale_img))))
    axes[4].set_title(f"Reg precision scale\n(rho={rho:.2f})", fontsize=11)
    axes[4].set_xlabel("arcsec")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04,
                 label=r"$\lambda_i / \lambda_{\rm global}$")

    axes[0].set_ylabel("arcsec")
    for ax in axes[:3]:
        ax.set_xlim(*xlim_unmasked)
        ax.set_ylim(*ylim_unmasked)

    lbl = "  ".join(f"{n}={medians[n]:+.4f}" for n in
                    ("theta_E", "gamma", "e1_mass", "e2_mass") if n in medians)
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
    lens_subtracted_image, noise_map, psf_kernel, feature_mask,
    medians_m1: dict, position_likelihood, s1_package, circular_mask=None,
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

    pix_src = PixelizedSourceModel(n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        adaptive_reg_rho=rho,
    )
    phys = PhysicalModel(
        lens_mass=[sie, shear],
        source_light=[pix_src],
        lens_light=[],
    )
    combined_mask = feature_mask
    if circular_mask is not None:
        combined_mask = combined_mask | circular_mask
    return _make_pixel_likelihood(
        image_data=lens_subtracted_image,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        solver_type=SOLVER_TYPE,
        source_bbox_padding=SOURCE_BBOX_PADDING,
        **_fixed_source_kwargs(s1_package),
    )


def run_stage_m2(image_data, noise_map, psf_kernel, feature_mask,
                  lens_light_model, medians_m1, position_likelihood,
                  s1_package, circular_mask=None):
    """Stage M2: fit adaptive source-regularization hyperparameters."""
    print("\n" + "=" * 60)
    print(" Stage M2 : fixed SIE + shear + adaptive pix source (fit λ, rho)")
    print("=" * 60)
    t0 = time.time()
    lens_subtracted = image_data - lens_light_model
    likelihood = build_stage_m2_likelihood(
        lens_subtracted, noise_map, psf_kernel, feature_mask,
        medians_m1, position_likelihood, s1_package,
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
        m2_mass_model="fixed-SIE+Shear",
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
    lens_subtracted_image, noise_map, psf_kernel, feature_mask,
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
    return _make_pixel_likelihood(
        image_data=lens_subtracted_image,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=position_likelihood,
        solver_type=SOLVER_TYPE,
        source_bbox_padding=SOURCE_BBOX_PADDING,
        **_fixed_source_kwargs(s1_package),
    )


def run_stage_m3(image_data, noise_map, psf_kernel, feature_mask,
                  lens_light_model, stage_m1: StagePosterior,
                  position_likelihood, reg_hyperparams_fixed: dict,
                  s1_package, circular_mask=None):
    """Stage M3: final EPL+shear sampling with source-reg fixed from M2."""
    print("\n" + "=" * 60)
    print(" Stage M3 : EPL + shear + adaptive pix source (final mass fit)")
    print("=" * 60)
    print(f"[stage-M3] fixed source reg: {_format_reg_hyperparams(reg_hyperparams_fixed)}")
    t0 = time.time()
    lens_subtracted = image_data - lens_light_model
    likelihood = build_stage_m3_likelihood(
        lens_subtracted, noise_map, psf_kernel, feature_mask,
        stage_m1, position_likelihood, reg_hyperparams_fixed,
        s1_package, circular_mask=circular_mask,
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
# Stages M4 / M5 — smooth null model vs smooth + Pseudo-Jaffe subhalo
# ------------------------------------------------------------------ #
def build_comparison_likelihood(
    lens_subtracted_image, noise_map, psf_kernel, feature_mask,
    stage_m2: StagePosterior, stage_m3: StagePosterior, position_likelihood, s1_package,
    circular_mask=None, with_subhalo: bool = False,
):
    """Build the M4 (``with_subhalo=False``) or M5 (``True``) likelihood.

    Both models use the same globals and stage-M2/M3 posteriors, so
    every shared ingredient — smooth-mass top-hats, the source-regularization
    strength prior, ``adaptive_reg_rho``, the fixed source grid/template/bbox,
    the mask, PSF, noise map and solver settings — is identical by construction.
    Only the subhalo component differs, which is what makes
    ``logZ(M5) - logZ(M4)`` a subhalo Bayes factor.
    """
    epl, shear = _smooth_mass_for_comparison(stage_m3)
    log_lam, rho = _source_reg_for_comparison(stage_m2)

    lens_mass = [epl, shear]
    if with_subhalo:
        lens_mass.append(
            _pseudo_jaffe_subhalo(epl.theta_E, lens_subtracted_image.shape)
        )

    pix_src = PixelizedSourceModel(
        n=NSRC,
        log_lambda_reg=log_lam,
        regularization_type=PIXEL_REGULARIZATION_TYPE,
        adaptive_reg_rho=rho,
    )
    phys = PhysicalModel(
        lens_mass=lens_mass,
        source_light=[pix_src],
        lens_light=[],
    )
    combined_mask = feature_mask
    if circular_mask is not None:
        combined_mask = combined_mask | circular_mask
    return _make_pixel_likelihood(
        image_data=lens_subtracted_image,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        nsub=NSUB_PIX,
        phys_model=phys,
        mask=combined_mask,
        position_likelihood=(
            position_likelihood if COMPARISON_USE_POSITION_LIKELIHOOD else None
        ),
        solver_type=SOLVER_TYPE,
        source_bbox_padding=SOURCE_BBOX_PADDING,
        **_fixed_source_kwargs(s1_package),
    )


def _run_comparison_stage(
    stage_id: str, title: str, with_subhalo: bool, n_live: int, n_eff: int,
    image_data, noise_map, psf_kernel, feature_mask, lens_light_model,
    stage_m2, stage_m3, position_likelihood, s1_package, circular_mask=None,
):
    """Sample one arm of the M4/M5 subhalo comparison."""
    tag = f"stage-{stage_id.upper()}"
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    print(f"[{tag}] role = " + (
        "subhalo model (smooth + Pseudo-Jaffe)" if with_subhalo else "null model (smooth only)"
    ))
    print(f"[{tag}] sampler: n_live={n_live}, n_eff={n_eff}")
    t0 = time.time()
    lens_subtracted = image_data - lens_light_model
    likelihood = build_comparison_likelihood(
        lens_subtracted, noise_map, psf_kernel, feature_mask,
        stage_m2, stage_m3, position_likelihood, s1_package,
        circular_mask=circular_mask, with_subhalo=with_subhalo,
    )
    stage = _run_sampler(
        likelihood, n_live=n_live, n_eff=n_eff, tag=tag, vectorized=True,
    )
    t1 = time.time()
    samples, weights = stage.samples, stage.weights
    names, logz = stage.param_names, stage.log_z
    _print_summary(tag, samples, weights, names)
    print(f"[{tag}] time taken: {t1 - t0:.2f} seconds")
    medians = stage.medians()
    edge_report = _prior_edge_report(tag, stage)
    strength_report = _subhalo_strength_report(tag, stage) if with_subhalo else None
    mass_report = _subhalo_mass_report(stage) if with_subhalo else None
    if mass_report is not None:
        _print_subhalo_mass_report(tag, mass_report)
    drift_report = _solution_drift_report(tag, stage, stage_m3)
    _dump_stage(
        stage_id, samples, weights, names, logz,
        extra=dict(
            medians=medians,
            pipeline_schema=_comparison_cache_schema(
                stage_m2, stage_m3, lens_subtracted, noise_map, psf_kernel,
                feature_mask, position_likelihood, s1_package,
                circular_mask=circular_mask,
            ),
            comparison_role="subhalo" if with_subhalo else "null",
            mass_model="EPL+Shear+PseudoJaffeSubhalo" if with_subhalo else "EPL+Shear",
            subhalo_position=(
                _subhalo_position_bounds(image_data.shape, DPIX) if with_subhalo else None
            ),
            prior_edge_report=edge_report,
            subhalo_strength_report=strength_report,
            subhalo_mass_report=mass_report,
            solution_drift_report=drift_report,
            position_likelihood_used=bool(COMPARISON_USE_POSITION_LIKELIHOOD),
            sampler_settings=dict(n_live=int(n_live), n_eff=int(n_eff)),
            time_taken=t1 - t0,
        ),
        stage=stage,
    )
    try:
        _plot_pix_stage(tag, likelihood, medians, names,
                        str(OUT_DIR / f"stage_{stage_id}_model.png"))
    except Exception as err:
        print(f"[{tag}] plotting failed (non-fatal): {err}")
    return stage, medians


def run_stage_m4(image_data, noise_map, psf_kernel, feature_mask,
                 lens_light_model, stage_m2: StagePosterior, stage_m3: StagePosterior,
                 position_likelihood, s1_package, circular_mask=None):
    """Stage M4: the smooth null model of the subhalo comparison."""
    return _run_comparison_stage(
        "m4", "Stage M4 : smooth EPL + shear + adaptive pix source (null model)",
        False, M4_N_LIVE, M4_N_EFF,
        image_data, noise_map, psf_kernel, feature_mask, lens_light_model,
        stage_m2, stage_m3, position_likelihood, s1_package,
        circular_mask=circular_mask,
    )


def run_stage_m5(image_data, noise_map, psf_kernel, feature_mask,
                 lens_light_model, stage_m2: StagePosterior, stage_m3: StagePosterior,
                 position_likelihood, s1_package, circular_mask=None):
    """Stage M5: add a Pseudo-Jaffe subhalo with free strength and position.

    The adopted approximate tidal radius is derived from ``b_subhalo`` and the
    sampled host Einstein radius ``theta_E``; it is not an additional inference
    parameter or an exact reproduction of the paper's host normalization.
    """
    return _run_comparison_stage(
        "m5", "Stage M5 : EPL + shear + Pseudo-Jaffe subhalo + adaptive pix source",
        True, M5_N_LIVE, M5_N_EFF,
        image_data, noise_map, psf_kernel, feature_mask, lens_light_model,
        stage_m2, stage_m3, position_likelihood, s1_package,
        circular_mask=circular_mask,
    )


def run_m5_from_cache(cache_dir: str, out_dir: str):
    """Run M5 alone, reusing the earlier stages and M4 null from a prior run."""
    global OUT_DIR
    cache_dir = Path(cache_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if cache_dir == out_dir:
        raise ValueError("M5 output must differ from the reference cache directory.")
    if (out_dir / "stage_m5.pkl").exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir / 'stage_m5.pkl'}")

    def cached(tag):
        with (cache_dir / f"stage_{tag}.pkl").open("rb") as f:
            return pickle.load(f)

    a, m1, m2, m3, m4, reference_m5 = (
        cached(tag) for tag in ("a", "m1", "m2", "m3", "m4", "m5")
    )
    if m4["extra"].get("pipeline_schema") != reference_m5["extra"].get("pipeline_schema"):
        raise ValueError("Reference M4 and M5 caches have different comparison schemas.")

    image_data, noise_map, psf_kernel, contamination_mask = _load_b1938_data()
    circular_mask = _make_circular_mask(image_data.shape, DPIX, MASK_RADIUS) | contamination_mask
    lens_light_model = a["extra"]["lens_light_model"]
    snr_map = (image_data - lens_light_model) / noise_map
    feature_mask = _make_feature_mask(snr_map, circular_mask=circular_mask)
    stage_m2, stage_m3 = _stage_from_payload(m2), _stage_from_payload(m3)
    s1_package = _validate_source_package(m1["extra"]["s1"])
    position_likelihood = _position_likelihood_from_stage_a(a["extra"]["medians"])

    # The null likelihood does not depend on the subhalo position prior. Verify
    # that the cached null used the same data and shared priors as this run.
    current_mode = SUBHALO_POSITION_MODE
    current_m4_counts = (M4_N_LIVE, M4_N_EFF)
    current_m5_counts = (M5_N_LIVE, M5_N_EFF)
    try:
        globals()["SUBHALO_POSITION_MODE"] = "gaussian_local"
        m4_settings = m4["extra"]["sampler_settings"]
        m5_settings = reference_m5["extra"]["sampler_settings"]
        globals()["M4_N_LIVE"] = int(m4_settings["n_live"])
        globals()["M4_N_EFF"] = int(m4_settings["n_eff"])
        globals()["M5_N_LIVE"] = int(m5_settings["n_live"])
        globals()["M5_N_EFF"] = int(m5_settings["n_eff"])
        reference_schema = _comparison_cache_schema(
            stage_m2, stage_m3, image_data - lens_light_model, noise_map,
            psf_kernel, feature_mask, position_likelihood, s1_package,
            circular_mask=circular_mask,
        )
    finally:
        globals()["SUBHALO_POSITION_MODE"] = current_mode
        globals()["M4_N_LIVE"], globals()["M4_N_EFF"] = current_m4_counts
        globals()["M5_N_LIVE"], globals()["M5_N_EFF"] = current_m5_counts
    null_schema_matches = reference_schema == m4["extra"].get("pipeline_schema")
    if not null_schema_matches:
        print("[M5-only] WARNING: the reference M4 cache has a different pipeline "
              "schema. Its logZ is not a valid Bayes-factor denominator for this run.")

    OUT_DIR = out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[M5-only] reference cache: {cache_dir}")
    print(f"[M5-only] output: {OUT_DIR}")
    print(f"[M5-only] position prior: {_subhalo_position_bounds(image_data.shape, DPIX)}")
    stage_m5, medians_m5 = run_stage_m5(
        image_data, noise_map, psf_kernel, feature_mask, lens_light_model,
        stage_m2, stage_m3, position_likelihood, s1_package,
        circular_mask=circular_mask,
    )
    mass_report = _subhalo_mass_report(stage_m5)
    with (OUT_DIR / "stage_m5_mass.json").open("w", encoding="utf-8") as f:
        json.dump(mass_report, f, indent=2)
        f.write("\n")
    reference = _stage_from_payload(reference_m5)
    position_names = ("center_x_subhalo", "center_y_subhalo")
    position_quantiles = {}
    for name in position_names:
        index = stage_m5.param_names.index(name)
        q16, q50, q84 = weighted_quantile(
            stage_m5.samples[:, index], stage_m5.weights, [0.16, 0.5, 0.84]
        )
        position_quantiles[name] = dict(q16=float(q16), q50=float(q50), q84=float(q84))
    reference_xy = np.array([reference.median(name) for name in position_names])
    sample_xy = stage_m5.samples[
        :, [stage_m5.param_names.index(name) for name in position_names]
    ]
    distance_from_reference = np.linalg.norm(sample_xy - reference_xy, axis=1)
    summary = dict(
        position_prior=_subhalo_position_bounds(image_data.shape, DPIX),
        sampler_settings=dict(n_live=M5_N_LIVE, n_eff=M5_N_EFF),
        reference_cache=str(cache_dir),
        reference_log_z_m4=float(m4["log_z"]),
        reference_log_z_m5=float(reference.log_z),
        log_z_m5=float(stage_m5.log_z),
        reference_m4_schema_matches=bool(null_schema_matches),
        delta_log_z_m5_m4=(
            float(stage_m5.log_z - m4["log_z"]) if null_schema_matches else None
        ),
        reference_medians={
            name: float(reference.median(name))
            for name in ("b_subhalo", "center_x_subhalo", "center_y_subhalo")
        },
        medians={
            name: float(medians_m5[name])
            for name in ("b_subhalo", "center_x_subhalo", "center_y_subhalo")
        },
        position_quantiles=position_quantiles,
        position_offset_from_reference_arcsec=float(np.linalg.norm(
            np.array([medians_m5[name] for name in position_names]) - reference_xy
        )),
        posterior_weight_within_0p1_arcsec_of_reference=float(
            stage_m5.weights[distance_from_reference < 0.1].sum()
        ),
        mass_msun=mass_report["mass_msun"],
    )
    with (OUT_DIR / "blind_search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"[M5-only] comparison saved to {OUT_DIR / 'blind_search_summary.json'}")
    return summary


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def main(skip_done: bool = False, out_dir: str | None = None):
    global OUT_DIR, RESUME_SAMPLERS
    RESUME_SAMPLERS = skip_done
    if out_dir is not None:
        requested_out_dir = Path(out_dir).expanduser()
        OUT_DIR = (
            requested_out_dir
            if requested_out_dir.is_absolute()
            else SCRIPT_DIR / requested_out_dir
        )

    image_data, noise_map, psf_kernel, contamination_mask = _load_b1938_data()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aperture_mask = _make_circular_mask(image_data.shape, DPIX, radius_arcsec=MASK_RADIUS)
    circular_mask = aperture_mask | contamination_mask
    print(f"[mask] contaminants: {int(contamination_mask.sum())}; "
          f"inside aperture: {int((contamination_mask & ~aperture_mask).sum())}; "
          f"kept: {int((~circular_mask).sum())} / {circular_mask.size}")
    fits.writeto(OUT_DIR / "stage_a_fit_mask.fits", circular_mask.astype(np.uint8),
                 header=fits.getheader(DATA_DIR / "B1938_666_F115W_mask.fits"),
                 overwrite=True, checksum=True)

    # ---- stage A ---------------------------------------------------- #
    time_a = 0.0
    if skip_done and (OUT_DIR / "stage_a.pkl").exists():
        print(f"[stage-A] loading cached {OUT_DIR}/stage_a.pkl")
        d = _load_stage("a")
        cached_mask = d.get("extra", {}).get("fit_mask")
        if cached_mask is None or not np.array_equal(cached_mask, circular_mask):
            raise ValueError("Cached stage A used a different mask; rerun without --skip-done")
        stage_a = _stage_from_payload(d)
        medians_a = d["extra"]["medians"]
        lens_light_model = d["extra"]["lens_light_model"]
        time_a = d["extra"].get("time_taken", 0.0)
    else:
        stage_a, medians_a, lens_light_model = run_stage_a(
            image_data, noise_map, psf_kernel, circular_mask=circular_mask,
        )
        time_a = _load_stage("a")["extra"].get("time_taken", 0.0)

    # ---- stage B ---------------------------------------------------- #
    feature_mask = run_stage_b(image_data, noise_map, lens_light_model, circular_mask=circular_mask)

    lens_subtracted = image_data - lens_light_model
    position_likelihood = _position_likelihood_from_stage_a(medians_a)
    source_bbox = _stage_a_source_bbox(stage_a)
    print(f"[stage-A] pixelized stages will use fixed source bbox {source_bbox}")

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
            s1_package = _validate_source_package(d["extra"]["s1"])
            time_m1 = d["extra"].get("time_taken", 0.0)
        except KeyError as err:
            print(f"[stage-M1] cached output has old/incomplete format ({err}); recomputing.")
            stage_m1, medians_m1, s1_package = run_stage_m1(
                image_data, noise_map, psf_kernel, feature_mask,
                lens_light_model, stage_a, position_likelihood,
                circular_mask=circular_mask,
            )
            names_m1 = stage_m1.param_names
            time_m1 = _load_stage("m1")["extra"].get("time_taken", 0.0)
    else:
        stage_m1, medians_m1, s1_package = run_stage_m1(
            image_data, noise_map, psf_kernel, feature_mask,
            lens_light_model, stage_a, position_likelihood,
            circular_mask=circular_mask,
        )
        names_m1 = stage_m1.param_names
        time_m1 = _load_stage("m1")["extra"].get("time_taken", 0.0)

    if s1_package is None:
        raise RuntimeError(
            "[stage-M1] Failed to determine S1 source package — "
            "stage_m1.pkl may be corrupted."
        )

    if (OUT_DIR / "stage_m1.pkl").exists() and not (OUT_DIR / "stage_m1_model.png").exists():
        print("[stage-M1] stage_m1_model.png missing — re-plotting")
        lkl_m1_replot = build_stage_m1_likelihood(
            lens_subtracted, noise_map, psf_kernel, feature_mask,
            stage_a, position_likelihood,
            circular_mask=circular_mask,
        )
        try:
            _plot_pix_stage("stage-M1", lkl_m1_replot, medians_m1, names_m1,
                            str(OUT_DIR / "stage_m1_model.png"))
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
            lens_light_model, medians_m1, position_likelihood,
            s1_package, circular_mask=circular_mask,
        )
        names_m2 = stage_m2.param_names
        time_m2 = _load_stage("m2")["extra"].get("time_taken", 0.0)

    if not (OUT_DIR / "stage_m2_model.png").exists():
        lkl_m2 = build_stage_m2_likelihood(
            lens_subtracted, noise_map, psf_kernel, feature_mask,
            medians_m1, position_likelihood, s1_package,
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
            lens_light_model, stage_m1, position_likelihood,
            reg_hyperparams_m2, s1_package, circular_mask=circular_mask,
        )
        names_m3 = stage_m3.param_names
        time_m3 = _load_stage("m3")["extra"].get("time_taken", 0.0)

    if not (OUT_DIR / "stage_m3_model.png").exists():
        lkl_m3 = build_stage_m3_likelihood(
            lens_subtracted, noise_map, psf_kernel, feature_mask,
            stage_m1, position_likelihood, reg_hyperparams_m2, s1_package,
            circular_mask=circular_mask,
        )
        try:
            _plot_pix_stage("stage-M3", lkl_m3, medians_m3, names_m3,
                            str(OUT_DIR / "stage_m3_model.png"))
        except Exception as err:
            print(f"[stage-M3] plotting failed (non-fatal): {err}")

    # Both arms read the same global settings; prior_specs are saved by the sampler.
    comparison_schema = _comparison_cache_schema(
        stage_m2, stage_m3, lens_subtracted, noise_map, psf_kernel,
        feature_mask, position_likelihood, s1_package,
        circular_mask=circular_mask,
    )
    print("\n[M4/M5] Vegetti-style Pseudo-Jaffe with the adopted Einstein-radius "
          "approximation rt = sqrt(b_sub * theta_E_host), not an exact "
          "reproduction of Vegetti et al. (2010).")
    print(f"[M4/M5] subhalo position mode = {SUBHALO_POSITION_MODE}")
    if COMPARISON_USE_POSITION_LIKELIHOOD:
        print("[M4/M5] Position penalty is active in both arms; its positions come "
              "from the smooth stage-A model. Set "
              "COMPARISON_USE_POSITION_LIKELIHOOD=False for the robustness check.")

    # ---- stage M4 (smooth null model) -------------------------------- #
    time_m4 = 0.0
    stage_m4 = None
    medians_m4 = None
    names_m4 = None
    cached_m4 = False
    if skip_done and (OUT_DIR / "stage_m4.pkl").exists():
        d = _load_stage("m4")
        if _comparison_cache_matches(d, comparison_schema, expected_role="null"):
            print(f"[stage-M4] loading cached {OUT_DIR}/stage_m4.pkl")
            stage_m4 = _stage_from_payload(d)
            names_m4 = stage_m4.param_names
            medians_m4 = d["extra"]["medians"]
            time_m4 = d["extra"].get("time_taken", 0.0)
            cached_m4 = True
            _refresh_cached_boundary_reports("m4", d, stage_m4)
        else:
            print(
                "[stage-M4] cached stage_m4.pkl does not match the exact null "
                f"model schema (expected {comparison_schema}); recomputing."
            )
    if stage_m4 is None:
        stage_m4, medians_m4 = run_stage_m4(
            image_data, noise_map, psf_kernel, feature_mask,
            lens_light_model, stage_m2, stage_m3, position_likelihood,
            s1_package, circular_mask=circular_mask,
        )
        names_m4 = stage_m4.param_names
        time_m4 = _load_stage("m4")["extra"].get("time_taken", 0.0)

    if medians_m4 is None or names_m4 is None:
        raise RuntimeError("[stage-M4] posterior medians or parameter names missing.")

    if cached_m4 and not (OUT_DIR / "stage_m4_model.png").exists():
        print("[stage-M4] stage_m4_model.png missing — re-plotting")
        lkl_m4 = build_comparison_likelihood(
            lens_subtracted, noise_map, psf_kernel, feature_mask,
            stage_m2, stage_m3, position_likelihood, s1_package,
            circular_mask=circular_mask, with_subhalo=False,
        )
        try:
            _plot_pix_stage(
                "stage-M4", lkl_m4, medians_m4, names_m4,
                str(OUT_DIR / "stage_m4_model.png"),
            )
        except Exception as err:
            print(f"[stage-M4] re-plotting failed (non-fatal): {err}")

    # ---- stage M5 (same model + Pseudo-Jaffe subhalo) ------------------------- #
    time_m5 = 0.0
    stage_m5 = None
    medians_m5 = None
    names_m5 = None
    cached_m5 = False
    if skip_done and (OUT_DIR / "stage_m5.pkl").exists():
        d = _load_stage("m5")
        if _comparison_cache_matches(d, comparison_schema, expected_role="subhalo"):
            print(f"[stage-M5] loading cached {OUT_DIR}/stage_m5.pkl")
            stage_m5 = _stage_from_payload(d)
            names_m5 = stage_m5.param_names
            medians_m5 = d["extra"]["medians"]
            time_m5 = d["extra"].get("time_taken", 0.0)
            cached_m5 = True
            _refresh_cached_boundary_reports("m5", d, stage_m5)
        else:
            print(
                "[stage-M5] cached stage_m5.pkl does not match the exact subhalo "
                f"model schema (expected {comparison_schema}); recomputing."
            )
    if stage_m5 is None:
        stage_m5, medians_m5 = run_stage_m5(
            image_data, noise_map, psf_kernel, feature_mask,
            lens_light_model, stage_m2, stage_m3, position_likelihood,
            s1_package, circular_mask=circular_mask,
        )
        names_m5 = stage_m5.param_names
        time_m5 = _load_stage("m5")["extra"].get("time_taken", 0.0)

    if medians_m5 is None or names_m5 is None:
        raise RuntimeError("[stage-M5] posterior medians or parameter names missing.")

    if cached_m5 and not (OUT_DIR / "stage_m5_model.png").exists():
        print("[stage-M5] stage_m5_model.png missing — re-plotting")
        lkl_m5 = build_comparison_likelihood(
            lens_subtracted, noise_map, psf_kernel, feature_mask,
            stage_m2, stage_m3, position_likelihood, s1_package,
            circular_mask=circular_mask, with_subhalo=True,
        )
        try:
            _plot_pix_stage(
                "stage-M5", lkl_m5, medians_m5, names_m5,
                str(OUT_DIR / "stage_m5_model.png"),
            )
        except Exception as err:
            print(f"[stage-M5] re-plotting failed (non-fatal): {err}")

    # ---- evidence comparison ---------------------------------------- #

    logz_m3 = float(stage_m3.log_z)
    logz_m4 = float(stage_m4.log_z)
    logz_m5 = float(stage_m5.log_z)
    delta_logz = logz_m5 - logz_m4
    mass_report = _subhalo_mass_report(stage_m5)
    mass_path = OUT_DIR / "stage_m5_mass.json"
    with mass_path.open("w", encoding="utf-8") as f:
        json.dump(mass_report, f, indent=2)
        f.write("\n")
    print(f"[stage-M5] mass summary saved to {mass_path}")

    print("\n" + "=" * 60)
    print(" Pipeline complete")
    print("=" * 60)
    print(" Time summary:")
    print(f"    Stage A:  {time_a/60:.2f} min")
    print(f"    Stage M1: {time_m1/60:.2f} min")
    print(f"    Stage M2: {time_m2/60:.2f} min")
    print(f"    Stage M3: {time_m3/60:.2f} min")
    print(f"    Stage M4: {time_m4/60:.2f} min")
    print(f"    Stage M5: {time_m5/60:.2f} min")
    total_time = (time_a + time_m1 + time_m2 + time_m3
                  + time_m4 + time_m5)
    print(f"    Total:    {total_time/60:.2f} min\n")
    print(f"    M3 median gamma        = {medians_m3['gamma']:+.4f}")
    print(f"    M2 median source reg   = {_format_reg_hyperparams(reg_hyperparams_m2)}")
    print("    Subhalo Bayes factor (both arms share every nuisance prior):")
    print(f"    logZ(M4, smooth null)  = {logz_m4:+.3f}")
    print(f"    logZ(M5, +Pseudo-Jaffe subhalo) = {logz_m5:+.3f}")
    print(f"    delta logZ (M5 - M4)   = {delta_logz:+.3f}")
    print(f"    logZ(M3) (reference only, NOT the comparison baseline) = {logz_m3:+.3f}")
    _print_subhalo_mass_report("stage-M5", mass_report)

    # Large smooth-parameter drift can signal a prior-width sensitivity. This
    # diagnostic alone does not establish how the Pseudo-Jaffe evidence behaves.
    drift_warn = []
    for name in SMOOTH_MASS_PRIOR_LIMITS:
        sigma = float(stage_m3.std(name))
        if sigma <= 0 or name not in medians_m5:
            continue
        drift = (float(medians_m5[name]) - float(stage_m3.median(name))) / sigma
        if abs(drift) > SOLUTION_DRIFT_WARN_SIGMA:
            drift_warn.append((name, drift))
    if drift_warn:
        print()
        print("    WARNING: the subhalo arm's smooth solution has drifted from the")
        print("    stage-M3 solution by more than "
              f"{SOLUTION_DRIFT_WARN_SIGMA:g} sigma_M3 in:")
        for name, drift in drift_warn:
            print(f"        {name:18s} {drift:+7.2f} sigma_M3")
        print("    This may indicate a trade between the subhalo and main lens.")
        print("    Prior-width dependence of the current Pseudo-Jaffe evidence has")
        print("    not been verified. Run both arms at multiple shared prior widths")
        print("    before interpreting this delta logZ as a subhalo detection.")

    for k in ("theta_E", "gamma", "e1_mass", "e2_mass",
              "center_x_mass", "center_y_mass", "gamma1", "gamma2",
              "log_lambda_reg"):
        if k in medians_m4:
            print(f"    M4     {k:20s} = {medians_m4[k]:+.4f}")
    for k in ("b_subhalo", "center_x_subhalo", "center_y_subhalo"):
        if k in medians_m5:
            print(f"    M5     {k:20s} = {medians_m5[k]:+.4f}")

    summary = dict(
        target="B1938+666", band="F115W", source_planes=1,
        mask_radius_arcsec=MASK_RADIUS,
        inner_mask_radius_arcsec=INNER_MASK_RADIUS,
        m1_lambda_prior_range=list(M1_LAMBDA_REG_PRIOR_RANGE),
        pcg_max_iter=PCG_MAX_ITER,
        subhalo_position_prior=_subhalo_position_bounds(image_data.shape, DPIX),
        source_grid_n=NSRC, source_bbox=list(source_bbox),
        stage_times_seconds=dict(a=time_a, m1=time_m1,
                                 m2=time_m2, m3=time_m3, m4=time_m4, m5=time_m5),
        log_z=dict(m3=logz_m3, m4=logz_m4, m5=logz_m5),
        delta_log_z_m5_minus_m4=delta_logz,
        medians=dict(a=medians_a, m1=medians_m1, m2=medians_m2,
                     m3=medians_m3, m4=medians_m4, m5=medians_m5),
        subhalo_mass=mass_report,
    )
    with (OUT_DIR / "pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit B1938+666 recentered F115W through stage M5.")
    parser.add_argument("--skip-done", action="store_true", help="Reuse completed stage posteriors and resume sampler checkpoints.")
    parser.add_argument("--out-dir", default=None, help="Output path, relative to this script.")
    args = parser.parse_args()
    main(skip_done=args.skip_done, out_dir=args.out_dir)
