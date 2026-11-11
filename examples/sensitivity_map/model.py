"""Fit the HST mock with Float32 evidence in four L-BFGS-B stages."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import minimize

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel
from TinyLensGpu.example_support import require_valid_reconstruction
from TinyLensGpu.visualizer import plot_pixelized_source_results

if __package__:
    from .sim import DATA_DIR, PIXEL_SCALE, lens_mass_components
else:
    from sim import DATA_DIR, PIXEL_SCALE, lens_mass_components


OUT_DIR = Path(__file__).resolve().parent / "output"
MASK_RADIUS = 2.5  # arcsec
SOURCE_N = 80
SOURCE_DPIX = 0.02  # arcsec/pixel
NSUB = 4
MAXITER = 200
LOG_LAMBDA_LIMIT = np.float32(6) * np.log(np.float32(10))

# Physical bounds and coordinate scales for L-BFGS-B.
PARAMETER_CONTROLS = {
    "epl_theta_E": ((0.5, 3.0), 0.04),
    "epl_gamma": ((1.1, 2.9), 0.12),
    "epl_e1": ((-0.5, 0.5), 0.04),
    "epl_e2": ((-0.5, 0.5), 0.04),
    "epl_center_x": ((-0.3, 0.3), 0.04),
    "epl_center_y": ((-0.3, 0.3), 0.04),
    "shear_gamma1": ((-0.3, 0.3), 0.03),
    "shear_gamma2": ((-0.3, 0.3), 0.03),
    "subhalo_b_sub": ((0.0, 0.3), 0.025),
    "subhalo_center_x": ((-MASK_RADIUS, MASK_RADIUS), PIXEL_SCALE / NSUB),
    "subhalo_center_y": ((-MASK_RADIUS, MASK_RADIUS), PIXEL_SCALE / NSUB),
    "log_lambda_reg": ((-LOG_LAMBDA_LIMIT, LOG_LAMBDA_LIMIT), 1.0),
    "adaptive_reg_rho": ((0.0, 8.0), 1.0),
}


def parameter(name, value, varying):
    if not varying:
        return value
    bounds, _ = PARAMETER_CONTROLS[name]
    param = ParamU(name, value, prior_settings=bounds, limits=bounds)
    param.to_dynamic()
    return param


def optimize(likelihood, stage, maxiter):
    params = likelihood.get_dynamic_params()
    names = [param.name for param in params]
    initial = np.asarray(likelihood.get_values(), dtype=np.float32)
    scales = np.array([PARAMETER_CONTROLS[name][1] for name in names], dtype=np.float32)
    bounds = np.array([param.limits for param in params], dtype=np.float32)
    lower, upper = ((bounds - initial[:, None]) / scales[:, None]).T
    start = np.zeros_like(initial)
    value_and_grad = jax.jit(jax.value_and_grad(lambda z: -likelihood(initial + scales * z)))
    invalid_trials = 0

    def objective(z):
        nonlocal invalid_trials
        # SciPy supplies double-precision coordinates to the Float32 objective.
        z = np.asarray(z, dtype=np.float32)
        value, gradient = value_and_grad(z)
        value, gradient = float(value), np.asarray(gradient)
        if not np.isfinite(value) or value >= 1e10 or not np.all(np.isfinite(gradient)):
            invalid_trials += 1
            return 1e10, np.zeros_like(z)
        return value, gradient

    print(f"[Stage {stage}] Optimizing {', '.join(names)}", flush=True)
    initial_loss, _ = objective(start)
    if initial_loss >= 1e10:
        _, diagnostics = likelihood.evaluate_with_diagnostics(jnp.asarray(initial))
        reason = diagnostics.invalid_reason or diagnostics.evidence_invalid_reason or "Nonfinite gradient."
        raise RuntimeError(f"Stage {stage} has an invalid starting point. {reason}")
    fit = minimize(
        objective, start, method="L-BFGS-B", jac=True, bounds=list(zip(lower, upper)),
        options=dict(maxiter=maxiter, maxls=40, maxcor=15, ftol=np.finfo(np.float32).eps, gtol=2e-5),
    )
    if not np.isfinite(fit.fun) or fit.fun >= 1e10 or fit.fun > initial_loss:
        raise RuntimeError(f"Stage {stage} failed to retain a valid fit: {fit.message}")
    scaled_point = np.asarray(fit.x, dtype=np.float32)
    point = initial + scales * scaled_point
    projected_gradient = scaled_point - np.clip(scaled_point - fit.jac.astype(np.float32), lower, upper)
    info = dict(
        success=bool(fit.success), message=str(fit.message), iterations=int(fit.nit),
        evaluations=int(fit.nfev), invalid_trials=invalid_trials,
        initial_log_evidence=-initial_loss, projected_gradient_inf=float(np.max(np.abs(projected_gradient))),
    )
    return point, dict(zip(names, map(float, point))), info


def fit_stage(stage, observation, out_dir, mass, regularization, *,
              vary_mass, vary_reg=(), adapt_image=None, maxiter):
    mass_params = {
        profile: {attr: parameter(f"{profile}_{attr}", value, vary_mass) for attr, value in values.items()}
        for profile, values in mass.items()
    }
    reg_params = {name: parameter(name, value, name in vary_reg) for name, value in regularization.items()}
    source = PixelizedSourceModel(
        SOURCE_N, regularization_type="first-order", interpolation="cubic_hermite", **reg_params,
    )
    physical_model = PhysicalModel(lens_mass=lens_mass_components(**mass_params), source_light=[source])
    likelihood = PixelizedImageProbModelOperator(
        **observation, phys_model=physical_model,
        nsub=NSUB, solver_type="pcg", evidence_logdet="overlap", adapt_image=adapt_image,
    )
    likelihood.pcg_max_iter = 1200
    likelihood.pcg_rtol = 1e-5

    point, fitted, optimizer = optimize(likelihood, stage, maxiter)
    result, diagnostics = likelihood.evaluate_with_diagnostics(jnp.asarray(point))
    require_valid_reconstruction(result, diagnostics, context=f"reconstructing Stage {stage}")
    if not diagnostics.evidence_valid:
        raise RuntimeError(f"Stage {stage} evidence failed: {diagnostics.evidence_invalid_reason}")

    model_image = np.asarray(result.model_image)
    source_image = np.asarray(result.source_pixels).reshape(SOURCE_N, SOURCE_N)
    bbox = np.asarray(result.source_bbox)
    residuals = np.asarray(diagnostics.normalized_residuals)
    log_evidence = float(result.log_evidence)
    chi2 = diagnostics.chi2
    mask = observation["mask"]
    stem = out_dir / f"stage_{stage}"
    np.savez_compressed(
        stem.with_suffix(".npz"), model_image=model_image, source_image=source_image,
        source_bbox=bbox, normalized_residuals=residuals, regularization_scale=diagnostics.regularization_scale,
        chi2=np.float32(chi2), log_evidence=np.float32(log_evidence),
    )
    if stage in (3, 4):
        extent = bbox + (SOURCE_DPIX / 2) * np.array([-1, 1, -1, 1])
        label = "Smooth lens" if stage == 3 else "Smooth lens + Pseudo-Jaffe subhalo"
        fig, axes = plot_pixelized_source_results(
            observation["image_data"], model_image, residuals, mask, source_image, extent,
            image_dpix=observation["dpix"], lambda_reg=diagnostics.lambda_reg, title=f"Stage {stage} · {label}",
        )
        for ax in axes[:3]:
            ax.set(xlim=(-MASK_RADIUS, MASK_RADIUS), ylim=(-MASK_RADIUS, MASK_RADIUS))
        fig.savefig(stem.with_suffix(".png"), dpi=150)
        plt.close(fig)

    chi2_per_pixel = chi2 / np.count_nonzero(~mask)
    print(f"[Stage {stage}] {optimizer['message']} after {optimizer['iterations']} iterations. "
          f"χ²/pixel = {chi2_per_pixel:.4f}, log evidence = {log_evidence:.4f}", flush=True)
    summary = dict(
        initial_mass=mass, initial_regularization=regularization,
        mass={
            profile: {attr: fitted.get(f"{profile}_{attr}", value) for attr, value in values.items()}
            for profile, values in mass.items()
        },
        regularization={name: fitted.get(name, value) for name, value in regularization.items()},
        free_parameters=list(fitted), chi2=chi2, chi2_per_pixel=chi2_per_pixel, log_evidence=log_evidence,
        pcg_iterations=int(result.solver_info.n_iter), optimizer=optimizer,
    )
    return summary, model_image


def compare_bic(stage3, stage4, n_pixels):
    """Compute BIC from the fitted source-marginalized log likelihoods.

    The Overlap log evidence integrates over the pixelized source conditional
    on the mass and regularization parameters. Count the nonlinear parameters
    varied in each stage and use the common number of unmasked image pixels.

    Consult README.md before changing or reviewing the likelihood convention.
    """
    log_n = np.log(np.float32(n_pixels))
    k3, k4 = len(stage3["free_parameters"]), len(stage4["free_parameters"])
    bic3 = np.float32(-2) * np.float32(stage3["log_evidence"]) + np.float32(k3) * log_n
    bic4 = np.float32(-2) * np.float32(stage4["log_evidence"]) + np.float32(k4) * log_n
    return dict(
        likelihood="source_marginalized_overlap_log_evidence", n=int(n_pixels),
        k_stage3=k3, k_stage4=k4, stage3=float(bic3), stage4=float(bic4),
        delta_stage4_minus_stage3=float(bic4 - bic3),
    )


def main(data_dir=DATA_DIR, out_dir=OUT_DIR, *, maxiter=MAXITER):
    if maxiter < 1:
        raise ValueError("maxiter must be positive.")
    jax.config.update("jax_enable_x64", False)
    image, header = fits.getdata(data_dir / "image.fits", header=True)
    image = image.astype(np.float32)
    noise = fits.getdata(data_dir / "noise.fits").astype(np.float32)
    psf = fits.getdata(data_dir / "psf.fits").astype(np.float32)
    truth = json.loads((data_dir / "truth.json").read_text())
    dpix = header["PIXSCALE"]
    axis = (np.arange(image.shape[0], dtype=np.float32) - (image.shape[0] - 1) / 2) * dpix
    circular_mask = np.hypot(axis[:, None], axis[None, :]) > MASK_RADIUS
    footprint = np.ones((3, 3), dtype=bool)
    labels, _ = ndimage.label((image / noise > 2.0) & ~circular_mask, structure=footprint)
    large_regions = np.bincount(labels.ravel()) >= 5
    large_regions[0] = False
    closed = ndimage.binary_closing(large_regions[labels], structure=footprint)
    arc = ndimage.binary_dilation(closed, structure=np.ones((5, 5), dtype=bool))
    mask = circular_mask | ~arc
    n_pixels = np.count_nonzero(~mask)
    if n_pixels == 0:
        raise ValueError("The arc mask excludes every image pixel.")

    half_width = (SOURCE_N - 1) * SOURCE_DPIX / 2
    x, y = truth["source"]["center_x"], truth["source"]["center_y"]
    bbox = (x - half_width, x + half_width, y - half_width, y + half_width)
    observation = dict(
        image_data=image, noise_map=noise, psf_kernel=psf, dpix=dpix, mask=mask, fixed_source_bbox=bbox,
    )
    smooth_truth = {profile: truth["lens_mass"][profile] for profile in ("epl", "shear")}

    out_dir.mkdir(parents=True, exist_ok=True)
    fits.writeto(out_dir / "mask.fits", mask.astype(np.uint8), overwrite=True)
    summary = dict(
        data_dir=str(data_dir.resolve()), dtype=str(image.dtype), source_n=SOURCE_N, source_bbox=bbox,
        interpolation="cubic_hermite", regularization_type="first-order", nsub=NSUB,
        adapt_image="adapt_image.fits", mask="mask.fits", stages={}, bic=None,
    )

    def save_summary():
        pending = out_dir / "results.json.tmp"
        pending.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        pending.replace(out_dir / "results.json")

    def run_stage(stage, mass, regularization, **options):
        record, model_image = fit_stage(
            stage, observation, out_dir, mass, regularization, maxiter=maxiter, **options,
        )
        summary["stages"][f"stage_{stage}"] = record
        save_summary()
        return record, model_image

    save_summary()
    print(f"Fitting {n_pixels} native arc pixels.", flush=True)
    stage1, adapt_image = run_stage(
        1, smooth_truth, dict(log_lambda_reg=-5.5, adaptive_reg_rho=0.0),
        vary_mass=True, vary_reg=("log_lambda_reg",),
    )
    fits.writeto(out_dir / "adapt_image.fits", adapt_image, header, overwrite=True)
    stage2, _ = run_stage(
        2, stage1["mass"], dict(stage1["regularization"], adaptive_reg_rho=2.0),
        vary_mass=False, vary_reg=("log_lambda_reg", "adaptive_reg_rho"), adapt_image=adapt_image,
    )
    stage3, _ = run_stage(
        3, stage1["mass"], stage2["regularization"], vary_mass=True, adapt_image=adapt_image,
    )
    stage4, _ = run_stage(
        4, truth["lens_mass"], stage3["regularization"],
        vary_mass=True, vary_reg=("log_lambda_reg",), adapt_image=adapt_image,
    )

    bic = compare_bic(stage3, stage4, n_pixels)
    summary["bic"] = bic
    save_summary()
    print(f"ΔBIC (Stage 4 − Stage 3) = {bic['delta_stage4_minus_stage3']:.4f}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--maxiter", type=int, default=MAXITER, help="L-BFGS-B iteration limit per stage.")
    main(**vars(parser.parse_args()))
