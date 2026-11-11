"""Fit the Euclid VIS mock with a pixelized source using Stages A, B, and M0–M3.

A fits SIE + shear and an MGE source. B selects arc pixels by S/N. M0
calibrates uniform regularization. M1 refits SIE + shear and produces a
reference arc. M2 calibrates adaptive regularization at the M1 mass. M3
fits EPL + shear with the M2 regularization strengths and the M1 arc.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "float32")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import ndimage

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.ObservationModel import ImageProbModel, PointSourceProbModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
from TinyLensGpu.PhysicalModel import EPL, SIE, GaussianEllipse, PhysicalModel, PixelizedSourceModel, Shear
from TinyLensGpu.example_support import SamplingConfig, require_valid_reconstruction, run_sampling
from TinyLensGpu.utils import generate_radial_basis_knots, load_lens_data
from TinyLensGpu.visualizer import plot_model_results, plot_pixelized_source_results


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output" / "pixelized"
MASK_RADIUS = 2.5  # arcsec
SOURCE_N = 80
SOURCE_DPIX = 0.02  # arcsec/pixel
NSUB = 4
CHUNK_SIZE = 5


def sampled(name, value, prior_type, settings, limits):
    param = ParamU(name, value, prior_type=prior_type, prior_settings=settings, limits=limits)
    param.to_dynamic()
    return param


def build_stage_a_likelihood(image, noise, psf, dpix, mask):
    mass = SIE(
        theta_E=sampled("theta_E", 1.5, "uniform", (0.5, 3.0), (0.5, 3.0)),
        e1=sampled("e1_mass", 0.0, "truncated_gaussian", (0.0, 0.2), (-1.0, 1.0)),
        e2=sampled("e2_mass", 0.0, "truncated_gaussian", (0.0, 0.2), (-1.0, 1.0)),
        center_x=sampled("center_x_mass", 0.0, "gaussian", (0.0, 0.1), (-1.0, 1.0)),
        center_y=sampled("center_y_mass", 0.0, "gaussian", (0.0, 0.1), (-1.0, 1.0)),
    )
    shear = Shear(
        gamma1=sampled("gamma1", 0.0, "uniform", (-0.2, 0.2), (-0.5, 0.5)),
        gamma2=sampled("gamma2", 0.0, "uniform", (-0.2, 0.2), (-0.5, 0.5)),
    )
    geometry = dict(
        center_x=sampled("center_x_src", 0.0, "gaussian", (0.0, 0.5), (-3.0, 3.0)),
        center_y=sampled("center_y_src", 0.0, "gaussian", (0.0, 0.5), (-3.0, 3.0)),
        e1=sampled("e1_src", 0.0, "truncated_gaussian", (0.0, 0.3), (-1.0, 1.0)),
        e2=sampled("e2_src", 0.0, "truncated_gaussian", (0.0, 0.3), (-1.0, 1.0)),
    )
    widths = generate_radial_basis_knots(dpix=dpix, n_sigmas=10, log_rmin=-2, log_rmax=0, mode="mge")
    source = [GaussianEllipse(sigma=float(sigma), flux=1.0, **geometry) for sigma in widths]
    return ImageProbModel(
        image_data=image, noise_map=noise, psf_kernel=psf, dpix=dpix, nsub=NSUB, mask=mask,
        phys_model=PhysicalModel(lens_mass=[mass, shear], source_light=source),
        use_linear=True, solver_type="nnls",
    )


def fixed_sie(stage, center):
    return [
        SIE(theta_E=stage.median("theta_E"), e1=stage.median("e1_mass"), e2=stage.median("e2_mass"),
            center_x=center[0], center_y=center[1]),
        Shear(gamma1=stage.median("gamma1"), gamma2=stage.median("gamma2")),
    ]


def sampled_mass(stage, center, profile, **parameters):
    profile_name = profile.__name__
    return [
        profile(
            theta_E=stage.gaussian("theta_E", model=profile_name, attr="theta_E", limits=(0.0, 5.0)),
            e1=stage.gaussian("e1_mass", model=profile_name, attr="e1", limits=(-1.0, 1.0)),
            e2=stage.gaussian("e2_mass", model=profile_name, attr="e2", limits=(-1.0, 1.0)),
            center_x=center[0], center_y=center[1], **parameters,
        ),
        Shear(
            gamma1=stage.gaussian("gamma1", model="Shear", attr="gamma1", limits=(-0.5, 0.5)),
            gamma2=stage.gaussian("gamma2", model="Shear", attr="gamma2", limits=(-0.5, 0.5)),
        ),
    ]


def sample_stage(tag, likelihood, out_dir, n_live, n_eff):
    config = SamplingConfig(
        n_live, n_eff, vectorized=True, n_batch=100, vectorized_chunk_size=CHUNK_SIZE,
        sampler_kwargs={"seed": 73},
    )
    stage = run_sampling(likelihood, config, tag=f"Stage {tag}")
    point = np.array(list(stage.medians().values()))
    np.savez_compressed(
        out_dir / f"stage_{tag.lower()}_posterior.npz", samples=stage.samples,
        weights=stage.weights, names=stage.param_names, log_z=stage.log_z, medians=point,
    )
    return stage, point


def calibrate_uniform_regularization(likelihood, out_dir):
    print("\n[Stage M0] Scanning uniform regularization strength")
    loglike = make_likelihood(likelihood, vectorized=True, vectorized_chunk_size=CHUNK_SIZE)
    lower, upper = likelihood.source_model.log_lambda_reg.limits
    coarse = np.linspace(lower, upper, 200)
    coarse_scores = np.asarray(loglike(coarse[:, None]))
    peak = coarse[np.argmax(coarse_scores)]
    half_width = 0.5 * np.log(10)
    fine = np.linspace(peak - half_width, peak + half_width, 200)
    fine_scores = np.asarray(loglike(fine[:, None]))
    best = fine[np.argmax(fine_scores)]
    np.savez_compressed(
        out_dir / "stage_m0_evidence.npz", log_lambda=best,
        coarse=coarse, coarse_scores=coarse_scores, fine=fine, fine_scores=fine_scores,
    )
    return float(best)


def save_reconstruction(tag, likelihood, point, dpix, out_dir):
    result, diagnostics = likelihood.evaluate_with_diagnostics(jnp.asarray(point))
    require_valid_reconstruction(result, diagnostics, context=f"reconstructing Stage {tag}")
    model = np.asarray(result.model_image)
    source = np.asarray(result.source_pixels).reshape(SOURCE_N, SOURCE_N)
    residuals = np.asarray(diagnostics.normalized_residuals)
    bbox = np.asarray(result.source_bbox)
    extent = bbox + (SOURCE_DPIX / 2) * np.array([-1, 1, -1, 1])
    mask = ~np.asarray(likelihood.unmask)
    print(f"[Stage {tag}] χ²/pixel = {diagnostics.chi2 / (~mask).sum():.3f}")

    stem = out_dir / f"stage_{tag.lower()}_model"
    np.savez_compressed(
        stem.with_suffix(".npz"), model_image=model, source_image=source, source_bbox=bbox,
        normalized_residuals=residuals, chi2=diagnostics.chi2, lambda_reg=diagnostics.lambda_reg,
        adaptive_reg_rho=diagnostics.adaptive_reg_rho, source_interpolation=likelihood.source_model.interpolation,
    )
    fig, axes = plot_pixelized_source_results(
        np.asarray(likelihood.image_data), model, residuals, mask, source, extent,
        image_dpix=dpix, lambda_reg=diagnostics.lambda_reg, title=f"Stage {tag}",
    )
    for ax in axes[:3]:
        ax.set(xlim=(-MASK_RADIUS, MASK_RADIUS), ylim=(-MASK_RADIUS, MASK_RADIUS))
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return model


def main(data_dir=DATA_DIR, out_dir=OUT_DIR, *, interpolation="bilinear"):
    image_path = data_dir / "image.fits"
    image, noise, psf, _ = load_lens_data(image_path, data_dir / "noise.fits", data_dir / "psf.fits")
    dpix = fits.getval(image_path, "PIXSCALE")
    out_dir.mkdir(parents=True, exist_ok=True)
    axis = (np.arange(image.shape[0]) - (image.shape[0] - 1) / 2) * dpix
    circular_mask = axis[:, None]**2 + axis[None, :]**2 > MASK_RADIUS**2

    likelihood_a = build_stage_a_likelihood(image, noise, psf, dpix, circular_mask)
    stage_a, point_a = sample_stage("A", likelihood_a, out_dir, 200, 800)
    fig, _ = plot_model_results(
        likelihood_a, point_a, save_path=out_dir / "stage_a_model.png", title="Stage A · MGE source",
    )
    plt.close(fig)

    footprint = np.ones((3, 3), dtype=bool)
    labels, _ = ndimage.label((image / noise > 2.0) & ~circular_mask, structure=footprint)
    large_regions = np.bincount(labels.ravel()) >= 5
    large_regions[0] = False
    closed = ndimage.binary_closing(large_regions[labels], structure=footprint)
    arc = ndimage.binary_dilation(closed, structure=np.ones((5, 5), dtype=bool))
    mask = circular_mask | ~arc
    fits.writeto(out_dir / "feature_mask.fits", mask.astype(np.uint8), overwrite=True)
    print(f"[Stage B] Arc pixels kept = {(~mask).sum()}")

    center = (stage_a.median("center_x_mass"), stage_a.median("center_y_mass"))
    src_x, src_y = stage_a.median("center_x_src"), stage_a.median("center_y_src")
    half_width = (SOURCE_N - 1) * SOURCE_DPIX / 2  # extent of pixel centers
    bbox = (src_x - half_width, src_x + half_width, src_y - half_width, src_y + half_width)
    position_solver = PointSourceProbModel(
        PhysicalModel(lens_mass=fixed_sie(stage_a, center)),
        observed_positions=[[0.0, 0.0]], position_sigma=0.01,
        source_x=src_x, source_y=src_y, source_position_fixed=True,
        solver_config=dict(initial_range=4.0, n_x=400, n_y=400, k_keep=30,
                           num_iters=20, tolerance=5e-4, cluster_tol=0.08),
    )
    positions, _ = position_solver.solve_image_positions()
    position_likelihood = dict(positions=np.asarray(positions), threshold_arcsec=0.1, min_log_like=-1e10)

    def pixel_likelihood(mass, log_lambda, *, rho=0.0, adapt_image=None):
        source = PixelizedSourceModel(
            SOURCE_N, log_lambda_reg=log_lambda, regularization_type="first-order", adaptive_reg_rho=rho,
            interpolation=interpolation,
        )
        return PixelizedImageProbModelOperator(
            image, noise, psf, dpix, PhysicalModel(lens_mass=mass, source_light=[source]),
            mask=mask, nsub=NSUB, solver_type="pcg", position_likelihood=position_likelihood,
            fixed_source_bbox=bbox, adapt_image=adapt_image,
        )

    m0_bounds = (-8 * np.log(10), 8 * np.log(10))
    m0 = pixel_likelihood(
        fixed_sie(stage_a, center), sampled("log_lambda_reg", 0.0, "uniform", m0_bounds, m0_bounds),
    )
    log_lambda = calibrate_uniform_regularization(m0, out_dir)
    save_reconstruction("M0", m0, np.array([log_lambda]), dpix, out_dir)

    m1 = pixel_likelihood(
        sampled_mass(stage_a, center, SIE),
        sampled("log_lambda_reg", log_lambda, "truncated_gaussian", (log_lambda, 0.15),
                (log_lambda - 0.5, log_lambda + 0.5)),
    )
    stage_m1, point_m1 = sample_stage("M1", m1, out_dir, 300, 600)
    reference_arc = save_reconstruction("M1", m1, point_m1, dpix, out_dir)

    m2_bounds = (-6 * np.log(10), 6 * np.log(10))
    m2 = pixel_likelihood(
        fixed_sie(stage_m1, center),
        sampled("log_lambda_reg", stage_m1.median("log_lambda_reg"), "uniform", m2_bounds, m2_bounds),
        rho=sampled("adaptive_reg_rho", 2.0, "uniform", (0.0, 8.0), (0.0, 8.0)),
        adapt_image=reference_arc,
    )
    stage_m2, point_m2 = sample_stage("M2", m2, out_dir, 250, 600)
    save_reconstruction("M2", m2, point_m2, dpix, out_dir)

    m3 = pixel_likelihood(
        sampled_mass(stage_m1, center, EPL, gamma=sampled("gamma", 2.0, "uniform", (1.0, 3.0), (1.0, 3.0))),
        stage_m2.fixed("log_lambda_reg"), rho=stage_m2.median("adaptive_reg_rho"), adapt_image=reference_arc,
    )
    _, point_m3 = sample_stage("M3", m3, out_dir, 300, 600)
    save_reconstruction("M3", m3, point_m3, dpix, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    main(**vars(parser.parse_args()))
