"""Fit the Euclid VIS mock with EPL + shear and an MGE source."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.ObservationModel import ImageProbModel
from TinyLensGpu.PhysicalModel import EPL, GaussianEllipse, PhysicalModel, Shear
from TinyLensGpu.example_support import SamplingConfig, posterior_quantiles, run_sampling
from TinyLensGpu.utils import generate_radial_basis_knots, load_lens_data
from TinyLensGpu.visualizer import plot_model_results


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output" / "parametric"
CROP_PIXELS = 64
TRUTH_KEYS = dict(
    theta_E="THETA_E", gamma="GAMMA", e1_mass="E1", e2_mass="E2",
    center_x_mass="CENTER_X", center_y_mass="CENTER_Y", gamma1="GAMMA1", gamma2="GAMMA2",
)


def sampled(name, low, high):
    param = ParamU(name, (low + high) / 2, prior_settings=(low, high), limits=(low, high))
    param.to_dynamic()
    return param


def build_likelihood(image, noise, psf, dpix, nsub):
    start = (image.shape[0] - CROP_PIXELS) // 2
    region = np.s_[start:start + CROP_PIXELS, start:start + CROP_PIXELS]
    mass = EPL(
        theta_E=sampled("theta_E", 1.0, 2.0), gamma=sampled("gamma", 1.4, 2.8),
        e1=sampled("e1_mass", -0.25, 0.25), e2=sampled("e2_mass", -0.25, 0.25),
        center_x=sampled("center_x_mass", -0.2, 0.2), center_y=sampled("center_y_mass", -0.2, 0.2),
    )
    shear = Shear(gamma1=sampled("gamma1", -0.2, 0.2), gamma2=sampled("gamma2", -0.2, 0.2))
    geometry = dict(
        center_x=sampled("center_x_src", -0.65, 0.65), center_y=sampled("center_y_src", -0.65, 0.65),
        e1=sampled("e1_src", -0.35, 0.35), e2=sampled("e2_src", -0.35, 0.35),
    )
    widths = generate_radial_basis_knots(dpix=dpix, n_sigmas=12, log_rmin=-2, log_rmax=0, mode="mge")
    source = [GaussianEllipse(sigma=float(sigma), flux=1.0, **geometry) for sigma in widths]
    return ImageProbModel(
        image_data=image[region], noise_map=noise[region], psf_kernel=psf, dpix=dpix, nsub=nsub,
        phys_model=PhysicalModel(lens_mass=[mass, shear], source_light=source),
        use_linear=True, solver_type="nnls",
    )


def main(data_dir=DATA_DIR, out_dir=OUT_DIR, *, n_live=160, n_eff=800, batch_size=32, nsub=4):
    image_path = data_dir / "image.fits"
    image, noise, psf, _ = load_lens_data(image_path, data_dir / "noise.fits", data_dir / "psf.fits")
    header = fits.getheader(image_path)
    likelihood = build_likelihood(image, noise, psf, header["PIXSCALE"], nsub)
    sampling = SamplingConfig(n_live, n_eff, vectorized=True, n_batch=batch_size, sampler_kwargs={"seed": 42})
    stage = run_sampling(likelihood, sampling, tag="Parametric")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "posterior.npz", samples=stage.samples, weights=stage.weights,
        names=stage.param_names, log_z=stage.log_z,
    )
    quantiles = posterior_quantiles(stage, q=(0.025, 0.16, 0.5, 0.84, 0.975))
    with (out_dir / "parameter_recovery.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "parameter", "truth", "median", "q16", "q84", "q025", "q975",
            "truth_in_68pct", "truth_in_95pct",
        ))
        for name, key in TRUTH_KEYS.items():
            truth = header[key]
            q025, q16, median, q84, q975 = quantiles[name]
            writer.writerow((
                name, truth, median, q16, q84, q025, q975, q16 <= truth <= q84, q025 <= truth <= q975,
            ))

    point = np.array([quantiles[name][2] for name in stage.param_names])
    fig, _ = plot_model_results(
        likelihood, point, save_path=out_dir / "model_visualization.png", title="EPL + shear · MGE source",
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-live", type=int, default=160)
    parser.add_argument("--n-eff", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nsub", type=int, default=4)
    main(**vars(parser.parse_args()))
