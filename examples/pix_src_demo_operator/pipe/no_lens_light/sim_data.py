"""Simulate a Euclid VIS mock with EPL + shear, a Sérsic source, and no lens light."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from TinyLensGpu.ForwardSimulation import LensSimulator, SimulatorConfig, make_grid_2d
from TinyLensGpu.PhysicalModel import EPL, PhysicalModel, SersicEllipse, Shear
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


DATA_DIR = Path(__file__).resolve().parent / "data"
PIXEL_SCALE = 0.1  # arcsec/pixel
PSF_FWHM = 0.18  # arcsec


def main(out_dir=DATA_DIR, *, add_noise=True):
    e1, e2 = phi_q2_ellipticity(np.pi / 2, 0.9)
    mass = dict(theta_E=1.5, gamma=2.2, e1=float(e1), e2=float(e2), center_x=0.0, center_y=0.0)
    shear = dict(gamma1=0.05, gamma2=0.05)
    source = SersicEllipse(
        R_sersic=0.3, n_sersic=1.0, e1=0.05, e2=0.05,
        center_x=0.0, center_y=0.3, Ie=1.0,
    )
    model = PhysicalModel(lens_mass=[EPL(**mass), Shear(**shear)], source_light=[source])

    psf_sigma = PSF_FWHM / np.sqrt(8 * np.log(2))
    x, y = make_grid_2d(21, PIXEL_SCALE)
    psf = np.exp(-np.asarray(x**2 + y**2) / (2 * psf_sigma**2))
    psf /= psf.sum()
    config = SimulatorConfig(dpix=PIXEL_SCALE, npix=200, psf_kernel=psf, nsub=16)
    image = np.asarray(LensSimulator(model, config).simulate())

    exposure, background_rms, seed = 300.0, 0.1, 42
    noise = np.sqrt(np.maximum(image, 0) / exposure + background_rms**2)
    if add_noise:
        image = image + np.random.default_rng(seed).normal(0, noise)

    header = fits.Header(dict(
        PIXSCALE=PIXEL_SCALE, PSFFWHM=PSF_FWHM, PSFSIGMA=psf_sigma,
        EXPTIME=exposure, BKGRMS=background_rms, NOISEADD=add_noise,
        RNGSEED=seed, **mass, **shear,
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, array in (("image", image), ("noise", noise), ("psf", psf)):
        fits.writeto(out_dir / f"{name}.fits", array, header, overwrite=True)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    for ax, (title, array) in zip(axes.flat, (
        ("Image", image), ("Noise RMS", noise), ("PSF", psf), ("Signal-to-noise", image / noise),
    )):
        half_width = array.shape[0] * PIXEL_SCALE / 2
        panel = ax.imshow(array, origin="lower", extent=(-half_width, half_width) * 2)
        ax.set(title=title, xlabel="x (arcsec)", ylabel="y (arcsec)")
        fig.colorbar(panel, ax=ax)
    fig.savefig(out_dir / "mock_diagnostics.png", dpi=150)
    plt.close(fig)
    print(f"Mock FITS and diagnostics saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-noise", action="store_true", help="Omit the random draw and keep the RMS map.")
    args = parser.parse_args()
    main(args.out_dir, add_noise=not args.no_noise)
