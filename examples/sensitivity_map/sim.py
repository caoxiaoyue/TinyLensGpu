"""Simulate a Float32 HST-like lensed arc with a Pseudo-Jaffe subhalo."""

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
import numpy as np
from astropy.io import fits

from TinyLensGpu.ForwardSimulation import LensSimulator, SimulatorConfig, make_grid_2d
from TinyLensGpu.PhysicalModel import EPL, PhysicalModel, PseudoJaffeSubhalo, SersicEllipse, Shear
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


DATA_DIR = Path(__file__).resolve().parent / "data"
PIXEL_SCALE = 0.05  # arcsec/pixel
PSF_SIGMA = 0.05  # arcsec
NPIX = 120
NSUB = 16
B_SUB = 0.08  # arcsec


def lens_mass_components(epl, shear, subhalo=None):
    host = EPL(**epl)
    components = [host, Shear(**shear)]
    if subhalo is not None:
        components.append(PseudoJaffeSubhalo(theta_E_host=host.theta_E, **subhalo))
    return components


def plot_mock(image, smooth, perturbed, noise, subhalo, path):
    import matplotlib.pyplot as plt

    half_width = image.shape[0] * PIXEL_SCALE / 2
    extent = (-half_width, half_width) * 2
    difference = (perturbed - smooth) / noise
    limit = np.max(np.abs(difference))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    for ax, (title, array, options) in zip(axes, (
        ("Observed image", image, {"vmin": 0, "vmax": smooth.max()}),
        ("Smooth lens", smooth, {"vmin": 0, "vmax": smooth.max()}),
        ("Subhalo signal / RMS", difference, {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit}),
    )):
        panel = ax.imshow(array, origin="lower", extent=extent, **options)
        ax.plot(subhalo["center_x"], subhalo["center_y"], "+", color="black", markersize=9)
        ax.set(title=title, xlabel="x (arcsec)", ylabel="y (arcsec)")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        fig.colorbar(panel, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(out_dir=DATA_DIR, *, add_noise=True):
    jax.config.update("jax_enable_x64", False)
    e1, e2 = phi_q2_ellipticity(np.pi / 2, 0.9)
    epl = dict(theta_E=1.5, gamma=2.2, e1=float(e1), e2=float(e2), center_x=0.0, center_y=0.0)
    shear = dict(gamma1=0.05, gamma2=0.05)
    source = dict(R_sersic=0.3, n_sersic=1.0, e1=0.05, e2=0.05, center_x=0.0, center_y=0.3, Ie=1.0)

    psf_x, psf_y = make_grid_2d(11, PIXEL_SCALE)
    psf = np.exp(-np.asarray(psf_x**2 + psf_y**2) / (2 * PSF_SIGMA**2))
    psf /= psf.sum()
    config = SimulatorConfig(dpix=PIXEL_SCALE, npix=NPIX, psf_kernel=psf, nsub=NSUB)

    def render(subhalo=None):
        model = PhysicalModel(
            lens_mass=lens_mass_components(epl, shear, subhalo),
            source_light=[SersicEllipse(**source)],
        )
        return np.asarray(LensSimulator(model, config).simulate())

    smooth = render()
    # Place the subhalo on the bright upper arc of the unperturbed mock.
    x, y = np.asarray(config.xgrid), np.asarray(config.ygrid)
    upper_arc = (y > 0) & (np.hypot(x, y) > 0.75 * epl["theta_E"])
    peak = np.unravel_index(np.argmax(np.where(upper_arc, smooth, -np.inf)), smooth.shape)
    subhalo = dict(b_sub=B_SUB, center_x=float(x[peak]), center_y=float(y[peak]))
    perturbed = render(subhalo)

    exposure, background_rms, seed = 300.0, 0.1, 42
    noise = np.sqrt(np.maximum(perturbed, 0) / exposure + background_rms**2)
    image = perturbed
    if add_noise:
        draw = np.random.default_rng(seed).standard_normal(perturbed.shape, dtype=np.float32)
        image = perturbed + noise * draw

    out_dir.mkdir(parents=True, exist_ok=True)
    header = fits.Header(dict(
        PIXSCALE=PIXEL_SCALE, PSFSIGMA=PSF_SIGMA, PSFFWHM=PSF_SIGMA * np.sqrt(8 * np.log(2)),
        NSUB=NSUB, EXPTIME=exposure, BKGRMS=background_rms, NOISEADD=add_noise, RNGSEED=seed,
    ))
    for name, array in (("image", image), ("noise", noise), ("psf", psf)):
        fits.writeto(out_dir / f"{name}.fits", array, header, overwrite=True)
    truth = dict(lens_mass=dict(epl=epl, shear=shear, subhalo=subhalo), source=source)
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    plot_mock(image, smooth, perturbed, noise, subhalo, out_dir / "mock_diagnostics.png")
    print(f"Saved mock to {out_dir}. Subhalo at ({x[peak]:.3f}, {y[peak]:.3f}) arcsec.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-noise", action="store_true", help="Keep the RMS map and omit the random draw.")
    args = parser.parse_args()
    main(args.out_dir, add_noise=not args.no_noise)
