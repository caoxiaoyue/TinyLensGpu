"""Simulate lens+source+lens-light imaging data for irregular-grid inversion demo.

This script follows the physical setup used in ``paper/demo/lens_src_mge/sim_data.py``
for mass/source/lens-light components, and saves FITS files consumed by
``single_step_inversion.py`` in the same folder.
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from TinyLensGpu.ForwardSimulation import LensSimulator, SimulatorConfig, make_grid_2d
from TinyLensGpu.PhysicalModel import GaussianEllipse, PhysicalModel, SIE, SersicEllipse, Shear
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


def build_physical_model() -> PhysicalModel:
    """Build the simulation-time physical model with Sersic source and lens light."""
    e1_lens, e2_lens = phi_q2_ellipticity(90.0 * np.pi / 180.0, 0.9)
    return PhysicalModel(
        lens_mass=[
            SIE(theta_E=1.5, e1=e1_lens, e2=e2_lens, center_x=0.0, center_y=0.0),
            Shear(gamma1=0.05, gamma2=0.05),
        ],
        source_light=[
            SersicEllipse(
                R_sersic=0.3,
                n_sersic=1.0,
                e1=0.05,
                e2=0.05,
                center_x=0.0,
                center_y=0.5,
                Ie=1.0,
            )
        ],
        lens_light=[
            SersicEllipse(
                R_sersic=1.0,
                n_sersic=4.0,
                e1=e1_lens,
                e2=e2_lens,
                center_x=0.0,
                center_y=0.0,
                Ie=1.0,
            )
        ],
    )


def build_psf_kernel(dpix: float) -> np.ndarray:
    """Create a normalized Gaussian PSF kernel."""
    x_psf, y_psf = make_grid_2d(21, dpix)
    psf_kernel = GaussianEllipse(
        flux=1.0,
        sigma=0.05,
        e1=0.0,
        e2=0.0,
        center_x=0.0,
        center_y=0.0,
    ).light(x=x_psf, y=y_psf)
    psf_kernel = np.array(psf_kernel, dtype=np.float64, copy=True)
    psf_kernel /= psf_kernel.sum()
    return psf_kernel


def add_mock_noise(ideal_image: np.ndarray, *, back_rms: float, exp_time: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise using a Poisson+background RMS approximation."""
    rng = np.random.default_rng(seed)
    noise_map = np.sqrt(ideal_image / exp_time + back_rms**2)
    noisy_image = ideal_image + rng.normal(0.0, noise_map)
    return noisy_image, noise_map


def simulate_lensing_data(seed: int = 0) -> dict[str, np.ndarray | float]:
    """Run the end-to-end simulation and return arrays required for inversion."""
    dpix = 0.074
    npix = 200
    psf_kernel = build_psf_kernel(dpix)

    simulator = LensSimulator(
        build_physical_model(),
        SimulatorConfig(dpix=dpix, npix=npix, psf_kernel=psf_kernel, nsub=16),
    )
    ideal_image = np.asarray(simulator.simulate())
    noisy_image, noise_map = add_mock_noise(ideal_image, back_rms=0.1, exp_time=300.0, seed=seed)

    xgrid, ygrid = make_grid_2d(npix, dpix)
    radius = np.sqrt(np.asarray(xgrid) ** 2 + np.asarray(ygrid) ** 2)
    mask = radius > 3.0

    return {
        "noisy_image": noisy_image,
        "noise_map": noise_map,
        "psf_kernel": psf_kernel,
        "mask": mask,
        "dpix": dpix,
    }


def main() -> None:
    """Generate FITS inputs for the irregular-grid joint inversion demo."""
    data = simulate_lensing_data(seed=0)
    os.makedirs("data", exist_ok=True)

    fits.writeto("data/image.fits", np.asarray(data["noisy_image"]), overwrite=True)
    fits.writeto("data/noise.fits", np.asarray(data["noise_map"]), overwrite=True)
    fits.writeto("data/psf.fits", np.asarray(data["psf_kernel"]), overwrite=True)
    fits.writeto("data/mask.fits", np.asarray(data["mask"]).astype(np.int32), overwrite=True)

    print("Saved simulated data to data/ directory.")


if __name__ == "__main__":
    main()
