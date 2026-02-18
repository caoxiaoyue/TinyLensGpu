"""
Simulate gravitational lensing data for shapelet source-only reconstruction demo.

Sersic source + SIE lens, same setup as pix_src/src_light_only/rect_src_grid/sim_data.py.
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from TinyLensGpu.ForwardSimulation import LensSimulator, SimulatorConfig, make_grid_2d
from TinyLensGpu.PhysicalModel import GaussianEllipse, PhysicalModel, SIE, SersicEllipse
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


def simulate_lensing_data(seed: int = 0) -> dict:
    """Simulate lensing data and return a dictionary of arrays."""
    np.random.seed(seed)

    e1_l, e2_l = phi_q2_ellipticity(90 * np.pi / 180, 0.9)
    phy_model = PhysicalModel(
        lens_mass=[SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0)],
        source_light=[
            SersicEllipse(
                R_sersic=0.3,
                n_sersic=1.0,
                e1=0.05,
                e2=0.05,
                center_x=0.0,
                center_y=0.3,
                Ie=1.0,
            )
        ],
        lens_light=[],
    )

    npix = 200
    dpix = 10.0 / npix  # 0.05 arcsec/pixel

    x_psf, y_psf = make_grid_2d(21, dpix)
    psf_kernel = GaussianEllipse(
        flux=1.0, sigma=0.05, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0
    ).light(x=x_psf, y=y_psf)
    psf_kernel = np.array(psf_kernel, dtype=np.float64, copy=True)
    psf_kernel /= psf_kernel.sum()

    sim_config = SimulatorConfig(dpix=dpix, npix=npix, psf_kernel=psf_kernel, nsub=16)
    ideal_image = np.asarray(LensSimulator(phy_model, sim_config).simulate())

    noise_map = np.sqrt(ideal_image / 300.0 + 0.1**2)
    noisy_image = ideal_image + np.random.normal(0.0, noise_map)

    xgrid, ygrid = make_grid_2d(npix, dpix)
    mask = np.asarray(np.sqrt(xgrid**2 + ygrid**2) > 2.7)

    return {
        "noisy_image": noisy_image,
        "noise_map": noise_map,
        "psf_kernel": psf_kernel,
        "mask": mask,
        "dpix": dpix,
    }


def main() -> None:
    data = simulate_lensing_data()
    os.makedirs("data", exist_ok=True)
    fits.writeto("data/image.fits", np.asarray(data["noisy_image"]), overwrite=True)
    fits.writeto("data/noise.fits", np.asarray(data["noise_map"]), overwrite=True)
    fits.writeto("data/psf.fits", np.asarray(data["psf_kernel"]), overwrite=True)
    fits.writeto("data/mask.fits", np.asarray(data["mask"]).astype(np.int32), overwrite=True)
    print("Saved simulated data to data/ directory.")


if __name__ == "__main__":
    main()
