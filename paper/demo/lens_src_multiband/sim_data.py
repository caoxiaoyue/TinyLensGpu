from pathlib import Path

import numpy as np
from astropy.io import fits

from TinyLensGpu.ForwardSimulation import LensSimulator, SimulatorConfig, make_grid_2d
from TinyLensGpu.PhysicalModel import GaussianEllipse, PhysicalModel, SersicEllipse, SIE
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


def mock_lens(ideal_image: np.ndarray, back_rms: float, exp_time: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    noise_map = np.sqrt(ideal_image / exp_time + back_rms**2)
    noisy_image = ideal_image + rng.normal(0.0, noise_map)
    return noisy_image, noise_map


def build_physical_model() -> PhysicalModel:
    e1_lens, e2_lens = phi_q2_ellipticity(90.0 * np.pi / 180.0, 0.9)
    return PhysicalModel(
        lens_mass=[
            SIE(theta_E=1.5, e1=e1_lens, e2=e2_lens, center_x=0.0, center_y=0.0),
        ],
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


def make_psf_kernel(dpix: float, sigma: float) -> np.ndarray:
    x_psf, y_psf = make_grid_2d(21, dpix)
    psf_kernel = GaussianEllipse(
        flux=1.0,
        sigma=sigma,
        e1=0.0,
        e2=0.0,
        center_x=0.0,
        center_y=0.0,
    ).light(x=x_psf, y=y_psf)
    psf_kernel /= psf_kernel.sum()
    return np.asarray(psf_kernel)


def main() -> None:
    dpix = 0.074
    npix = 200
    nsub = 16
    back_rms = 0.1
    exp_time = 300.0

    band_psf_sigma = {
        "g": 0.03,
        "r": 0.05,
        "i": 0.07,
    }

    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    phy_model = build_physical_model()

    for idx, (band, sigma) in enumerate(band_psf_sigma.items()):
        psf_kernel = make_psf_kernel(dpix=dpix, sigma=sigma)
        sim_config = SimulatorConfig(
            dpix=dpix,
            npix=npix,
            psf_kernel=psf_kernel,
            nsub=nsub,
        )
        simulator = LensSimulator(phy_model, sim_config)
        ideal_image = np.asarray(simulator.simulate())

        rng = np.random.default_rng(2026 + idx)
        noisy_image, noise_map = mock_lens(ideal_image, back_rms=back_rms, exp_time=exp_time, rng=rng)

        fits.writeto(output_dir / f"{band}_image.fits", np.asarray(noisy_image), overwrite=True)
        fits.writeto(output_dir / f"{band}_noise.fits", np.asarray(noise_map), overwrite=True)
        fits.writeto(output_dir / f"{band}_psf.fits", np.asarray(psf_kernel), overwrite=True)


if __name__ == "__main__":
    main()
