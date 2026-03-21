from pathlib import Path

import numpy as np
from astropy.io import fits

from TinyLensGpu.ForwardSimulation import LensSimulator, SimulatorConfig, make_grid_2d
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d_transformed
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
    # Heterogeneous geometry: different square sizes and pixel scales per band
    # g band is the reference (default alignment)
    band_configs = {
        "g": {"npix": 200, "dpix": 0.074, "nsub": 16, "shift_x": 0.0, "shift_y": 0.0, "rotation": 0.0},
        "r": {"npix": 180, "dpix": 0.08, "nsub": 16, "shift_x": 0.02, "shift_y": -0.015, "rotation": 0.01},
        "i": {"npix": 160, "dpix": 0.09, "nsub": 16, "shift_x": 0.0, "shift_y": 0.0, "rotation": 0.0},
    }

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
        config = band_configs[band]
        npix = config["npix"]
        dpix = config["dpix"]
        nsub = config["nsub"]

        psf_kernel = make_psf_kernel(dpix=dpix, sigma=sigma)
        sim_config = SimulatorConfig(
            dpix=dpix,
            npix=npix,
            psf_kernel=psf_kernel,
            nsub=nsub,
        )
        simulator = LensSimulator(phy_model, sim_config)
        shift_x = config["shift_x"]
        shift_y = config["shift_y"]
        rotation = config["rotation"]
        if shift_x != 0.0 or shift_y != 0.0 or rotation != 0.0:
            xgrid_sub, ygrid_sub = make_grid_2d_transformed(
                npix=npix, dpix=dpix, nsub=nsub,
                shift_x=shift_x, shift_y=shift_y, rotation=rotation
            )
            ideal_image = np.asarray(simulator.simulate(xgrid_sub=xgrid_sub, ygrid_sub=ygrid_sub))
        else:
            ideal_image = np.asarray(simulator.simulate())

        rng = np.random.default_rng(2026 + idx)
        noisy_image, noise_map = mock_lens(ideal_image, back_rms=back_rms, exp_time=exp_time, rng=rng)

        fits.writeto(output_dir / f"{band}_image.fits", np.asarray(noisy_image), overwrite=True)
        fits.writeto(output_dir / f"{band}_noise.fits", np.asarray(noise_map), overwrite=True)
        fits.writeto(output_dir / f"{band}_psf.fits", np.asarray(psf_kernel), overwrite=True)

    # Save the alignment configuration for reference
    print("Generated multiband data with heterogeneous geometry:")
    for band, config in band_configs.items():
        print(f"  {band}: npix={config['npix']}, dpix={config['dpix']}, "
              f"shift=({config['shift_x']}, {config['shift_y']}), rotation={config['rotation']}")


if __name__ == "__main__":
    main()
