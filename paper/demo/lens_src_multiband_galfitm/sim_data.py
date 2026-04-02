from pathlib import Path

import numpy as np
from astropy.io import fits

from TinyLensGpu.ForwardSimulation import LensSimulator, SimulatorConfig, make_grid_2d
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d_transformed
from TinyLensGpu.PhysicalModel import GaussianEllipse, PhysicalModel, SersicEllipse, SIE
from TinyLensGpu.utils.geometry import phi_q2_ellipticity
from TinyLensGpu.utils.chebyshev import (
    chebyshev_node,
    evaluate_chebyshev_series,
    compute_wavelength_range,
)


def mock_lens(ideal_image: np.ndarray, back_rms: float, exp_time: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    noise_map = np.sqrt(ideal_image / exp_time + back_rms**2)
    noisy_image = ideal_image + rng.normal(0.0, noise_map)
    return noisy_image, noise_map


def compute_band_parameters(
    band: str,
    band_wavelengths: dict[str, float],
    lambda_min: float,
    lambda_max: float,
) -> dict:
    """
    Compute Sersic parameters for a given band using Chebyshev polynomials.
    
    Following GALFITM method:
    - R_sersic and n_sersic evolve with wavelength via Chebyshev polynomials
    - Position and ellipticity remain constant across bands
    
    Parameters
    ----------
    band : str
        Band name (e.g., 'g', 'r', 'i')
    band_wavelengths : dict
        Dictionary mapping band names to wavelengths
    lambda_min, lambda_max : float
        Wavelength range for Chebyshev normalization
        
    Returns
    -------
    params : dict
        Dictionary with R_sersic, n_sersic, etc. for this band
    """
    wavelength = band_wavelengths[band]
    z = chebyshev_node(wavelength, lambda_min, lambda_max)
    
    # Source galaxy: blue, exponential-like disk
    # R_sersic decreases slightly with wavelength (galaxy appears smaller in redder bands)
    # Using 2nd-order Chebyshev: c0=0.35, c1=-0.05, c2=0.02
    src_R_coeffs = [0.35, -0.05, 0.02]
    src_R_sersic = float(evaluate_chebyshev_series(z, src_R_coeffs))
    
    # n_sersic increases with wavelength (disk more prominent in blue, bulge in red)
    # Using 2nd-order Chebyshev: c0=0.9, c1=0.15, c2=-0.03
    src_n_coeffs = [0.9, 0.15, -0.03]
    src_n_sersic = float(evaluate_chebyshev_series(z, src_n_coeffs))
    
    # Lens galaxy: red, bulge-dominated
    # R_sersic nearly constant (bulge dominates)
    # Using 2nd-order Chebyshev: c0=1.05, c1=0.02, c2=-0.01
    lens_R_coeffs = [1.05, 0.02, -0.01]
    lens_R_sersic = float(evaluate_chebyshev_series(z, lens_R_coeffs))
    
    # n_sersic ~4 (de Vaucouleurs) with slight evolution
    # Using 2nd-order Chebyshev: c0=4.1, c1=-0.1, c2=0.05
    lens_n_coeffs = [4.1, -0.1, 0.05]
    lens_n_sersic = float(evaluate_chebyshev_series(z, lens_n_coeffs))
    
    return {
        "src_R_sersic": src_R_sersic,
        "src_n_sersic": src_n_sersic,
        "lens_R_sersic": lens_R_sersic,
        "lens_n_sersic": lens_n_sersic,
    }


def build_band_physical_model(
    band: str,
    band_parameters: dict,
    e1_lens: float,
    e2_lens: float,
) -> PhysicalModel:
    """Build physical model with band-specific parameters."""
    return PhysicalModel(
        lens_mass=[
            SIE(theta_E=1.5, e1=e1_lens, e2=e2_lens, center_x=0.0, center_y=0.0),
        ],
        source_light=[
            SersicEllipse(
                R_sersic=band_parameters["src_R_sersic"],
                n_sersic=band_parameters["src_n_sersic"],
                e1=0.05,
                e2=0.05,
                center_x=0.0,
                center_y=0.3,
                Ie=1.0,
            )
        ],
        lens_light=[
            SersicEllipse(
                R_sersic=band_parameters["lens_R_sersic"],
                n_sersic=band_parameters["lens_n_sersic"],
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
        "r": {"npix": 180, "dpix": 0.08, "nsub": 16, "shift_x": 0.02, "shift_y": -0.015, "rotation": 0.573},
        "i": {"npix": 160, "dpix": 0.09, "nsub": 16, "shift_x": 0.0, "shift_y": 0.0, "rotation": 0.0},
    }

    back_rms = 0.1
    exp_time = 300.0

    band_psf_sigma = {
        "g": 0.03,
        "r": 0.05,
        "i": 0.07,
    }

    # Get band wavelengths and compute wavelength range for Chebyshev normalization
    bands = list(band_configs.keys())
    band_wavelengths = {"g": 4770.0, "r": 6231.0, "i": 7625.0}
    lambda_min, lambda_max = compute_wavelength_range(list(band_wavelengths.values()))
    
    print("Chebyshev polynomial wavelength evolution:")
    print(f"  Wavelength range: [{lambda_min:.0f}, {lambda_max:.0f}] Angstroms")
    for band in bands:
        z = chebyshev_node(band_wavelengths[band], lambda_min, lambda_max)
        print(f"  {band}-band ({band_wavelengths[band]:.0f}A): z = {z:.3f}")
    
    # Print Chebyshev coefficients
    print("\nChebyshev coefficients (2nd-order):")
    print("  Source R_sersic: c0=0.35, c1=-0.05, c2=0.02")
    print("  Source n_sersic: c0=0.90, c1=0.15, c2=-0.03")
    print("  Lens R_sersic:   c0=1.05, c1=0.02, c2=-0.01")
    print("  Lens n_sersic:   c0=4.10, c1=-0.10, c2=0.05")

    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute lens ellipticity (constant across bands)
    e1_lens, e2_lens = phi_q2_ellipticity(90.0 * np.pi / 180.0, 0.9)

    for idx, (band, sigma) in enumerate(band_psf_sigma.items()):
        config = band_configs[band]
        npix = config["npix"]
        dpix = config["dpix"]
        nsub = config["nsub"]
        
        # Compute band-specific parameters using Chebyshev polynomials
        band_params = compute_band_parameters(band, band_wavelengths, lambda_min, lambda_max)
        
        print(f"\n{band}-band parameters:")
        print(f"  Source: R_sersic={band_params['src_R_sersic']:.3f}, n_sersic={band_params['src_n_sersic']:.3f}")
        print(f"  Lens:   R_sersic={band_params['lens_R_sersic']:.3f}, n_sersic={band_params['lens_n_sersic']:.3f}")

        # Build physical model for this band
        phy_model = build_band_physical_model(band, band_params, e1_lens, e2_lens)

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
    print("\nGenerated multiband data with heterogeneous geometry:")
    for band, config in band_configs.items():
        print(f"  {band}: npix={config['npix']}, dpix={config['dpix']}, "
              f"shift=({config['shift_x']}, {config['shift_y']}), rotation={config['rotation']}")
    
    # Save true parameters for validation
    print("\nTrue band parameters (for validation):")
    for band in bands:
        band_params = compute_band_parameters(band, band_wavelengths, lambda_min, lambda_max)
        print(f"  {band}: src_R={band_params['src_R_sersic']:.4f}, src_n={band_params['src_n_sersic']:.4f}, "
              f"lens_R={band_params['lens_R_sersic']:.4f}, lens_n={band_params['lens_n_sersic']:.4f}")


if __name__ == "__main__":
    main()
