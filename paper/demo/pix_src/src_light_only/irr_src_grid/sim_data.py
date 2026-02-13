"""
Simulate gravitational lensing data for irregular grid pixelized source demo.

This script simulates a lensing system with a Sersic source and an SIE lens,
and saves the noisy image, noise map, PSF, and mask to FITS files.
"""

import os
import numpy as np
from astropy.io import fits

from TinyLensGpu.PhysicalModel import PhysicalModel, SersicEllipse, SIE, GaussianEllipse
from TinyLensGpu.ForwardSimulation import SimulatorConfig, LensSimulator, make_grid_2d
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


def simulate_lensing_data():
    """Simulate lensing data and return a dictionary of arrays."""
    print("Simulating lensing data...")
    
    # 1. Define physical model
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

    # 2. Simulation configuration
    npix = 200
    image_size = 10.0
    dpix = image_size / npix

    # PSF
    x_psf, y_psf = make_grid_2d(21, dpix)
    psf_kernel = GaussianEllipse(
        flux=1.0,
        sigma=0.05,
        e1=0.0,
        e2=0.0,
        center_x=0.0,
        center_y=0.0,
    ).light(x=x_psf, y=y_psf)
    psf_kernel /= psf_kernel.sum()
    psf_kernel = np.asarray(psf_kernel)

    sim_config = SimulatorConfig(dpix=dpix, npix=npix, psf_kernel=psf_kernel, nsub=16)
    sim_obj = LensSimulator(phy_model, sim_config)
    img_2d = sim_obj.simulate()

    # 3. Add noise
    def mock_lens(ideal_image, back_rms, exp_time):
        noise_map = np.sqrt(ideal_image / exp_time + back_rms**2)
        noisy_image = ideal_image + np.random.normal(0, noise_map)
        return noisy_image, noise_map

    noisy_image, noise_map = mock_lens(img_2d, 0.1, 300)

    # 4. Create mask
    xgrid_image, ygrid_image = make_grid_2d(npix, dpix)
    rgrid_image = np.sqrt(xgrid_image**2 + ygrid_image**2)
    mask = rgrid_image > 3.0

    return {
        "noisy_image": noisy_image,
        "noise_map": noise_map,
        "psf_kernel": psf_kernel,
        "mask": mask,
        "dpix": dpix,
    }


def main():
    # Set random seed
    np.random.seed(0)
    
    # Simulate data
    data_dict = simulate_lensing_data()
    
    # Save to disk
    os.makedirs('data', exist_ok=True)
    
    print("Saving data to data/ directory...")
    fits.writeto('data/image.fits', np.asarray(data_dict['noisy_image']), overwrite=True)
    fits.writeto('data/noise.fits', np.asarray(data_dict['noise_map']), overwrite=True)
    fits.writeto('data/psf.fits', np.asarray(data_dict['psf_kernel']), overwrite=True)
    fits.writeto('data/mask.fits', np.asarray(data_dict['mask']).astype(np.int32), overwrite=True)
    
    print("Done.")


if __name__ == "__main__":
    main()
