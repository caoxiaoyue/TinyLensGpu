"""Generate reproducible mock lens-light data with a constant sky background."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

from TinyLensGpu.ForwardSimulation import LensSimulator, SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d
from TinyLensGpu.PhysicalModel import ConstantBackground, GaussianEllipse, PhysicalModel, SersicEllipse
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


RNG_SEED = 42
SKY_INTENSITY = 0.5
BACKGROUND_RMS = 0.1
EXPOSURE_TIME = 300
DATA_DIR = Path("data")

e1_l, e2_l = phi_q2_ellipticity(90 * np.pi / 180, 0.9)

phy_model = PhysicalModel(
    lens_mass=[],
    source_light=[],
    lens_light=[
        SersicEllipse(R_sersic=1.0, n_sersic=4.0, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0, Ie=1.0),
        ConstantBackground(intensity=SKY_INTENSITY),
    ],
)

x_psf, y_psf = make_grid_2d(21, 0.074)
psf_kernel = GaussianEllipse(flux=1.0, sigma=0.05, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0).light(x=x_psf, y=y_psf)
psf_kernel /= psf_kernel.sum()
psf_kernel = np.asarray(psf_kernel)
sim_config = SimulatorConfig(
    dpix=0.074,
    npix=200,
    psf_kernel=psf_kernel,
    nsub=16,
)

sim_obj = LensSimulator(
    phy_model,
    sim_config,
)

img_2d = sim_obj.simulate()
plt.figure()
plt.imshow(img_2d, origin='lower')
plt.colorbar()
plt.close()

def mock_lens(ideal_image, back_rms, exp_time, rng):
    """Add reproducible Gaussian noise using a Poisson plus background model."""
    noise_map = np.sqrt(ideal_image / exp_time + back_rms ** 2)
    noisy_image = ideal_image + rng.normal(0.0, noise_map)
    return noisy_image, noise_map


rng = np.random.default_rng(RNG_SEED)
noisy_image, noise_map = mock_lens(img_2d, BACKGROUND_RMS, EXPOSURE_TIME, rng)
plt.figure()
plt.imshow(noisy_image / noise_map, origin='lower')
plt.colorbar()
plt.close()

DATA_DIR.mkdir(exist_ok=True)
fits.writeto(DATA_DIR / 'image.fits', np.asarray(noisy_image), overwrite=True)
fits.writeto(DATA_DIR / 'noise.fits', np.asarray(noise_map), overwrite=True)
fits.writeto(DATA_DIR / 'psf.fits', np.asarray(psf_kernel), overwrite=True)

xgrid, ygrid = make_grid_2d(200, 0.074)
rgrid = np.sqrt(xgrid ** 2 + ygrid ** 2)
mask = rgrid > 5.0
fits.writeto(DATA_DIR / 'mask.fits', np.asarray(mask).astype(np.int32), overwrite=True)
