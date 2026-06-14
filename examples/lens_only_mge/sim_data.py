#%%
from TinyLensGpu.PhysicalModel import PhysicalModel, SersicEllipse, SIE, Shear
from TinyLensGpu.utils.geometry import phi_q2_ellipticity
import numpy as np

# %%
e1_l, e2_l = phi_q2_ellipticity(90*np.pi/180, 0.9)

phy_model = PhysicalModel(
    lens_mass=[],
    source_light=[],
    lens_light=[
        SersicEllipse(R_sersic=1.0, n_sersic=4.0, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0, Ie=1.0)
    ],
)

# %%
from TinyLensGpu.ForwardSimulation import SimulatorConfig, LensSimulator
from TinyLensGpu.ForwardSimulation.LensImage import make_grid_2d
from TinyLensGpu.PhysicalModel import GaussianEllipse

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
    sim_config
)


# %%
img_2d = sim_obj.simulate()
from matplotlib import pyplot as plt
plt.figure()
plt.imshow(img_2d, origin='lower')
# plt.imshow(np.log10(img_2d), origin='lower')
plt.colorbar()
plt.show()

# %%
import numpy as np
def mock_lens(ideal_image, back_rms, exp_time):
    noise_map = np.sqrt(ideal_image/exp_time + back_rms**2)
    noisy_image = ideal_image + np.random.normal(0, noise_map)
    return noisy_image, noise_map

noisy_image, noise_map = mock_lens(img_2d, 0.1, 300)
plt.figure()
plt.imshow(noisy_image/noise_map, origin='lower')
plt.colorbar()
plt.show()

#%%
import os
os.makedirs('data', exist_ok=True)
from astropy.io import fits
fits.writeto('data/image.fits', np.asarray(noisy_image), overwrite=True)
fits.writeto('data/noise.fits', np.asarray(noise_map), overwrite=True)
fits.writeto('data/psf.fits', np.asarray(psf_kernel), overwrite=True)

# %%
