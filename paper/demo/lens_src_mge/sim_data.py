#%%
from TinyLensGpu.Simulator.Image import Simulator
from TinyLensGpu.Profile.Light.Sersic import SersicEllipse
from TinyLensGpu.Profile.Mass.Sie import SIE
from TinyLensGpu.Profile.Mass.Shear import Shear

phy_model = Simulator.PhysicalModel(
    lens_mass=[SIE(), Shear()],
    source_light=[SersicEllipse()],
    lens_light=[SersicEllipse()],
)

# %%
from TinyLensGpu.Simulator.Image.Simulator import SimulatorConfig
from TinyLensGpu.Simulator.Image.Simulator import util as su
from TinyLensGpu.Profile.Light.Gaussian import Gaussian
import numpy as np

x_psf, y_psf = su.make_grid_2d(21, 0.074)
psf_kernel = Gaussian().light(x_psf, y_psf, 1.0, 0.05, 0.0, 0.0)
psf_kernel /= psf_kernel.sum()
psf_kernel = np.asarray(psf_kernel)
sim_config = SimulatorConfig(
    dpix=0.074,
    npix=200,
    psf_kernel=psf_kernel,
    nsub=16,
)

from TinyLensGpu.Simulator.Image import Simulator
sim_obj = Simulator.LensSimulator(
    phy_model,
    sim_config
)


# %%
from TinyLensGpu.Profile import util
e1_l, e2_l = util.phi_q2_ellipticity(90*np.pi/180, 0.9)
lens_mass_par_list = [
    dict(
        theta_E=1.5,
        e1=e1_l,
        e2=e2_l,
        center_x=0.0,
        center_y=0.0,
    ),
    dict(
        gamma1=0.05,
        gamma2=0.05,
    ),
]

source_light_par_list = [
    dict(
        R_sersic=0.3,
        n_sersic=1.0,
        e1=0.05,
        e2=0.05,
        center_x=0.0,
        center_y=0.5,
        Ie=1.0,
    )
]

lens_light_par_list = [
    dict(
        R_sersic=1.0,
        n_sersic=4.0,
        e1=e1_l,
        e2=e2_l,
        center_x=0.0,
        center_y=0.0,
        Ie=1.0,  
    )
]

params = [
    lens_mass_par_list,
    source_light_par_list,
    lens_light_par_list,
]

#%%
img_2d = sim_obj.simulate(
    params,
    xgrid_sub=sim_obj.xgrid_sub,
    ygrid_sub=sim_obj.ygrid_sub,
    psf_kernel=psf_kernel
)
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
