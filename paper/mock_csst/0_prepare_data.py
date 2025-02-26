#%%
from astropy.io import fits
import numpy as np
import pickle
import gzip
import os

#load csst_mocks.pklz
with gzip.open('csst_mocks.pklz', 'rb') as f:
    csst_mocks = pickle.load(f)

#%%
image = csst_mocks['image']
arc_image = csst_mocks['arc_image']
noise_map = csst_mocks['noise_map']
arc_snr = csst_mocks['arc_snr']
sample_table = csst_mocks['sample_table']
psf_kernel = csst_mocks['psf_kernel']

#%%
from matplotlib import pyplot as plt
os.makedirs('dataset', exist_ok=True)

def save_lens_data(index):
    image = csst_mocks['image'][index]
    arc_image = csst_mocks['arc_image'][index]
    noise_map = csst_mocks['noise_map'][index]
    arc_snr = str(csst_mocks['arc_snr'][index])
    psf_kernel = csst_mocks['psf_kernel']

    os.makedirs(f'dataset/lens_{index}', exist_ok=True)
    fits.writeto(f'dataset/lens_{index}/image.fits', image, overwrite=True)
    fits.writeto(f'dataset/lens_{index}/arc_image.fits', arc_image, overwrite=True)
    fits.writeto(f'dataset/lens_{index}/noise_map.fits', noise_map, overwrite=True)
    fits.writeto(f'dataset/lens_{index}/psf_kernel.fits', psf_kernel, overwrite=True)
    with open(f'dataset/lens_{index}/arc_snr.txt', 'w') as f:
        f.write(arc_snr)

    plt.figure()
    plt.subplot(221)
    plt.imshow(image, origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('image')
    plt.subplot(222)
    plt.imshow(noise_map, origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('noise_map')
    plt.subplot(223)
    plt.imshow(arc_image, origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('arc_image')
    plt.subplot(224)
    plt.imshow(psf_kernel, origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('psf_kernel')
    plt.tight_layout()
    plt.savefig(f'dataset/lens_{index}/lens_subplot.png', bbox_inches='tight')
    plt.close()
    
#%%
save_lens_data(0)

# %%
from multiprocessing import Pool
pool = Pool(processes=24)
pool.map(save_lens_data, range(len(csst_mocks['image'])))
pool.close()

#%%
sample_table.write('dataset/sample_table.csv', format='csv', overwrite=True)

# %%
