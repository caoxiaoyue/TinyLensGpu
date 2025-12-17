#%%
from astropy.io import fits
import numpy as np
import pickle
import gzip
import os
from astropy.table import Table

def MTF(x, m):
    x = np.clip(x, 0, 1)
    y = ((m - 1) * x) / (((2 * m - 1) * x - m))
    return np.where(x == 0, 0,
                    np.where(x == m, 0.5,
                             np.where(x == 1, 1, y)))

def auto_mtf(image, mean = 0.125, clip_low = 0.000, clip_high = 0.99):
    image = np.clip(image, clip_low*np.max(image), clip_high*np.max(image))
    x = np.mean(image)
    m = (x-mean*x)/(x-2*mean*x+mean)
    return MTF(image, m)

#load csst_mocks.pklz
with gzip.open('csst_mocks.pklz', 'rb') as f:
    csst_mocks = pickle.load(f)

table_truth = Table.read("dataset/sample_table.csv")
lens_re_truth = table_truth['re_l'].data

#%%
image = csst_mocks['image']
arc_image = csst_mocks['arc_image']
noise_map = csst_mocks['noise_map']
arc_snr = csst_mocks['arc_snr']
sample_table = csst_mocks['sample_table']
psf_kernel = csst_mocks['psf_kernel']


#%%
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

imgWD = 2
nimgs = 30
ncols = 5
nrows = int(nimgs/ncols)
fig = plt.figure(figsize = (imgWD*ncols,imgWD*nrows))
gds = gridspec.GridSpec(nrows, ncols)
gds.update(wspace=0.0, hspace=0.0) 

np.random.seed(42)  # for reproducibility
lens_indices = np.random.choice(len(image), size=nimgs, replace=False)  # generate 50 unique random numbers from available image indices
image_atlas = []
for lens_index in lens_indices:
    image_atlas.append(auto_mtf(image[lens_index], mean = 0.125, clip_low = 0.000, clip_high = 0.99))
image_atlas = np.array(image_atlas)
vmin = np.min(image_atlas)

for j in range(nimgs):
    lens_index = lens_indices[j]
    axs = plt.subplot(gds[int(j/(ncols)), j%(ncols)])
    axs.imshow(image_atlas[j], origin='lower', cmap='gray', vmin=vmin)
    
    # Add scale bar (1 arcsec = 1/0.074 pixels)
    scale_bar_length = int(1/0.074)  # length in pixels for 1 arcsec
    start_x = 5  # position from left
    start_y = 5  # position from bottom
    axs.plot([start_x, start_x + scale_bar_length], [start_y, start_y], '-w', linewidth=2)
    axs.text(start_x, start_y + 3, '1"', color='white', fontsize=8)
    
    axs.text(0.05, 0.9, f"Arc-SNR: {arc_snr[lens_index]['max']:.2f}", transform = axs.transAxes, color='white', fontsize=8)
    axs.text(0.05, 0.8, r"$r_l$: "+f"{lens_re_truth[lens_index]:.2f}"+r"$^{\prime\prime}$", transform = axs.transAxes, color='white', fontsize=8)
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect('equal')
plt.savefig('csst_atlas.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)

# %%
