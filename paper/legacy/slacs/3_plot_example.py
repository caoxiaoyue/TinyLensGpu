#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import scienceplots
import numpy as np
import gzip
import pickle
from matplotlib.colors import LogNorm

# Use scientific journal style
plt.style.use(['science', 'nature', "no-latex"])

# Set global font to Times New Roman (MNRAS requirement)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'axes.grid': False,  # No grid for clarity
    'figure.dpi': 300  # High resolution for publication
})

from TinyLensGpu.Simulator.Image import util

# Load lens models
lens_names = ["J0029-0055", "J0044+0113", "J0157-0056"]
# lens_names = ["J0029-0055", "J0044+0113", "J0216-0813"]
lens_model_list = []

for this_lens_name in lens_names:
    with gzip.open(f'results/{this_lens_name}/lens_model.pkl.gz', 'rb') as f:
        lens_model = pickle.load(f)
    lens_model_list.append(lens_model)

with gzip.open(f'../test2/results/{lens_names[2]}/lens_model.pkl.gz', 'rb') as f:
    lens_model = pickle.load(f)
lens_model_list.append(lens_model)

# Generate source images
source_image_list = []
for lens_model in lens_model_list:
    sim_obj = lens_model.prob_model.sim_obj
    sim_obj.phys_model.lens_mass = []
    sim_obj.phys_model.lens_light = []
    _, source_image_sub = sim_obj._generate_ideal_model(
        np.expand_dims(sim_obj.xgrid_sub, -1),
        np.expand_dims(sim_obj.ygrid_sub, -1),
        lens_model.param_dict,
        len(sim_obj.phys_model.source_light),
        len(sim_obj.phys_model.lens_light),
        len(sim_obj.phys_model.lens_mass)
    )
    source_image_sub *= lens_model.intensity_list[np.newaxis, np.newaxis, np.newaxis, :len(sim_obj.phys_model.source_light)]
    source_image_sub = np.sum(source_image_sub, axis=-1)
    source_image = util.bin_image_general(source_image_sub, lens_model.data_config.get('nsub', 4))
    source_image = np.squeeze(source_image, axis=-1)
    source_image_list.append(source_image)

#%%
# Define figure size
columnwidth = 3.33  # Standard column width (inches)
aspect_ratio = 1
fig, axes = plt.subplots(4, 4, figsize=(columnwidth*4 + 3*0.5, 4*columnwidth*aspect_ratio + 3*0.2), constrained_layout=True)

this_cmap = plt.cm.inferno.copy()
this_cmap.set_bad('white')

for i, lens_model in enumerate(lens_model_list):
    half_width = lens_model.image_map.shape[0] * 0.5 * lens_model.data_config['pixel_scale']
    extent = [-half_width, half_width, -half_width, half_width]
    
    # Determine data range for consistent scaling
    # data_min = np.min(lens_model.image_map[lens_model.image_map > 0])  # Minimum non-zero value
    data_min = np.percentile(np.abs(lens_model.image_map), 5)
    data_max = np.percentile(np.abs(lens_model.image_map), 100)
    
    # Panel 1: Observed Data with log scale to show both faint arcs and bright lens
    if i == 1: print(lens_model.image_map.min(), lens_model.image_map.max())
    ax = axes[i, 0]
    # Use SymLogNorm for better visualization of bright and dark features
    norm = LogNorm(vmin=data_min, vmax=data_max)
    im = ax.imshow(lens_model.image_map, origin='lower', cmap=this_cmap, extent=extent, norm=norm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Data", fontsize=12)
    
    # Panel 2: Lens Light Subtracted Data with vmin=0
    ax = axes[i, 1]
    lens_sub_img = lens_model.image_map - lens_model.image_lens_light
    # Set minimum to 0 to avoid negative values
    vmin = np.min(lens_model.image_lensed_source)
    vmax = np.max(lens_model.image_lensed_source)
    im2 = ax.imshow(lens_sub_img, origin='lower', cmap=this_cmap, extent=extent, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Lens Light Subtracted", fontsize=12)
    
    # Panel 3: Model Lensed Source with same vmin/vmax as col 2
    ax = axes[i, 2]
    # Use same vmin/vmax as the lens-subtracted image for direct comparison
    im3 = ax.imshow(lens_model.image_lensed_source, origin='lower', cmap=this_cmap, 
                   extent=extent, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Model Lensed Source", fontsize=12)

    # Panel 4: Model Source
    ax = axes[i, 3]
    im = ax.imshow(source_image_list[i], origin='lower', cmap=this_cmap, extent=extent)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Model Source", fontsize=12)

    # Formatting for all subplots
    for ax in axes[i, :]:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xticks(np.linspace(-3, 3, 5))
        ax.set_yticks(np.linspace(-3, 3, 5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.tick_params(axis='both', which='both', direction='in', right=True, top=True)
        ax.set_xlabel("Arcsec", fontsize=11)
        if ax == axes[i, 0]:  # Only first column has y-labels
            ax.set_ylabel("Arcsec", fontsize=11)

# Save figure
plt.savefig("slacs_example.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%