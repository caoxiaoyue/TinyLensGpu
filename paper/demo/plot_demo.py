# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import gzip
import pickle
import scienceplots

# Use a high-quality journal style
plt.style.use(['science', 'nature', 'no-latex'])

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

# Load Lens Models
lens_types = ['lens_only', 'src_only', 'lens_src']
lens_model_list = []

for lens_type in lens_types:
    with gzip.open(f'{lens_type}/output/lens_model.pkl.gz', 'rb') as f:
        lens_model = pickle.load(f)
    lens_model_list.append(lens_model)

# Figure Settings
columnwidth = 3.33  # Column width in inches
fig, axes = plt.subplots(3, 3, figsize=(columnwidth * 3 + 2 * 0.5, 3 * columnwidth + 2 * 0.2), constrained_layout=True)

for i, lens_model in enumerate(lens_model_list):
    half_width = lens_model.image_map.shape[0] * 0.5 * lens_model.data_config['pixel_scale']
    extent = [-half_width, half_width, -half_width, half_width]

    # Panel 1: Data
    ax = axes[i, 0]
    im = ax.imshow(lens_model.image_map, origin='lower', cmap='inferno', extent=extent)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Data", fontsize=12)

    # Panel 2: Model
    ax = axes[i, 1]
    im = ax.imshow(lens_model.image_model, origin='lower', cmap='inferno', extent=extent)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Model", fontsize=12)

    # Panel 3: Normalized Residuals
    ax = axes[i, 2]
    im = ax.imshow((lens_model.image_model - lens_model.image_map) / lens_model.noise_map, origin='lower', cmap='inferno', extent=extent)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Normalized Residuals", fontsize=12)

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

# Save the figure
plt.savefig("demo_cases.pdf", bbox_inches='tight', dpi=300)
plt.show()

# %%
from dynesty.utils import resample_equal
import corner

# Resample posterior samples
lens_model = lens_model_list[2]
samples = resample_equal(lens_model.inference.samples, lens_model.inference.weights)

# Configure plot aesthetics
plt.rcParams.update({
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.labelsize': 13,
})

# Select parameters for corner plot
sample_plot = samples[:, 0:5]
labels = [r"$\theta_E$", r"$e_1$", r"$e_2$", r"$\gamma_1$", r"$\gamma_2$"]

# Define confidence levels
levels = [1 - np.exp(-0.5), 1 - np.exp(-0.5 * 4), 1 - np.exp(-0.5 * 9)]
credible_level = 0.95
quantiles = [0.5 - credible_level / 2.0, 0.5, 0.5 + credible_level / 2.0]

# Generate corner plot
corner.corner(
    sample_plot,
    levels=levels,
    labels=labels,
    truth_color="black",
    truths=[1.5, -0.052631, 0.0, 0.05, 0.05],  # Adjusted for clarity
    quantiles=quantiles,
    # show_titles=True,
    smooth=1.0,
    smooth1d=1.0,
    title_kwargs={"fontsize": 12},
    plot_datapoints=False,
    plot_density=False,
)

# Save the figure
plt.savefig("demo_posterior.pdf", bbox_inches='tight', dpi=300)
plt.show()
# %%
