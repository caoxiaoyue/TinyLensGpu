#%%
import os
import pickle
import gzip
import numpy as np
from astropy.table import Table, join
from dynesty.utils import resample_equal
from scipy.stats import norm

# Define the Z-scores for 3-sigma
z_lower = -3
z_upper = 3
# Compute the percentile values
p_lower = norm.cdf(z_lower)  # CDF at -3σ
p_upper = norm.cdf(z_upper)  # CDF at +3σ


def model_table_from(lens_name):
    file_name = f'results/{lens_name}/lens_model.pkl.gz'
    if not os.path.exists(file_name):
        print(f'{lens_name} does not exist')
        return lens_name
    with gzip.open(file_name, 'rb') as f:
        lens_model = pickle.load(f)

    samples = resample_equal(lens_model.inference.samples, lens_model.inference.weights)
    lens_thetaE_samps = samples[:, 4]

    result_dict = {}
    result_dict['lens_name'] = lens_name
    #get the 16, 50, 84 percentile of lens: thetaE, e1, e2; and src: re, q, magnitude
    result_dict['thetaE_lens_l'] = np.percentile(lens_thetaE_samps, p_lower*100)
    result_dict['thetaE_lens_m'] = np.median(lens_thetaE_samps)
    result_dict['thetaE_lens_u'] = np.percentile(lens_thetaE_samps, p_upper*100)

    return result_dict

#load lens_names.txt as list
with open("lens_names.txt", "r") as f:
    lens_names = f.readlines()
lens_names = [name.strip() for name in lens_names]

result_list = []
for lens_name in lens_names:
    result_list.append(model_table_from(lens_name))

unfinished = [item for item in result_list if type(item) == str]

my_tab = Table()
for key in result_list[0].keys():
    my_tab[key] = [item[key] for item in result_list if type(item) != str]

    
# %%
slacs_tab = Table.read('vizier_votable.vot')
joined_table = join(my_tab, slacs_tab, keys_left='lens_name', keys_right='Name')

my_thetaE_l = joined_table['thetaE_lens_l'].data
my_thetaE_m = joined_table['thetaE_lens_m'].data
my_thetaE_u = joined_table['thetaE_lens_u'].data
my_thetaE_err = (my_thetaE_u - my_thetaE_l) / 2
lens_names = joined_table['lens_name'].data
slacs_thetaE = joined_table['bSIE'].data

#%%
#write joint table in csv format
joined_table.write('joint_table.csv', format='csv', overwrite=True)

#%%
from matplotlib import pyplot as plt
import scienceplots
import sys
sys.path.append("../../")
from analyze_util import nmad_plot, nmad_plot_single_truth

plt.style.use(['science','no-latex', 'nature'])
# Set the font family and size for all text
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
# Set the background color and grid style
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['grid.color'] = 'black'
plt.rcParams['grid.alpha'] = 0.5
# Set the tick sizes
plt.rcParams['xtick.major.size'] = 6  # Length of major ticks on the x-axis
plt.rcParams['xtick.minor.size'] = 3  # Length of minor ticks on the x-axis
plt.rcParams['ytick.major.size'] = 6  # Length of major ticks on the y-axis
plt.rcParams['ytick.minor.size'] = 3  # Length of minor ticks on the y-axis

columnwidth = 3.33
aspect_ratio = 1

threshold = 4

fig, (ax_main, ax_residual) = plt.subplots(2, 1, 
                                          figsize=(columnwidth, columnwidth*1.4),
                                          gridspec_kw={'height_ratios': [4, 1]})
nmad_plot(ax_main, ax_residual, slacs_thetaE, my_thetaE_m, my_thetaE_err, threshold=threshold, normalize=True)
ax_main.set_aspect('equal', adjustable='box')
ax_main.set_xlim(0.5, 2.0)
ax_main.set_ylim(0.5, 2.0)

# Set consistent ticks for both axes
ticks = np.linspace(0.5, 2.0, 4)  # 5 evenly spaced ticks from 0.5 to 2.0
ax_main.set_xticks(ticks)
ax_main.set_yticks(ticks[1:])
ax_residual.set_xlim(0.5, 2.0)  # Ensure same x limits
ax_residual.set_ylim(-0.5, 0.5)  # Ensure same y limits
ax_residual.set_xticks(ticks[1:])    # Ensure same x ticks

ax_residual.set_xlabel(r"$\theta_\mathrm{E}^\mathrm{Bolton}$ [arcsec]")
ax_residual.set_ylabel(r"$\Delta \theta_\mathrm{E}/\theta_\mathrm{E}^\mathrm{Bolton}$")
ax_main.set_ylabel(r"$\theta_\mathrm{E}^\mathrm{M}$")
# Adjust the spacing between subplots
plt.subplots_adjust(hspace=-0.2, top=0.95, bottom=0.1, left=0.15, right=0.95)
plt.savefig('slacs_summary.pdf', bbox_inches='tight', dpi=300)


#%%




