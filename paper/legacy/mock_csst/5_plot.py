# %%
from matplotlib import pyplot as plt
import scienceplots

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

# %%
from astropy.table import Table
import numpy as np

tab_matched = Table.read("table_matched.csv")

#------------get lens parameters------------
thetaE_true = tab_matched["thetaE_lens_truth"].data
thetaE_m = tab_matched["thetaE_lens_50"].data
thetaE_err = (tab_matched["thetaE_lens_84"].data - tab_matched["thetaE_lens_16"].data) / 2

e1_lens_true = tab_matched["e1_lens_truth"].data
e1_lens_m = tab_matched["e1_lens_50"].data
e1_lens_err = (tab_matched["e1_lens_84"].data - tab_matched["e1_lens_16"].data) / 2

e2_lens_true = tab_matched["e2_lens_truth"].data
e2_lens_m = tab_matched["e2_lens_50"].data
e2_lens_err = (tab_matched["e2_lens_84"].data - tab_matched["e2_lens_16"].data) / 2

q_lens_true = tab_matched["q_lens_truth"].data
q_lens_m = tab_matched["q_lens_50"].data
q_lens_err = (tab_matched["q_lens_84"].data - tab_matched["q_lens_16"].data) / 2

phi_lens_m = tab_matched["phi_lens_50"].data
phi_lens_err = (tab_matched["phi_lens_84"].data - tab_matched["phi_lens_16"].data) / 2
phi_lens_true = np.full_like(phi_lens_m, 90.0)

#------------get source parameters------------
re_src_true = tab_matched["re_src_truth"].data
re_src_m = tab_matched["re_src_50"].data
re_src_err = (tab_matched["re_src_84"].data - tab_matched["re_src_16"].data) / 2

q_src_true = tab_matched["q_src_truth"].data
q_src_m = tab_matched["q_src_50"].data
q_src_err = (tab_matched["q_src_84"].data - tab_matched["q_src_16"].data) / 2

mag_src_true = tab_matched["magnitude_src_truth"].data
mag_src_m = tab_matched["magnitude_src_50"].data
mag_src_err = (tab_matched["magnitude_src_84"].data - tab_matched["magnitude_src_16"].data) / 2

# %%
import sys
sys.path.append("..")
from analyze_util import nmad_plot, nmad_plot_single_truth

threshold = 4

# %%
columnwidth = 3.33
aspect_ratio = 1
fig = plt.figure(figsize=(columnwidth*3+2*1.0, 2*columnwidth*aspect_ratio+1*0.2))

gs1 = plt.GridSpec(2, 3, height_ratios=[3, 1])
gs2 = plt.GridSpec(2, 3, height_ratios=[3, 1])
gs1.update(hspace=0.0, wspace=0.45, top=0.95, bottom=0.55, left=0.1, right=0.95)
gs2.update(hspace=0.0, wspace=0.45, top=0.45, bottom=0.05, left=0.1, right=0.95)

#the first rowL: lens parameters
#thetaE
ax_main = fig.add_subplot(gs1[0, 0])
ax_residual = fig.add_subplot(gs1[1, 0], sharex=ax_main)
nmad_plot(ax_main, ax_residual, thetaE_true, thetaE_m, thetaE_err, threshold=threshold, normalize=True)
ax_main.set_ylabel(r"$\theta_{\rm E}^{\rm M}$ [arcsec]")
ax_residual.set_xlabel(r"$\theta_{\rm E}^{\rm T}$ [arcsec]")
ax_residual.set_ylabel(r"$\Delta \theta_{\rm E}/\theta_{\rm E}^{\rm T}$")
ax_main.set_xlim(0.5, 3.5)
ax_main.set_ylim(0.5, 3.5)
ax_residual.set_ylim(-0.1, 0.1)
ax_residual.set_yticks([-0.1, -0.05, 0.0, 0.05])

#q
ax_main = fig.add_subplot(gs1[0, 1])
ax_residual = fig.add_subplot(gs1[1, 1], sharex=ax_main)
nmad_plot(ax_main, ax_residual, q_lens_true, q_lens_m, q_lens_err, threshold=threshold, normalize=False)
ax_main.set_ylabel(r"$q_1^{\rm M}$")
ax_main.set_xlim(0.2, 1.0)
ax_main.set_ylim(0.2, 1.0)
ax_residual.set_ylim(-0.3, 0.3)
ax_residual.set_yticks([-0.3, -0.15, 0.0, 0.15])
ax_residual.set_xlabel(r"$q_1^{\rm T}$")
ax_residual.set_ylabel(r"$\Delta q_1$")

#phi
ax_main = fig.add_subplot(gs1[0, 2])
ax_residual = fig.add_subplot(gs1[1, 2], sharex=ax_main)
nmad_plot_single_truth(ax_main, ax_residual, 90.0, phi_lens_m, phi_lens_err, threshold=threshold, normalize=False)
ax_main.set_ylabel(r"$\phi_1^{\rm M} \, [\circ]$")
ax_main.set_xlim(1, 1000)
ax_main.set_ylim(0.0, 180.0)
ax_residual.set_ylim(-20, 20)
ax_residual.set_yticks([-20, -10, 0, 10])
ax_residual.set_xlabel(r"Lens ID")
ax_residual.set_ylabel(r"$\Delta \phi_1$")

#the second row: source parameters
#re
ax_main = fig.add_subplot(gs2[0, 0])
ax_residual = fig.add_subplot(gs2[1, 0], sharex=ax_main)
nmad_plot(ax_main, ax_residual, re_src_true, re_src_m, re_src_err, threshold=threshold, normalize=True)
ax_main.set_ylabel(r"$r_{\rm s}^{\rm M}$ [arcsec]")
ax_main.set_xlim(0.0, 1.2)
ax_main.set_ylim(0.0, 1.2)
ax_residual.set_ylim(-0.1, 0.1)
ax_residual.set_yticks([-0.1, -0.05, 0.0, 0.05])
ax_residual.set_xlabel(r"$r_{\rm s}^{\rm T}$ [arcsec]")
ax_residual.set_ylabel(r"$\Delta r_{\rm s}/r_{\rm s}^{\rm T}$")

#q
ax_main = fig.add_subplot(gs2[0, 1])
ax_residual = fig.add_subplot(gs2[1, 1], sharex=ax_main)
nmad_plot(ax_main, ax_residual, q_src_true, q_src_m, q_src_err, threshold=threshold, normalize=False)
ax_main.set_ylabel(r"$q_s^{\rm M}$")
ax_main.set_xlim(0.2, 1.0)
ax_main.set_ylim(0.2, 1.0)
ax_residual.set_ylim(-0.3, 0.3)
ax_residual.set_yticks([-0.3, -0.15, 0.0, 0.15])
ax_residual.set_xlabel(r"$q_s^{\rm T}$")
ax_residual.set_ylabel(r"$\Delta q_s$")

#mag
ax_main = fig.add_subplot(gs2[0, 2])
ax_residual = fig.add_subplot(gs2[1, 2], sharex=ax_main)
nmad_plot(ax_main, ax_residual, mag_src_true, mag_src_m, mag_src_err, threshold=threshold, normalize=False)
ax_main.set_ylabel(r"$m_s^{\rm M}$ [mag]")
ax_main.set_xlim(20.5, 23.5)
ax_main.set_ylim(20.5, 23.5)
ax_residual.set_ylim(-1, 1)
ax_residual.set_yticks([-1, -0.5, 0.0, 0.5])
ax_residual.set_xlabel(r"$m_s^{\rm T}$ [mag]")
ax_residual.set_ylabel(r"$\Delta m_s$")

plt.savefig("mock_summary.pdf", bbox_inches="tight", dpi=300)

# %%



