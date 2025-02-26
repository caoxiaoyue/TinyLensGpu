#%%
from astropy.table import Table, join
import numpy as np
import matplotlib.pyplot as plt

votable = Table.read('vizier_votable_ra_dec.vot')
joint_table = Table.read('joint_table.csv', format='csv')
#match the two tables with "lens_name" 

# %%
matched_table = join(joint_table, votable, keys_left='lens_name', keys_right='Name1')

# %%
new_tab = matched_table['lens_name', '_RA1', '_DE1', "zFG1", "zBG1", "bSIE", "thetaE_lens_l", "thetaE_lens_m", "thetaE_lens_u"]

# %%
new_tab.write('slacs_with_ra_dec.csv', format='csv')

# %%
