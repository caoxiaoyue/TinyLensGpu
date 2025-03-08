#%%
import gzip
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dynesty.utils import resample_equal
from TinyLensGpu.Profile import util
import sys
sys.path.append('..')
from analyze_util import SersicLight, cps2mag
from astropy.table import Table, join
from scipy.stats import norm

# Define the Z-scores for 3-sigma
z_lower = -3
z_upper = 3
# Compute the percentile values
p_lower = norm.cdf(z_lower) * 100  # CDF at -3σ
p_upper = norm.cdf(z_upper) * 100  # CDF at +3σ

def model_table_from(i):
    file_name = f'results/lens_{i}/lens_model.pkl.gz'
    if not os.path.exists(file_name):
        return i
    with gzip.open(file_name, 'rb') as f:
        lens_model = pickle.load(f)

    samples = resample_equal(lens_model.inference.samples, lens_model.inference.weights, rstate=np.random.RandomState(0))
    idx = np.random.choice(len(samples), size=300, replace=False)
    samples_subset = samples[idx]
    src_e1_samps = samples_subset[:, 8]
    src_e2_samps = samples_subset[:, 9]
    src_re_samps = samples_subset[:, 5]
    src_phi_samps, src_q_samps = util.ellipticity2phi_q(src_e1_samps, src_e2_samps)
    src_n_samps = samples_subset[:, 10]
    src_phi_samps *= 180.0 / np.pi #convert src_phi_samps from [-90, 90] to [0, 180]
    src_phi_samps = np.where(src_phi_samps < 0, src_phi_samps + 180.0, src_phi_samps)

    lens_thetaE_samps = samples_subset[:, 4]
    lens_e1_samps = samples_subset[:, 2]
    lens_e2_samps = samples_subset[:, 3]
    lens_phi_samps, lens_q_samps = util.ellipticity2phi_q(lens_e1_samps, lens_e2_samps)
    lens_phi_samps *= 180.0 / np.pi #convert lens_phi_samps from [-90, 90] to [0, 180]
    lens_phi_samps = np.where(lens_phi_samps < 0, lens_phi_samps + 180.0, lens_phi_samps)

    params_dict_list = lens_model.inference.params_array2kargs(samples_subset)
    image_model, intensity_list = lens_model.prob_model.sim_obj.simulate(
        params_dict_list, 
        bs=300,
        use_linear=(lens_model.model_parser.n_linear_params>0), 
        return_intensity=True, 
        image_map=lens_model.image_map, 
        noise_map=lens_model.noise_map,
        xgrid_sub=lens_model.prob_model.sim_obj.xgrid_sub,
        ygrid_sub=lens_model.prob_model.sim_obj.ygrid_sub,
        psf_kernel=lens_model.prob_model.sim_obj.psf_kernel,
    )
    src_Ie_samps = intensity_list[0]
    lens_Ie_samps = intensity_list[1]

    src_flux_samps = SersicLight.total_flux_analytic_from(
        Re=src_re_samps,
        Ie=src_Ie_samps*1/0.074**2,
        n=src_n_samps,
    )
    src_mag_samps = cps2mag(src_flux_samps, 26.23)

    #dict saving the median and std of lens: thetaE, e1, e2; and src: re, q, magnitude
    #easy to build astropy table from this dict
    result_dict = {}
    result_dict['lens_id'] = i
    #get the 16, 50, 84 percentile of lens: thetaE, e1, e2; and src: re, q, magnitude
    result_dict['thetaE_lens_16'] = np.percentile(lens_thetaE_samps, p_lower)
    result_dict['thetaE_lens_50'] = np.median(lens_thetaE_samps)
    result_dict['thetaE_lens_84'] = np.percentile(lens_thetaE_samps, p_upper)
    result_dict['e1_lens_16'] = np.percentile(lens_e1_samps, p_lower)
    result_dict['e1_lens_50'] = np.median(lens_e1_samps)
    result_dict['e1_lens_84'] = np.percentile(lens_e1_samps, p_upper)
    result_dict['e2_lens_16'] = np.percentile(lens_e2_samps, p_lower)
    result_dict['e2_lens_50'] = np.median(lens_e2_samps)
    result_dict['e2_lens_84'] = np.percentile(lens_e2_samps, p_upper)
    result_dict['phi_lens_16'] = np.percentile(lens_phi_samps, p_lower)
    result_dict['phi_lens_50'] = np.median(lens_phi_samps)
    result_dict['phi_lens_84'] = np.percentile(lens_phi_samps, p_upper)
    result_dict['q_lens_16'] = np.percentile(lens_q_samps, p_lower)
    result_dict['q_lens_50'] = np.median(lens_q_samps)
    result_dict['q_lens_84'] = np.percentile(lens_q_samps, p_upper)

    result_dict['re_src_16'] = np.percentile(src_re_samps, p_lower)
    result_dict['re_src_50'] = np.median(src_re_samps)
    result_dict['re_src_84'] = np.percentile(src_re_samps, p_upper)
    result_dict['q_src_16'] = np.percentile(src_q_samps, p_lower)
    result_dict['q_src_50'] = np.median(src_q_samps)
    result_dict['q_src_84'] = np.percentile(src_q_samps, p_upper)
    result_dict['magnitude_src_16'] = np.percentile(src_mag_samps, p_lower)
    result_dict['magnitude_src_50'] = np.median(src_mag_samps)
    result_dict['magnitude_src_84'] = np.percentile(src_mag_samps, p_upper)
    result_dict['phi_src_16'] = np.percentile(src_phi_samps, p_lower)
    result_dict['phi_src_50'] = np.median(src_phi_samps)
    result_dict['phi_src_84'] = np.percentile(src_phi_samps, p_upper)

    lens_model = None

    print(f'{i} done')

    return result_dict


#%%
from multiprocessing import Pool
with Pool(64) as p:
    result_list = p.map(model_table_from, range(1000))

unfinished = [item for item in result_list if type(item) == int]

#build astropy table from result_list
table = Table()
for key in result_list[0].keys():
    table[key] = [item[key] for item in result_list if type(item) != int]
table.sort('lens_id')


# %%
table_truth = Table.read("dataset/sample_table.csv")
src_mag_truth = table_truth['mag_g_s0'].data
src_re_truth = table_truth['re_s0'].data
src_q_truth = table_truth['q_s0'].data
src_phi_truth = table_truth['pa_s0'].data

lens_q_truth = table_truth['q_l'].data
lens_phi_truth = 90.0 * np.ones_like(lens_q_truth)
lens_thetaE_truth = table_truth['thetaE_s0'].data
lens_e1_truth, lens_e2_truth = util.phi_q2_ellipticity(
    lens_phi_truth*np.pi/180.0, 
    lens_q_truth
)

table_truth = Table()
table_truth['lens_id'] = np.arange(len(lens_thetaE_truth), dtype=int)
table_truth['thetaE_lens_truth'] = lens_thetaE_truth
table_truth['e1_lens_truth'] = lens_e1_truth
table_truth['e2_lens_truth'] = lens_e2_truth
table_truth['phi_lens_truth'] = lens_phi_truth
table_truth['q_lens_truth'] = lens_q_truth

table_truth['re_src_truth'] = src_re_truth
table_truth['q_src_truth'] = src_q_truth
table_truth['magnitude_src_truth'] = src_mag_truth
table_truth['phi_src_truth'] = src_phi_truth


# %%
# match table_truth and table using lens_id
table_matched = join(table_truth, table, keys='lens_id')
table_matched.write('table_matched.csv', overwrite=True)


# %%
