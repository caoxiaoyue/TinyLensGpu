#%%
import yaml
import numpy as np

# Dataset configuration
dataset_config = {
    "data_path": "data/image.fits",
    "noise_path": "data/noise.fits",
    "psf_path": "data/psf.fits",
    "pixel_scale": 0.074
}

# Model components configuration
lens_mass_list = []
source_light_list = []

N_mge = 1
N_gaussians_per_mge = 15
sigma_list = 10**(np.linspace(-2.0, np.log10(3.0), N_gaussians_per_mge))
lens_light_list = []
for i in range(N_mge):
    for j in range(N_gaussians_per_mge):
        if j == 0:
            center_x = {
                "fixed": False,
                "prior_type": "gaussian",
                "prior_settings": [0.0, 0.1],
                "limits": [-3.0, 3.0],
            }
            center_y = {
                "fixed": False,
                "prior_type": "gaussian",
                "prior_settings": [0.0, 0.1],
                "limits": [-3.0, 3.0],
            }
            e1 = {
                "fixed": False,
                "prior_type": "gaussian",
                "prior_settings": [0.0, 0.3],
                "limits": [-1.0, 1.0],
            }
            e2 = {
                "fixed": False,
                "prior_type": "gaussian",
                "prior_settings": [0.0, 0.3],
                "limits": [-1.0, 1.0],
            }
        else:
            center_x = {
                "fixed": True,
                "fixed_value": {
                    "component_type": "lens_light_list",
                    "component_idx": i*N_gaussians_per_mge,
                    "parameter": "center_x"
                }
            }
            center_y = {
                "fixed": True,
                "fixed_value": {
                    "component_type": "lens_light_list",
                    "component_idx": i*N_gaussians_per_mge,
                    "parameter": "center_y"
                }
            }
            e1 = {
                "fixed": True,
                "fixed_value": {
                    "component_type": "lens_light_list",
                    "component_idx": i*N_gaussians_per_mge,
                    "parameter": "e1"
                }
            }
            e2 = {
                "fixed": True,
                "fixed_value": {
                    "component_type": "lens_light_list",
                    "component_idx": i*N_gaussians_per_mge,
                    "parameter": "e2"
                }
            }

        this_lens_light = {
            "type": "Gaussian",
            "params": {
                "sigma": {
                    "fixed": True,
                    "fixed_value": float(sigma_list[j])    
                },
                "center_x": center_x,
                "center_y": center_y,
                "e1": e1,
                "e2": e2,
                "flux": {
                    "use_linear": True
                }
            }
        }
        lens_light_list.append(this_lens_light)



model_components_config = {
    "lens_mass_list": lens_mass_list,
    "source_light_list": source_light_list,
    "lens_light_list": lens_light_list
}


inference_config = {
    "type": "sampler",
    "method": "nautilus",
    "settings": {
        "nlive": 200,
        "batch_size": 200
    }
}

output_config = {
    "path": "output",
    "figures": {
        "results": "model_results.png",
        "corner": "model_corner.png"
    },
    "tables": {
        "samples": "result_samples.csv",
        "summary": "result_summary.csv"
    },
    "datasets": {
        "subplot": "dataset_subplot.png"
    }
}

# Combine all configurations
config_dict = {
    "dataset": dataset_config,
    "model_components": model_components_config,
    "inference": inference_config,
    "output": output_config,
    "solver_type": "nnls", #or "normal"; "nnls" is recommended for MGE models
}

# Write the config_dict to the specified path
with open("model_config.yaml", 'w') as f:
    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

# %%
