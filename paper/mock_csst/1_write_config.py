# %%
import yaml
import os

# Model components configuration
lens_mass_list = [
    {
        'type': "SIE",
        'params': {
            'theta_E': {
                'prior_type': "uniform",
                'prior_settings': [0.1, 3.5],
                'limits': [0.0, 10.0],
                'fixed': False
            },
            'e1': {
                'prior_type': "gaussian",
                'prior_settings': [0.0, 0.3],
                'limits': [-1.0, 1.0],
                'fixed': False
            },
            'e2': {
                'prior_type': "gaussian",
                'prior_settings': [0.0, 0.3],
                'limits': [-1.0, 1.0],
                'fixed': False
            },
            'center_x': {
                'prior_type': "gaussian",
                'prior_settings': [0.0, 0.1],
                'limits': [-1.0, 1.0],
                'fixed': False
            },
            'center_y': {
                'prior_type': "gaussian",
                'prior_settings': [0.0, 0.1],
                'limits': [-1.0, 1.0],
                'fixed': False
            }
        }
    },
]

source_light_list = [
    {   
        'type': "Sersic",
        'params': {
            'R_sersic': {
                'prior_type': "uniform",
                'prior_settings': [0.001, 2.00],
                'limits': [0.0, 5.0],
                'fixed': False
            },
            'n_sersic': {
                'prior_type': "uniform",
                'prior_settings': [0.3, 2.5],
                'limits': [0.3, 8.0],
                'fixed': False
            },
            'e1': {
                'prior_type': "gaussian",
                'prior_settings': [0.0, 0.3],
                'limits': [-1.0, 1.0],
                'fixed': False
            },
            'e2': {
                'prior_type': "gaussian",
                'prior_settings': [0.0, 0.3],
                'limits': [-1.0, 1.0],
                'fixed': False
            },
            'center_x': {
                'prior_type': "gaussian",
                'prior_settings': [0.0, 1.0],
                'limits': [-3.0, 3.0],
                'fixed': False
            },
            'center_y': {
                'prior_type': "gaussian",
                'prior_settings': [0.0, 1.0],
                'limits': [-3.0, 3.0],
                'fixed': False
            },
            'Ie': {
                'use_linear': True
            }
        }
    }
]

lens_light_list = [
    {
        'type': "Sersic",
        'params': {
            'R_sersic': {
                'prior_type': "uniform",
                'prior_settings': [0.001, 4.0],
                'limits': [0.0, 5.0],
                'fixed': False
            },
            'n_sersic': {
                'prior_type': "gaussian",
                'prior_settings': [4.0, 1.0],
                'limits': [0.3, 8.0],
                'fixed': False
            },
            'e1': {
                'prior_type': "gaussian",
                'prior_settings': [0.0, 0.3],
                'limits': [-1.0, 1.0],
                'fixed': False
            },
            'e2': {
                'prior_type': "gaussian",
                'prior_settings': [0.0, 0.3],
                'limits': [-1.0, 1.0],
                'fixed': False
            },
            'center_x': {
                'fixed': True,
                'fixed_value': {
                    'component_type': "lens_mass_list",
                    'component_idx': 0,
                    'parameter': "center_x"
                }   
            },
            'center_y': {
                'fixed': True,
                'fixed_value': {
                    'component_type': "lens_mass_list",
                    'component_idx': 0,
                    'parameter': "center_y"
                }
            },
            'Ie': {
                'use_linear': True
            }
        }
    }
]

model_components_config = {
    'lens_mass_list': lens_mass_list,
    'source_light_list': source_light_list,
    'lens_light_list': lens_light_list
}

# Inference configuration
inference_config = {
    'type': "sampler",
    'method': "nautilus",
    'settings': {
        'nlive': 200,
        'batch_size': 600,
        'verbose': True
    }
}
# inference_config = {
#     "type": "sampler",
#     "method": "dynesty",
#     "settings": {
#         "nlive": 200,
#         "dlogz": 0.05,
#         "bound": "multi",
#         "sample": "rwalk"
#     }
# }
# inference_config = {
#     "type": "optimizer",
#     "method": "differential_evolution",
#     "settings": {
#         "strategy": "best1bin",
#         "maxiter": 1000,
#         "popsize": 15,
#         "tol": 0.01,
#         "mutation": [0.5, 1.0],
#         "recombination": 0.7,
#         "seed": 42,
#         "polish": True,
#         "init": "latinhypercube",
#         "atol": 0,
#         "updating": "immediate"
#     }
# }

def save_modeling_config(index):
    # Dataset configuration
    dataset_config = {
        'data_path': f"dataset/lens_{index}/image.fits",
        'noise_path': f"dataset/lens_{index}/noise_map.fits",
        'psf_path': f"dataset/lens_{index}/psf_kernel.fits",
        'pixel_scale': 0.074
    }

    # Output configuration
    output_config = {
        'path': f"results/lens_{index}",
        'figures': {
            'results': "model_results.png",
            'corner': "model_corner.png"
        },
        'tables': {
            'samples': "result_samples.csv",
            'summary': "result_summary.csv"
        },
        'datasets': {
            'subplot': "dataset_subplot.png"
        }
    }

    # Combine all configurations
    config = {
        'dataset': dataset_config,
        'model_components': model_components_config,
        'inference': inference_config,
        'output': output_config,
        "solver_type": "nnls",
    }

    os.makedirs(f"results/lens_{index}", exist_ok=True)
    # Save configuration
    with open(f"results/lens_{index}/modeling_config.yaml", "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(24) as p:
        p.map(save_modeling_config, range(1000))


#%%