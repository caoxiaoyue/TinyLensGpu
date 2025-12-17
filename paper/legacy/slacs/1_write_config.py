# %%
import yaml
import os

#load lens_names.txt as list
with open("lens_names.txt", "r") as f:
    lens_names = f.readlines()
lens_names = [name.strip() for name in lens_names]


# Model components configuration
lens_mass_list = [
    {
        'type': "SIE",
        'params': {
            'theta_E': {
                'prior_type': "uniform",
                'prior_settings': [0.3, 3.5],
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
    {
        'type': "SHEAR",
        'params': {
            'gamma1': {
                'prior_type': "uniform",
                'prior_settings': [-0.2, 0.2],
                'limits': [-0.5, 0.5],
                'fixed': False
            },
            'gamma2': {
                'prior_type': "uniform",
                'prior_settings': [-0.2, 0.2],
                'limits': [-0.5, 0.5],
                'fixed': False
            }
        }
    }
]

source_light_list = [
    {   
        'type': "Sersic",
        'params': {
            'R_sersic': {
                'prior_type': "uniform",
                'prior_settings': [0.001, 10.0],
                'limits': [0.0, 10.0],
                'fixed': False
            },
            'n_sersic': {
                # 'prior_type': "uniform",
                # 'prior_settings': [0.3, 6.0],
                # 'limits': [0.3, 6.0],
                'fixed': True,
                'fixed_value': 4.0
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
                'prior_settings': [0.0, 0.5],
                'limits': [-3.5, 3.5],
                'fixed': False
            },
            'center_y': {
                'prior_type': "gaussian",
                'prior_settings': [0.0, 0.5],
                'limits': [-3.5, 3.5],
                'fixed': False
            },
            'Ie': {
                'use_linear': True
            }
        }
    },
    {   
        'type': "Sersic",
        'params': {
            'R_sersic': {
                'prior_type': "uniform",
                'prior_settings': [0.001, 10.0],
                'limits': [0.0, 10.0],
                'fixed': False
            },
            'n_sersic': {
                # 'prior_type': "uniform",
                # 'prior_settings': [0.3, 6.0],
                # 'limits': [0.3, 6.0],
                'fixed': True,
                'fixed_value': 1.0
            },
            'e1': {
                # 'prior_type': "gaussian",
                # 'prior_settings': [0.0, 0.3],
                # 'limits': [-1.0, 1.0],
                'fixed': True,
                'fixed_value': {
                    'component_type': "source_light_list",
                    'component_idx': 0,
                    'parameter': "e1"
                }
            },
            'e2': {
                # 'prior_type': "gaussian",
                # 'prior_settings': [0.0, 0.3],
                # 'limits': [-1.0, 1.0],
                'fixed': True,
                'fixed_value': {
                    'component_type': "source_light_list",
                    'component_idx': 0,
                    'parameter': "e2"
                }
            },
            'center_x': {
                # 'prior_type': "gaussian",
                # 'prior_settings': [0.0, 0.5],
                # 'limits': [-3.5, 3.5],
                'fixed': True,
                'fixed_value': {
                    'component_type': "source_light_list",
                    'component_idx': 0,
                    'parameter': "center_x"
                }
            },
            'center_y': {
                # 'prior_type': "gaussian",
                # 'prior_settings': [0.0, 0.5],
                # 'limits': [-3.5, 3.5],
                'fixed': True,
                'fixed_value': {
                    'component_type': "source_light_list",
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

lens_light_list = [
    {
        'type': "Sersic",
        'params': {
            'R_sersic': {
                'prior_type': "uniform",
                'prior_settings': [0.01, 10.0],
                'limits': [0.0, 10.0],
                'fixed': False
            },
            'n_sersic': {
                # 'prior_type': "uniform",
                # 'prior_settings': [0.3, 6.0],
                # 'limits': [0.3, 6.0],
                'fixed': True, #NOTE here we fix the Sersic index to 4.0
                'fixed_value': 4.0
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
    },
    {
        'type': "Sersic",
        'params': {
            'R_sersic': {
                'prior_type': "uniform",
                'prior_settings': [0.01, 10.0],
                'limits': [0.0, 10.0],
                'fixed': False
            },
            'n_sersic': {
                # 'prior_type': "uniform",
                # 'prior_settings': [0.3, 6.0],
                # 'limits': [0.3, 6.0],   
                'fixed': True, #NOTE here we fix the Sersic index to 1.0
                'fixed_value': 1.0
            },
            'e1': {
                # 'prior_type': "gaussian",
                # 'prior_settings': [0.0, 0.3],
                # 'limits': [-1.0, 1.0],
                'fixed': True,
                'fixed_value': {
                    'component_type': "lens_light_list",
                    'component_idx': 0,
                    'parameter': "e1"
                }
            },
            'e2': {
                # 'prior_type': "gaussian",
                # 'prior_settings': [0.0, 0.3],
                # 'limits': [-1.0, 1.0],
                'fixed': True,
                'fixed_value': {
                    'component_type': "lens_light_list",
                    'component_idx': 0,
                    'parameter': "e2"
                }
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
        'batch_size': 100,
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


def save_modeling_config(lens_name):
    # Dataset configuration
    dataset_config = {
        'data_path': f"dataset/{lens_name}/image.fits",
        'noise_path': f"dataset/{lens_name}/noise.fits",
        'psf_path': f"dataset/{lens_name}/psf.fits",
        'pixel_scale': 0.05
    }

    # Output configuration
    output_config = {
        'path': f"results/{lens_name}",
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

    os.makedirs(f"results/{lens_name}", exist_ok=True)
    # Save configuration
    with open(f"results/{lens_name}/modeling_config.yaml", "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    # save_modeling_config(lens_names[0])
    from multiprocessing import Pool
    with Pool(24) as p:
        p.map(save_modeling_config, lens_names)


#%%