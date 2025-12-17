"""
Prior transformation for nested sampling and optimization.

This module implements prior transformations that map unit cube samples
to physical parameter space, supporting uniform, Gaussian, and log-uniform
distributions.
"""

import jax.numpy as jnp
from scipy.stats import norm
from typing import Dict, List, Tuple, Any
import numpy as np


class PriorTransform:
    """
    Prior transformation for nested sampling.

    This class transforms samples from the unit hypercube [0,1]^n to
    the physical parameter space according to specified prior distributions.

    Supported prior types:
    - uniform: Linear transformation from [0,1] to [min, max]
    - gaussian: Gaussian distribution with mean and std, optionally truncated
    - log_uniform: Uniform in log-space (Jeffreys prior)

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model_components with prior specifications

    Attributes
    ----------
    dynamic_params : list of dict
        List of dynamic parameter specifications
    param_names : list of str
        Names of all dynamic parameters
    ndim : int
        Total number of dynamic parameters
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model_components = config['model_components']

        # Collect all dynamic parameters
        self.dynamic_params: List[Dict[str, Any]] = []
        self.param_names: List[str] = []
        self.ndim = 0

        self._collect_dynamic_params()

    def _collect_dynamic_params(self):
        """Collect all dynamic parameters from configuration."""
        param_idx = 0

        # Iterate through all component types
        for comp_type in ['lens_mass_list', 'source_light_list', 'lens_light_list']:
            if comp_type not in self.model_components:
                continue

            for comp_idx, comp_cfg in enumerate(self.model_components[comp_type]):
                module_name = comp_cfg['module']

                for param_name, param_cfg in comp_cfg['params'].items():
                    mode = param_cfg.get('mode', 'static')

                    # Only include dynamic parameters in prior transform
                    if mode == 'dynamic':
                        prior_cfg = param_cfg['prior']

                        self.dynamic_params.append({
                            'comp_type': comp_type,
                            'comp_idx': comp_idx,
                            'module_name': module_name,
                            'param_name': param_name,
                            'prior': prior_cfg,
                            'index': param_idx
                        })

                        # Create parameter name: index-name format
                        self.param_names.append(f"{param_idx}-{param_name}")
                        param_idx += 1

        self.ndim = param_idx

    def transform(self, cube_unit):
        """
        Transform unit cube samples to physical parameter space.

        Parameters
        ----------
        cube_unit : array_like
            Samples from unit hypercube, shape (n_samples, ndim) or (ndim,)

        Returns
        -------
        pt_array : array_like
            Transformed parameters in physical space, same shape as input
        """
        # Ensure 2D array
        input_1d = False
        if cube_unit.ndim == 1:
            cube_unit = cube_unit.reshape(1, -1)
            input_1d = True

        bs = cube_unit.shape[0]
        pt_array = jnp.zeros((bs, self.ndim))

        # Transform each parameter according to its prior
        for param_info in self.dynamic_params:
            idx = param_info['index']
            prior_cfg = param_info['prior']
            prior_type = prior_cfg['type']

            if prior_type == 'uniform':
                # Uniform prior: linear transformation
                low, high = prior_cfg['range']
                pt_array = pt_array.at[:, idx].set(
                    cube_unit[:, idx] * (high - low) + low
                )

            elif prior_type == 'gaussian':
                # Gaussian prior: inverse CDF transformation
                mean = prior_cfg['mean']
                std = prior_cfg['std']
                limits = prior_cfg.get('limits')

                # Use scipy for inverse CDF (not jittable, but only called during sampling)
                # Convert to numpy for scipy
                cube_np = np.array(cube_unit[:, idx])
                transformed = norm.ppf(cube_np, mean, std)

                # Apply hard limits if specified
                if limits is not None:
                    transformed = np.clip(transformed, limits[0], limits[1])

                pt_array = pt_array.at[:, idx].set(jnp.array(transformed))

            elif prior_type == 'log_uniform':
                # Log-uniform prior: uniform in log-space
                low, high = prior_cfg['range']
                log_min = np.log(float(low))
                log_max = np.log(float(high))

                log_val = cube_unit[:, idx] * (log_max - log_min) + log_min
                pt_array = pt_array.at[:, idx].set(jnp.exp(log_val))

            else:
                raise ValueError(f"Unknown prior type: {prior_type}")

        # Return 1D if input was 1D
        if input_1d:
            pt_array = pt_array.squeeze(0)

        return pt_array

    def get_param_info(self, param_idx: int) -> Dict[str, Any]:
        """
        Get information about a specific parameter.

        Parameters
        ----------
        param_idx : int
            Index of the parameter

        Returns
        -------
        dict
            Parameter information including name, prior, and location
        """
        if param_idx < 0 or param_idx >= self.ndim:
            raise IndexError(f"Parameter index {param_idx} out of range [0, {self.ndim})")

        return self.dynamic_params[param_idx]

    def get_param_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for all dynamic parameters (for optimizers).

        Returns
        -------
        list of tuple
            List of (min, max) bounds for each parameter
        """
        bounds = []
        for param_info in self.dynamic_params:
            prior_cfg = param_info['prior']
            prior_type = prior_cfg['type']

            if prior_type == 'uniform':
                bounds.append(tuple(prior_cfg['range']))
            elif prior_type == 'gaussian':
                limits = prior_cfg.get('limits')
                if limits is not None:
                    bounds.append(tuple(limits))
                else:
                    # Use mean ± 5*sigma as bounds
                    mean = prior_cfg['mean']
                    std = prior_cfg['std']
                    bounds.append((mean - 5*std, mean + 5*std))
            elif prior_type == 'log_uniform':
                bounds.append(tuple(prior_cfg['range']))
            else:
                raise ValueError(f"Unknown prior type: {prior_type}")

        return bounds

    def __repr__(self):
        return f"PriorTransform(ndim={self.ndim}, n_params={len(self.param_names)})"
