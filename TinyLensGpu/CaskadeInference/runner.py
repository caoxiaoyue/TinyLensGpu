"""
Caskade-based model inference runner for gravitational lensing.

This module provides the main runner for performing Bayesian inference
on gravitational lens models using caskade-based implementations.
"""

from matplotlib import pyplot as plt
from astropy.io import fits
import os
import numpy as np
import yaml
from astropy.table import Table
from jax.extend import backend
import time
import jax.numpy as jnp
import jax.scipy as jsp
from corner import corner
from dynesty.utils import resample_equal

from TinyLensGpu.ProbModel.Image.caskade_model import CaskadeImageProbModel
from TinyLensGpu.CaskadeInference.config_parser import CaskadeConfigParser
from TinyLensGpu.Inference.NestedSampler.nautilus_sampler import NautilusSampler
from TinyLensGpu.Inference.NestedSampler.dynesty_sampler import DynestySampler
from TinyLensGpu.Inference.Optimizer import (
    DifferentialEvolutionOptimizer,
    BasinHoppingOptimizer,
    DirectOptimizer
)


class CaskadeModelInference:
    """
    Base class for caskade-based model inference.

    This class handles parameter transformations using CaskadeConfigParser
    and manages the interface between inference methods and the probability model.

    Parameters
    ----------
    prob_model : CaskadeImageProbModel
        Probability model for computing likelihoods
    config_path : str
        Path to configuration file

    Attributes
    ----------
    config_parser : CaskadeConfigParser
        Configuration parser managing physical model and parameters
    ndim : int
        Number of dynamic (non-linear) parameters
    prob_model : CaskadeImageProbModel
        Probability model instance
    """

    def __init__(self, prob_model, config_path):
        self.config_parser = CaskadeConfigParser(config_path)
        self.ndim = self.config_parser.ndim
        self.prob_model = prob_model

    def params_array2kargs(self, array):
        """
        Convert parameter array to caskade parameter values.

        With caskade, we don't return keyword arguments. Instead, we set
        the parameter values directly in the PhysicalModel's caskade Parameters.

        Parameters
        ----------
        array : array_like
            Parameter array, shape (ndim,) or (bs, ndim)

        Returns
        -------
        None
            Parameters are set in the PhysicalModel, nothing returned
        """
        # Ensure 2D array
        if array.ndim == 1:
            array = array.reshape(1, -1)

        bs = array.shape[0]

        # Set each dynamic parameter
        for param_info in self.config_parser.prior_transform.dynamic_params:
            idx = param_info['index']
            comp_type = param_info['comp_type']
            comp_idx = param_info['comp_idx']
            param_name = param_info['param_name']

            # Get the component
            if comp_type == 'lens_mass_list':
                component = self.prob_model.sim_obj.phys_model.lens_mass[comp_idx]
            elif comp_type == 'source_light_list':
                component = self.prob_model.sim_obj.phys_model.source_light[comp_idx]
            elif comp_type == 'lens_light_list':
                component = self.prob_model.sim_obj.phys_model.lens_light[comp_idx]
            else:
                raise ValueError(f"Unknown component type: {comp_type}")

            # Get the parameter object
            param_obj = getattr(component, param_name)

            # Set the value
            # If bs == 1, set scalar value; otherwise set array
            if bs == 1:
                param_obj.to_static(float(array[0, idx]))
            else:
                param_obj.to_static(array[:, idx])

        # Return None - parameters are set in place
        return None

    def params_kargs2array(self, kargs):
        """
        Convert caskade parameters to array.

        Parameters
        ----------
        kargs : ignored
            Not used for caskade (parameters are in PhysicalModel)

        Returns
        -------
        array : np.ndarray
            Parameter array, shape (ndim,)
        """
        array = np.zeros(self.ndim)

        for param_info in self.config_parser.prior_transform.dynamic_params:
            idx = param_info['index']
            comp_type = param_info['comp_type']
            comp_idx = param_info['comp_idx']
            param_name = param_info['param_name']

            # Get the component
            if comp_type == 'lens_mass_list':
                component = self.prob_model.sim_obj.phys_model.lens_mass[comp_idx]
            elif comp_type == 'source_light_list':
                component = self.prob_model.sim_obj.phys_model.source_light[comp_idx]
            elif comp_type == 'lens_light_list':
                component = self.prob_model.sim_obj.phys_model.lens_light[comp_idx]

            param_obj = getattr(component, param_name)
            array[idx] = param_obj.value

        return array

    def prior(self, array):
        """
        Transform unit cube to physical parameter space.

        Parameters
        ----------
        array : array_like
            Samples from unit hypercube [0,1]^ndim

        Returns
        -------
        array : array_like
            Transformed parameters in physical space
        """
        return self.config_parser.prior_transform.transform(array)


class NautilusCaskadeModelSampler(CaskadeModelInference, NautilusSampler):
    """Nautilus sampler for caskade models."""

    def __init__(self, prob_model, config_path):
        CaskadeModelInference.__init__(self, prob_model, config_path)
        NautilusSampler.__init__(self, prob_model=prob_model, ndim=self.ndim)

    def likelihood(self, array):
        """Compute log-likelihood with caskade parameter system."""
        if array.ndim == 1:
            bs = 1
            array = array.reshape(1, -1)
        else:
            bs = array.shape[0]

        # Set parameters in PhysicalModel
        self.params_array2kargs(array)

        # Compute likelihood (no kwargs needed, params already set)
        return self.prob_model.likelihood(bs=bs, debug=True)


class DynestyCaskadeModelSampler(CaskadeModelInference, DynestySampler):
    """Dynesty sampler for caskade models."""

    def __init__(self, prob_model, config_path):
        CaskadeModelInference.__init__(self, prob_model, config_path)
        DynestySampler.__init__(self, prob_model=prob_model, ndim=self.ndim)

    def likelihood(self, array):
        """Compute log-likelihood with caskade parameter system."""
        if array.ndim == 1:
            bs = 1
            array = array.reshape(1, -1)
        else:
            bs = array.shape[0]

        # Set parameters in PhysicalModel
        self.params_array2kargs(array)

        # Compute likelihood
        return self.prob_model.likelihood(bs=bs, debug=True)


class DifferentialEvolutionCaskadeModelOptimizer(CaskadeModelInference, DifferentialEvolutionOptimizer):
    """Differential Evolution optimizer for caskade models."""

    def __init__(self, prob_model, config_path):
        CaskadeModelInference.__init__(self, prob_model, config_path)
        DifferentialEvolutionOptimizer.__init__(self, prob_model=prob_model, ndim=self.ndim)

    def likelihood(self, array):
        """Compute log-likelihood with caskade parameter system."""
        if array.ndim == 1:
            bs = 1
            array = array.reshape(1, -1)
        else:
            bs = array.shape[0]

        self.params_array2kargs(array)
        return self.prob_model.likelihood(bs=bs, debug=True)


class BasinHoppingCaskadeModelOptimizer(CaskadeModelInference, BasinHoppingOptimizer):
    """Basin Hopping optimizer for caskade models."""

    def __init__(self, prob_model, config_path):
        CaskadeModelInference.__init__(self, prob_model, config_path)
        BasinHoppingOptimizer.__init__(self, prob_model=prob_model, ndim=self.ndim)

    def likelihood(self, array):
        """Compute log-likelihood with caskade parameter system."""
        if array.ndim == 1:
            bs = 1
            array = array.reshape(1, -1)
        else:
            bs = array.shape[0]

        self.params_array2kargs(array)
        return self.prob_model.likelihood(bs=bs, debug=True)


class DirectCaskadeModelOptimizer(CaskadeModelInference, DirectOptimizer):
    """DIRECT optimizer for caskade models."""

    def __init__(self, prob_model, config_path):
        CaskadeModelInference.__init__(self, prob_model, config_path)
        DirectOptimizer.__init__(self, prob_model=prob_model, ndim=self.ndim)

    def likelihood(self, array):
        """Compute log-likelihood with caskade parameter system."""
        if array.ndim == 1:
            bs = 1
            array = array.reshape(1, -1)
        else:
            bs = array.shape[0]

        self.params_array2kargs(array)
        return self.prob_model.likelihood(bs=bs, debug=True)


class RunCaskadeLensModel:
    """
    Main runner for caskade-based gravitational lens model inference.

    This class manages the full workflow:
    1. Load configuration and data
    2. Setup caskade physical model and probability model
    3. Setup inference method (sampler or optimizer)
    4. Run inference
    5. Analyze and visualize results

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file

    Attributes
    ----------
    config_path : str
        Configuration file path
    config : dict
        Full configuration dictionary
    data_config : dict
        Dataset configuration
    inference_config : dict
        Inference configuration
    output_config : dict
        Output configuration
    """

    def __init__(self, config_path):
        """Initialize with configuration file."""
        self.config_path = config_path
        self.load_config()
        self.setup_jax()

    def setup_jax(self):
        """Setup JAX configuration."""
        print(f"JAX backend platform: {backend.get_backend().platform}")

    def load_config(self):
        """Load and parse configuration file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract configurations
        self.data_config = self.config['dataset']

        # Handle inference config
        if 'inference' in self.config:
            self.inference_config = self.config['inference']
        else:
            # Backward compatibility
            self.inference_config = {
                'type': 'sampler',
                'method': self.config.get('sampling', {}).get('sampler', 'nautilus'),
                'settings': self.config.get('sampling', {})
            }

        self.output_config = self.config['output']

        # Create output directory
        os.makedirs(self.output_config['path'], exist_ok=True)

    def load_data(self):
        """Load image, noise, and PSF data."""
        self.image_map = fits.getdata(self.data_config['data_path']).astype('float64')
        self.noise_map = fits.getdata(self.data_config['noise_path']).astype('float64')
        self.psf_map = fits.getdata(self.data_config['psf_path']).astype('float64')

        try:
            self.mask = fits.getdata(self.data_config['mask_path']).astype('bool')
        except:
            print("No mask used, fit the whole image.")
            self.mask = None

        # Mask out noise
        if self.mask is not None:
            self.noise_map = np.where(self.mask, 1e8, self.noise_map)

    def plot_data(self):
        """Plot loaded dataset."""
        hw = self.image_map.shape[0] * self.data_config['pixel_scale'] * 0.5
        extent = [-hw, hw, -hw, hw]
        hw_psf = self.psf_map.shape[0] * self.data_config['pixel_scale'] * 0.5
        extent_psf = [-hw_psf, hw_psf, -hw_psf, hw_psf]

        position_likelihood = self.config.get('position_likelihood', None)
        if position_likelihood is not None:
            positions = np.array(position_likelihood['positions'])

        plt.figure(figsize=(6, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(self.image_map, origin='lower', extent=extent, cmap="jet")
        if position_likelihood is not None:
            plt.scatter(positions[:, 0], positions[:, 1], marker='x', color='red')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Image')
        plt.ylabel('arcsec')

        plt.subplot(2, 2, 2)
        plt.imshow(self.noise_map, origin='lower', extent=extent, cmap="jet")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Noise')

        plt.subplot(2, 2, 3)
        plt.imshow(self.psf_map, origin='lower', extent=extent_psf, cmap="jet")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.ylabel('arcsec')
        plt.xlabel('arcsec')
        plt.title('PSF')

        plt.subplot(2, 2, 4)
        plt.imshow(self.image_map / self.noise_map, origin='lower', extent=extent, cmap="jet")
        plt.colorbar(fraction=0.046, pad=0.04)
        if position_likelihood is not None:
            plt.scatter(positions[:, 0], positions[:, 1], marker='x', color='red')
        plt.title('SNR')
        plt.xlabel('arcsec')
        plt.tight_layout()

        # Save
        dataset_path = os.path.join(self.output_config['path'],
                                    self.output_config['datasets']['subplot'])
        plt.savefig(dataset_path, bbox_inches='tight')
        plt.show()

    def setup_model(self):
        """Setup caskade physical model and probability model."""
        # Initialize config parser and get physical model
        self.config_parser = CaskadeConfigParser(self.config_path)
        self.phys_model = self.config_parser.phys_model

        # Set static parameters
        self.config_parser.set_static_params()

        # Initialize probability model
        self.prob_model = CaskadeImageProbModel(
            image_data=self.image_map,
            noise_map=self.noise_map,
            psf_kernel=self.psf_map,
            dpix=self.data_config['pixel_scale'],
            nsub=4,
            phys_model=self.phys_model,
            use_linear=(self.config_parser.n_linear_params > 0),
            mask=self.mask,
            solver_type=self.config_parser.solver_type,
            position_likelihood=self.config.get('position_likelihood', None),
        )

    def setup_inference(self):
        """Setup inference method based on configuration."""
        inference_type = self.inference_config['type'].lower()
        method = self.inference_config['method'].lower()

        if inference_type == 'sampler':
            sampler_map = {
                'nautilus': NautilusCaskadeModelSampler,
                'dynesty': DynestyCaskadeModelSampler,
            }

            if method not in sampler_map:
                raise ValueError(f"Unsupported sampler: {method}. "
                               f"Supported: {list(sampler_map.keys())}")

            inference_class = sampler_map[method]

        elif inference_type == 'optimizer':
            optimizer_map = {
                'differential_evolution': DifferentialEvolutionCaskadeModelOptimizer,
                'basin_hopping': BasinHoppingCaskadeModelOptimizer,
                'direct': DirectCaskadeModelOptimizer,
            }

            if method not in optimizer_map:
                raise ValueError(f"Unsupported optimizer: {method}. "
                               f"Supported: {list(optimizer_map.keys())}")

            inference_class = optimizer_map[method]

        else:
            raise ValueError(f"Unsupported inference type: {inference_type}. "
                           "Supported: 'sampler', 'optimizer'")

        self.inference = inference_class(self.prob_model, self.config_path)

    def init_jit_likelihood(self):
        """Initialize JIT compilation of likelihood function."""
        # Get parameter bounds
        x0 = []
        bounds = self.config_parser.prior_transform.get_param_bounds()
        for low, high in bounds:
            x0.append((low + high) / 2.0)
        x0 = np.array(x0)

        if self.inference_config['type'].lower() == 'sampler':
            bs = self.inference_config['settings'].get('batch_size', 1)
            x0 = x0.reshape(1, -1)
            x0 = np.repeat(x0, bs, axis=0)
        else:
            bs = 1

        print("Starting JIT compilation of likelihood...")
        t0 = time.time()

        # Set parameters and compute likelihood
        self.inference.params_array2kargs(x0)
        self.prob_model.likelihood(bs=bs)

        t1 = time.time()
        print(f"JIT compilation took: {t1-t0:.2f} seconds")

    def run_inference(self):
        """Run inference with method-specific settings."""
        t0 = time.time()

        if self.inference_config['type'].lower() == 'sampler':
            # Run nested sampler
            self.results = self.inference.run(**self.inference_config['settings'])
        else:
            # Run optimizer
            self.results = self.inference.run(**self.inference_config['settings'])

        t1 = time.time()
        print(f"Inference took: {(t1-t0)/60.0:.2f} minutes")

    def run(self):
        """Run full inference workflow."""
        print("="*60)
        print("Caskade Gravitational Lens Model Inference")
        print("="*60)

        self.load_data()
        self.plot_data()
        self.setup_model()
        self.setup_inference()
        self.init_jit_likelihood()
        self.run_inference()

        print("\n" + "="*60)
        print("Inference complete!")
        print("="*60)

    def __repr__(self):
        return f"RunCaskadeLensModel(config={self.config_path})"
