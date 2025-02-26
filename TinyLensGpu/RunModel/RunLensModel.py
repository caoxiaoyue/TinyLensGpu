# %%
from matplotlib import pyplot as plt
from astropy.io import fits
import os
import sys
import jax
import numpy as np
import yaml
from astropy.table import Table
from TinyLensGpu.ProbModel.Image.Model import ImageProbModel
from TinyLensGpu.Inference.NestedSampler.nautilus_sampler import NautilusSampler
from TinyLensGpu.Inference.NestedSampler.dynesty_sampler import DynestySampler
from TinyLensGpu.Inference.Optimizer import (
    DifferentialEvolutionOptimizer,
    BasinHoppingOptimizer,
    DirectOptimizer
)
from dynesty.utils import resample_equal
from corner import corner
from TinyLensGpu.ModelParser.Parser import ModelParser
import time
from jax.lib import xla_bridge
import jax.numpy as jnp
import jax.scipy as jsp
from TinyLensGpu.Simulator.Image import util

class BaseModelInference:
    """Base class for model inference that handles parameter transformations"""
    def __init__(self, prob_model, config_path):
        self.model_parser = ModelParser(config_path)
        self.ndim = self.model_parser.ndim
        self.prob_model = prob_model

    def params_array2kargs(self, array):
        """
        Converts an array of parameters into a dictionary of keyword arguments
        using ModelParser's array_to_kwargs_list method
        """
        return self.model_parser.array_to_kwargs_list(array)

    def params_kargs2array(self, kargs):
        """
        Converts a dictionary of keyword arguments into an array of parameters
        using ModelParser's kwargs_list_to_array method
        """
        return self.model_parser.kwargs_list_to_array(kargs)

    def prior(self, array):
        """
        Returns the prior probability of the parameters
        using ModelParser's prior_transform method
        """
        return self.model_parser.prior_transform(array)

class NautilusModelSampler(BaseModelInference, NautilusSampler):
    def __init__(self, prob_model, config_path):
        BaseModelInference.__init__(self, prob_model, config_path)
        NautilusSampler.__init__(self, prob_model=prob_model, ndim=self.ndim)

class DynestyModelSampler(BaseModelInference, DynestySampler):
    def __init__(self, prob_model, config_path):
        BaseModelInference.__init__(self, prob_model, config_path)
        DynestySampler.__init__(self, prob_model=prob_model, ndim=self.ndim)

class DifferentialEvolutionModelOptimizer(BaseModelInference, DifferentialEvolutionOptimizer):
    def __init__(self, prob_model, config_path):
        BaseModelInference.__init__(self, prob_model, config_path)
        DifferentialEvolutionOptimizer.__init__(self, prob_model=prob_model, ndim=self.ndim)

class BasinHoppingModelOptimizer(BaseModelInference, BasinHoppingOptimizer):
    def __init__(self, prob_model, config_path):
        BaseModelInference.__init__(self, prob_model, config_path)
        BasinHoppingOptimizer.__init__(self, prob_model=prob_model, ndim=self.ndim)

class DirectModelOptimizer(BaseModelInference, DirectOptimizer):
    def __init__(self, prob_model, config_path):
        BaseModelInference.__init__(self, prob_model, config_path)
        DirectOptimizer.__init__(self, prob_model=prob_model, ndim=self.ndim)

class RunLensModel:
    def __init__(self, config_path):
        """Initialize the RunLensModel with a config file path"""
        self.config_path = config_path
        self.load_config()
        self.setup_jax()
        
    def setup_jax(self):
        """Setup JAX configuration"""
        print(f"JAX backend platform: {xla_bridge.get_backend().platform}")
        
    def load_config(self):
        """Load and parse the configuration file"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Extract dataset info
        self.data_config = self.config['dataset']
        
        # Handle both old and new config formats
        if 'inference' in self.config:
            self.inference_config = self.config['inference']
        else:
            # For backward compatibility with old sampling config
            self.inference_config = {
                'type': 'sampler',
                'method': self.config.get('sampling', {}).get('sampler', 'nautilus'),
                'settings': self.config.get('sampling', {})
            }
            
        self.output_config = self.config['output']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_config['path'], exist_ok=True)
        
    def load_data(self):
        """Load the image, noise, and PSF data"""
        self.image_map = fits.getdata(self.data_config['data_path']).astype('float64')
        self.noise_map = fits.getdata(self.data_config['noise_path']).astype('float64')
        self.psf_map = fits.getdata(self.data_config['psf_path']).astype('float64')
        try:
            self.mask = fits.getdata(self.data_config['mask_path']).astype('bool')
        except:
            print("No mask used, fit the whole image.")
            self.mask = None

        
    def plot_data(self):
        """Plot the loaded data"""
        hw = self.image_map.shape[0] * self.data_config['pixel_scale'] * 0.5
        extent = [-hw, hw, -hw, hw]
        hw_psf = self.psf_map.shape[0] * self.data_config['pixel_scale'] * 0.5
        extent_psf = [-hw_psf, hw_psf, -hw_psf, hw_psf]

        plt.figure(figsize=(6, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(self.image_map, origin='lower', extent=extent, cmap="jet")
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
        plt.imshow(self.image_map/self.noise_map, origin='lower', extent=extent, cmap="jet")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('SNR')
        plt.xlabel('arcsec')
        plt.tight_layout()
        
        # Save the dataset subplot
        dataset_path = os.path.join(self.output_config['path'], self.output_config['datasets']['subplot'])
        plt.savefig(dataset_path, bbox_inches='tight')
        plt.show()
        
    def setup_model(self):
        """Setup the physical and probability models"""
        # Initialize model parser and get physical model
        self.model_parser = ModelParser(self.config_path)
        self.phys_model = self.model_parser.phys_model

        # Initialize probability model
        self.prob_model = ImageProbModel(
            image_data=self.image_map,
            noise_map=self.noise_map,
            psf_kernel=self.psf_map,
            dpix=self.data_config['pixel_scale'],
            nsub=4,
            phys_model=self.phys_model,
            use_linear=(self.model_parser.n_linear_params>0),
            mask=self.mask,
            solver_type=self.model_parser.solver_type,
        )
        
    def setup_inference(self):
        """Setup the inference method (sampler or optimizer) based on configuration"""
        inference_type = self.inference_config['type'].lower()
        method = self.inference_config['method'].lower()
        
        if inference_type == 'sampler':
            sampler_map = {
                'nautilus': NautilusModelSampler,
                'dynesty': DynestyModelSampler,
            }
            
            if method not in sampler_map:
                raise ValueError(f"Unsupported sampler type: {method}. "
                               f"Supported types are: {list(sampler_map.keys())}")
            
            inference_class = sampler_map[method]
            
        elif inference_type == 'optimizer':
            optimizer_map = {
                'differential_evolution': DifferentialEvolutionModelOptimizer,
                'basin_hopping': BasinHoppingModelOptimizer,
                'direct': DirectModelOptimizer,
            }
            
            if method not in optimizer_map:
                raise ValueError(f"Unsupported optimizer type: {method}. "
                               f"Supported types are: {list(optimizer_map.keys())}")
            
            inference_class = optimizer_map[method]
            
        else:
            raise ValueError(f"Unsupported inference type: {inference_type}. "
                           "Supported types are: 'sampler' and 'optimizer'")
            
        self.inference = inference_class(self.prob_model, self.config_path)

    def init_jit_likelihood(self):
        """Initialize the likelihood function with JIT compilation"""
        x0 = []
        for this_param_setting in self.model_parser.param_settings:
            x0.append(np.mean(this_param_setting['limits']))
        x0=np.array(x0)
        if self.inference_config['type'].lower() == 'sampler':
            x0=x0.reshape(1, -1)
            x0=np.repeat(x0, self.inference_config['settings'].get('batch_size', 1), axis=0)
        print("start to jit compile likelihood")
        param_dict = self.inference.params_array2kargs(x0)
        t0 = time.time()
        self.prob_model.likelihood(param_dict, bs=self.inference_config['settings'].get('batch_size', 1))
        t1 = time.time()
        print(f"JIT compilation took: {t1-t0:.2f} seconds")

        
    def run_inference(self):
        """Run the inference with method-specific configuration"""
        t0 = time.time()
        
        inference_type = self.inference_config['type'].lower()
        method = self.inference_config['method'].lower()
        settings = self.inference_config.get('settings', {})
        
        if inference_type == 'sampler':
            # Get common parameters for samplers
            nlive = settings.get('nlive', 1000)
            
            if method == 'nautilus':
                # Nautilus specific parameters
                bs = settings.get('batch_size', 1)
                verbose = settings.get('verbose', True)
                self.inference.run(
                    nlive=nlive,
                    bs=bs,
                    pool=(None, None),
                    verbose=verbose
                )
            elif method == 'dynesty':
                # Dynesty specific parameters
                dlogz = settings.get('dlogz', None)
                bound = settings.get('bound', 'multi')
                sample = settings.get('sample', 'auto')
                self.inference.run(
                    nlive=nlive,
                    dlogz=dlogz,
                    bound=bound,
                    sample=sample
                )
                
        elif inference_type == 'optimizer':
            if method == 'differential_evolution':
                # Get bounds from prior limits
                bounds = []
                for this_param_setting in self.model_parser.param_settings:
                    bounds.append(this_param_setting.get('limits', [-np.inf, np.inf]))
                self.result = self.inference.run(bounds=bounds, **settings)
                
            elif method == 'basin_hopping':
                # Use prior means or centers as initial guess
                x0 = []
                for this_param_setting in self.model_parser.param_settings:
                    x0.append(np.mean(this_param_setting['limits']))
                self.result = self.inference.run(x0=np.array(x0), **settings)
                
            elif method == 'direct':
                # Get bounds from prior limits
                bounds = []
                for this_param_setting in self.model_parser.param_settings:
                    bounds.append(this_param_setting['limits'])
                self.result = self.inference.run(bounds=bounds, **settings)
            
        t1 = time.time()
        print(f"Lens modeling with {method} {inference_type} took: {t1-t0:.2f} seconds")
        
    def get_results(self):
        """Get the results from the inference method and compute model"""
        inference_type = self.inference_config['type'].lower()
        
        if inference_type == 'sampler':
            # Get median and error values for sampler
            self.med_values = np.asarray([item[1] for item in self.inference.quantiles])
            self.high_values = np.asarray([item[2] for item in self.inference.quantiles])
            self.low_values = np.asarray([item[0] for item in self.inference.quantiles])
            self.sigma_values = (self.high_values - self.low_values)*0.5
            
        elif inference_type == 'optimizer':
            # Get best values for optimizer
            self.med_values = self.inference.best_params
            self.best_value = self.inference.best_value

        # Get model using best/median parameters
        self.param_dict = self.inference.params_array2kargs(self.med_values)
        self.image_model, self.intensity_list = self.prob_model.sim_obj.simulate(
            self.param_dict, 
            use_linear=(self.model_parser.n_linear_params>0), 
            return_intensity=True, 
            image_map=self.image_map, 
            noise_map=self.noise_map,
            xgrid_sub=self.prob_model.sim_obj.xgrid_sub,
            ygrid_sub=self.prob_model.sim_obj.ygrid_sub,
            psf_kernel=self.prob_model.sim_obj.psf_kernel,
        )

        #get best-fit model lens light and lensed source images
        # Get lens light and source components separately
        img_lens_sub, img_arc_sub = self.prob_model.sim_obj._generate_ideal_model(
            jnp.expand_dims(self.prob_model.sim_obj.xgrid_sub, -1),
            jnp.expand_dims(self.prob_model.sim_obj.ygrid_sub, -1),
            self.param_dict,
            len(self.prob_model.sim_obj.phys_model.source_light),
            len(self.prob_model.sim_obj.phys_model.lens_light),
            len(self.prob_model.sim_obj.phys_model.lens_mass)
        ) #shape: [n_y_pix_sub, n_x_pix_sub, bs, n_lens_light] or [n_y_pix_sub, n_x_pix_sub, bs, n_src]
        # Process lens light component
        img_arc_sub = img_arc_sub * self.intensity_list[jnp.newaxis, jnp.newaxis, jnp.newaxis, :len(self.prob_model.sim_obj.phys_model.source_light)]
        img_lens_sub = img_lens_sub * self.intensity_list[jnp.newaxis, jnp.newaxis, jnp.newaxis, len(self.prob_model.sim_obj.phys_model.source_light):]
        lens_light = jnp.sum(img_lens_sub, axis=-1)  # Sum over all lens light components
        lens_light = util.bin_image_general(lens_light, self.data_config.get('nsub', 4))
        lens_light = jsp.signal.fftconvolve(lens_light, jnp.expand_dims(self.prob_model.sim_obj.psf_kernel, -1), mode='same', axes=(0, 1))
        self.image_lens_light = jnp.squeeze(lens_light, axis=-1)
        # Process lensed source component
        lensed_source = jnp.sum(img_arc_sub, axis=-1)  # Sum over all source components
        lensed_source = util.bin_image_general(lensed_source, self.data_config.get('nsub', 4))
        lensed_source = jsp.signal.fftconvolve(lensed_source, jnp.expand_dims(self.prob_model.sim_obj.psf_kernel, -1), mode='same', axes=(0, 1))
        self.image_lensed_source = jnp.squeeze(lensed_source, axis=-1)

        # Print results
        if inference_type == 'sampler':
            print("likelihood:", self.inference.likelihood(self.med_values))
        else:
            print("best objective value:", self.best_value)
        print("linear parameter names:", self.model_parser.linear_param_names)
        print("linear parameter values:", self.intensity_list)
        
    def save_results(self):
        """Save inference results to CSV files"""
        inference_type = self.inference_config['type'].lower()
        output_path = self.output_config['path']
        
        if inference_type == 'sampler':
            # Save samples for sampler
            samples = resample_equal(self.inference.samples, self.inference.weights)
            param_names = self.model_parser.param_names
            samples_table = Table(samples, names=param_names)
            samples_path = os.path.join(output_path, self.output_config['tables']['samples'])
            samples_table.write(samples_path, overwrite=True)
            
            # Save summary statistics
            summary_data = {
                'parameter': param_names,
                'median': self.med_values,
                'percentile_16': self.low_values,
                'percentile_84': self.high_values,
            }
            summary_table = Table(summary_data)
            summary_table.meta['log_evidence'] = float(self.inference.log_z)
            summary_table.meta['best_likelihood'] = float(self.inference.log_l_max)
            
        else:
            # Save optimization results
            param_names = self.model_parser.param_names
            summary_data = {
                'parameter': param_names,
                'best_value': self.med_values,
                'objective_value': np.full(len(param_names), float(self.best_value)),
            }
            summary_table = Table(summary_data)
            
        summary_path = os.path.join(output_path, self.output_config['tables']['summary'])
        summary_table.write(summary_path, overwrite=True)
        
    def plot_model_results(self):
        """Plot the model results comparison including lens light and source components"""
        half_width = self.image_map.shape[0]*0.5*self.data_config['pixel_scale']
        extent = [-half_width, half_width, -half_width, half_width]
        
        # Create figure with 6 subplots
        plt.figure(figsize=(17, 10))
        
        # Data
        plt.subplot(231)
        plt.imshow(self.image_map, origin='lower', cmap='jet', extent=extent)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("Arcsec")
        plt.ylabel("Arcsec")
        plt.title("Data")
        
        # Total model
        plt.subplot(232)
        plt.imshow(self.image_model, origin='lower', cmap='jet', extent=extent)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("Arcsec")
        plt.title("Total Model")
        
        # Residuals
        plt.subplot(233)
        plt.imshow((self.image_model-self.image_map)/self.noise_map, origin='lower', cmap='jet', extent=extent)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("Arcsec")
        plt.title("Normalized residuals")
        
        # Lens light component
        plt.subplot(234)
        plt.imshow(self.image_lens_light, origin='lower', cmap='jet', extent=extent)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("Arcsec")
        plt.ylabel("Arcsec")
        plt.title("Lens Light")
        
        # Lensed source component
        plt.subplot(235)
        plt.imshow(self.image_lensed_source, origin='lower', cmap='jet', extent=extent)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("Arcsec")
        plt.title("Lensed Source")
        
        # Data - Lens light (to highlight arcs)
        plt.subplot(236)
        plt.imshow(self.image_map - self.image_lens_light, origin='lower', cmap='jet', extent=extent)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("Arcsec")
        plt.title("Data - Lens Light")
        
        # Save the model results figure
        results_path = os.path.join(self.output_config['path'], self.output_config['figures']['results'])
        plt.savefig(results_path, bbox_inches='tight')
        plt.show()
        
    def plot_corner(self):
        """Create and plot the corner plot"""
        if self.inference_config['type'].lower() == 'sampler':
            samples = resample_equal(self.inference.samples, self.inference.weights)
            labels = self.model_parser.param_names
            levels = [1-np.exp(-0.5), 1-np.exp(-0.5*4), 1-np.exp(-0.5*9)]
            credible_level = 0.95
            quantiles = [0.5-credible_level/2.0, 0.5, 0.5+credible_level/2.0]
            
            corner(
                samples,
                levels=levels,
                labels=labels,
                truth_color="black",
                quantiles=quantiles,
                show_titles=False,
                smooth=0.8,
                smooth1d=0.8,
                title_kwargs={"fontsize": 12},
                plot_datapoints=False,
                plot_density=False,
            )
            
            # Save the corner plot
            corner_path = os.path.join(self.output_config['path'], self.output_config['figures']['corner'])
            plt.savefig(corner_path, bbox_inches='tight')
            plt.show()
        else:
            print("Corner plot is only available for sampling methods")
        
    def run(self):
        """Run the complete lens modeling pipeline"""
        self.load_data()
        self.plot_data()
        self.setup_model()
        self.setup_inference()
        self.init_jit_likelihood() #Buggy now for optimizer?
        self.run_inference()
        self.get_results()
        self.save_results()
        self.plot_model_results()
        if self.inference_config['type'].lower() == 'sampler':
            self.plot_corner()
        
if __name__ == "__main__":
    # Example usage
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'model_config.yaml'
    lens_model = RunLensModel(config_path)
    lens_model.run()
    
# %%
