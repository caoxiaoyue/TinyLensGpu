#%%
import yaml
import numpy as np
import os
from scipy.stats import norm
from TinyLensGpu.Profile.Light.Gaussian import GaussianEllipse
from TinyLensGpu.Simulator.Image import Simulator
from TinyLensGpu.Profile.Light.Sersic import SersicEllipse
from TinyLensGpu.Profile.Mass.Sie import SIE
from TinyLensGpu.Profile.Mass.Shear import Shear

class ModelParser:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_components = self.config['model_components']
        # Get solver type from config, default to 'nnls'
        self.solver_type = self.config.get('solver_type', 'nnls')
        if self.solver_type not in ['nnls', 'normal']:
            raise ValueError("solver_type must be either 'nnls' or 'normal'")
            
        self.ndim, self.param_names, self.param_settings = self._count_free_parameters()
        self.n_linear_params, self.linear_param_names = self._count_linear_parameters()
        self.phys_model = self._build_physical_model()
        
    def _count_linear_parameters(self):
        """Count number of parameters that use linear fitting"""
        count = 0
        linear_param_names = []
        count_profiles = 0
        for component_list in ['source_light_list', 'lens_light_list']:
            if component_list not in self.model_components:
                continue
            count_profiles += len(self.model_components[component_list])
            for component_id, component in enumerate(self.model_components[component_list]):
                for param_name, settings in component['params'].items():
                    if settings.get('use_linear', False):
                        count += 1
                        linear_param_names.append(component_list + f"[{str(component_id)}]" + '-' + param_name)
        assert count == count_profiles, f"Number of linear parameters ({count}) does not match number of light profiles ({count_profiles})"
        return count, linear_param_names
    
    def _count_free_parameters(self):
        """Count number of free parameters across all components"""
        count = 0
        param_names = []
        param_settings = []
        for component_list in ['lens_mass_list', 'source_light_list', 'lens_light_list']:
            if component_list not in self.model_components:
                continue
            for component in self.model_components[component_list]:
                for param_name, settings in component['params'].items():
                    # Skip parameters that use linear fitting - they are automatically fixed to 1.0
                    if settings.get('use_linear', False):
                        settings['fixed'] = True
                        settings['fixed_value'] = 1.0
                        continue
                    if not settings.get('fixed', False):
                        param_names.append(f"{count}-{param_name}")
                        param_settings.append(settings)
                        count += 1
        return count, param_names, param_settings
    
    def _build_physical_model(self):
        """Build PhysicalModel based on YAML configuration"""
        lens_mass = []
        source_light = []
        lens_light = []
        
        # Parse lens mass components
        if 'lens_mass_list' in self.model_components:
            for component in self.model_components['lens_mass_list']:
                if component['type'] == 'SIE':
                    lens_mass.append(SIE())
                elif component['type'] == 'SHEAR':
                    lens_mass.append(Shear())
                    
        # Parse source light components
        if 'source_light_list' in self.model_components:
            for component in self.model_components['source_light_list']:
                if component['type'] == 'Sersic':
                    source_light.append(SersicEllipse())
                elif component['type'] == 'Gaussian':
                    source_light.append(GaussianEllipse())
                    
        # Parse lens light components
        if 'lens_light_list' in self.model_components:
            for component in self.model_components['lens_light_list']:
                if component['type'] == 'Sersic':
                    lens_light.append(SersicEllipse())
                elif component['type'] == 'Gaussian':
                    lens_light.append(GaussianEllipse())
                    
        return Simulator.PhysicalModel(
            lens_mass=lens_mass,
            source_light=source_light,
            lens_light=lens_light
        )
    
    def array_to_kwargs_list(self, array):
        """Convert parameter array to model kwargs format"""
        if array.ndim == 1:
            array = array.reshape(1, -1)
            
        param_idx = 0
        lens_mass_par_list = []
        source_light_par_list = []
        lens_light_par_list = []
        
        # First pass: Parse all non-linked parameters
        # Parse lens mass parameters
        if 'lens_mass_list' in self.model_components:
            for component in self.model_components['lens_mass_list']:
                params = {}
                for param_name, settings in component['params'].items():
                    if settings.get('fixed', False):
                        if isinstance(settings['fixed_value'], dict):
                            # Skip linked parameters in first pass
                            continue
                        else:
                            params[param_name] = settings['fixed_value']
                    else:
                        params[param_name] = array[:, param_idx]
                        param_idx += 1
                lens_mass_par_list.append(params)
                
        # Parse source light parameters
        if 'source_light_list' in self.model_components:
            for component in self.model_components['source_light_list']:
                params = {}
                for param_name, settings in component['params'].items():
                    if settings.get('fixed', False):
                        if isinstance(settings['fixed_value'], dict):
                            # Skip linked parameters in first pass
                            continue
                        else:
                            params[param_name] = settings['fixed_value']
                    else:
                        params[param_name] = array[:, param_idx]
                        param_idx += 1
                source_light_par_list.append(params)
                
        # Parse lens light parameters
        if 'lens_light_list' in self.model_components:
            for component in self.model_components['lens_light_list']:
                params = {}
                for param_name, settings in component['params'].items():
                    if settings.get('fixed', False):
                        if isinstance(settings['fixed_value'], dict):
                            # Skip linked parameters in first pass
                            continue
                        else:
                            params[param_name] = settings['fixed_value']
                    else:
                        params[param_name] = array[:, param_idx]
                        param_idx += 1
                lens_light_par_list.append(params)
        
        # Second pass: Handle linked parameters
        kwargs_list = [lens_mass_par_list, source_light_par_list, lens_light_par_list]
        component_types = ['lens_mass_list', 'source_light_list', 'lens_light_list']
        
        for comp_type_idx, comp_type in enumerate(component_types):
            if comp_type not in self.model_components:
                continue
            for comp_idx, component in enumerate(self.model_components[comp_type]):
                for param_name, settings in component['params'].items():
                    if settings.get('fixed', False) and isinstance(settings['fixed_value'], dict):
                        # Handle linked parameter
                        link_info = settings['fixed_value']
                        target_comp_type = link_info['component_type']  # e.g., 'lens_mass_list'
                        target_comp_idx = link_info['component_idx']    # e.g., 0
                        target_param = link_info['parameter']          # e.g., 'center_x'
                        
                        # Map component type to list index
                        type_to_idx = {
                            'lens_mass_list': 0,
                            'source_light_list': 1,
                            'lens_light_list': 2
                        }
                        
                        # Get the linked parameter value
                        target_value = kwargs_list[type_to_idx[target_comp_type]][target_comp_idx][target_param]
                        kwargs_list[comp_type_idx][comp_idx][param_name] = target_value
                
        return kwargs_list
    
    def prior_transform(self, cube_unit):
        """Convert model parameters from kwargs format to array"""
        if cube_unit.ndim == 1:
            bs = 1
            cube_unit = cube_unit.reshape(1, -1)
        else:
            bs = cube_unit.shape[0]
            
        pt_array = np.empty((bs, self.ndim))
        param_idx = 0
        
        # Process lens mass parameters
        if 'lens_mass_list' in self.model_components:
            for component in self.model_components['lens_mass_list']:
                for param_name, settings in component['params'].items():
                    if not settings.get('fixed', False):
                        prior_type = settings['prior_type']
                        prior_settings = settings['prior_settings']
                        limits = settings.get('limits', None)
                        
                        if prior_type == 'uniform':
                            pt_array[:, param_idx] = cube_unit[:, param_idx] * (prior_settings[1] - prior_settings[0]) + prior_settings[0]
                        elif prior_type == 'gaussian':
                            # First apply the Gaussian transform
                            transformed = norm.ppf(cube_unit[:, param_idx], prior_settings[0], prior_settings[1])
                            # Then clip to limits if they exist
                            if limits is not None:
                                transformed = np.clip(transformed, limits[0], limits[1])
                            pt_array[:, param_idx] = transformed
                        elif prior_type == 'log_uniform':
                            # Convert to float64 for log operations
                            log_min = np.log(np.float64(prior_settings[0]))
                            log_max = np.log(np.float64(prior_settings[1]))
                            pt_array[:, param_idx] = np.exp(cube_unit[:, param_idx].astype(np.float64) * (log_max - log_min) + log_min)
                        param_idx += 1
                        
        # Process source light parameters
        if 'source_light_list' in self.model_components:
            for component in self.model_components['source_light_list']:
                for param_name, settings in component['params'].items():
                    if not settings.get('fixed', False):
                        prior_type = settings['prior_type']
                        prior_settings = settings['prior_settings']
                        limits = settings.get('limits', None)
                        
                        if prior_type == 'uniform':
                            pt_array[:, param_idx] = cube_unit[:, param_idx] * (prior_settings[1] - prior_settings[0]) + prior_settings[0]
                        elif prior_type == 'gaussian':
                            # First apply the Gaussian transform
                            transformed = norm.ppf(cube_unit[:, param_idx], prior_settings[0], prior_settings[1])
                            # Then clip to limits if they exist
                            if limits is not None:
                                transformed = np.clip(transformed, limits[0], limits[1])
                            pt_array[:, param_idx] = transformed
                        elif prior_type == 'log_uniform':
                            # Convert to float64 for log operations
                            log_min = np.log(np.float64(prior_settings[0]))
                            log_max = np.log(np.float64(prior_settings[1]))
                            pt_array[:, param_idx] = np.exp(cube_unit[:, param_idx].astype(np.float64) * (log_max - log_min) + log_min)
                        param_idx += 1
                        
        # Process lens light parameters
        if 'lens_light_list' in self.model_components:
            for component in self.model_components['lens_light_list']:
                for param_name, settings in component['params'].items():
                    if not settings.get('fixed', False):
                        prior_type = settings['prior_type']
                        prior_settings = settings['prior_settings']
                        limits = settings.get('limits', None)
                        
                        if prior_type == 'uniform':
                            pt_array[:, param_idx] = cube_unit[:, param_idx] * (prior_settings[1] - prior_settings[0]) + prior_settings[0]
                        elif prior_type == 'gaussian':
                            # First apply the Gaussian transform
                            transformed = norm.ppf(cube_unit[:, param_idx], prior_settings[0], prior_settings[1])
                            # Then clip to limits if they exist
                            if limits is not None:
                                transformed = np.clip(transformed, limits[0], limits[1])
                            pt_array[:, param_idx] = transformed
                        elif prior_type == 'log_uniform':
                            # Convert to float64 for log operations
                            log_min = np.log(np.float64(prior_settings[0]))
                            log_max = np.log(np.float64(prior_settings[1]))
                            pt_array[:, param_idx] = np.exp(cube_unit[:, param_idx].astype(np.float64) * (log_max - log_min) + log_min)
                        param_idx += 1

        if bs == 1:
            pt_array = pt_array.squeeze()

        return pt_array 
    
    def kwargs_list_to_array(self, kwargs_list):
        """Convert model kwargs format back to parameter array
        
        Args:
            kwargs: List containing [lens_mass_par_list, source_light_par_list, lens_light_par_list]
            
        Returns:
            array: numpy array of shape (1, ndim) containing the parameter values
        """
        array = [None] * self.ndim
        param_idx = 0
        
        # Process lens mass parameters
        if 'lens_mass_list' in self.model_components:
            for comp_idx, component in enumerate(self.model_components['lens_mass_list']):
                for param_name, settings in component['params'].items():
                    if not settings.get('fixed', False):
                        # Extract scalar value if it's a numpy array
                        value = kwargs_list[0][comp_idx][param_name]
                        array[param_idx] = value
                        param_idx += 1
        
        # Process source light parameters
        if 'source_light_list' in self.model_components:
            for comp_idx, component in enumerate(self.model_components['source_light_list']):
                for param_name, settings in component['params'].items():
                    if not settings.get('fixed', False):
                        # Extract scalar value if it's a numpy array
                        value = kwargs_list[1][comp_idx][param_name]
                        array[param_idx] = value
                        param_idx += 1
        
        # Process lens light parameters
        if 'lens_light_list' in self.model_components:
            for comp_idx, component in enumerate(self.model_components['lens_light_list']):
                for param_name, settings in component['params'].items():
                    if not settings.get('fixed', False):
                        # Extract scalar value if it's a numpy array
                        value = kwargs_list[2][comp_idx][param_name]
                        array[param_idx] = value
                        param_idx += 1
        
        return np.column_stack(array)

if __name__ == '__main__':
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'example_modeling_config.yaml')
    
    # Simple test run
    parser = ModelParser(config_path)
    print(f"Number of free parameters: {parser.ndim}")
    
    # Print physical model structure
    print("\nPhysical Model Structure:")
    print(f"Lens Mass components: {len(parser.phys_model.lens_mass)}")
    print(f"Source Light components: {len(parser.phys_model.source_light)}")
    print(f"Lens Light components: {len(parser.phys_model.lens_light)}")

    # Test parameter conversion batch
    test_array = np.random.rand(10, parser.ndim)
    print("\nConverting random array to kwargs format...")
    kwargs_list = parser.array_to_kwargs_list(test_array)
    reconstructed_array = parser.kwargs_list_to_array(kwargs_list)
    assert np.allclose(test_array, reconstructed_array), "Conversion failed"
    
    # Test parameter conversion single
    test_array = np.random.rand(1, parser.ndim)
    print("\nConverting random array to kwargs format...")
    kwargs_list = parser.array_to_kwargs_list(test_array)
    reconstructed_array = parser.kwargs_list_to_array(kwargs_list)
    assert np.allclose(test_array, reconstructed_array), "Conversion failed"
        
    # Print first few parameters from each component
    print("\nSample parameters:")
    print("Lens Mass (SIE):", {k: v[0] for k, v in kwargs_list[0][0].items() if k in ['theta_E', 'e1', 'e2']})
    print("Lens Mass (Shear):", {k: v[0] for k, v in kwargs_list[0][1].items()})
    print("Source Light:", {k: v[0] for k, v in kwargs_list[1][0].items() if k in ['R_sersic', 'n_sersic']})
    print("Lens Light:", {k: v[0] for k, v in kwargs_list[2][0].items() if k in ['R_sersic', 'n_sersic']})
    
    # Test prior transformations
    print("\nTesting prior transformations...")
    cube_unit = np.random.rand(10, parser.ndim)
    transformed_array = parser.prior_transform(cube_unit)
    print("Original array shape:", cube_unit.shape)
    print("Transformed array shape:", transformed_array.shape)
    pt_array_fiducial_0 = 3.0 * cube_unit[:, 0]
    assert np.allclose(pt_array_fiducial_0, transformed_array[:, 0]), "Uniform Transformation failed"
    pt_array_fiducial_1 = norm.ppf(cube_unit[:, 1], 0.0, 0.3)
    assert np.allclose(pt_array_fiducial_1, transformed_array[:, 1]), "Gaussian Transformation failed"   
    pt_array_fiducial_7 = np.exp(cube_unit[:, 7] * (np.log(1e5) - np.log(1e-5)) + np.log(1e-5))
    assert np.allclose(pt_array_fiducial_7, transformed_array[:, 7]), "Log Uniform Transformation failed"

    print("\nChecking physical model...")
    print(parser.phys_model.lens_mass[0])
    print(parser.phys_model.lens_mass[1])
    print(parser.phys_model.source_light[0])
    print(parser.phys_model.source_light[1])
    print(parser.phys_model.lens_light[0])
    print("All tests passed")


# %%
