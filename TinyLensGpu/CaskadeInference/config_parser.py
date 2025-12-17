"""
Configuration parser for caskade-based lensing models.

This module parses YAML configuration files and constructs caskade models,
managing parameter states (dynamic/static/linear/pointer) and prior transformations.
"""

import yaml
import caskade as ck
from typing import Dict, List, Any, Tuple, Optional
import jax.numpy as jnp

from ..CaskadeModels.mass import SIE, Shear
from ..CaskadeModels.light import SersicEllipse, GaussianEllipse
from ..CaskadeModels.composite import PhysicalModel
from ..CaskadeModels.priors import PriorTransform


class CaskadeConfigParser:
    """
    Parse YAML configuration and build caskade models.

    This class handles:
    - Loading and parsing YAML configuration files
    - Building PhysicalModel from configuration
    - Setting up parameter management (dynamic/static/linear/pointer)
    - Creating prior transformation module
    - Managing parameter links between components

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file

    Attributes
    ----------
    config : dict
        Full configuration dictionary
    model_components : dict
        Model components configuration
    phys_model : PhysicalModel
        Constructed physical model
    prior_transform : PriorTransform
        Prior transformation module
    ndim : int
        Number of dynamic (non-linear) parameters
    n_linear_params : int
        Number of linear parameters
    param_names : list of str
        Names of all dynamic parameters
    linear_params : list of tuple
        List of (comp_type, comp_idx, param_name) for linear parameters
    solver_type : str
        Linear solver type ('nnls' or 'normal')
    """

    # Registry mapping configuration names to caskade module classes
    MODULE_REGISTRY = {
        # Mass models
        'SIE': SIE,
        'SHEAR': Shear,
        'Shear': Shear,
        # Light models
        'SersicEllipse': SersicEllipse,
        'Sersic': SersicEllipse,
        'GaussianEllipse': GaussianEllipse,
        'Gaussian': GaussianEllipse,
    }

    def __init__(self, config_path: str):
        """
        Initialize configuration parser.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_components = self.config['model_components']
        self.solver_type = self.config.get('solver_type', 'nnls')
        self.parameter_links = self.config.get('parameter_links', [])

        # Build physical model
        self.phys_model = self._build_physical_model()

        # Build prior transformation
        # First normalize config to new format if needed
        self._normalize_config_format()
        self.prior_transform = PriorTransform(self.config)

        # Extract parameter information
        self.ndim = self.prior_transform.ndim
        self.param_names = self.prior_transform.param_names

        # Collect linear parameters
        self.linear_params = self._collect_linear_params()
        self.n_linear_params = len(self.linear_params)

    def _normalize_config_format(self):
        """
        Normalize configuration to new format.

        Converts old format (type + fixed/use_linear) to new format (module + mode).
        This allows backward compatibility with existing configs.
        """
        for comp_type in ['lens_mass_list', 'source_light_list', 'lens_light_list']:
            if comp_type not in self.model_components:
                continue

            for comp_cfg in self.model_components[comp_type]:
                # Handle old 'type' field
                if 'type' in comp_cfg and 'module' not in comp_cfg:
                    comp_cfg['module'] = comp_cfg['type']

                # Normalize parameters
                for param_name, param_cfg in comp_cfg['params'].items():
                    if 'mode' not in param_cfg:
                        # Determine mode from old fields
                        if param_cfg.get('use_linear', False):
                            param_cfg['mode'] = 'linear'
                        elif param_cfg.get('fixed', False):
                            param_cfg['mode'] = 'static'
                        else:
                            param_cfg['mode'] = 'dynamic'

                    # Convert prior format
                    if param_cfg.get('mode') == 'dynamic' and 'prior' not in param_cfg:
                        prior_type = param_cfg.get('prior_type', 'uniform')
                        prior_settings = param_cfg.get('prior_settings', [0.0, 1.0])
                        limits = param_cfg.get('limits')

                        prior_cfg = {'type': prior_type, 'limits': limits}

                        if prior_type == 'uniform':
                            prior_cfg['range'] = prior_settings
                        elif prior_type == 'gaussian':
                            prior_cfg['mean'] = prior_settings[0]
                            prior_cfg['std'] = prior_settings[1]
                        elif prior_type == 'log_uniform':
                            prior_cfg['range'] = prior_settings

                        param_cfg['prior'] = prior_cfg

    def _build_physical_model(self) -> PhysicalModel:
        """
        Build PhysicalModel from configuration.

        Returns
        -------
        PhysicalModel
            Constructed physical model with all components
        """
        lens_mass = []
        source_light = []
        lens_light = []

        # Build all modules first with default params
        if 'lens_mass_list' in self.model_components:
            for comp_cfg in self.model_components['lens_mass_list']:
                module_type = comp_cfg.get('module', comp_cfg.get('type'))
                module_cls = self.MODULE_REGISTRY[module_type]
                params = self._parse_params(comp_cfg['params'])
                lens_mass.append(module_cls(**params))

        if 'source_light_list' in self.model_components:
            for comp_cfg in self.model_components['source_light_list']:
                module_type = comp_cfg.get('module', comp_cfg.get('type'))
                module_cls = self.MODULE_REGISTRY[module_type]
                params = self._parse_params(comp_cfg['params'])
                source_light.append(module_cls(**params))

        if 'lens_light_list' in self.model_components:
            for comp_cfg in self.model_components['lens_light_list']:
                module_type = comp_cfg.get('module', comp_cfg.get('type'))
                module_cls = self.MODULE_REGISTRY[module_type]
                params = self._parse_params(comp_cfg['params'])
                lens_light.append(module_cls(**params))

        # Apply parameter links by direct assignment
        link_count = self._apply_parameter_links(lens_mass, source_light, lens_light)

        if link_count > 0:
            print(f"Applied {link_count} parameter links")

        return PhysicalModel(
            lens_mass=lens_mass,
            source_light=source_light,
            lens_light=lens_light
        )

    def _parse_params(self, params_cfg: Dict) -> Dict:
        """
        Parse parameter configuration.

        Parameters
        ----------
        params_cfg : dict
            Parameter configuration dictionary

        Returns
        -------
        dict
            Dictionary of parameter values for module initialization
        """
        result = {}

        for param_name, param_cfg in params_cfg.items():
            mode = param_cfg.get('mode', 'static')

            if mode == 'static':
                # Fixed parameter - set initial value
                value = param_cfg.get('fixed_value', param_cfg.get('value', 0.0))
                # Check if it's a link (dict with component_type)
                if isinstance(value, dict):
                    # This is a parameter link - will be handled later
                    result[param_name] = None
                else:
                    result[param_name] = value

            elif mode == 'linear':
                # Linear parameter - initialize with 1.0
                result[param_name] = 1.0

            elif mode == 'dynamic':
                # Dynamic parameter - initialize with None (will be set during sampling)
                result[param_name] = None

            elif mode == 'pointer':
                # Pointer parameter - will be set during _apply_parameter_links
                result[param_name] = None

        return result

    def _apply_parameter_links(self, lens_mass: List, source_light: List, lens_light: List) -> int:
        """
        Apply parameter links using caskade's native pointer mechanism.

        When you assign a Param from one module to another (e.g., `module2.param = module1.param`),
        caskade creates an internal link so that both modules share the same parameter value.
        This is caskade's native way of implementing parameter sharing.

        Parameters
        ----------
        lens_mass : list
            List of lens mass modules
        source_light : list
            List of source light modules
        lens_light : list
            List of lens light modules

        Returns
        -------
        int
            Number of links applied
        """
        # Create component registry
        component_lists = {
            'lens_mass_list': lens_mass,
            'source_light_list': source_light,
            'lens_light_list': lens_light
        }

        link_count = 0

        # Apply fixed_value links from params
        for comp_type in ['lens_mass_list', 'source_light_list', 'lens_light_list']:
            if comp_type not in self.model_components:
                continue

            for comp_idx, comp_cfg in enumerate(self.model_components[comp_type]):
                for param_name, param_cfg in comp_cfg['params'].items():
                    mode = param_cfg.get('mode', 'static')
                    if mode == 'static':
                        fixed_value = param_cfg.get('fixed_value', param_cfg.get('value'))
                        if isinstance(fixed_value, dict):
                            # This is a parameter link
                            source_type = fixed_value['component_type']
                            source_idx = fixed_value['component_idx']
                            source_param = fixed_value['parameter']

                            source_module = component_lists[source_type][source_idx]
                            source_param_obj = getattr(source_module, source_param)

                            target_module = component_lists[comp_type][comp_idx]

                            # Use caskade's native parameter linking via assignment
                            # This creates an internal link so both modules share the same value
                            setattr(target_module, param_name, source_param_obj)

                            link_count += 1

        return link_count

    def _collect_linear_params(self) -> List[Tuple[str, int, str]]:
        """
        Collect parameters marked as linear.

        Returns
        -------
        list of tuple
            List of (comp_type, comp_idx, param_name) for each linear parameter
        """
        linear_params = []

        for comp_type in ['source_light_list', 'lens_light_list']:
            if comp_type not in self.model_components:
                continue

            for idx, comp_cfg in enumerate(self.model_components[comp_type]):
                for param_name, param_cfg in comp_cfg['params'].items():
                    mode = param_cfg.get('mode')
                    # Check both new and old format
                    if mode == 'linear' or param_cfg.get('use_linear', False):
                        linear_params.append((comp_type, idx, param_name))

        return linear_params

    def set_static_params(self):
        """
        Set all non-dynamic parameters to static mode in caskade.

        This should be called after model construction to properly initialize
        parameter states.
        """
        for comp_type in ['lens_mass_list', 'source_light_list', 'lens_light_list']:
            if comp_type not in self.model_components:
                continue

            # Get component list
            if comp_type == 'lens_mass_list':
                comp_list = self.phys_model.lens_mass
            elif comp_type == 'source_light_list':
                comp_list = self.phys_model.source_light
            else:
                comp_list = self.phys_model.lens_light

            for comp_idx, comp_cfg in enumerate(self.model_components[comp_type]):
                module = comp_list[comp_idx]

                for param_name, param_cfg in comp_cfg['params'].items():
                    mode = param_cfg.get('mode', 'static')

                    if mode == 'static':
                        fixed_value = param_cfg.get('fixed_value', param_cfg.get('value', 0.0))
                        if not isinstance(fixed_value, dict):
                            # Set to static (skip pointer params)
                            param_obj = getattr(module, param_name)
                            if hasattr(param_obj, 'to_static'):
                                param_obj.to_static(fixed_value)

                    elif mode == 'linear':
                        # Linear params stay at 1.0 (static)
                        param_obj = getattr(module, param_name)
                        if hasattr(param_obj, 'to_static'):
                            param_obj.to_static(1.0)

    def __repr__(self):
        return (f"CaskadeConfigParser("
                f"ndim={self.ndim}, "
                f"n_linear={self.n_linear_params}, "
                f"n_mass={len(self.phys_model.lens_mass)}, "
                f"n_source={len(self.phys_model.source_light)}, "
                f"n_lens_light={len(self.phys_model.lens_light)})")
