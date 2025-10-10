"""
Integration tests for complete TinyLensGpu workflows.
"""
import pytest
import numpy as np
import yaml
import os
from astropy.io import fits
from TinyLensGpu.ModelParser.Parser import ModelParser
from TinyLensGpu.Simulator.Image.Simulator import PhysicalModel, SimulatorConfig, LensSimulator
from TinyLensGpu.Profile.Light.Sersic import SersicEllipse
from TinyLensGpu.Profile.Mass.Sie import SIE


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test end-to-end workflows."""
    
    def test_simple_lens_light_model(self, sample_config_simple, tmp_path):
        """Test a simple lens light modeling workflow."""
        # Parse configuration
        parser = ModelParser(sample_config_simple)
        
        # Check that parser initialized correctly
        assert parser.ndim > 0
        assert parser.phys_model is not None
        
        # Create simulator configuration
        npix = 50
        dpix = 0.05
        psf = np.ones((15, 15)) / 225.0
        
        sim_config = SimulatorConfig(
            dpix=dpix,
            npix=npix,
            psf_kernel=psf,
            nsub=4
        )
        
        # Create simulator
        simulator = LensSimulator(
            phys_model=parser.phys_model,
            sim_config=sim_config
        )
        
        # Check that simulator initialized correctly
        assert simulator is not None
    
    def test_full_lens_model(self, sample_config_full, tmp_path):
        """Test a full lens model with all components."""
        # Parse configuration
        parser = ModelParser(sample_config_full)
        
        # Check all components are loaded
        assert len(parser.phys_model.lens_mass) == 2
        assert len(parser.phys_model.source_light) == 1
        assert len(parser.phys_model.lens_light) == 1
        
        # Create random parameters
        test_params = np.random.rand(parser.ndim)
        
        # Transform to physical space
        transformed_params = parser.prior_transform(test_params)
        
        # Convert to kwargs
        kwargs_list = parser.array_to_kwargs_list(transformed_params)
        
        # Check that kwargs have expected structure
        assert len(kwargs_list) == 3  # lens_mass, source_light, lens_light
        assert len(kwargs_list[0]) == 2  # Two mass components
        assert len(kwargs_list[1]) == 1  # One source light
        assert len(kwargs_list[2]) == 1  # One lens light
    
    def test_config_roundtrip(self, tmp_path):
        """Test configuration file creation and parsing."""
        config = {
            'dataset': {
                'data_path': 'test.fits',
                'noise_path': 'test.fits',
                'psf_path': 'test.fits',
                'pixel_scale': 0.05
            },
            'model_components': {
                'lens_mass_list': [
                    {
                        'type': 'SIE',
                        'params': {
                            'theta_E': {
                                'prior_type': 'uniform',
                                'prior_settings': [0.5, 2.0],
                                'limits': [0.0, 5.0],
                                'fixed': False
                            },
                            'e1': {
                                'prior_type': 'gaussian',
                                'prior_settings': [0.0, 0.3],
                                'limits': [-1.0, 1.0],
                                'fixed': False
                            },
                            'e2': {
                                'prior_type': 'gaussian',
                                'prior_settings': [0.0, 0.3],
                                'limits': [-1.0, 1.0],
                                'fixed': False
                            },
                            'center_x': {
                                'fixed': True,
                                'fixed_value': 0.0
                            },
                            'center_y': {
                                'fixed': True,
                                'fixed_value': 0.0
                            }
                        }
                    }
                ],
                'source_light_list': [],
                'lens_light_list': [
                    {
                        'type': 'Sersic',
                        'params': {
                            'R_sersic': {
                                'prior_type': 'uniform',
                                'prior_settings': [0.5, 2.0],
                                'limits': [0.0, 5.0],
                                'fixed': False
                            },
                            'n_sersic': {
                                'prior_type': 'uniform',
                                'prior_settings': [2.0, 5.0],
                                'limits': [0.3, 6.0],
                                'fixed': False
                            },
                            'e1': {
                                'prior_type': 'gaussian',
                                'prior_settings': [0.0, 0.3],
                                'limits': [-1.0, 1.0],
                                'fixed': False
                            },
                            'e2': {
                                'prior_type': 'gaussian',
                                'prior_settings': [0.0, 0.3],
                                'limits': [-1.0, 1.0],
                                'fixed': False
                            },
                            'center_x': {
                                'fixed': True,
                                'fixed_value': 0.0
                            },
                            'center_y': {
                                'fixed': True,
                                'fixed_value': 0.0
                            },
                            'Ie': {
                                'use_linear': True
                            }
                        }
                    }
                ]
            }
        }
        
        # Write config
        config_path = tmp_path / "test_roundtrip.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Parse config
        parser = ModelParser(str(config_path))
        
        # Check that parsing worked
        # SIE: 3 params (theta_E, e1, e2), Sersic: 4 params (R_sersic, n_sersic, e1, e2)
        # Total: 7 params (Ie is linear so fixed to 1.0)
        assert parser.ndim == 7
        assert len(parser.phys_model.lens_mass) == 1
        assert len(parser.phys_model.lens_light) == 1


@pytest.mark.integration
@pytest.mark.slow
class TestParameterTransformations:
    """Test parameter transformations across the pipeline."""
    
    def test_parameter_flow(self, sample_config_full):
        """Test parameter flow through the entire pipeline."""
        parser = ModelParser(sample_config_full)
        
        # Generate random unit cube samples
        n_samples = 10
        cube_unit = np.random.rand(n_samples, parser.ndim)
        
        # Transform to physical space
        physical_params = parser.prior_transform(cube_unit)
        
        # Check that parameters are within bounds
        for i, setting in enumerate(parser.param_settings):
            if 'limits' in setting:
                limits = setting['limits']
                assert np.all(physical_params[:, i] >= limits[0])
                assert np.all(physical_params[:, i] <= limits[1])
        
        # Convert to kwargs
        for i in range(n_samples):
            kwargs_list = parser.array_to_kwargs_list(physical_params[i:i+1])
            
            # Check that all required parameters are present
            # Lens mass (SIE and Shear)
            assert len(kwargs_list[0]) == 2
            # SIE component
            assert 'theta_E' in kwargs_list[0][0]
            assert 'e1' in kwargs_list[0][0]
            assert 'e2' in kwargs_list[0][0]
            # Shear component
            assert 'gamma1' in kwargs_list[0][1]
            assert 'gamma2' in kwargs_list[0][1]
            
            # Source light
            for comp in kwargs_list[1]:
                assert 'R_sersic' in comp
                assert 'n_sersic' in comp
            
            # Lens light
            for comp in kwargs_list[2]:
                assert 'R_sersic' in comp
                assert 'n_sersic' in comp
            
            # Convert back to array
            reconstructed = parser.kwargs_list_to_array(kwargs_list)
            
            # Check round-trip accuracy
            assert np.allclose(physical_params[i:i+1], reconstructed, atol=1e-6)


@pytest.mark.integration
class TestModelConsistency:
    """Test model consistency and physical constraints."""
    
    def test_profile_positivity(self):
        """Test that light profiles produce positive values."""
        profile = SersicEllipse()
        
        # Create test coordinates
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        xx, yy = np.meshgrid(x, y)
        
        # Evaluate profile
        light = profile.light(
            xx, yy,
            R_sersic=1.0,
            n_sersic=4.0,
            e1=0.2,
            e2=0.1,
            center_x=0.0,
            center_y=0.0,
            Ie=1.0
        )
        
        # Check that all values are positive
        assert np.all(light >= 0)
    
    def test_mass_deflection_direction(self):
        """Test that mass deflections point in expected directions."""
        profile = SIE()
        
        # Test point on positive x-axis
        x = 1.0
        y = 0.0
        
        alpha_x, alpha_y = profile.deriv(
            x, y,
            theta_E=1.0,
            e1=0.0,
            e2=0.0,
            center_x=0.0,
            center_y=0.0
        )
        
        # For circular SIE, deflection should point toward center
        # So for positive x, alpha_x should be positive
        assert alpha_x > 0
        assert abs(alpha_y) < 0.1  # Should be small for point on axis
    
    def test_ellipticity_range(self):
        """Test that ellipticity parameters produce valid profiles."""
        profile = SersicEllipse()
        
        # Test various ellipticity values
        e_values = [0.0, 0.3, 0.6, 0.9]
        
        for e1 in e_values:
            for e2 in [0.0, 0.1]:
                # Should not raise error for valid ellipticities
                light = profile.light(
                    0.0, 1.0,
                    R_sersic=1.0,
                    n_sersic=4.0,
                    e1=e1,
                    e2=e2,
                    center_x=0.0,
                    center_y=0.0,
                    Ie=1.0
                )
                
                # Light should be positive and finite
                assert np.isfinite(light)
                assert light >= 0

