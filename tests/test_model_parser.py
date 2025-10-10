"""
Unit tests for ModelParser class.
"""
import pytest
import numpy as np
from TinyLensGpu.ModelParser.Parser import ModelParser


@pytest.mark.unit
class TestModelParserInitialization:
    """Test ModelParser initialization and configuration parsing."""
    
    def test_parser_initialization_simple(self, sample_config_simple):
        """Test parser initialization with simple config."""
        parser = ModelParser(sample_config_simple)
        
        # Check that parser initializes correctly
        assert parser.ndim >= 0
        assert parser.n_linear_params >= 0
        assert len(parser.param_names) == parser.ndim
        assert len(parser.param_settings) == parser.ndim
    
    def test_parser_initialization_full(self, sample_config_full):
        """Test parser initialization with full config."""
        parser = ModelParser(sample_config_full)
        
        # Check that all components are loaded
        assert len(parser.phys_model.lens_mass) == 2
        assert len(parser.phys_model.source_light) == 1
        assert len(parser.phys_model.lens_light) == 1
    
    def test_parser_counts_free_parameters(self, sample_config_simple):
        """Test that parser correctly counts free parameters."""
        parser = ModelParser(sample_config_simple)
        
        # Simple config has 4 free params (R_sersic, n_sersic, e1, e2)
        # and 1 linear param (Ie)
        assert parser.ndim == 4
        assert parser.n_linear_params == 1
    
    def test_parser_linear_parameters(self, sample_config_simple):
        """Test that linear parameters are identified correctly."""
        parser = ModelParser(sample_config_simple)
        
        # Check that linear parameters are marked as fixed
        assert parser.n_linear_params == 1
        assert 'lens_light_list[0]-Ie' in parser.linear_param_names


@pytest.mark.unit
class TestArrayKwargsConversion:
    """Test array to kwargs conversion and vice versa."""
    
    def test_array_to_kwargs_single(self, sample_config_simple):
        """Test conversion of single parameter array to kwargs."""
        parser = ModelParser(sample_config_simple)
        
        # Create a test array with correct dimensions
        test_array = np.random.rand(1, parser.ndim)
        
        kwargs_list = parser.array_to_kwargs_list(test_array)
        
        # Check that we get three lists (lens_mass, source_light, lens_light)
        assert len(kwargs_list) == 3
        
        # For simple config, only lens_light should have parameters
        assert len(kwargs_list[2]) == 1  # One lens light component
        
        # Check that parameters are present
        assert 'R_sersic' in kwargs_list[2][0]
        assert 'n_sersic' in kwargs_list[2][0]
    
    def test_array_to_kwargs_batch(self, sample_config_simple):
        """Test conversion of batched parameter arrays to kwargs."""
        parser = ModelParser(sample_config_simple)
        
        # Create a batch of test arrays
        batch_size = 10
        test_array = np.random.rand(batch_size, parser.ndim)
        
        kwargs_list = parser.array_to_kwargs_list(test_array)
        
        # Check that batch dimension is preserved
        for param_value in kwargs_list[2][0].values():
            if not isinstance(param_value, (int, float)):
                assert len(param_value) == batch_size
    
    def test_kwargs_to_array_roundtrip_single(self, sample_config_simple):
        """Test round-trip conversion array->kwargs->array."""
        parser = ModelParser(sample_config_simple)
        
        # Create a test array
        test_array = np.random.rand(1, parser.ndim)
        
        # Convert to kwargs and back
        kwargs_list = parser.array_to_kwargs_list(test_array)
        reconstructed = parser.kwargs_list_to_array(kwargs_list)
        
        # Check that we get back the same array
        assert np.allclose(test_array, reconstructed, atol=1e-6)
    
    def test_kwargs_to_array_roundtrip_batch(self, sample_config_full):
        """Test round-trip conversion with batched arrays."""
        parser = ModelParser(sample_config_full)
        
        # Create a batch of test arrays
        batch_size = 10
        test_array = np.random.rand(batch_size, parser.ndim)
        
        # Convert to kwargs and back
        kwargs_list = parser.array_to_kwargs_list(test_array)
        reconstructed = parser.kwargs_list_to_array(kwargs_list)
        
        # Check that we get back the same array
        assert np.allclose(test_array, reconstructed, atol=1e-6)


@pytest.mark.unit
class TestPriorTransform:
    """Test prior transformation functions."""
    
    def test_uniform_prior_transform(self, sample_config_simple):
        """Test uniform prior transformation."""
        parser = ModelParser(sample_config_simple)
        
        # Create unit cube samples
        cube_unit = np.random.rand(10, parser.ndim)
        
        # Transform
        transformed = parser.prior_transform(cube_unit)
        
        # Check shape
        assert transformed.shape == cube_unit.shape
        
        # Check that transformed values are within prior ranges
        for i, setting in enumerate(parser.param_settings):
            if setting['prior_type'] == 'uniform':
                prior_min, prior_max = setting['prior_settings']
                assert np.all(transformed[:, i] >= prior_min)
                assert np.all(transformed[:, i] <= prior_max)
    
    def test_gaussian_prior_transform(self, sample_config_full):
        """Test gaussian prior transformation."""
        parser = ModelParser(sample_config_full)
        
        # Create unit cube samples
        cube_unit = np.random.rand(10, parser.ndim)
        
        # Transform
        transformed = parser.prior_transform(cube_unit)
        
        # Check shape
        assert transformed.shape == cube_unit.shape
        
        # Check that values with limits are clipped
        for i, setting in enumerate(parser.param_settings):
            if setting['prior_type'] == 'gaussian' and 'limits' in setting:
                limits = setting['limits']
                assert np.all(transformed[:, i] >= limits[0])
                assert np.all(transformed[:, i] <= limits[1])
    
    def test_prior_transform_single(self, sample_config_simple):
        """Test prior transformation with single sample."""
        parser = ModelParser(sample_config_simple)
        
        # Create single unit cube sample
        cube_unit = np.random.rand(parser.ndim)
        
        # Transform
        transformed = parser.prior_transform(cube_unit)
        
        # Check shape (should be squeezed to 1D)
        assert transformed.shape == (parser.ndim,)
    
    def test_prior_transform_deterministic(self, sample_config_simple):
        """Test that prior transform is deterministic."""
        parser = ModelParser(sample_config_simple)
        
        # Create unit cube samples
        cube_unit = np.random.rand(5, parser.ndim)
        
        # Transform twice
        transformed1 = parser.prior_transform(cube_unit)
        transformed2 = parser.prior_transform(cube_unit)
        
        # Should be identical
        assert np.allclose(transformed1, transformed2, atol=1e-8)


@pytest.mark.unit
class TestPhysicalModel:
    """Test physical model construction."""
    
    def test_empty_components(self, sample_config_simple):
        """Test model with empty components."""
        parser = ModelParser(sample_config_simple)
        
        # Simple config has no lens mass or source light
        assert len(parser.phys_model.lens_mass) == 0
        assert len(parser.phys_model.source_light) == 0
        assert len(parser.phys_model.lens_light) > 0
    
    def test_all_components(self, sample_config_full):
        """Test model with all components."""
        parser = ModelParser(sample_config_full)
        
        # Full config has all components
        assert len(parser.phys_model.lens_mass) > 0
        assert len(parser.phys_model.source_light) > 0
        assert len(parser.phys_model.lens_light) > 0
    
    def test_model_component_types(self, sample_config_full):
        """Test that model components have correct types."""
        parser = ModelParser(sample_config_full)
        
        # Check that components are correctly instantiated
        from TinyLensGpu.Profile.Mass.Sie import SIE
        from TinyLensGpu.Profile.Mass.Shear import Shear
        from TinyLensGpu.Profile.Light.Sersic import SersicEllipse
        
        # Check types
        assert isinstance(parser.phys_model.lens_mass[0], SIE)
        assert isinstance(parser.phys_model.lens_mass[1], Shear)
        assert isinstance(parser.phys_model.source_light[0], SersicEllipse)
        assert isinstance(parser.phys_model.lens_light[0], SersicEllipse)


@pytest.mark.unit
class TestSolverType:
    """Test solver type configuration."""
    
    def test_default_solver_type(self, sample_config_simple):
        """Test default solver type is nnls."""
        parser = ModelParser(sample_config_simple)
        assert parser.solver_type == 'nnls'
    
    def test_invalid_solver_type(self, tmp_path):
        """Test that invalid solver type raises error."""
        import yaml
        
        config = {
            'dataset': {'data_path': 'test.fits', 'noise_path': 'test.fits', 
                       'psf_path': 'test.fits', 'pixel_scale': 0.05},
            'model_components': {'lens_mass_list': [], 'source_light_list': [], 
                                'lens_light_list': []},
            'solver_type': 'invalid_solver'
        }
        
        config_path = tmp_path / "invalid_solver.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(ValueError, match="solver_type must be either"):
            ModelParser(str(config_path))

