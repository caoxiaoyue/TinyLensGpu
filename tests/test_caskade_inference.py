"""
Unit tests for caskade-based inference system.

This module tests the integration of:
1. CaskadeImageProbModel
2. CaskadeModelInference and adapters
3. Parameter conversion and prior transformation
"""

import pytest
import numpy as np
import jax.numpy as jnp

from TinyLensGpu.CaskadeInference.config_parser import CaskadeConfigParser
from TinyLensGpu.CaskadeInference.runner import (
    CaskadeModelInference,
    NautilusCaskadeModelSampler,
    RunCaskadeLensModel,
)
from TinyLensGpu.ProbModel.Image.caskade_model import CaskadeImageProbModel
from TinyLensGpu.CaskadeSimulator.config import SimulatorConfig


class TestCaskadeImageProbModel:
    """Test CaskadeImageProbModel"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config_path = 'paper/demo/lens_src/model_config.yaml'

        # Create synthetic data
        self.npix = 64
        self.dpix = 0.074
        self.image_data = np.random.randn(self.npix, self.npix) * 0.1 + 1.0
        self.noise_map = np.ones((self.npix, self.npix)) * 0.1
        self.psf_kernel = np.array([[0.0, 0.1, 0.0],
                                    [0.1, 0.6, 0.1],
                                    [0.0, 0.1, 0.0]])

    def test_prob_model_creation(self):
        """Test creating probability model"""
        # Parse configuration
        parser = CaskadeConfigParser(self.config_path)

        # Set static params
        parser.set_static_params()

        # Create probability model
        prob_model = CaskadeImageProbModel(
            image_data=self.image_data,
            noise_map=self.noise_map,
            psf_kernel=self.psf_kernel,
            dpix=self.dpix,
            nsub=2,
            phys_model=parser.phys_model,
            use_linear=True,
            solver_type='nnls',
        )

        # Check attributes
        assert prob_model.image_data.shape == (self.npix, self.npix)
        assert prob_model.use_linear == True
        assert prob_model.sim_obj.solver_type == 'nnls'

        print("✓ Probability model creation test passed")

    def test_forward_model(self):
        """Test forward model simulation"""
        parser = CaskadeConfigParser(self.config_path)
        parser.set_static_params()

        prob_model = CaskadeImageProbModel(
            image_data=self.image_data,
            noise_map=self.noise_map,
            psf_kernel=self.psf_kernel,
            dpix=self.dpix,
            nsub=2,
            phys_model=parser.phys_model,
            use_linear=False,  # Non-linear for simpler test
            solver_type='nnls',
        )

        # Set reasonable parameter values using inference interface
        inference = CaskadeModelInference(prob_model, self.config_path)
        bounds = parser.prior_transform.get_param_bounds()
        param_array = np.array([(low + high) / 2.0 for low, high in bounds])
        inference.params_array2kargs(param_array)

        # Run forward model
        img, intensity = prob_model.forward_model(bs=1)

        # Check outputs
        assert img.shape == (self.npix, self.npix)
        assert not np.isnan(img).any(), "Forward model produced NaN values"
        print(f"Forward model image range: [{img.min():.6f}, {img.max():.6f}]")

        print("✓ Forward model test passed")

    def test_likelihood_computation(self):
        """Test likelihood computation"""
        parser = CaskadeConfigParser(self.config_path)
        parser.set_static_params()

        prob_model = CaskadeImageProbModel(
            image_data=self.image_data,
            noise_map=self.noise_map,
            psf_kernel=self.psf_kernel,
            dpix=self.dpix,
            nsub=2,
            phys_model=parser.phys_model,
            use_linear=False,
            solver_type='nnls',
        )

        # Set reasonable parameter values using inference interface
        inference = CaskadeModelInference(prob_model, self.config_path)
        bounds = parser.prior_transform.get_param_bounds()
        param_array = np.array([(low + high) / 2.0 for low, high in bounds])
        inference.params_array2kargs(param_array)

        # Compute likelihood
        log_like = prob_model.likelihood(bs=1, debug=False)  # Turn off debug to see actual value

        # Check output
        assert isinstance(log_like, (float, np.ndarray))
        print(f"Log-likelihood: {log_like}")

        # Only check for non-NaN, may be -inf due to poor parameters
        if not np.isnan(log_like).any():
            print("✓ Likelihood computation test passed")
        else:
            print("! Likelihood produced NaN (parameter issue)")
            raise ValueError("NaN in likelihood")


class TestCaskadeModelInference:
    """Test parameter conversion and inference interface"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config_path = 'paper/demo/lens_src/model_config.yaml'
        self.npix = 64
        self.dpix = 0.074
        self.image_data = np.random.randn(self.npix, self.npix) * 0.1 + 1.0
        self.noise_map = np.ones((self.npix, self.npix)) * 0.1
        self.psf_kernel = np.array([[0.0, 0.1, 0.0],
                                    [0.1, 0.6, 0.1],
                                    [0.0, 0.1, 0.0]])

    def test_inference_creation(self):
        """Test creating inference object"""
        # Setup models
        parser = CaskadeConfigParser(self.config_path)
        parser.set_static_params()

        prob_model = CaskadeImageProbModel(
            image_data=self.image_data,
            noise_map=self.noise_map,
            psf_kernel=self.psf_kernel,
            dpix=self.dpix,
            nsub=2,
            phys_model=parser.phys_model,
            use_linear=False,
            solver_type='nnls',
        )

        # Create inference object
        inference = CaskadeModelInference(prob_model, self.config_path)

        # Check attributes
        assert inference.ndim == parser.ndim
        assert inference.prob_model is prob_model

        print(f"Inference object: ndim={inference.ndim}")
        print("✓ Inference creation test passed")

    def test_params_array2kargs(self):
        """Test parameter array to caskade conversion"""
        parser = CaskadeConfigParser(self.config_path)
        parser.set_static_params()

        prob_model = CaskadeImageProbModel(
            image_data=self.image_data,
            noise_map=self.noise_map,
            psf_kernel=self.psf_kernel,
            dpix=self.dpix,
            nsub=2,
            phys_model=parser.phys_model,
            use_linear=False,
            solver_type='nnls',
        )

        inference = CaskadeModelInference(prob_model, self.config_path)

        # Create parameter array
        bounds = parser.prior_transform.get_param_bounds()
        param_array = np.array([(low + high) / 2.0 for low, high in bounds])

        # Convert to caskade parameters
        result = inference.params_array2kargs(param_array)

        # Should return None (parameters set in place)
        assert result is None

        # Check that parameters were actually set
        # Get first dynamic param
        param_info = parser.prior_transform.dynamic_params[0]
        comp_type = param_info['comp_type']
        comp_idx = param_info['comp_idx']
        param_name = param_info['param_name']

        if comp_type == 'lens_mass_list':
            component = prob_model.sim_obj.phys_model.lens_mass[comp_idx]
        param_obj = getattr(component, param_name)

        # Check value was set
        assert param_obj.value is not None
        print(f"Parameter {param_name} value: {param_obj.value}")

        print("✓ Parameter conversion test passed")

    def test_prior_transform(self):
        """Test prior transformation"""
        parser = CaskadeConfigParser(self.config_path)
        parser.set_static_params()

        prob_model = CaskadeImageProbModel(
            image_data=self.image_data,
            noise_map=self.noise_map,
            psf_kernel=self.psf_kernel,
            dpix=self.dpix,
            nsub=2,
            phys_model=parser.phys_model,
            use_linear=False,
            solver_type='nnls',
        )

        inference = CaskadeModelInference(prob_model, self.config_path)

        # Create unit cube samples
        n_samples = 5
        cube_unit = np.random.rand(n_samples, inference.ndim)

        # Transform
        physical_params = inference.prior(cube_unit)

        # Check shape
        assert physical_params.shape == (n_samples, inference.ndim)

        # Check values are transformed (not all in [0,1])
        assert np.any(physical_params > 1.0) or np.any(physical_params < 0.0)

        print(f"Prior transform output shape: {physical_params.shape}")
        print("✓ Prior transform test passed")

    def test_likelihood_with_batch(self):
        """Test likelihood with batch processing"""
        parser = CaskadeConfigParser(self.config_path)
        parser.set_static_params()

        prob_model = CaskadeImageProbModel(
            image_data=self.image_data,
            noise_map=self.noise_map,
            psf_kernel=self.psf_kernel,
            dpix=self.dpix,
            nsub=2,
            phys_model=parser.phys_model,
            use_linear=False,
            solver_type='nnls',
        )

        inference = NautilusCaskadeModelSampler(prob_model, self.config_path)

        # Create parameter batch
        bounds = parser.prior_transform.get_param_bounds()
        bs = 5
        param_arrays = np.zeros((bs, inference.ndim))
        for i in range(bs):
            param_arrays[i, :] = np.array([(low + high) / 2.0 for low, high in bounds])

        # Compute likelihood
        log_like = inference.likelihood(param_arrays)

        # Check output
        assert log_like.shape == (bs,)
        assert not np.isnan(log_like).any()
        print(f"Batch likelihood shape: {log_like.shape}")
        print(f"Likelihood values: {log_like}")

        print("✓ Batch likelihood test passed")


if __name__ == "__main__":
    # Run tests
    print("Running Caskade Inference tests...\n")

    print("="*60)
    print("Testing CaskadeImageProbModel...")
    print("="*60)
    test_prob = TestCaskadeImageProbModel()
    test_prob.setup_method()
    test_prob.test_prob_model_creation()
    test_prob.test_forward_model()
    test_prob.test_likelihood_computation()

    print("\n" + "="*60)
    print("Testing CaskadeModelInference...")
    print("="*60)
    test_inference = TestCaskadeModelInference()
    test_inference.setup_method()
    test_inference.test_inference_creation()
    test_inference.test_params_array2kargs()
    test_inference.test_prior_transform()
    test_inference.test_likelihood_with_batch()

    print("\n" + "="*60)
    print("All Caskade Inference tests passed! ✓")
    print("="*60)
