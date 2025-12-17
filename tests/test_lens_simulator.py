"""
Unit tests for LensSimulator comparing caskade implementation with original.

This module tests the CaskadeSimulator.LensSimulator against the original
Simulator.Image.Simulator.LensSimulator to ensure:
1. Numerical accuracy in forward simulation
2. Correct linear solver integration
3. Batch processing functionality
4. PSF convolution correctness
"""

import pytest
import numpy as np
import jax.numpy as jnp

# Original simulator
from TinyLensGpu.Simulator.Image.Simulator import (
    LensSimulator as OriginalLensSimulator,
    PhysicalModel as OriginalPhysicalModel,
    SimulatorConfig as OriginalSimulatorConfig
)
from TinyLensGpu.Profile.Mass.Sie import SIE as OriginalSIE
from TinyLensGpu.Profile.Mass.Shear import Shear as OriginalShear
from TinyLensGpu.Profile.Light.Sersic import SersicEllipse as OriginalSersic

# Caskade-based implementations
from TinyLensGpu.CaskadeSimulator.lens_simulator import LensSimulator as CaskadeLensSimulator
from TinyLensGpu.CaskadeSimulator.config import SimulatorConfig
from TinyLensGpu.CaskadeModels.composite import PhysicalModel
from TinyLensGpu.CaskadeModels.mass import SIE, Shear
from TinyLensGpu.CaskadeModels.light import SersicEllipse


class TestLensSimulatorNonLinear:
    """Test non-linear simulation (without linear solver)"""

    def setup_method(self):
        """Set up test fixtures"""
        # Simulation parameters
        self.npix = 64
        self.dpix = 0.074
        self.nsub = 2

        # Create PSF kernel
        self.psf_kernel = np.array([[0.0, 0.1, 0.0],
                                    [0.1, 0.6, 0.1],
                                    [0.0, 0.1, 0.0]])

        # Model parameters
        self.sie_params = {
            'theta_E': 1.5,
            'e1': 0.1,
            'e2': -0.05,
            'center_x': 0.0,
            'center_y': 0.0
        }

        self.shear_params = {
            'gamma1': 0.02,
            'gamma2': -0.01
        }

        self.source_params = {
            'R_sersic': 0.3,
            'n_sersic': 2.0,
            'e1': 0.15,
            'e2': 0.1,
            'center_x': 0.05,
            'center_y': -0.02,
            'Ie': 5.0
        }

    def test_forward_simulation_simple(self):
        """Test simple forward simulation with one source component"""
        # Create caskade model
        sie = SIE(**self.sie_params)
        shear = Shear(**self.shear_params)
        source = SersicEllipse(**self.source_params)

        # Set all params to static
        for param_name, param_value in self.sie_params.items():
            getattr(sie, param_name).to_static(param_value)
        for param_name, param_value in self.shear_params.items():
            getattr(shear, param_name).to_static(param_value)
        for param_name, param_value in self.source_params.items():
            getattr(source, param_name).to_static(param_value)

        phys_model = PhysicalModel(
            lens_mass=[sie, shear],
            source_light=[source],
            lens_light=[]
        )

        # Create simulators
        sim_config = SimulatorConfig(
            dpix=self.dpix,
            npix=self.npix,
            psf_kernel=self.psf_kernel,
            nsub=self.nsub
        )

        caskade_sim = CaskadeLensSimulator(
            phys_model=phys_model,
            sim_config=sim_config,
            solver_type='nnls'
        )

        # Create original simulator
        original_sie = OriginalSIE(**self.sie_params)
        original_shear = OriginalShear(**self.shear_params)
        original_source = OriginalSersic(**self.source_params)

        original_phys_model = OriginalPhysicalModel(
            lens_mass=[original_sie, original_shear],
            source_light=[original_source],
            lens_light=[]
        )

        original_sim_config = OriginalSimulatorConfig(
            dpix=self.dpix,
            npix=self.npix,
            psf_kernel=self.psf_kernel,
            nsub=self.nsub
        )

        original_sim = OriginalLensSimulator(
            phys_model=original_phys_model,
            sim_config=original_sim_config
        )

        # Prepare params for original simulator
        params = [
            [self.sie_params, self.shear_params],  # lens_mass
            [self.source_params],  # source_light
            []  # lens_light
        ]

        # Run simulations
        caskade_img = caskade_sim.simulate(bs=1, use_linear=False)
        original_img = original_sim.simulate(
            params=params,
            bs=1,
            use_linear=False,
            xgrid_sub=original_sim_config.xgrid_sub,
            ygrid_sub=original_sim_config.ygrid_sub,
            psf_kernel=original_sim_config.psf_kernel
        )

        # Convert to numpy for comparison
        caskade_img_np = np.array(caskade_img)
        original_img_np = np.array(original_img).squeeze()

        # Compare
        max_diff = np.max(np.abs(caskade_img_np - original_img_np))
        print(f"\nMax difference (forward simulation): {max_diff}")

        assert caskade_img_np.shape == original_img_np.shape, \
            f"Shape mismatch: caskade {caskade_img_np.shape} vs original {original_img_np.shape}"

        np.testing.assert_allclose(
            caskade_img_np, original_img_np, rtol=3e-4, atol=1e-5,
            err_msg="Caskade and original simulator outputs differ"
        )

        print("✓ Forward simulation test passed")

    def test_batch_processing(self):
        """Test batch processing with bs > 1"""
        # Create simple model with only source light
        source = SersicEllipse(**self.source_params)
        for param_name, param_value in self.source_params.items():
            getattr(source, param_name).to_static(param_value)

        phys_model = PhysicalModel(
            lens_mass=[],
            source_light=[source],
            lens_light=[]
        )

        sim_config = SimulatorConfig(
            dpix=self.dpix,
            npix=self.npix,
            psf_kernel=self.psf_kernel,
            nsub=self.nsub
        )

        caskade_sim = CaskadeLensSimulator(
            phys_model=phys_model,
            sim_config=sim_config
        )

        # Test different batch sizes
        for bs in [1, 5, 10]:
            img = caskade_sim.simulate(bs=bs, use_linear=False)

            if bs == 1:
                assert img.ndim == 2, f"bs=1 should return 2D image, got {img.ndim}D"
                assert img.shape == (self.npix, self.npix)
            else:
                assert img.ndim == 3, f"bs={bs} should return 3D image, got {img.ndim}D"
                assert img.shape == (self.npix, self.npix, bs)

        print("✓ Batch processing test passed")

    def test_with_lens_light(self):
        """Test simulation with lens light component"""
        lens_light_params = {
            'R_sersic': 0.8,
            'n_sersic': 4.0,
            'e1': 0.2,
            'e2': -0.1,
            'center_x': 0.0,
            'center_y': 0.0,
            'Ie': 3.0
        }

        # Create caskade model
        source = SersicEllipse(**self.source_params)
        lens_light = SersicEllipse(**lens_light_params)

        for param_name, param_value in self.source_params.items():
            getattr(source, param_name).to_static(param_value)
        for param_name, param_value in lens_light_params.items():
            getattr(lens_light, param_name).to_static(param_value)

        phys_model = PhysicalModel(
            lens_mass=[],
            source_light=[source],
            lens_light=[lens_light]
        )

        sim_config = SimulatorConfig(
            dpix=self.dpix,
            npix=self.npix,
            psf_kernel=self.psf_kernel,
            nsub=self.nsub
        )

        caskade_sim = CaskadeLensSimulator(
            phys_model=phys_model,
            sim_config=sim_config
        )

        # Create original simulator
        original_source = OriginalSersic(**self.source_params)
        original_lens = OriginalSersic(**lens_light_params)

        original_phys_model = OriginalPhysicalModel(
            lens_mass=[],
            source_light=[original_source],
            lens_light=[original_lens]
        )

        original_sim_config = OriginalSimulatorConfig(
            dpix=self.dpix,
            npix=self.npix,
            psf_kernel=self.psf_kernel,
            nsub=self.nsub
        )

        original_sim = OriginalLensSimulator(
            phys_model=original_phys_model,
            sim_config=original_sim_config
        )

        # Prepare params for original simulator
        params = [
            [],  # lens_mass (empty)
            [self.source_params],  # source_light
            [lens_light_params]  # lens_light
        ]

        # Run simulations
        caskade_img = caskade_sim.simulate(bs=1, use_linear=False)
        original_img = original_sim.simulate(
            params=params,
            bs=1,
            use_linear=False,
            xgrid_sub=original_sim_config.xgrid_sub,
            ygrid_sub=original_sim_config.ygrid_sub,
            psf_kernel=original_sim_config.psf_kernel
        )

        # Convert and compare
        caskade_img_np = np.array(caskade_img)
        original_img_np = np.array(original_img).squeeze()

        max_diff = np.max(np.abs(caskade_img_np - original_img_np))
        print(f"\nMax difference (with lens light): {max_diff}")

        np.testing.assert_allclose(
            caskade_img_np, original_img_np, rtol=3e-4, atol=1e-5,
            err_msg="Results differ with lens light component"
        )

        print("✓ Lens light test passed")


class TestLensSimulatorLinear:
    """Test linear simulation with intensity optimization"""

    def setup_method(self):
        """Set up test fixtures"""
        self.npix = 64
        self.dpix = 0.074
        self.nsub = 2

        # Simple PSF
        self.psf_kernel = np.array([[0.0, 0.1, 0.0],
                                    [0.1, 0.6, 0.1],
                                    [0.0, 0.1, 0.0]])

        # Model parameters (without Ie - will be solved)
        self.source_params = {
            'R_sersic': 0.3,
            'n_sersic': 2.0,
            'e1': 0.0,
            'e2': 0.0,
            'center_x': 0.0,
            'center_y': 0.0,
            'Ie': 1.0  # Initial value, will be solved
        }

    def test_linear_solver_nnls(self):
        """Test linear solver with NNLS"""
        # Create model
        source = SersicEllipse(**self.source_params)
        for param_name, param_value in self.source_params.items():
            if param_name != 'Ie':
                getattr(source, param_name).to_static(param_value)

        phys_model = PhysicalModel(
            lens_mass=[],
            source_light=[source],
            lens_light=[]
        )

        sim_config = SimulatorConfig(
            dpix=self.dpix,
            npix=self.npix,
            psf_kernel=self.psf_kernel,
            nsub=self.nsub
        )

        caskade_sim = CaskadeLensSimulator(
            phys_model=phys_model,
            sim_config=sim_config,
            solver_type='nnls'
        )

        # Create mock data (use the model itself with Ie=5.0)
        source_for_data = SersicEllipse(**{**self.source_params, 'Ie': 5.0})
        for param_name, param_value in {**self.source_params, 'Ie': 5.0}.items():
            getattr(source_for_data, param_name).to_static(param_value)

        phys_model_for_data = PhysicalModel(
            lens_mass=[],
            source_light=[source_for_data],
            lens_light=[]
        )

        sim_for_data = CaskadeLensSimulator(
            phys_model=phys_model_for_data,
            sim_config=sim_config
        )

        mock_image = sim_for_data.simulate(bs=1, use_linear=False)
        mock_noise = np.ones_like(mock_image) * 0.1

        # Run linear simulation
        img, intensity = caskade_sim.simulate(
            bs=1,
            use_linear=True,
            return_intensity=True,
            image_map=mock_image,
            noise_map=mock_noise
        )

        # Check that solved intensity is close to 5.0
        print(f"\nSolved intensity: {intensity}")
        print(f"Expected intensity: 5.0")

        assert intensity.shape == (1,), f"Intensity shape should be (1,), got {intensity.shape}"
        np.testing.assert_allclose(
            intensity, 5.0, rtol=0.1, atol=0.5,
            err_msg="Solved intensity differs from true value"
        )

        print("✓ Linear solver NNLS test passed")

    def test_linear_vs_normal_solver(self):
        """Test that NNLS and normal solver give similar results for positive-definite case"""
        source = SersicEllipse(**self.source_params)
        for param_name, param_value in self.source_params.items():
            if param_name != 'Ie':
                getattr(source, param_name).to_static(param_value)

        phys_model = PhysicalModel(
            lens_mass=[],
            source_light=[source],
            lens_light=[]
        )

        sim_config = SimulatorConfig(
            dpix=self.dpix,
            npix=self.npix,
            psf_kernel=self.psf_kernel,
            nsub=self.nsub
        )

        # Create mock data
        source_for_data = SersicEllipse(**{**self.source_params, 'Ie': 3.0})
        for param_name, param_value in {**self.source_params, 'Ie': 3.0}.items():
            getattr(source_for_data, param_name).to_static(param_value)

        phys_model_for_data = PhysicalModel(
            lens_mass=[],
            source_light=[source_for_data],
            lens_light=[]
        )

        sim_for_data = CaskadeLensSimulator(
            phys_model=phys_model_for_data,
            sim_config=sim_config
        )

        mock_image = sim_for_data.simulate(bs=1, use_linear=False)
        mock_noise = np.ones_like(mock_image) * 0.1

        # Test both solvers
        for solver_type in ['nnls', 'normal']:
            sim = CaskadeLensSimulator(
                phys_model=phys_model,
                sim_config=sim_config,
                solver_type=solver_type
            )

            img, intensity = sim.simulate(
                bs=1,
                use_linear=True,
                return_intensity=True,
                image_map=mock_image,
                noise_map=mock_noise
            )

            print(f"\n{solver_type} solver intensity: {intensity}")

            # Both should be close to 3.0
            np.testing.assert_allclose(
                intensity, 3.0, rtol=0.15, atol=0.5,
                err_msg=f"{solver_type} solver failed"
            )

        print("✓ Linear vs normal solver test passed")


if __name__ == "__main__":
    # Run tests
    print("Running LensSimulator tests...\n")

    print("="*60)
    print("Testing Non-Linear Simulation...")
    print("="*60)
    test_nonlinear = TestLensSimulatorNonLinear()
    test_nonlinear.setup_method()
    test_nonlinear.test_forward_simulation_simple()
    test_nonlinear.test_batch_processing()
    test_nonlinear.test_with_lens_light()

    print("\n" + "="*60)
    print("Testing Linear Simulation...")
    print("="*60)
    test_linear = TestLensSimulatorLinear()
    test_linear.setup_method()
    test_linear.test_linear_solver_nnls()
    test_linear.test_linear_vs_normal_solver()

    print("\n" + "="*60)
    print("All LensSimulator tests passed! ✓")
    print("="*60)
