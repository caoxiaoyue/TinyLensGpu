"""
Unit tests for Simulator module.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from TinyLensGpu.Simulator.Image.Simulator import (
    PhysicalModel, SimulatorConfig, LensSimulator, solve_linear_vec
)
from TinyLensGpu.Profile.Light.Sersic import SersicEllipse
from TinyLensGpu.Profile.Light.Gaussian import GaussianEllipse
from TinyLensGpu.Profile.Mass.Sie import SIE
from TinyLensGpu.Profile.Mass.Shear import Shear
from TinyLensGpu.Simulator.Image import util


@pytest.mark.unit
class TestPhysicalModel:
    """Test PhysicalModel class."""
    
    def test_physical_model_initialization(self):
        """Test PhysicalModel initialization."""
        lens_mass = [SIE(), Shear()]
        source_light = [SersicEllipse()]
        lens_light = [SersicEllipse()]
        
        model = PhysicalModel(
            lens_mass=lens_mass,
            source_light=source_light,
            lens_light=lens_light
        )
        
        assert len(model.lens_mass) == 2
        assert len(model.source_light) == 1
        assert len(model.lens_light) == 1
    
    def test_physical_model_empty_components(self):
        """Test PhysicalModel with empty components."""
        model = PhysicalModel(
            lens_mass=[],
            source_light=[],
            lens_light=[]
        )
        
        assert len(model.lens_mass) == 0
        assert len(model.source_light) == 0
        assert len(model.lens_light) == 0


@pytest.mark.unit
class TestSimulatorConfig:
    """Test SimulatorConfig class."""
    
    def test_simulator_config_initialization(self):
        """Test SimulatorConfig initialization."""
        dpix = 0.05
        npix = 50
        nsub = 4
        psf = np.ones((15, 15)) / 225.0
        
        config = SimulatorConfig(
            dpix=dpix,
            npix=npix,
            psf_kernel=psf,
            nsub=nsub
        )
        
        assert config.dpix == dpix
        assert config.npix == npix
        assert config.nsub == nsub
        assert config.psf_kernel.shape == psf.shape
    
    def test_simulator_config_coordinate_grids(self):
        """Test that coordinate grids are created correctly."""
        dpix = 0.05
        npix = 50
        nsub = 4
        
        config = SimulatorConfig(dpix=dpix, npix=npix, nsub=nsub)
        
        # Check that grids have correct shapes
        assert config.xgrid.shape == (npix, npix)
        assert config.ygrid.shape == (npix, npix)
        assert config.xgrid_sub.shape == (npix * nsub, npix * nsub)
        assert config.ygrid_sub.shape == (npix * nsub, npix * nsub)
        
        # Check that grid spacing is correct
        dx = config.xgrid[0, 1] - config.xgrid[0, 0]
        assert np.allclose(dx, dpix, atol=1e-6)
    
    def test_simulator_config_default_mask(self):
        """Test default mask creation."""
        config = SimulatorConfig(dpix=0.05, npix=50)
        
        # Default mask should be all False (no masking)
        assert config.mask.shape == (50, 50)
        assert config.mask.dtype == bool
        assert np.sum(config.mask) == 0  # All False
    
    def test_simulator_config_custom_mask(self):
        """Test custom mask."""
        npix = 50
        custom_mask = np.zeros((npix, npix))
        custom_mask[20:30, 20:30] = 1  # Mask central region
        
        config = SimulatorConfig(dpix=0.05, npix=npix, mask=custom_mask)
        
        assert np.sum(config.mask) == 100  # 10x10 masked region


@pytest.mark.unit
class TestLinearSolver:
    """Test linear solver functions."""
    
    def test_solve_linear_vec_basic(self):
        """Test basic linear system solving."""
        # Create a simple linear system
        bs = 5
        m = 10
        n = 3
        
        # A: [m, n, bs]
        A = np.random.randn(m, n, bs)
        # x_true: [n, bs]
        x_true = np.random.randn(n, bs)
        # b = A @ x_true: [m, bs]
        b = np.einsum('ijk,jk->ik', A, x_true)
        
        # Solve for x
        x_solved = solve_linear_vec(jnp.array(A), jnp.array(b))
        
        # Check that solution is close to true values
        assert x_solved.shape == (n, bs)
        assert np.allclose(x_solved, x_true, atol=1e-4, rtol=1e-3)
    
    def test_solve_linear_vec_overdetermined(self):
        """Test solving overdetermined system."""
        bs = 3
        m = 20
        n = 5
        
        # Create overdetermined system
        A = np.random.randn(m, n, bs)
        x_true = np.random.randn(n, bs)
        b = np.einsum('ijk,jk->ik', A, x_true)
        
        # Solve
        x_solved = solve_linear_vec(jnp.array(A), jnp.array(b))
        
        # Solution should be close (least squares)
        assert x_solved.shape == (n, bs)
        assert np.allclose(x_solved, x_true, atol=1e-3, rtol=1e-2)


@pytest.mark.unit
class TestLensSimulator:
    """Test LensSimulator class."""
    
    def test_lens_simulator_initialization(self):
        """Test LensSimulator initialization."""
        phys_model = PhysicalModel(
            lens_mass=[SIE()],
            source_light=[SersicEllipse()],
            lens_light=[SersicEllipse()]
        )
        
        sim_config = SimulatorConfig(dpix=0.05, npix=50, nsub=4)
        
        simulator = LensSimulator(phys_model=phys_model, sim_config=sim_config)
        
        assert simulator.phys_model == phys_model
        assert simulator.sim_config == sim_config
        assert simulator.solver_type == 'nnls'
    
    def test_lens_simulator_invalid_solver(self):
        """Test that invalid solver type raises error."""
        phys_model = PhysicalModel(
            lens_mass=[],
            source_light=[],
            lens_light=[]
        )
        
        sim_config = SimulatorConfig(dpix=0.05, npix=50)
        
        with pytest.raises(ValueError, match="solver_type must be either"):
            LensSimulator(
                phys_model=phys_model,
                sim_config=sim_config,
                solver_type='invalid'
            )


@pytest.mark.unit
class TestSimulatorUtil:
    """Test utility functions in Simulator module."""
    
    def test_make_grid_2d(self):
        """Test 2D grid generation."""
        npix = 50
        dpix = 0.05
        nsub = 1
        
        xgrid, ygrid = util.make_grid_2d(npix, dpix, nsub)
        
        # Check shape
        assert xgrid.shape == (npix, npix)
        assert ygrid.shape == (npix, npix)
        
        # Check that grid is centered at origin
        assert np.allclose(xgrid.mean(), 0.0, atol=dpix)
        assert np.allclose(ygrid.mean(), 0.0, atol=dpix)
        
        # Check spacing
        dx = xgrid[0, 1] - xgrid[0, 0]
        dy = ygrid[1, 0] - ygrid[0, 0]
        assert np.allclose(dx, dpix, atol=1e-6)
        assert np.allclose(dy, dpix, atol=1e-6)
    
    def test_make_grid_2d_subsampled(self):
        """Test 2D grid generation with subsampling."""
        npix = 50
        dpix = 0.05
        nsub = 4
        
        xgrid, ygrid = util.make_grid_2d(npix, dpix, nsub)
        
        # Check shape (should be nsub times larger)
        assert xgrid.shape == (npix * nsub, npix * nsub)
        assert ygrid.shape == (npix * nsub, npix * nsub)
        
        # Check spacing (should be nsub times smaller)
        dx = xgrid[0, 1] - xgrid[0, 0]
        expected_dx = dpix / nsub
        assert np.allclose(dx, expected_dx, atol=1e-6)
    
    def test_bin_image_general(self):
        """Test image binning function."""
        # Create a test image with subsampling
        nsub = 4
        npix = 10
        npix_sub = npix * nsub
        
        # Create a simple pattern (ones)
        image_sub = jnp.ones((npix_sub, npix_sub))
        
        # Bin the image
        image_binned = util.bin_image_general(image_sub, nsub)
        
        # Check shape
        assert image_binned.shape == (npix, npix)
        
        # Check that binning takes the mean (averages over subpixels)
        # Each binned pixel should be 1.0 (mean of nsub x nsub ones)
        assert jnp.allclose(image_binned, 1.0, atol=1e-6)
    
    def test_bin_image_general_with_pattern(self):
        """Test image binning with non-uniform pattern."""
        nsub = 2
        npix = 4
        npix_sub = npix * nsub
        
        # Create a checkerboard pattern
        image_sub = jnp.zeros((npix_sub, npix_sub))
        image_sub = image_sub.at[::2, ::2].set(1.0)
        image_sub = image_sub.at[1::2, 1::2].set(1.0)
        
        # Bin the image
        image_binned = util.bin_image_general(image_sub, nsub)
        
        # Check shape
        assert image_binned.shape == (npix, npix)
        
        # Each binned pixel should average to 0.5 (two 1s and two 0s in 2x2 grid)
        assert jnp.allclose(image_binned, 0.5, atol=1e-6)

