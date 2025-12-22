"""
Performance and benchmark tests for TinyLensGpu.

This module tests performance characteristics and provides benchmarks
for key operations to ensure efficiency and track performance regressions.
"""

import pytest
import time
import jax.numpy as jnp
import numpy as np
from TinyLensGpu.Models import SIE, Shear, SersicEllipse, GaussianEllipse, PhysicalModel
from TinyLensGpu.Simulator import LensSimulator, SimulatorConfig
from TinyLensGpu.LinearSolver import LinearSolver


@pytest.mark.performance
class TestModelPerformance:
    """Benchmark model evaluation performance."""
    
    def test_sie_deflection_performance(self, benchmark_if_available):
        """Benchmark SIE deflection calculation."""
        sie = SIE(theta_E=1.5, e1=0.1, e2=0.05, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        # Medium-sized grid
        x = jnp.linspace(-5, 5, 100)
        y = jnp.linspace(-5, 5, 100)
        X, Y = jnp.meshgrid(x, y)
        
        # Warm-up (JIT compilation)
        _ = sie.deriv(X, Y)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            alpha_x, alpha_y = sie.deriv(X, Y)
            alpha_x.block_until_ready()  # Ensure computation completes
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        print(f"\nSIE deflection (100x100): {avg_time*1000:.2f} ms")
        
        # Should be fast (< 100ms on GPU, < 500ms on CPU)
        assert avg_time < 0.5, f"SIE deflection too slow: {avg_time:.3f}s"
    
    def test_sersic_light_performance(self, benchmark_if_available):
        """Benchmark Sersic light calculation."""
        sersic = SersicEllipse(R_sersic=1.0, n_sersic=4.0, e1=0.2, e2=0.1,
                              center_x=0.0, center_y=0.0, Ie=1.0)
        for param in [sersic.R_sersic, sersic.n_sersic, sersic.e1, sersic.e2,
                      sersic.center_x, sersic.center_y, sersic.Ie]:
            param.to_static()
        
        x = jnp.linspace(-5, 5, 100)
        y = jnp.linspace(-5, 5, 100)
        X, Y = jnp.meshgrid(x, y)
        
        # Warm-up
        _ = sersic.light(X, Y)
        
        start = time.time()
        for _ in range(10):
            light = sersic.light(X, Y)
            light.block_until_ready()
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        print(f"\nSersic light (100x100): {avg_time*1000:.2f} ms")
        
        assert avg_time < 0.5


@pytest.mark.performance
class TestSimulatorPerformance:
    """Benchmark simulator performance."""
    
    def test_nonlinear_simulation_performance(self, benchmark_if_available):
        """Benchmark non-linear simulation."""
        sie = SIE(theta_E=1.5, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
        source = GaussianEllipse(flux=10.0, sigma=0.5, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0)
        
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [source.flux, source.sigma, source.e1, source.e2,
                      source.center_x, source.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
        config = SimulatorConfig(dpix=0.05, npix=60, nsub=3)
        simulator = LensSimulator(model, config)
        
        # Warm-up
        _ = simulator.simulate(use_linear=False)
        
        start = time.time()
        for _ in range(5):
            img = simulator.simulate(use_linear=False)
            img.block_until_ready()
        elapsed = time.time() - start
        
        avg_time = elapsed / 5
        print(f"\nNon-linear simulation (60x60, nsub=3): {avg_time*1000:.2f} ms")
        
        # Should complete in reasonable time
        assert avg_time < 2.0, f"Simulation too slow: {avg_time:.3f}s"
    
    def test_linear_simulation_performance(self, benchmark_if_available):
        """Benchmark linear simulation with NNLS."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        source = GaussianEllipse(flux=1.0, sigma=0.5, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0)
        
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
            param.to_static()
        source.flux.to_dynamic()
        
        model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
        config = SimulatorConfig(dpix=0.05, npix=50, nsub=2)
        simulator = LensSimulator(model, config, solver_type='nnls')
        
        image_data = jnp.ones((50, 50))
        noise_map = jnp.ones((50, 50)) * 0.1
        
        # Warm-up
        _ = simulator.simulate(use_linear=True, image_map=image_data, noise_map=noise_map)
        
        start = time.time()
        for _ in range(5):
            img, _ = simulator.simulate(
                use_linear=True,
                return_intensity=True,
                image_map=image_data,
                noise_map=noise_map
            )
            img.block_until_ready()
        elapsed = time.time() - start
        
        avg_time = elapsed / 5
        print(f"\nLinear simulation with NNLS (50x50): {avg_time*1000:.2f} ms")
        
        assert avg_time < 3.0


@pytest.mark.performance
class TestLinearSolverPerformance:
    """Benchmark linear solver performance."""
    
    def test_nnls_solver_performance(self, benchmark_if_available):
        """Benchmark NNLS solver."""
        from TinyLensGpu.LinearSolver.linear_solver import fnnls_jax
        
        # Create test problem
        m, n = 1000, 50
        A = jnp.array(np.random.randn(m, n))
        b = jnp.array(np.random.randn(m))
        
        # Warm-up
        _ = fnnls_jax(A, b)
        
        start = time.time()
        for _ in range(10):
            x, _ = fnnls_jax(A, b)
            x.block_until_ready()
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        print(f"\nNNLS solver (1000x50): {avg_time*1000:.2f} ms")
        
        assert avg_time < 0.5
    
    def test_normal_solver_performance(self, benchmark_if_available):
        """Benchmark normal least squares solver."""
        from TinyLensGpu.LinearSolver.linear_solver import solve_linear
        
        m, n = 1000, 50
        A = jnp.array(np.random.randn(m, n))
        b = jnp.array(np.random.randn(m))
        
        # Warm-up
        _ = solve_linear(A, b)
        
        start = time.time()
        for _ in range(10):
            x = solve_linear(A, b)
            x.block_until_ready()
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        print(f"\nNormal LS solver (1000x50): {avg_time*1000:.2f} ms")
        
        assert avg_time < 0.2


@pytest.mark.performance
class TestScalability:
    """Test performance scaling with problem size."""
    
    def test_grid_size_scaling(self, benchmark_if_available):
        """Test how performance scales with grid size."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        sizes = [50, 100, 200]
        times = []
        
        for size in sizes:
            x = jnp.linspace(-5, 5, size)
            y = jnp.linspace(-5, 5, size)
            X, Y = jnp.meshgrid(x, y)
            
            # Warm-up
            _ = sie.deriv(X, Y)
            
            start = time.time()
            for _ in range(3):
                alpha_x, alpha_y = sie.deriv(X, Y)
                alpha_x.block_until_ready()
            elapsed = time.time() - start
            
            avg_time = elapsed / 3
            times.append(avg_time)
            print(f"\nGrid size {size}x{size}: {avg_time*1000:.2f} ms")
        
        # Check that scaling is reasonable (should be roughly quadratic)
        # Time ratio should be approximately (size_ratio)^2
        ratio_50_100 = times[1] / times[0]
        ratio_100_200 = times[2] / times[1]
        
        # Allow some overhead, so check if ratio is between 2 and 6
        assert 2 < ratio_50_100 < 6, f"Unexpected scaling 50->100: {ratio_50_100:.2f}"
        assert 2 < ratio_100_200 < 6, f"Unexpected scaling 100->200: {ratio_100_200:.2f}"
    
    def test_component_count_scaling(self, benchmark_if_available):
        """Test how performance scales with number of components."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        
        component_counts = [1, 5, 10]
        times = []
        
        for n_components in component_counts:
            # Create n Gaussian components
            gaussians = []
            for i in range(n_components):
                g = GaussianEllipse(flux=1.0, sigma=0.5, e1=0.0, e2=0.0,
                                   center_x=0.0, center_y=0.0)
                for param in [g.flux, g.sigma, g.e1, g.e2, g.center_x, g.center_y]:
                    param.to_static()
                gaussians.append(g)
            
            model = PhysicalModel(lens_mass=[sie], source_light=gaussians, lens_light=[])
            config = SimulatorConfig(dpix=0.05, npix=50, nsub=2)
            simulator = LensSimulator(model, config)
            
            # Warm-up
            _ = simulator.simulate(use_linear=False)
            
            start = time.time()
            for _ in range(3):
                img = simulator.simulate(use_linear=False)
                img.block_until_ready()
            elapsed = time.time() - start
            
            avg_time = elapsed / 3
            times.append(avg_time)
            print(f"\n{n_components} components: {avg_time*1000:.2f} ms")
        
        # Scaling should be roughly linear with component count
        ratio_1_5 = times[1] / times[0]
        ratio_5_10 = times[2] / times[1]
        
        # Check reasonable scaling (between 3 and 8 for 5x increase)
        assert 3 < ratio_1_5 < 8, f"Unexpected scaling 1->5: {ratio_1_5:.2f}"


@pytest.mark.performance
class TestMemoryEfficiency:
    """Test memory efficiency."""
    
    def test_large_simulation_memory(self, benchmark_if_available):
        """Test that large simulations don't cause memory issues."""
        sie = SIE(theta_E=1.5, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
        source = GaussianEllipse(flux=10.0, sigma=0.5, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0)
        
        for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
            param.to_static()
        for param in [source.flux, source.sigma, source.e1, source.e2,
                      source.center_x, source.center_y]:
            param.to_static()
        
        model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
        
        # Large simulation
        config = SimulatorConfig(dpix=0.05, npix=200, nsub=4)
        simulator = LensSimulator(model, config)
        
        # Should complete without memory errors
        img = simulator.simulate(use_linear=False)
        
        assert img.shape == (200, 200)
        assert not jnp.isnan(img).any()


@pytest.mark.performance
class TestJITCompilation:
    """Test JIT compilation behavior."""
    
    def test_jit_warmup_time(self, benchmark_if_available):
        """Test JIT compilation overhead."""
        sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
        sie.theta_E.to_static()
        sie.e1.to_static()
        sie.e2.to_static()
        sie.center_x.to_static()
        sie.center_y.to_static()
        
        x = jnp.linspace(-5, 5, 100)
        y = jnp.linspace(-5, 5, 100)
        X, Y = jnp.meshgrid(x, y)
        
        # First call (includes JIT compilation)
        start = time.time()
        alpha_x, alpha_y = sie.deriv(X, Y)
        alpha_x.block_until_ready()
        first_call_time = time.time() - start
        
        # Second call (JIT compiled)
        start = time.time()
        alpha_x, alpha_y = sie.deriv(X, Y)
        alpha_x.block_until_ready()
        second_call_time = time.time() - start
        
        print(f"\nFirst call (with JIT): {first_call_time*1000:.2f} ms")
        print(f"Second call (cached): {second_call_time*1000:.2f} ms")
        
        # Second call should be significantly faster
        assert second_call_time < first_call_time


# Fixture for optional benchmarking
@pytest.fixture
def benchmark_if_available():
    """Provide benchmark fixture if pytest-benchmark is available."""
    try:
        import pytest_benchmark
        return True
    except ImportError:
        return False


# Performance regression thresholds
PERFORMANCE_THRESHOLDS = {
    'sie_deflection_100x100': 0.5,  # seconds
    'sersic_light_100x100': 0.5,
    'simulation_60x60_nsub3': 2.0,
    'linear_simulation_50x50': 3.0,
    'nnls_1000x50': 0.5,
    'normal_ls_1000x50': 0.2,
}


def save_benchmark_results(results, filename='benchmark_results.json'):
    """Save benchmark results for tracking performance over time."""
    import json
    import os
    from datetime import datetime
    
    result_dict = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    output_dir = 'benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=2)
