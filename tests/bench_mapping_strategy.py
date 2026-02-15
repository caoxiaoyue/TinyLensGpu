
import time
import jax
import jax.numpy as jnp
import numpy as np
from TinyLensGpu.ForwardSimulation.LensImage.pixelized_core.mapping_strategies import KnnKernelMappingStrategy, RectBilinearMappingStrategy
from TinyLensGpu.ForwardSimulation.LensImage.pixelized_core.artifacts import GridArtifacts
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import MappingConfig

def bench_knn():
    print("Benchmarking KnnKernelMappingStrategy...")
    n_data = 10000
    n_source = 1000
    
    key = jax.random.PRNGKey(0)
    source_mesh_beta = jax.random.uniform(key, (n_source, 2))
    data_mesh_beta = jax.random.uniform(key, (n_data, 2))
    
    grid = GridArtifacts(
        source_mesh=jnp.zeros((n_source, 2)), # Dummy
        source_mesh_beta=source_mesh_beta,
        data_mesh_beta=data_mesh_beta,
        source_grid_shape=None,
        source_grid_bounds=None
    )
    
    config = MappingConfig(k_neighbors=5, interp_kernel="wendland_c4", radius_scale=1.5)
    strategy = KnnKernelMappingStrategy(config=config)
    
    # Define JIT-ed version
    @jax.jit
    def run_build_dense(source_mesh_beta, data_mesh_beta):
        # Reconstruct grid inside to avoid pytree registration issues if any
        # Although dataclasses are usually fine if registered.
        # But let's just use the strategy method on reconstructed grid or assume strategy is static.
        g = GridArtifacts(
            source_mesh=jnp.zeros((n_source, 2)),
            source_mesh_beta=source_mesh_beta,
            data_mesh_beta=data_mesh_beta,
            source_grid_shape=None,
            source_grid_bounds=None
        )
        return strategy.build_dense(g)

    # Warmup
    print("  Warmup (JIT)...")
    _ = run_build_dense(source_mesh_beta, data_mesh_beta).block_until_ready()
    
    print("  Running (JIT)...")
    start = time.time()
    for _ in range(100):
        _ = run_build_dense(source_mesh_beta, data_mesh_beta).block_until_ready()
    end = time.time()
    
    print(f"  Knn build_dense (JIT) average time: {(end - start) / 100:.6f} s")

def bench_rect():
    print("Benchmarking RectBilinearMappingStrategy...")
    nx, ny = 64, 64
    n_data = 10000
    
    key = jax.random.PRNGKey(1)
    data_mesh_beta = jax.random.uniform(key, (n_data, 2))
    
    # Define JIT-ed version
    strategy = RectBilinearMappingStrategy()

    @jax.jit
    def run_build_dense(data_mesh_beta):
        g = GridArtifacts(
            source_mesh=jnp.zeros((nx*ny, 2)),
            source_mesh_beta=jnp.zeros((nx*ny, 2)),
            data_mesh_beta=data_mesh_beta,
            source_grid_shape=(ny, nx),
            source_grid_bounds=(0.0, 1.0, 0.0, 1.0)
        )
        return strategy.build_dense(g)
    
    # Warmup
    print("  Warmup (JIT)...")
    _ = run_build_dense(data_mesh_beta).block_until_ready()
    
    print("  Running (JIT)...")
    start = time.time()
    for _ in range(100):
        _ = run_build_dense(data_mesh_beta).block_until_ready()
    end = time.time()
    
    print(f"  Rect build_dense (JIT) average time: {(end - start) / 100:.6f} s")

if __name__ == "__main__":
    bench_knn()
    bench_rect()
