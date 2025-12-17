"""
Test script for running caskade-based lens model inference.

This script demonstrates how to use RunCaskadeLensModel with
the lens_src demo configuration.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel


def test_caskade_lens_src_quick():
    """
    Quick test of lens_src demo with optimizer (faster than sampler).

    This test verifies that:
    1. Configuration can be loaded
    2. Data can be loaded
    3. Caskade models can be built
    4. Inference can be setup
    5. Likelihood can be computed
    """
    print("="*70)
    print("Testing Caskade lens_src Demo (Quick Test with Optimizer)")
    print("="*70)

    # Use absolute path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'paper/demo/lens_src/model_config.yaml')

    # Change to demo directory so relative paths work
    demo_dir = os.path.join(base_dir, 'paper/demo/lens_src')
    original_dir = os.getcwd()
    os.chdir(demo_dir)

    try:
        # Initialize runner
        print("\n1. Initializing runner...")
        runner = RunCaskadeLensModel('model_config.yaml')
        print(f"   Config loaded from: {config_path}")

        # Load data
        print("\n2. Loading data...")
        runner.load_data()
        print(f"   Image shape: {runner.image_map.shape}")
        print(f"   Noise shape: {runner.noise_map.shape}")
        print(f"   PSF shape: {runner.psf_map.shape}")

        # Setup model
        print("\n3. Setting up caskade model...")
        runner.setup_model()
        print(f"   Config parser: {runner.config_parser}")
        print(f"   Physical model: {runner.phys_model}")
        print(f"   Probability model: {runner.prob_model}")
        print(f"   Number of dynamic params: {runner.config_parser.ndim}")
        print(f"   Number of linear params: {runner.config_parser.n_linear_params}")

        # Modify config to use optimizer for quick test
        print("\n4. Setting up optimizer (quick test)...")
        bounds = runner.config_parser.prior_transform.get_param_bounds()
        runner.inference_config = {
            'type': 'optimizer',
            'method': 'differential_evolution',
            'settings': {
                'maxiter': 10,  # Very few iterations for quick test
                'popsize': 5,   # Small population
                'bounds': bounds,  # Add bounds for optimizer
            }
        }
        runner.setup_inference()
        print(f"   Inference method: {type(runner.inference).__name__}")

        # Initialize JIT compilation
        print("\n5. Initializing JIT compilation...")
        runner.init_jit_likelihood()

        # Test parameter conversion
        print("\n6. Testing parameter conversion...")
        import numpy as np
        bounds = runner.config_parser.prior_transform.get_param_bounds()
        test_params = np.array([(low + high) / 2.0 for low, high in bounds])
        runner.inference.params_array2kargs(test_params)
        log_like = runner.prob_model.likelihood(bs=1, debug=False)
        print(f"   Test log-likelihood: {log_like}")

        # Run quick optimization
        print("\n7. Running quick optimization...")
        runner.run_inference()
        print(f"   Results keys: {runner.results.keys()}")
        if 'best_merit' in runner.results:
            print(f"   Best merit (negative log-like): {runner.results['best_merit']}")
        if 'best_params' in runner.results:
            print(f"   Best params shape: {runner.results['best_params'].shape}")

        print("\n" + "="*70)
        print("Caskade lens_src Demo Test PASSED! ✓")
        print("="*70)

        return runner

    finally:
        # Restore original directory
        os.chdir(original_dir)


def test_caskade_lens_src_sampler_init():
    """
    Test initialization with sampler (don't run full sampling).

    This verifies that the sampler can be properly initialized.
    """
    print("\n"*2 + "="*70)
    print("Testing Caskade lens_src Demo with Sampler (Init Only)")
    print("="*70)

    # Use absolute path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'paper/demo/lens_src/model_config.yaml')

    # Change to demo directory
    demo_dir = os.path.join(base_dir, 'paper/demo/lens_src')
    original_dir = os.getcwd()
    os.chdir(demo_dir)

    try:
        # Initialize runner
        print("\n1. Initializing runner with sampler config...")
        runner = RunCaskadeLensModel('model_config.yaml')
        runner.load_data()
        runner.setup_model()

        # Keep the sampler settings from config
        print(f"\n2. Sampler settings:")
        print(f"   Method: {runner.inference_config['method']}")
        print(f"   Settings: {runner.inference_config['settings']}")

        runner.setup_inference()
        print(f"   Inference type: {type(runner.inference).__name__}")

        # JIT compile but don't run sampling
        print("\n3. JIT compiling likelihood...")
        runner.init_jit_likelihood()

        print("\n" + "="*70)
        print("Caskade lens_src Sampler Initialization PASSED! ✓")
        print("="*70)

        return runner

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    # Quick test with optimizer
    runner_opt = test_caskade_lens_src_quick()

    # Sampler initialization test
    runner_sampler = test_caskade_lens_src_sampler_init()

    print("\n\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    print("\nThe caskade-based inference system is working correctly.")
    print("To run full sampling, use: runner.run_inference()")
