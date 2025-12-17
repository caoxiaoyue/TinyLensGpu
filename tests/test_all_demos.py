"""
Comprehensive test suite for all TinyLensGpu demo cases.

This script tests all 6 demo configurations with the caskade-based system:
1. lens_only - Single Sersic lens light
2. lens_only_mge - 15 Gaussian MGE lens light
3. lens_src - SIE+Shear+source+lens (already tested in test_demo_lens_src.py)
4. lens_src_mge - Full system with MGE
5. src_only - Source light only
6. src_only_poslike - Source with position likelihood constraint
"""

import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel


def run_demo_test(demo_name, description):
    """
    Generic function to test a demo configuration.

    Parameters:
        demo_name (str): Name of demo directory
        description (str): Description of what the demo tests

    Returns:
        runner: RunCaskadeLensModel instance with results
    """
    print("\n" + "="*70)
    print(f"Testing Demo: {demo_name}")
    print(f"Description: {description}")
    print("="*70)

    # Get paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    demo_dir = os.path.join(base_dir, 'paper/demo', demo_name)

    # Check if demo directory exists
    if not os.path.exists(demo_dir):
        print(f"⚠️  Demo directory not found: {demo_dir}")
        pytest.skip(f"Demo {demo_name} not found")
        return None

    # Check if config file exists
    config_path = os.path.join(demo_dir, 'model_config.yaml')
    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found: {config_path}")
        pytest.skip(f"Config for {demo_name} not found")
        return None

    # Change to demo directory
    original_dir = os.getcwd()
    os.chdir(demo_dir)

    try:
        # Initialize runner
        print("\n1. Initializing runner...")
        runner = RunCaskadeLensModel('model_config.yaml')
        print(f"   ✓ Config loaded")

        # Load data
        print("\n2. Loading data...")
        runner.load_data()
        print(f"   ✓ Image shape: {runner.image_map.shape}")
        print(f"   ✓ Noise shape: {runner.noise_map.shape}")
        print(f"   ✓ PSF shape: {runner.psf_map.shape}")

        # Setup model
        print("\n3. Setting up caskade model...")
        runner.setup_model()
        print(f"   ✓ Dynamic params: {runner.config_parser.ndim}")
        print(f"   ✓ Linear params: {runner.config_parser.n_linear_params}")

        # Print model components
        phys_model = runner.phys_model
        print(f"   ✓ Lens mass components: {len(phys_model.lens_mass)}")
        print(f"   ✓ Source light components: {len(phys_model.source_light)}")
        print(f"   ✓ Lens light components: {len(phys_model.lens_light)}")

        # Run quick optimization test
        print("\n4. Running quick optimization test...")
        bounds = runner.config_parser.prior_transform.get_param_bounds()
        runner.inference_config = {
            'type': 'optimizer',
            'method': 'differential_evolution',
            'settings': {
                'maxiter': 5,   # Very quick test
                'popsize': 3,
                'bounds': bounds,
            }
        }
        runner.setup_inference()
        print(f"   ✓ Inference method: {type(runner.inference).__name__}")

        # JIT compile
        print("\n5. JIT compiling likelihood...")
        runner.init_jit_likelihood()
        print(f"   ✓ JIT compilation complete")

        # Test likelihood computation
        print("\n6. Testing likelihood computation...")
        import numpy as np
        test_params = np.array([(low + high) / 2.0 for low, high in bounds])
        runner.inference.params_array2kargs(test_params)
        log_like = runner.prob_model.likelihood(bs=1, debug=False)
        print(f"   ✓ Test log-likelihood: {log_like:.2f}")

        # Run optimization
        print("\n7. Running optimization...")
        runner.run_inference()
        print(f"   ✓ Optimization complete")
        if hasattr(runner.results, 'fun'):
            print(f"   ✓ Best merit: {runner.results.fun:.2f}")

        print("\n" + "="*70)
        print(f"✅ Demo '{demo_name}' PASSED!")
        print("="*70)

        return runner

    except Exception as e:
        print(f"\n❌ Demo '{demo_name}' FAILED with error:")
        print(f"   {type(e).__name__}: {e}")
        raise

    finally:
        # Restore original directory
        os.chdir(original_dir)


def test_lens_only():
    """
    Test lens_only demo: Single Sersic lens light profile.

    This demo tests:
    - Single Sersic light model
    - Parameter fitting without mass/source
    - Linear intensity solving (NNLS)
    """
    runner = run_demo_test(
        demo_name='lens_only',
        description='Single Sersic lens light (no lensing)'
    )

    # Validate model structure
    assert runner.config_parser.ndim > 0, "Should have dynamic parameters"
    assert len(runner.phys_model.lens_mass) == 0, "Should have no mass components"
    assert len(runner.phys_model.source_light) == 0, "Should have no source components"
    assert len(runner.phys_model.lens_light) > 0, "Should have lens light components"


def test_lens_only_mge():
    """
    Test lens_only_mge demo: 15-component Gaussian MGE lens light.

    This demo tests:
    - Multi-Gaussian expansion (MGE) with 15 components
    - Parameter linking (shared center and ellipticity)
    - Linear amplitude solving for all components
    """
    runner = run_demo_test(
        demo_name='lens_only_mge',
        description='15 Gaussian MGE lens light (parameter linking)'
    )

    # Validate model structure
    assert len(runner.phys_model.lens_light) == 15, "Should have 15 Gaussian components"
    assert runner.config_parser.n_linear_params == 15, "Should have 15 linear params (amplitudes)"

    # Check parameter linking (Gaussians 1-14 should share params with Gaussian 0)
    lens_light = runner.phys_model.lens_light
    for i in range(1, 15):
        # These should be the same object (pointer mode)
        assert lens_light[i].center_x is lens_light[0].center_x, \
            f"Gaussian {i} center_x should link to Gaussian 0"
        assert lens_light[i].center_y is lens_light[0].center_y, \
            f"Gaussian {i} center_y should link to Gaussian 0"
        assert lens_light[i].e1 is lens_light[0].e1, \
            f"Gaussian {i} e1 should link to Gaussian 0"
        assert lens_light[i].e2 is lens_light[0].e2, \
            f"Gaussian {i} e2 should link to Gaussian 0"


def test_lens_src():
    """
    Test lens_src demo: Full lens system (SIE+Shear+source+lens).

    This demo tests:
    - SIE mass model
    - External shear
    - Sersic source light
    - Sersic lens light
    - Linear intensity solving

    Note: This is already tested in test_demo_lens_src.py, but included
    here for completeness of the demo suite.
    """
    runner = run_demo_test(
        demo_name='lens_src',
        description='Full lens system (SIE+Shear+source+lens)'
    )

    # Validate model structure
    assert len(runner.phys_model.lens_mass) == 2, "Should have SIE + Shear"
    assert len(runner.phys_model.source_light) >= 1, "Should have source light"
    assert len(runner.phys_model.lens_light) >= 1, "Should have lens light"


def test_lens_src_mge():
    """
    Test lens_src_mge demo: Full lens system with MGE.

    This demo tests:
    - SIE mass + Shear
    - Multiple Gaussian source light (MGE)
    - Multiple Gaussian lens light (MGE)
    - Complex parameter linking
    - Large number of linear parameters
    """
    runner = run_demo_test(
        demo_name='lens_src_mge',
        description='Full lens system with MGE (complex parameter linking)'
    )

    # Validate model structure
    assert len(runner.phys_model.lens_mass) >= 1, "Should have mass components"
    assert len(runner.phys_model.source_light) > 1, "Should have multiple source Gaussians"
    assert len(runner.phys_model.lens_light) > 1, "Should have multiple lens Gaussians"


def test_src_only():
    """
    Test src_only demo: Source light only (no lensing).

    This demo tests:
    - Source light modeling without mass
    - Simplified workflow
    """
    runner = run_demo_test(
        demo_name='src_only',
        description='Source light only (no lensing)'
    )

    # Validate model structure
    assert len(runner.phys_model.lens_mass) == 0, "Should have no mass components"
    assert len(runner.phys_model.source_light) > 0, "Should have source light"


def test_src_only_poslike():
    """
    Test src_only_poslike demo: Source with position likelihood constraint.

    This demo tests:
    - Position likelihood constraints
    - Multiple image positions mapping to same source
    - Penalty-based likelihood
    """
    runner = run_demo_test(
        demo_name='src_only_poslike',
        description='Source with position likelihood constraint'
    )

    # Validate model structure
    assert len(runner.phys_model.lens_mass) > 0, "Should have mass for lensing"
    assert len(runner.phys_model.source_light) > 0, "Should have source light"

    # Check if position likelihood is configured
    if hasattr(runner.prob_model, 'position_like_config'):
        assert runner.prob_model.position_like_config is not None, \
            "Should have position likelihood configured"


def test_all_demos_summary():
    """
    Run all demos and generate summary report.

    This is not a pytest test but a convenient function to run all demos
    and see a comprehensive report.
    """
    demos = [
        ('lens_only', 'Single Sersic lens light'),
        ('lens_only_mge', '15 Gaussian MGE lens light'),
        ('lens_src', 'Full lens system (SIE+Shear+source+lens)'),
        ('lens_src_mge', 'Full system with MGE'),
        ('src_only', 'Source light only'),
        ('src_only_poslike', 'Source with position likelihood'),
    ]

    print("\n" + "="*70)
    print("COMPREHENSIVE DEMO TEST SUITE")
    print("="*70)

    results = {}

    for demo_name, description in demos:
        try:
            runner = run_demo_test(demo_name, description)
            results[demo_name] = {
                'status': '✅ PASSED',
                'ndim': runner.config_parser.ndim,
                'n_linear': runner.config_parser.n_linear_params,
                'image_shape': runner.image_map.shape,
            }
        except Exception as e:
            results[demo_name] = {
                'status': f'❌ FAILED: {type(e).__name__}',
                'error': str(e)
            }

    # Print summary
    print("\n\n" + "="*70)
    print("DEMO TEST SUMMARY")
    print("="*70)

    for demo_name, result in results.items():
        print(f"\n{demo_name}:")
        print(f"  Status: {result['status']}")
        if 'ndim' in result:
            print(f"  Dynamic params: {result['ndim']}")
            print(f"  Linear params: {result['n_linear']}")
            print(f"  Image shape: {result['image_shape']}")
        elif 'error' in result:
            print(f"  Error: {result['error']}")

    # Overall status
    passed = sum(1 for r in results.values() if '✅' in r['status'])
    total = len(results)

    print("\n" + "="*70)
    print(f"OVERALL: {passed}/{total} demos passed")
    print("="*70)

    return results


if __name__ == "__main__":
    # Run all demos with summary
    results = test_all_demos_summary()

    # Exit with appropriate code
    all_passed = all('✅' in r['status'] for r in results.values())
    sys.exit(0 if all_passed else 1)
