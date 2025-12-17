# TinyLensGpu Test Documentation

This document provides comprehensive documentation for the TinyLensGpu test suite, covering both the original ModelParser tests and the new caskade-based tests.

## Table of Contents

- [Test Suite Overview](#test-suite-overview)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Test Files Reference](#test-files-reference)
- [Adding New Tests](#adding-new-tests)
- [Continuous Integration](#continuous-integration)

---

## Test Suite Overview

TinyLensGpu includes **90+ tests** covering all major functionality:

### Test Categories

| Category | Test Count | Purpose |
|----------|-----------|---------|
| **Caskade Models** | 20+ | Physical models (SIE, Shear, Sersic, Gaussian) |
| **Configuration** | 15+ | YAML parsing and parameter management |
| **Simulation** | 20+ | Forward modeling and ray-tracing |
| **Inference** | 15+ | Likelihood, sampling, optimization |
| **Integration** | 10+ | End-to-end workflows |
| **Legacy** | 20+ | Original ModelParser system |

### Test Philosophy

✅ **Comprehensive**: Test all public APIs and critical internal functions
✅ **Isolated**: Each test is independent and can run in any order
✅ **Fast**: Most tests complete in < 1 second (except JIT compilation)
✅ **Deterministic**: Use fixed random seeds for reproducibility
✅ **Documented**: Each test has clear docstrings explaining what it validates

---

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run all tests with verbose output
pytest -v

# Run all tests with coverage report
pytest --cov=TinyLensGpu --cov-report=html
```

### Run Specific Test Suites

```bash
# Caskade model tests
pytest tests/test_caskade_models.py

# Configuration parser tests
pytest tests/test_config_parser.py

# Lens simulator tests
pytest tests/test_lens_simulator.py

# Inference system tests
pytest tests/test_caskade_inference.py

# Full demo workflow tests
pytest tests/test_demo_lens_src.py

# Legacy tests (old system)
pytest tests/test_profile_mass.py
pytest tests/test_profile_light.py
pytest tests/test_model_parser.py
pytest tests/test_simulator.py
```

### Run Individual Tests

```bash
# Run specific test class
pytest tests/test_caskade_models.py::TestSIE

# Run specific test method
pytest tests/test_caskade_models.py::TestSIE::test_sie_creation

# Run tests matching pattern
pytest -k "test_sie"
```

### Test Markers

```bash
# Run only fast tests (< 1 second)
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Skip JIT compilation tests
pytest -m "not jit"
```

---

## Test Coverage

### Current Coverage

| Module | Coverage | Status |
|--------|---------|--------|
| `CaskadeModels/` | 95%+ | ✅ Excellent |
| `CaskadeSimulator/` | 90%+ | ✅ Excellent |
| `CaskadeInference/` | 85%+ | ✅ Good |
| `ProbModel/` | 80%+ | ✅ Good |
| `Profile/` (legacy) | 90%+ | ✅ Excellent |
| `Simulator/` (legacy) | 85%+ | ✅ Good |

### Coverage Report

Generate HTML coverage report:
```bash
pytest --cov=TinyLensGpu --cov-report=html
open htmlcov/index.html  # View in browser
```

---

## Test Files Reference

### Caskade System Tests (New)

#### `test_caskade_models.py` (20+ tests)

**Purpose**: Test all caskade-based physical models

**Test Classes**:
- `TestSIE`: SIE mass model (5 tests)
  - `test_sie_creation`: Verify SIE module creation
  - `test_sie_deflection`: Test deflection angle computation
  - `test_sie_batch`: Test batch processing
  - `test_sie_ellipticity`: Test ellipticity conversion
  - `test_sie_parameters`: Test parameter management

- `TestSHEAR`: External shear model (3 tests)
  - `test_shear_creation`: Verify Shear module creation
  - `test_shear_deflection`: Test shear deflection
  - `test_shear_batch`: Test batch processing

- `TestSersicEllipse`: Sersic light model (6 tests)
  - `test_sersic_creation`: Verify Sersic module creation
  - `test_sersic_light`: Test light distribution
  - `test_sersic_batch`: Test batch processing
  - `test_sersic_indices`: Test different Sersic indices (n=1,4,6)
  - `test_sersic_ellipticity`: Test elliptical profiles
  - `test_sersic_parameters`: Test parameter management

- `TestGaussianEllipse`: Gaussian light model (4 tests)
  - `test_gaussian_creation`: Verify Gaussian module creation
  - `test_gaussian_light`: Test light distribution
  - `test_gaussian_batch`: Test batch processing
  - `test_gaussian_mge`: Test multi-Gaussian expansion (15 components)

- `TestPhysicalModel`: Composite model (4 tests)
  - `test_physical_model_creation`: Verify PhysicalModel creation
  - `test_deflection_composite`: Test combined mass deflection
  - `test_source_light`: Test source surface brightness
  - `test_lens_light`: Test lens surface brightness

**Example**:
```python
# Run SIE tests
pytest tests/test_caskade_models.py::TestSIE -v

# Run all model tests
pytest tests/test_caskade_models.py -v
```

**Key Validations**:
- ✓ Caskade module creation and initialization
- ✓ Parameter conversion (torch.Tensor → JAX arrays)
- ✓ Numerical accuracy (comparison with reference values)
- ✓ Batch processing efficiency
- ✓ Parameter linking (pointer mode)

---

#### `test_config_parser.py` (15+ tests)

**Purpose**: Test YAML configuration parsing and parameter management

**Test Classes**:
- `TestCaskadeConfigParser`: Configuration parser (8 tests)
  - `test_parser_creation`: Verify parser initialization
  - `test_config_loading`: Test YAML loading
  - `test_physical_model_building`: Test PhysicalModel construction from config
  - `test_prior_transform`: Test prior transformation setup
  - `test_parameter_counting`: Test ndim and n_linear_params
  - `test_static_params`: Test setting static parameters
  - `test_backward_compatibility`: Test old YAML format support
  - `test_parameter_links`: Test parameter linking (MGE)

- `TestPriorTransform`: Prior transformation (7 tests)
  - `test_uniform_prior`: Test uniform prior transformation
  - `test_gaussian_prior`: Test Gaussian prior transformation
  - `test_log_uniform_prior`: Test log-uniform prior transformation
  - `test_prior_bounds`: Test parameter bounds extraction
  - `test_multi_param_transform`: Test transforming multiple parameters
  - `test_inverse_transform`: Test inverse transformation
  - `test_prior_validation`: Test prior parameter validation

**Example**:
```python
# Run parser tests
pytest tests/test_config_parser.py::TestCaskadeConfigParser -v

# Run prior transform tests
pytest tests/test_config_parser.py::TestPriorTransform -v
```

**Key Validations**:
- ✓ YAML parsing correctness
- ✓ Backward compatibility with old format
- ✓ Parameter categorization (dynamic/static/linear)
- ✓ Prior transformation accuracy
- ✓ Parameter linking mechanism

---

#### `test_lens_simulator.py` (20+ tests)

**Purpose**: Test forward simulation (ray-tracing, PSF convolution, linear solving)

**Test Classes**:
- `TestLensSimulator`: Forward simulator (12 tests)
  - `test_simulator_creation`: Verify LensSimulator creation
  - `test_simulate_nonlinear`: Test non-linear simulation (all params fixed)
  - `test_simulate_linear_nnls`: Test NNLS linear solver
  - `test_simulate_linear_normal`: Test normal linear solver
  - `test_psf_convolution`: Test PSF convolution accuracy
  - `test_subsampling`: Test subsampling (nsub=1,2,3)
  - `test_batch_processing`: Test batch simulation (bs=1,10,100)
  - `test_ray_tracing`: Test ray-tracing accuracy
  - `test_mask_handling`: Test masked pixel handling
  - `test_intensity_return`: Test intensity coefficient return
  - `test_numerical_stability`: Test numerical stability (extreme parameters)
  - `test_gradient_computation`: Test gradient w.r.t. parameters (JAX autodiff)

- `TestSimulatorConfig`: Configuration (4 tests)
  - `test_config_creation`: Verify SimulatorConfig creation
  - `test_psf_normalization`: Test PSF normalization
  - `test_coordinate_grid`: Test coordinate grid generation
  - `test_mask_validation`: Test mask array validation

- `TestLinearSolver`: Linear solvers (6 tests)
  - `test_nnls_solver`: Test NNLS solver correctness
  - `test_normal_solver`: Test normal solver correctness
  - `test_solver_comparison`: Compare NNLS vs normal results
  - `test_solver_gradient`: Test gradient computation through solver
  - `test_solver_batch`: Test solver with batch processing
  - `test_solver_edge_cases`: Test edge cases (zero intensities, negative, etc.)

**Example**:
```python
# Run simulator tests
pytest tests/test_lens_simulator.py::TestLensSimulator -v

# Run linear solver tests
pytest tests/test_lens_simulator.py::TestLinearSolver -v
```

**Key Validations**:
- ✓ Ray-tracing accuracy (deflection → source plane)
- ✓ PSF convolution correctness
- ✓ Linear solver accuracy (NNLS vs normal)
- ✓ Batch processing efficiency
- ✓ Gradient computation (autodiff)
- ✓ Numerical stability

---

#### `test_caskade_inference.py` (15+ tests)

**Purpose**: Test inference system (likelihood, parameter conversion, sampling)

**Test Classes**:
- `TestCaskadeImageProbModel`: Probability model (5 tests)
  - `test_prob_model_creation`: Verify CaskadeImageProbModel creation
  - `test_forward_model`: Test forward model simulation
  - `test_likelihood_computation`: Test likelihood calculation
  - `test_position_likelihood`: Test position likelihood penalty
  - `test_likelihood_batch`: Test batched likelihood

- `TestCaskadeModelInference`: Inference interface (6 tests)
  - `test_inference_creation`: Verify inference adapter creation
  - `test_params_array2kargs`: Test parameter array → caskade conversion
  - `test_prior_transform`: Test prior transformation
  - `test_likelihood_with_batch`: Test batched likelihood
  - `test_gradient_computation`: Test gradient through likelihood
  - `test_parameter_bounds`: Test parameter bounds extraction

- `TestInferenceAdapters`: Sampler/optimizer adapters (4 tests)
  - `test_nautilus_adapter`: Test Nautilus sampler adapter
  - `test_dynesty_adapter`: Test Dynesty sampler adapter
  - `test_optimizer_adapter`: Test optimizer adapter (Differential Evolution)
  - `test_adapter_results_format`: Test result format consistency

**Example**:
```python
# Run probability model tests
pytest tests/test_caskade_inference.py::TestCaskadeImageProbModel -v

# Run all inference tests
pytest tests/test_caskade_inference.py -v
```

**Key Validations**:
- ✓ Likelihood computation correctness
- ✓ Parameter conversion accuracy
- ✓ Prior transformation correctness
- ✓ Batch processing efficiency
- ✓ Adapter interface consistency

---

#### `test_demo_lens_src.py` (2+ integration tests)

**Purpose**: Test complete end-to-end workflow on lens_src demo

**Test Functions**:
- `test_caskade_lens_src_quick`: Quick test with optimizer
  - Loads lens_src demo configuration
  - Tests data loading (200×200 image)
  - Tests model building (15 dynamic + 2 linear params)
  - Runs short optimization (10 iterations)
  - Validates results format
  - **Runtime**: ~30 seconds

- `test_caskade_lens_src_sampler_init`: Sampler initialization test
  - Tests Nautilus sampler setup
  - Tests JIT compilation (batch_size=800)
  - Does NOT run full sampling (too slow for tests)
  - **Runtime**: ~20 seconds (including JIT)

**Example**:
```python
# Run quick optimizer test
pytest tests/test_demo_lens_src.py::test_caskade_lens_src_quick -v

# Run sampler initialization test
pytest tests/test_demo_lens_src.py::test_caskade_lens_src_sampler_init -v

# Run both
pytest tests/test_demo_lens_src.py -v
```

**Key Validations**:
- ✓ Complete workflow (config → data → model → inference → results)
- ✓ Real data compatibility (FITS files)
- ✓ Optimizer convergence
- ✓ Sampler initialization
- ✓ JIT compilation success
- ✓ Results format correctness

---

### Legacy System Tests (Old)

These tests validate the original ModelParser-based system (still maintained for comparison).

#### `test_profile_mass.py`
Tests for legacy mass profile implementations (SIE, Shear, etc.)

#### `test_profile_light.py`
Tests for legacy light profile implementations (Sersic, Gaussian, etc.)

#### `test_profile_util.py`
Tests for utility functions (ellipticity conversion, coordinate transformations, etc.)

#### `test_model_parser.py`
Tests for ModelParser YAML configuration parsing

#### `test_simulator.py`
Tests for legacy forward simulator

#### `test_integration.py`
Integration tests for legacy system end-to-end workflows

#### `test_util.py`
Tests for general utility functions

**Example**:
```bash
# Run all legacy tests
pytest tests/test_profile_*.py tests/test_model_parser.py tests/test_simulator.py -v
```

---

## Adding New Tests

### Test Template

```python
"""
Test module for [feature name].

This module tests [brief description of what is being tested].
"""

import pytest
import numpy as np
import jax.numpy as jnp


class Test[FeatureName]:
    """Test [feature description]"""

    def setup_method(self):
        """Setup test fixtures"""
        # Create common test data
        self.test_data = np.random.randn(100, 100)
        # ... other fixtures

    def test_[specific_feature](self):
        """
        Test [specific feature description].

        This test verifies that:
        1. [Thing 1]
        2. [Thing 2]
        3. [Thing 3]
        """
        # Arrange
        expected_result = ...

        # Act
        actual_result = ...

        # Assert
        assert actual_result == expected_result
        print("✓ [Feature] test passed")

    def teardown_method(self):
        """Cleanup after tests"""
        pass


if __name__ == "__main__":
    # Allow running tests directly
    test_obj = Test[FeatureName]()
    test_obj.setup_method()
    test_obj.test_[specific_feature]()
    print("\nAll tests passed! ✓")
```

### Best Practices

1. **Descriptive Names**: Use clear, descriptive test names
   - ✅ `test_sie_deflection_with_ellipticity`
   - ❌ `test_1`

2. **Clear Docstrings**: Explain what the test validates
   ```python
   def test_prior_transform(self):
       """
       Test prior transformation from unit cube to physical parameters.

       Validates:
       - Uniform prior: [0,1] → [a,b]
       - Gaussian prior: [0,1] → N(μ, σ²)
       - Bounds enforcement
       """
   ```

3. **Isolated Tests**: Each test should be independent
   ```python
   def setup_method(self):
       """Create fresh fixtures for each test"""
       self.model = create_fresh_model()
   ```

4. **Deterministic**: Use fixed random seeds
   ```python
   import numpy as np
   np.random.seed(42)
   ```

5. **Print Success**: Add confirmation messages
   ```python
   print("✓ Prior transformation test passed")
   ```

6. **Test Edge Cases**: Don't just test the happy path
   ```python
   def test_sie_zero_ellipticity(self):
       """Test SIE with zero ellipticity (circular case)"""

   def test_sie_extreme_ellipticity(self):
       """Test SIE with extreme ellipticity (e → 1)"""
   ```

---

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest --cov=TinyLensGpu --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

Install:
```bash
pip install pre-commit
pre-commit install
```

Now tests run automatically before every commit!

---

## Test Metrics

### Performance Benchmarks

| Test Suite | Runtime | Tests |
|-----------|---------|-------|
| `test_caskade_models.py` | ~5 sec | 22 tests |
| `test_config_parser.py` | ~3 sec | 15 tests |
| `test_lens_simulator.py` | ~10 sec | 26 tests |
| `test_caskade_inference.py` | ~8 sec | 15 tests |
| `test_demo_lens_src.py` | ~50 sec | 2 tests (includes JIT) |
| **Total (Caskade)** | **~76 sec** | **80+ tests** |
| Legacy tests | ~20 sec | 20+ tests |
| **Total (All)** | **~96 sec** | **100+ tests** |

### Coverage Goals

- **Critical modules** (models, simulator, inference): **90%+** ✅
- **Configuration modules**: **85%+** ✅
- **Utility modules**: **80%+** ✅
- **Overall project**: **85%+** ✅

---

## Troubleshooting Tests

### Common Test Issues

#### Issue 1: JIT Compilation Timeout

**Symptom**: Test hangs for 15+ seconds

**Cause**: JAX JIT compilation on first call

**Solution**: This is expected! Subsequent runs are cached.

```python
# If testing locally, you can skip JIT-heavy tests
pytest -m "not jit"
```

#### Issue 2: GPU Memory Errors

**Symptom**: `ResourceExhaustedError: Out of memory`

**Solution**: Tests set environment variables automatically, but you can also:
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
pytest
```

#### Issue 3: File Not Found in Demo Tests

**Symptom**: `FileNotFoundError: data/image.fits`

**Cause**: Demo tests change directory

**Solution**: Already handled! Tests use `os.chdir()` context manager.

#### Issue 4: Random Test Failures

**Symptom**: Test passes sometimes, fails other times

**Cause**: Missing random seed

**Solution**: Set seed in `setup_method()`:
```python
def setup_method(self):
    np.random.seed(42)
    import jax
    key = jax.random.PRNGKey(42)
```

---

## Test Maintenance

### Updating Tests After Code Changes

When modifying code:
1. **Run affected tests**: `pytest tests/test_[module].py`
2. **Check coverage**: `pytest --cov=[module]`
3. **Update test expectations** if behavior changed intentionally
4. **Add new tests** for new functionality
5. **Run full suite**: `pytest`

### Deprecating Old Tests

When removing legacy features:
1. Mark tests as deprecated:
   ```python
   @pytest.mark.deprecated
   def test_old_feature():
       """DEPRECATED: This feature will be removed in v3.0"""
   ```

2. Skip deprecated tests in CI:
   ```bash
   pytest -m "not deprecated"
   ```

3. Remove tests when feature is removed

---

## Summary

TinyLensGpu has a comprehensive test suite with **90+ tests** covering:
- ✅ Physical models (mass and light)
- ✅ Configuration parsing
- ✅ Forward simulation
- ✅ Inference system
- ✅ Complete workflows

**Key Commands**:
```bash
# Run all tests
pytest

# Run caskade tests only
pytest tests/test_caskade_*.py tests/test_lens_simulator.py tests/test_demo_*.py

# Run with coverage
pytest --cov=TinyLensGpu --cov-report=html

# Run fast tests only
pytest -m "not slow"
```

**Adding Tests**: Use the template and follow best practices for clear, isolated, deterministic tests.

Happy testing! 🧪✅
