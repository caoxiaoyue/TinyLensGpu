# Testing Specification

## Test Organization

Tests are located in `tests/` directory with the following structure:

- `conftest.py` - Shared fixtures (sample_image_data, sample_noise_map, sample_psf_kernel, coordinate_grids)
- `test_*.py` - Individual test modules by feature area

## Test Categories

| Marker | Description | Example Files |
|--------|-------------|---------------|
| `unit` | Unit tests for individual components | `test_mass_profile.py`, `test_light_profile.py`, `test_linear_solver.py` |
| `integration` | End-to-end workflows | `test_integration.py`, `test_caskade_models.py` |
| `slow` | Time-consuming tests | `test_performance.py`, `test_pixelized_inversion.py` |
| `performance` | Benchmark/performance tests | `test_performance.py` |
| `boundary` | Edge cases and boundary conditions | `test_boundary.py` |

## Running Tests

```bash
pytest                          # all tests
pytest -m "not slow"           # exclude slow tests
pytest -m integration          # only integration tests
pytest -m unit                 # only unit tests
pytest -k "pattern"            # filter by name pattern
pytest -n auto                 # parallel execution (requires pytest-xdist)
```

## Test Fixtures

Defined in `conftest.py`:

- `sample_image_data` - 50x50 random positive image
- `sample_noise_map` - 50x50 positive noise map
- `sample_psf_kernel` - 15x15 Gaussian PSF (normalized)
- `coordinate_grids` - JAX coordinate grids for 50x50 image at 0.05 arcsec/pixel

## Integration Test Patterns

Integration tests verify:
- Forward simulation matches expected outputs
- Likelihood models produce finite values
- Prior transformations are invertible
- Sampler interfaces are compatible
- Multi-band models handle band-specific data correctly

## Testing Guidelines

- Use `pytest.mark` decorators to categorize tests
- Keep unit tests fast (< 1s each)
- Integration tests should verify end-to-end workflows
- Performance tests should include baseline assertions
- Use fixtures for common setup patterns
- Prefer explicit JAX array creation over numpy where possible

## GPU Testing

- Tests should run on CPU by default
- GPU-specific tests should be marked `slow` or `performance`
- Use `jax.devices()` to check GPU availability
- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` in CI environment
