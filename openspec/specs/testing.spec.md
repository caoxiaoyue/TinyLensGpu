# Testing Specification

## Test Organization

Tests are located in `tests/` directory with the following structure:

- `conftest.py` - Shared fixtures (sample_image_data, sample_noise_map, sample_psf_kernel, coordinate_grids)
- `test_*.py` - 17 individual test modules by feature area

## Test Categories

Registered markers in `pytest.ini` (with `--strict-markers`): `unit`, `integration`, `slow`,
`performance`, `boundary`. Note: `boundary` is registered but currently unused by any test.

| Marker | Description | Example Files |
|--------|-------------|---------------|
| `unit` | Unit tests for individual components | `test_boundary.py`, `test_linear_solver.py`, `test_pixelized_inversion.py`, `test_pixelized_operator.py`, `test_pixelized_source_model.py`, `test_pixelized_source_utils.py`, `test_point_source_model.py`, `test_regularization.py`, `test_util.py` |
| `integration` | End-to-end workflows | `test_integration.py`, `test_point_source_model.py` |
| `slow` | Time-consuming tests | `test_integration.py` |
| `performance` | Benchmark/performance tests | `test_performance.py` (class-level on `TestModelPerformance`) |
| `boundary` | Edge cases and boundary conditions | (registered, no current usage) |

### Full test module inventory

| Test file | Tests | Markers | Coverage |
|-----------|------:|---------|----------|
| `test_boundary.py` | 19 | `unit` | Zero/extreme params for mass & light profiles, simulator boundaries |
| `test_bspline_multipole.py` | 32 | `skipif` (scipy) | B-spline multipole lens-light basis |
| `test_caskade_models.py` | 19 | — | SIE, Shear, Sersic, Gaussian, ConstantBackground, PhysicalModel |
| `test_integration.py` | 26 | `integration`, `slow` | Forward simulation, multi-band, likelihood wiring |
| `test_light_profile.py` | 4 | `skipif` (lenstronomy) | Light-profile migration vs lenstronomy |
| `test_linear_solver.py` | 2 | `unit` | `fnnls_jax` numerical safeguards |
| `test_mass_profile.py` | 17 | `skipif` (lenstronomy) | Mass-profile migration vs lenstronomy |
| `test_multiband_parametric.py` | 31 | — | `MultiBandImageProbModel` validation, alignment, priors, vectorization |
| `test_performance.py` | 10 | `performance` | SIE/Sersic eval, NNLS/normal solver, scaling, JIT warmup |
| `test_pixelized_inversion.py` | 35 | `unit` | `PixelizedLensSimulator` + `PixelizedImageProbModel` TDD specs |
| `test_pixelized_operator.py` | 36 | `unit` | Matrix-free `PixelizedLensOperator`: PCG, matvec/logdet, reg, GP |
| `test_pixelized_source_model.py` | 8 | `unit` | `PixelizedSourceModel` construction, import paths, validation |
| `test_pixelized_source_utils.py` | 14 | `unit` | Source-grid & mapping utilities |
| `test_point_source_model.py` | 6 | `unit`, `integration` | `PointSourceProbModel` position likelihood, AMR solvers |
| `test_prior.py` | 41 | — | `PriorSpec.transform`, `extract_prior_specs`, `make_prior_transformation` |
| `test_regularization.py` | 19 | `unit` | `DenseRegularizationBuilder` finite-difference & GP matrices |
| `test_util.py` | 1 | `unit` | `load_lens_data` smoke test |

Total: ~320 test functions across 17 modules. Three modules (`test_caskade_models.py`,
`test_multiband_parametric.py`, `test_prior.py`) carry no custom marker, so `-m` filters do not
select/deselect them by category.

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

Defined in `tests/conftest.py`:

- `sample_image_data` - 50x50 positive random image (seed 42)
- `sample_noise_map` - 50x50 positive noise map (seed 43)
- `sample_psf_kernel` - 15x15 normalized Gaussian PSF, sigma=2.0
- `coordinate_grids` - 50x50 JAX meshgrid at 0.05 arcsec/pixel (returns `xx, yy`)

A local `benchmark_if_available` fixture is also defined inside `test_performance.py` (not shared).

## Integration Test Patterns

Integration tests verify:
- Forward simulation matches expected outputs (`simulate` vs `forward`)
- Likelihood models produce finite values
- Prior transformations are invertible
- Sampler/optimizer interfaces are compatible
- Multi-band models handle band-specific data and alignment geometry correctly
- Matrix-free pixelized operator matches dense backend (matvec, logdet)

## Optional-Dependency Gating

- `test_light_profile.py` and `test_mass_profile.py` skip without `lenstronomy` (not in core deps)
- `test_bspline_multipole.py` skips without `scipy`
- `UltraNestSampler` requires the `ultranest` extra

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
