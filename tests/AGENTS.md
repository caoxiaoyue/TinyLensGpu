# TEST SUITE KB

## OVERVIEW
`tests/` is a flat pytest tree with strong regression coverage plus a few standalone benchmark scripts. It is not mirrored by package structure, so file naming matters.

## STRUCTURE
```text
tests/
|- conftest.py                    # shared fixtures
|- test_mass_profile.py           # lenstronomy-backed mass checks
|- test_light_profile.py          # lenstronomy-backed light checks
|- test_integration.py            # end-to-end workflows
|- test_performance.py            # timing thresholds
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Shared fixtures | `tests/conftest.py` | image, noise, PSF, coordinate grids |
| End-to-end regression | `tests/test_integration.py` | broadest workflow coverage |

## CONVENTIONS
- Use pytest markers defined in `pytest.ini`: `unit`, `integration`, `slow`, `performance`, `boundary`.
- Activate the environment with `source ~/anaconda3/bin/activate && conda activate tinylens_gpu` before running test programs.
- When running a specific test file, run it from the directory where that file lives unless a task says otherwise.
- JAX performance tests should warm up first and call `block_until_ready()` before timing assertions.
- Lenstronomy-backed migration tests skip cleanly when the optional dependency is absent.
- GPU perf tests should skip when JAX GPU or `nvidia-smi` is unavailable.

## ANTI-PATTERNS
- Do not add performance assertions without warmup and synchronization.
- Do not forget markers on slow or performance-heavy tests; the suite relies on marker-based selection.
- Do not turn standalone benchmarks into pytest tests unless they become deterministic enough for CI.

## NOTES
- `test_integration.py` is the best executable doc for major architectural flows.
