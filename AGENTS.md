# TinyLensGpu — Agent Guidance

GPU-accelerated gravitational lens modeling with JAX. Python 3.10+.

## Setup

```bash
pip install -r requirements-dev.txt
pip install -e .
```

Core deps: `jax[cuda12]`, `caskade[jax]`, `numpy<2.0`, `jaxnnls`. If JAX sees only CPU, check CUDA runtime compatibility with `nvidia-smi`.

## Test & Verify

```bash
pytest                          # all tests (90+)
pytest -m "not slow"           # fast subset
pytest -m integration          # end-to-end only
pytest -k "pattern"            # specific test
```

Test markers: `unit`, `integration`, `slow`, `performance`, `boundary`. Defined in `pytest.ini`. No CI workflows exist in this repo yet.

**Environment:** Activate the environment with `source ~/anaconda3/bin/activate && conda activate tinylens_gpu` before running test programs.

**Running specific tests:** When running a specific test, run it from the directory where that file lives unless a task says otherwise.

## Architecture (4 layers)

| Directory | Role | Key imports |
|---|---|---|
| `TinyLensGpu/PhysicalModel/` | Mass & light components | `SIE`, `Shear`, `SersicEllipse`, `PhysicalModel` |
| `TinyLensGpu/ForwardSimulation/` | Ray-tracing + PSF convolution | `LensSimulator`, `SimulatorConfig` |
| `TinyLensGpu/ObservationModel/` | Likelihood / evidence | `ImageProbModel`, `MultiBandImageProbModel`, `PointSourceProbModel` |
| `TinyLensGpu/Inference/` | Priors, likelihood builders, samplers | `ParamU`, `make_prior_transformation`, `make_likelihood` |

Both shallow (`from TinyLensGpu.PhysicalModel import SIE`) and deep (`from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE`) imports work. The package `__init__.py` re-exports common symbols.

## Canonical Workflow

1. **Load data** — `load_lens_data()` wraps FITS image/noise/PSF loading.
2. **Build components** — Instantiate `ParamU` parameters inside `SIE`, `Shear`, `SersicEllipse`, etc.
3. **Select modes** — Call `.to_dynamic()` (sampled), `.to_static(value)` (fixed), or leave as linear (solved during forward).
4. **Assemble** — `PhysicalModel(lens_mass=[...], source_light=[...], lens_light=[...])`
5. **Likelihood** — `ImageProbModel(..., use_linear=True, solver_type="nnls")`
6. **Sample** — `prior, specs = make_prior_transformation(prob_model)` → `loglike = make_likelihood(prob_model, vectorized=True)` → feed to Nautilus/Dynesty.

Runnable examples: `examples/*/run_model.py` (parametric) or `single_step_inversion.py` (pixelized/shapelet).

## JAX & Runtime Quirks

- **Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`** when working on memory-constrained GPUs (common in demos/scripts).
- **First JIT call takes 10–15s** — expected; `init_jit_likelihood()` can separate compilation from sampling.
- **GPU memory exhaustion** — reduce `batch_size` or `n_batch` in samplers.
- **NaNs in likelihood** — check prior bounds, ensure FITS inputs have no NaN/Inf, use masks.

## Codebase Conventions

- **Caskade modules**: All physical components inherit from `caskade.Module`. The backend is forced to `'jax'` in `TinyLensGpu/__init__.py`.
- **Jax Array bool**: `TinyLensGpu/__init__.py` monkey-patches `jax.Array.__bool__` for legacy compatibility with `array or default` patterns. Prefer explicit `None` checks in new code.
- **Linear solvers**: `solver_type="nnls"` is physical (non-negative) but slightly slower; `"normal"` is faster but may produce negative fluxes.
- **nsub**: 1 (fast) → 3+ (high accuracy). Sub-pixel integration factor.
- **No formatting configs**: `black`, `flake8`, `isort`, `ruff` are in dev extras but no `.flake8`, `pyproject.toml`, or `ruff.toml` exists. Use defaults.

## OpenSpec Workflow

This repo uses OpenSpec for structured changes. Skills are in `.opencode/skills/` and `.claude/skills/`. Use `openspec-new-change`, `openspec-apply-change`, `openspec-verify-change`, etc. for planned features. See `openspec/config.yaml`.

## References

- `doc/GUIDE.md` — authoritative installation, troubleshooting, and migration guide.
- `README.md` — capabilities matrix, demo index, and citation.
- `setup.py` — dependency matrix and extras (`dev`, `docs`, `notebooks`, `ultranest`).
