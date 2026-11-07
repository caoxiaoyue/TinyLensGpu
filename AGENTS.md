# TinyLensGpu — Agent Guidance

GPU-accelerated gravitational lens modeling with JAX. Python 3.10+.

## Setup

```bash
pip install -r requirements-dev.txt
pip install -e .
```

Core deps: `jax[cuda12]`, `caskade[jax]`, `numpy<2.0`, `jaxnnls`. If JAX sees only CPU, check CUDA runtime compatibility with `nvidia-smi`.

## Test & Verify

Activate the environment before running tests:

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
```

From the repository root:

```bash
pytest                         # core library tests
pytest -m "not slow"           # core tests excluding slow cases
pytest -m integration          # core integration tests
pytest benchmarks              # timing and scaling benchmarks
```

B1938+666 is outside the scope of agent work, including its dedicated workflow,
modeling, replays, and tests. For example regressions, select specific test files
or pytest node IDs after checking shared modules for B1938 cases.

Place library behavior tests in `tests/`, including cross-module integration
and large-input correctness. Place example-specific assembly, exports,
plotting, and stage-flow tests in `examples/tests/`; place timing assertions
in `benchmarks/`. Run these suites in separate processes because example
imports can change JAX and plotting configuration. Default collection is
restricted to `tests/`; see `docs/guides/guide.md` for fixture and path rules.

Markers: `unit`, `integration`, `slow`, `performance`, `boundary` (registered
in `pytest.ini`). No CI workflows exist in this repo yet.

**Running specific tests:** Run a specific test from the directory where its
file lives unless a task says otherwise.

## Architecture (4 layers)

| Directory | Role | Key imports |
|---|---|---|
| `TinyLensGpu/PhysicalModel/` | Mass & light components | `SIE`, `Shear`, `SersicEllipse`, `PhysicalModel` |
| `TinyLensGpu/ForwardSimulation/` | Ray-tracing + PSF convolution | `LensSimulator`, `SimulatorConfig` |
| `TinyLensGpu/ObservationModel/` | Likelihood / evidence | `ImageProbModel`, `MultiBandImageProbModel`, `PointSourceProbModel`, `PixelizedImageProbModel`, `PixelizedImageProbModelOperator` |
| `TinyLensGpu/Inference/` | Priors, likelihood builders, posterior transfer, samplers | `ParamU`, `make_prior_transformation`, `make_likelihood`, `StagePosterior` |

The layer packages re-export common models; `TinyLensGpu/__init__.py` initializes backend and compatibility behavior. When locating a backend, public import, or regression test, read `docs/agents/codebase-map.md`.

## Canonical Workflow (Parametric Images)

1. **Load data** — `load_lens_data()` wraps FITS image/noise/PSF loading.
2. **Build components** — Instantiate `ParamU` parameters inside `SIE`, `Shear`, `SersicEllipse`, etc.
3. **Select modes** — Call `.to_dynamic()` (sampled), `.to_static(value)` (fixed), or leave as linear (solved during forward).
4. **Assemble** — `PhysicalModel(lens_mass=[...], source_light=[...], lens_light=[...])`
5. **Likelihood** — `ImageProbModel(..., use_linear=True, solver_type="nnls")`
6. **Sample** — `prior, specs = make_prior_transformation(prob_model)` → `loglike = make_likelihood(prob_model, vectorized=True)` → feed to Nautilus/Dynesty.

Runnable examples: `examples/*/run_model.py` (parametric) or `single_step_inversion.py` (pixelized/shapelet).

## JAX & Runtime Quirks

- **Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`** when working on memory-constrained GPUs (common in demos/scripts).
- **First JIT call takes 10–15s** — expected; warm up the likelihood with a representative call before timing or sampling.
- **GPU memory exhaustion** — reduce `batch_size` or `n_batch` in samplers.
- **NaNs in likelihood** — check prior bounds, ensure FITS inputs have no NaN/Inf, use masks.

## Codebase Conventions

- **Caskade modules**: All physical components inherit from `caskade.Module`. The backend is forced to `'jax'` in `TinyLensGpu/__init__.py`.
- **Jax Array bool**: JAX arrays use native truthiness; select optional arrays with explicit `None` checks (`x if x is not None else default`). Do not reintroduce a `jax.Array.__bool__` patch.
- **Linear solvers**: Parametric image models use `"nnls"` for non-negative amplitudes or `"normal"` for unconstrained amplitudes. Pixelized models use different solver choices; see `docs/agents/codebase-map.md` before selecting one.
- **nsub**: 1 (fast) → 3+ (high accuracy). Sub-pixel integration factor.
- **No formatting configs**: `black`, `flake8`, `isort`, `ruff` are in dev extras but no `.flake8`, `pyproject.toml`, or `ruff.toml` exists. Use defaults.

## References

- `docs/guides/guide.md` — authoritative installation, troubleshooting, and migration guide.
- `README.md` — capabilities matrix, demo index, and citation.
- `setup.py` — dependency matrix and extras (`dev`, `docs`, `notebooks`, `ultranest`).

## Specs and experiment records

For spec-based reviews, use an explicitly designated spec or an issue-tracker
`spec.md`/issue as the requirements source. `docs/technical-notes/` and research
notes under `.scratch/` record experiments, hypotheses, and recommendations;
use them as evidence and context unless the user explicitly designates one as a
spec. If no spec is identified, report that the spec axis is unavailable.
Attribute numerical findings to the model and configuration that produced them
before using them to assess a changed model.

## Agent skills

### Issue tracker

Before creating, claiming, or closing specs and tickets, read `docs/agents/issue-tracker.md`. Work lives in `.scratch/<feature>/`; `.scratch/README.md` indexes existing material.

### Triage labels

When triaging incoming reports, read `docs/agents/triage-labels.md` for the five canonical roles. Implementation progress follows the separate lifecycle in `docs/agents/issue-tracker.md`.

### Domain docs

Before exploring domain behavior or changing terminology, read `docs/agents/domain.md` for the single-context glossary and relevant ADRs.
