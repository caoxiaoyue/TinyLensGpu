# TinyLensGpu Documentation

_Last updated: 2026-01-05_

This single guide consolidates the information that previously lived in multiple Markdown files. It focuses on the essentials required to install TinyLensGpu, run models (config-driven or programmatic), verify the setup, and keep the system healthy.

---

## 1. Overview

TinyLensGpu is a JAX-powered, GPU-accelerated framework for galaxy–galaxy strong-lensing analysis. All modern workflows are based on the **Caskade** module system:

| Layer | Purpose | Key Objects |
| --- | --- | --- |
| Models | Mass & light components implemented as `caskade.Module`s | `SIE`, `Shear`, `SersicEllipse`, `GaussianEllipse`, `PhysicalModel` |
| ObservationModel | Likelihood / evidence evaluation | `ImageProbModel` |
| ForwardSimulation | Ray-tracing + PSF convolution + (optional) linear intensity solving | `LensSimulator` |
| Inference | Prior/likelihood wrappers + samplers/optimizers | `make_prior_transformation`, `make_likelihood`, Nautilus/Dynesty/SciPy optimizers |

This repository snapshot focuses on the programmatic (pure-Python) workflow used in `paper/demo/**`.

---

## 2. Installation

### 2.1 Quick installation (pip)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt        # runtime deps
pip install -r requirements-dev.txt    # optional dev/test deps
pip install -e .                       # editable install
```

### 2.2 Conda + GPU (recommended for CUDA users)

```bash
conda create -n tinylens_gpu python=3.11
conda activate tinylens_gpu
conda install -c conda-forge cudatoolkit=12.0  # or match your driver
pip install -U "jax[cuda12]"
pip install -r requirements.txt
pip install -e .
```

### 2.3 Verify the environment

```python
import jax
print(jax.devices())   # Expect GPU info if CUDA is available
```

Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` in your shell when working on memory-constrained GPUs.

---

## 3. Quickstart

### 3.1 Configuration-driven workflow (YAML + runner)

The YAML runner workflow is not included in the current codebase layout. Use the programmatic API and the runnable scripts under `paper/demo/**` as the source of truth.

### 3.2 Programmatic workflow (direct module construction)

```python
import os
import jax.numpy as jnp
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE, Shear
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.ObservationModel.LensImage import ImageProbModel
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from nautilus import Sampler

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Load your FITS data (see paper/demo utilities for reference)
image_data = jnp.load("data/image.npy")
noise_map = jnp.load("data/noise.npy")
psf_kernel = jnp.load("data/psf.npy")

sie = SIE(theta_E=ParamU("theta_E", 1.5, prior_type="uniform",
                         prior_settings=[0.5, 3.0], limits=[0.0, 10.0]))
shear = Shear(gamma1=ParamU("gamma1", 0.0), gamma2=ParamU("gamma2", 0.0))
source = SersicEllipse(
    R_sersic=ParamU("R_src", 1.0, prior_type="uniform", prior_settings=[0.1, 2.0]),
    n_sersic=ParamU("n_src", 2.0, prior_type="uniform", prior_settings=[0.3, 6.0]),
    Ie=ParamU("Ie_src", 1.0)  # solved linearly when `use_linear=True`
)

sie.theta_E.to_dynamic()
source.R_sersic.to_dynamic()
source.n_sersic.to_dynamic()

phys_model = PhysicalModel(lens_mass=[sie, shear], source_light=[source])

prob_model = ImageProbModel(
    image_data=image_data,
    noise_map=noise_map,
    psf_kernel=psf_kernel,
    dpix=0.074,
    nsub=3,
    phys_model=phys_model,
    use_linear=True,
    solver_type="nnls",
)

prior, prior_specs = make_prior_transformation(prob_model)
loglike = make_likelihood(prob_model, vectorized=True)

sampler = Sampler(prior, loglike, n_dim=len(prior_specs), n_live=200, vectorized=True)
sampler.run(verbose=True, n_eff=800)
```

This path keeps everything in Python and avoids helper wrappers, matching the lightweight philosophy of the project.

---

## 4. Configuration & Parameter Management

- **Parameter modes**
  - `to_dynamic()`: sampled by samplers/optimizers
  - `to_static(value)`: fixed values
  - `use_linear: true`: solved via NNLS/normal solver during forward modeling
  - Pointer mode: link parameters in YAML by referencing another component; internally implemented with Caskade’s native `setattr` linking

- **Batching**
  - Samplers (e.g., Nautilus) support `batch_size` up to 800 for high throughput
  - Optimizers typically run with batch size 1
  - Use `make_likelihood(..., vectorized=True)` for batched likelihood evaluation

- **Common tuning knobs**
  - `nsub`: 1 (fast) → 3+ (high accuracy)
  - `solver_type`: `nnls` (physical, slightly slower), `normal` (fast, may produce negatives)
  - `position_likelihood`: optional block enforcing consistency between observed image positions

---

## 5. Testing & Quality

The repository ships with a comprehensive pytest suite.

```bash
pytest                       # run everything
pytest -m "integration"      # only integration tests
pytest tests/test_boundary.py::TestParameterBoundaries::test_sie_zero_einstein_radius
pytest --cov=TinyLensGpu --cov-report=term-missing
```

Useful markers & options:
- `-m "not slow"` to skip time-consuming cases
- `-k "pattern"` to match test names
- `-n auto` (requires `pytest-xdist`) for parallel execution

Continuous Integration example (GitHub Actions):
1. Install dependencies from `requirements-dev.txt`
2. `pytest -m unit`
3. `pytest -m "integration and not slow"`
4. `pytest --cov=TinyLensGpu --cov-report=xml`
5. Upload coverage (e.g., Codecov)

---

## 6. Migration & Compatibility

- Legacy ModelParser/Profile code has been fully removed (Dec 2025).
- To migrate old scripts:
  1. Replace `RunLensModel` with `RunCaskadeLensModel`.
  2. Keep using the same YAML files; the parser remains backward compatible.
  3. Trigger `runner.init_jit_likelihood()` once before long sampling runs to separate compilation time.
- For historical reference, see git history of the `doc/` folder (previous `MIGRATION_GUIDE.md`, `LEGACY_REMOVAL_SUMMARY.md`, etc.).

---

## 7. Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: caskade` | Missing dependency | `pip install "caskade[jax]"` |
| JAX sees CPU only | CUDA runtime mismatch | Install matching `jax[cudaXX]`, verify `nvidia-smi` |
| First likelihood call takes 10–15s | JIT compilation | Expected; keep batch size modest while debugging |
| `ResourceExhaustedError` | GPU memory | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, reduce `batch_size` |
| NaNs in likelihood | Parameters out of bounds or bad data | Check prior ranges, ensure FITS inputs have no NaN/Inf, enable masks |

---

## 8. Resources

- **README**: Project overview and citation instructions.
- **paper/demo/**: End-to-end runnable examples (`lens_only`, `lens_src`, `lens_src_mge`, etc.).
- **tests/**: Reference implementations for new components or regression reproduction.
- **GitHub Issues**: https://github.com/caoxiaoyue/TinyLensGpu/issues

Contributions should update this single guide when behavior or workflows change.

---

**Maintainer**: TinyLensGpu development team  
**Contact**: Open an issue or pull request on GitHub.
