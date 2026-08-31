# TinyLensGpu User Guide

This guide covers installation, the programmatic modeling workflow, testing, and troubleshooting. Runnable scripts under `examples/` are the source of truth for complete models.

---

## 1. Overview

TinyLensGpu is a JAX-powered, GPU-accelerated framework for galaxy–galaxy strong-lensing analysis. All modern workflows are based on the **Caskade** module system:

| Layer | Purpose | Key Objects |
| --- | --- | --- |
| Models | Mass & light components implemented as `caskade.Module`s | `SIE`, `Shear`, `SersicEllipse`, `GaussianEllipse`, `PhysicalModel` |
| ObservationModel | Likelihood / evidence evaluation | `ImageProbModel` |
| ForwardSimulation | Ray-tracing + PSF convolution + (optional) linear intensity solving | `LensSimulator` |
| Inference | Prior/likelihood wrappers + samplers/optimizers | `make_prior_transformation`, `make_likelihood`, Nautilus/Dynesty/SciPy optimizers |

This repository snapshot focuses on the programmatic (pure-Python) workflow used in `examples/**`.

---

## 2. Installation

### 2.1 Quick installation (pip)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt        # runtime deps, including jaxnnls for NNLS solving
pip install -r requirements-dev.txt    # optional dev/test deps
pip install -e .                       # editable install
```

### 2.2 Conda + GPU (recommended for CUDA users)

```bash
conda create -n tinylens_gpu python=3.11
conda activate tinylens_gpu
pip install -U "jax[cuda12]"
pip install -r requirements.txt        # includes jaxnnls for the linear NNLS backend
pip install -e .
```

Notes:
- `cudatoolkit` from `conda-forge` is no longer required: the `jax[cuda12]` wheels bundle their own CUDA runtime.
- `requirements.txt` pins `numpy<2.0`, so pip will resolve a JAX version compatible with NumPy 1.x (e.g. `jax==0.7.1`).
- If `pip install -e .` fails inside a build-isolation step with SSL/proxy errors, use the already-installed build tools:
  ```bash
  pip install -e . --no-build-isolation
  ```

### 2.3 Verify the environment

```python
import jax
print(jax.devices())   # Expect GPU info if CUDA is available
```

Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` in your shell when working on memory-constrained GPUs.

---

## 3. Quickstart

TinyLensGpu models are constructed directly with the programmatic Python API.

```python
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE, Shear, SersicEllipse
from TinyLensGpu.ObservationModel import ImageProbModel
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.utils import load_lens_data
from nautilus import Sampler

image_data, noise_map, psf_kernel, mask = load_lens_data(
    image_path="data/image.fits",
    noise_path="data/noise.fits",
    psf_path="data/psf.fits",
)

sie = SIE(theta_E=ParamU("theta_E", 1.5, prior_type="uniform",
                         prior_settings=[0.5, 3.0], limits=[0.0, 10.0]))
shear = Shear(gamma1=ParamU("gamma1", 0.0), gamma2=ParamU("gamma2", 0.0))
source = SersicEllipse(
    R_sersic=ParamU("R_src", 1.0, prior_type="uniform",
                    prior_settings=[0.1, 2.0], limits=[0.1, 2.0]),
    n_sersic=ParamU("n_src", 2.0, prior_type="uniform",
                    prior_settings=[0.3, 6.0], limits=[0.3, 6.0]),
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
    mask=mask,
    solver_type="nnls",
)

prior, prior_specs = make_prior_transformation(prob_model)
loglike = make_likelihood(prob_model, vectorized=True)

sampler = Sampler(prior, loglike, n_dim=len(prior_specs), n_live=200, vectorized=True)
sampler.run(verbose=True, n_eff=800)
```

Run the complete example from its own directory so its relative data paths resolve correctly; see `examples/lens_src/run_model.py`.

---

## 4. Configuration & Parameter Management

- **Parameter modes**
  - `to_dynamic()`: sampled by samplers/optimizers
  - `to_static(value)`: fixed values
  - `use_linear: true`: solved via NNLS/normal solver during forward modeling
  - Shared parameters: reuse or link Caskade parameters explicitly in Python

- **Robust mixture prior**

  Use a bounded truncated-Gaussian core plus a uniform escape component when an
  informative estimate should guide, but not exclude, a wider parameter range:

  ```python
  theta_e = ParamU(
      "theta_E",
      1.35,
      prior_type="truncated_gaussian_uniform_mixture",
      prior_settings=[1.35, 0.405, 0.8],
      limits=[0.5, 3.0],
  )
  ```

  The settings are `[core_mean, core_std, core_weight]`; the remaining weight
  belongs to a uniform component on `limits`. Both components share that finite
  support. This is the inference prior itself, so it changes the posterior and
  evidence measure rather than acting only as a sampler proposal.

- **Batching**
  - Choose the sampler's `n_batch` or equivalent batch setting according to available GPU memory
  - Use `make_likelihood(..., vectorized=True)` for batched likelihood evaluation

- **Common tuning knobs**
  - `nsub`: 1 (fast) → 3+ (high accuracy)
  - `solver_type`: `nnls` (physical, slightly slower), `normal` (fast, may produce negatives)
  - `position_likelihood`: optional block enforcing consistency between observed image positions

- **Operator pixel-source solves**
  - `pcg`: fast unconstrained solve; linear intensities may be negative
  - `pnpg`: preferred non-negative solve for ill-conditioned joint pixel-source
    and MGE lens-light systems; uses matrix-free diagonal equilibration,
    backtracking, and a componentwise KKT convergence gate
  - `fista`: retained for compatibility with existing scalar-step workflows
  - Tune `pnpg_max_iter`, `pnpg_rtol`, and `pnpg_power_iter` on representative
    likelihood points before sampling; failed convergence is gated rather than
    accepted as a finite likelihood

---

## 5. Testing & Quality

The repository ships with a comprehensive pytest suite.

```bash
pytest                       # run everything
pytest -m "integration"      # only integration tests
pytest --cov=TinyLensGpu --cov-report=term-missing

# Run a specific test from the directory containing its file
cd tests
pytest test_boundary.py::TestParameterBoundaries::test_sie_zero_einstein_radius
```

Useful markers & options:
- `-m "not slow"` to skip time-consuming cases
- `-k "pattern"` to match test names
- `-n auto` (requires `pytest-xdist`) for parallel execution

---

## 6. Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: caskade` | Missing dependency | `pip install "caskade[jax]"` |
| JAX sees CPU only | CUDA runtime mismatch | Install matching `jax[cudaXX]`, verify `nvidia-smi` |
| First likelihood call takes 10–15s | JIT compilation | Expected; keep batch size modest while debugging |
| `ResourceExhaustedError` | GPU memory | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, reduce `batch_size` |
| NaNs in likelihood | Parameters out of bounds or bad data | Check prior ranges, ensure FITS inputs have no NaN/Inf, enable masks |

---

## 7. Resources

- **README**: Project overview and citation instructions.
- **examples/**: End-to-end runnable examples (`lens_only`, `lens_src`, `lens_src_mge`, etc.).
- **tests/**: Reference implementations for new components or regression reproduction.
- **Point-source guide**: `docs/guides/point-source-model.md`

Contributions should update the relevant guide when behavior or workflows change.

---

**Maintainer**: TinyLensGpu development team  
**Contact**: Open an issue or pull request on GitHub.
