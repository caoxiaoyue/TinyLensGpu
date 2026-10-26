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
  - `pcg` and `pnpg` are the only supported operator solvers. The former
    `fista` option and all `fista_*` constructor arguments have been removed.
    Migrate non-negative workflows to `solver_type="pnpg"` and the
    `pnpg_max_iter`, `pnpg_rtol`, `pnpg_power_iter` controls; retune convergence
    instead of copying the old tolerance, because the metrics differ.
  - Tune `pnpg_max_iter`, `pnpg_rtol`, and `pnpg_power_iter` on representative
    likelihood points before sampling; failed convergence is gated rather than
    accepted as a finite likelihood
  - Configure parameter modes before constructing `PixelizedImageProbModelOperator`.
    It precomputes fixed observation terms; with `fixed_source_bbox`, it also
    caches fixed regularization blocks/log determinants and, when lens mass is
    fixed, mapping data, the source right-hand side and data curvature blocks.
    Registered pointer dependencies are included in the fixedness check.
    Rebuild the model **and its compiled likelihood** after changing fixed physical
    parameters, parameter modes, observation data or grid settings. Dynamic
    parameter values and solver controls are not part of the data cache.
  - `PixelizedImageProbModel` accepts the same square `fixed_source_bbox`.
    Without it, every evaluation still infers the source bbox from seed rays.
    With it — provided `adaptive_reg_rho` is fixed — the model stores the
    finite-difference regularization matrix and
    its `logdet(R)`, and the evidence reads that stored determinant.
    `log_lambda_reg` stays outside the cache and is applied as `lambda * R` at
    each evaluation. When that bbox,
    every lens-mass component, and every unit-amplitude lens-light basis are
    also fixed, it stores the joint design matrix, the noise-weighted data Gram
    matrix, and the data right-hand side. Each evaluation then adds
    `lambda * R` and the weak lens-light regularization and factorizes again.
    Lens-light intensities remain solved amplitudes. Empty mass and lens-light
    lists count as fixed. A dynamic mass or lens-light basis skips the design
    cache. A pointer is fixed only when it has
    children and those dependencies are fixed. Set parameter modes before
    construction, and rebuild the evidence model **and its compiled likelihood**
    after changing a fixed value, a parameter mode, the observations, or the bbox.

  - Adaptive source regularization uses the same inputs on both backends:
    pass the stage-m0 reconstruction as `fixed_reg_template` (flat `(n*n,)`
    or `(n,n)`) together with `fixed_source_bbox`, and keep the strength on
    `PixelizedSourceModel.adaptive_reg_rho`. A fixed rho of approximately zero
    disables adaptation and needs neither input. A positive fixed rho or a
    rho with dynamic or unresolved dependencies — even one initialized at
    zero — requires both
    inputs at construction. (A pointer counts as fixed when its registered
    dependencies are fixed.) The normalized template is prepared once; a fixed
    rho additionally caches the weighted regularizer and its log determinant,
    while a dynamic rho recomputes both on every evaluation. The design and
    mapping caches are independent of rho. The dense backend keeps its exact
    `slogdet` determinant while the operator backend keeps its block-diagonal
    approximation, so their log evidences are not expected to be identical.
  - `fixed_reg_scale` was removed: migrate
    `fixed_reg_scale=source_template_scale_map(S0, n, rho)` to
    `fixed_reg_template=S0` with the same `rho` stored on
    `PixelizedSourceModel`. The old keyword now raises `TypeError`, and
    arbitrary external weight maps are no longer an observation-model input.

  - Both the dense and operator backends support the three finite-difference
    regularization types (`"zero-order"`, `"first-order"`, `"second-order"`).
    A standalone Gaussian-process regularization utility remains available for
    direct use outside the pixelized-source model backends:

    ```python
    from TinyLensGpu.utils.inversion.gp_regularization import GaussianProcessRegularization
    regularizer = GaussianProcessRegularization(n=5, regularization_type='matern32')
    precision, logdet_precision = regularizer.matrix(-1.0,1.0,-1.0,1.0,kernel_scale=0.3)
    ```

    This utility is independent of `PixelizedSourceModel`; it is not a GP model
    workflow.
  - Both backends support `source_bbox_padding` and
    `source_bbox_outlier_frac` for inferred source-plane bboxes; the settings
    are ignored when `fixed_source_bbox` is supplied.
  - Use `result = prob_model.evaluate(theta)` on either backend to obtain
    `log_evidence`, `model_image`, `source_pixels`, `lens_amplitudes`,
    `source_bbox` and `solver_info` from one inversion (`InversionResult`).
    It includes any position penalty. On the operator backend,
    `solver_info` carries the PCG/PNPG convergence diagnostics and failed
    solves return the existing evidence penalty with zero reconstruction
    arrays; on the dense backend `solver_info` is `None` and the existing
    nonfinite-evidence penalty behavior is unchanged (reconstructions are
    returned as computed). `prob_model(theta)` and
    `forward_model(theta, return_source=True)` remain compatible but are
    independent evaluations — reuse `result` if you need several fields from
    the same solve. Reconstruction alone does not compute evidence
    determinants. All three interfaces support JIT/vmap.
  - `reconstruct_component_images(source_pixels, lens_amplitudes)` maps
    solved amplitudes back to masked 2D source and lens-light images plus the
    source bbox on either backend, without solving or evaluating evidence.
    It uses the parameter values current at call time — fill dynamic
    parameters first to match a chosen proposal:

    ```python
    import caskade as ck

    with ck.ActiveContext(prob_model):
        prob_model.fill_params(theta)
        result = prob_model.evaluate()
        source_img, lens_img, bbox = prob_model.reconstruct_component_images(
            result.source_pixels, result.lens_amplitudes,
        )
    ```

    Masked pixels (`mask=True`) are zero. Source-only models accept `None` or
    an empty vector for `lens_amplitudes` and return a zero lens image; joint
    models raise `ValueError` if `lens_amplitudes` is `None`.
  - Choosing the entry point: call `prob_model(theta)` when a sampler needs
    only the scalar log evidence, `forward_model(theta, ...)` when only
    reconstructions are needed, `evaluate(theta)` when evidence and
    reconstructions should come from one solve, and
    `evaluate_with_diagnostics(theta, ...)` when plots, exports, or stage
    consumers need the reconstruction plus host-side diagnostics:

    ```python
    result, diagnostics = prob_model.evaluate_with_diagnostics(
        theta, include_effective_dof=True, include_components=True,
    )
    if not diagnostics.valid:
        from TinyLensGpu.ObservationModel.LensImage.inversion_result import InvalidReconstructionError
        raise InvalidReconstructionError(diagnostics.invalid_reason)
    # Use result.model_image/source_pixels/source_bbox and diagnostics.source_image/
    # lens_light_image; no second reconstruction is needed.
    ```

    The single-parameter host-side diagnostic entry point returns the same
    `InversionResult` plus an `InversionDiagnostics` from one reconstruction.
    It is not the JIT/vmap interface; `evaluate`, `forward_model`, and the
    likelihood are unchanged. `include_effective_dof` and
    `include_components` default to ``False`` and skip the optional
    statistics/component work; unrequested fields stay `None`. Basic
    diagnostics always include `normalized_residuals`, `chi2`, `lambda_reg`,
    `adaptive_reg_rho`, `regularization_scale`, and an enabled position
    record. The dense `effective_dof_method` is `"dense_curvature"` and keeps
    the existing lens-light-regularized curvature convention (NNLS gets no
    new constrained-DOF interpretation); the operator source-only method is
    `"operator_source_block_preconditioner"`, the existing approximation.
    Requested operator *joint* effective/residual/reduced statistics stay
    `None` with `effective_dof_method="operator_joint_unsupported"` and an
    explanatory `effective_dof_reason`; a valid reconstruction remains
    usable and raw `chi2`/component images are still returned. `valid` is
    `False` only on explicit solver non-convergence or nonfinite
    reconstruction arrays (dense `solver_info` stays `None`). Raising
    `InvalidReconstructionError` stops the current stage before formal
    output while previously saved artifacts are retained; unrelated plot
    errors keep their legacy handling. Application plots and exports should
    use this public diagnostic endpoint; internal profiling scripts remain
    legitimate and existing public APIs are retained.
  - With a **fixed source bbox**, the operator backend's `pcg` and `pnpg`
    solvers also support first-order
    reverse-mode gradients of source/lens-light reconstructions and evidence.
    Pass the parameter vector through the model's public interface:

    ```python
    theta = prob_model.get_values()  # order: prob_model.get_dynamic_params()
    value, gradient = jax.jit(jax.value_and_grad(prob_model))(theta)
    source_jacobian = jax.jit(jax.jacrev(
        lambda p: prob_model.forward_model(p, return_source=True)[1]
    ))(theta)
    ```

    Use this array-returning model interface for autodiff; the scalar
    `make_likelihood` convenience wrapper converts its result to a Python float.
    The solver gradients use implicit differentiation, not differentiation of
    the finite iteration sequence. PNPG requires a locally stable active set
    with strict complementarity. Forward/adjoint non-convergence or an invalid
    active set produces **NaN gradients**, while forward failure retains the
    existing evidence penalty and zero reconstruction. Check `solver_info`
    before accepting a gradient; the PNPG wrapper also reports
    `kkt_violation`, the largest componentwise KKT residual normalized by
    `1 + abs(b_i) + abs((Ax)_i)`, and the unnormalized `min_active_residual`
    (infinity when every variable is free). KKT tolerances are checked per
    component so a bright coefficient cannot hide another component's error.
    Its complementarity check uses component scales
    and the free-variable stationarity error. Stop or adjust solver settings
    when gradients are non-finite. A successful forward convergence flag alone
    does not guarantee an accurate derivative: PNPG also checks per-component
    KKT residuals and strict complementarity, and may need a tighter tolerance
    and larger iteration budget for gradients than for forward-only sampling.
    The derivative does not pass through PNPG backtracking, the PCG warm start,
    or the numerical preconditioner. Only first-order reverse-mode derivatives
    with a fixed source bbox are supported; higher-order derivatives remain
    outside this gradient interface.
    `pnpg_comp_rel_tol` controls the relative strict-complementarity margin
    (default `1e-4`) independently of forward convergence. Bright redundant
    MGE bases can require a smaller margin after converging in float64; retain
    the positive dual/KKT checks and verify finite-difference gradients when
    changing it. See the [20-MGE gradient validation](../validation/pnpg-mge-fixed-bbox-gradients.md)
    for the precision, iteration budgets, and local optimization checks.
    That model passes reference-point gradient checks, but its full tested
    optimizer rejected a gradient at the configured complementarity safety
    margin after 22 accepted steps; stable optimization convergence is not
    established for the large example.

---

## 5. Testing & Quality

From the repository root, activate the environment and select a suite:

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
pytest                         # core library tests (default collection)
pytest -m "integration"        # core integration tests
pytest examples/tests          # example and pipeline regressions
pytest benchmarks              # timing and scaling benchmarks
pytest --cov=TinyLensGpu --cov-report=term-missing
```

Run the suites in separate processes: some example modules change JAX
precision, working directories, or plotting configuration when imported.
Use `XLA_PYTHON_CLIENT_PREALLOCATE=false` on memory-constrained GPUs and
`MPLBACKEND=Agg` for headless plotting tests. Target `examples/tests`
explicitly; runnable examples elsewhere may execute work at import time.

| Location | Responsibility |
| --- | --- |
| `tests/` | Library behavior, numerical regressions, cross-module integration, and large-input correctness |
| `examples/tests/` | Example assembly, source-package exports, plotting, tools, and pipeline stage flow |
| `benchmarks/` | Timing thresholds and performance scaling |
| Example or experiment directory | Manual scientific comparisons, parameter scans, and full sampling runs |

Place a test according to the behavior it guarantees. Using a real model,
several library modules, or a large input does not by itself make a test an
example test or a benchmark.

Core tests must run without importing or reading `examples/`. Keep shared
fixtures in each suite's own `conftest.py` and file-specific fixtures beside
their tests. Suites must not import each other's test helpers. The
`examples_dir` fixture resolves example paths independently of the working
directory. Example tests should restore import-time working-directory
changes, write outputs to `tmp_path`, and use small generated inputs for
automated runs. Keep shared example fixtures free of global JAX imports so
standalone tool tests can run with their own dependencies. Source checks
scan runnable examples and Markdown Python blocks, excluding `examples/tests`.

Run a specific test from the directory containing its file:

```bash
cd tests
pytest test_boundary.py::TestParameterBoundaries::test_sie_zero_einstein_radius
```

Useful markers & options:

- `-m "not slow"` excludes slow cases within the selected suite.
- `-k "pattern"` matches test names.
- `-n auto` (requires `pytest-xdist`) enables parallel execution; benchmark
  timings should be measured without concurrent test workers.

Markers describe test characteristics; paths select suites. The root
`pytest.ini` is shared by all suites and keeps `testpaths = tests`.
The shared `pythonpath = .` setting makes each suite import the current checkout.

---

## 6. Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: caskade` | Missing dependency | `pip install "caskade[jax]"` |
| JAX sees CPU only | CUDA runtime mismatch | Install matching `jax[cudaXX]`, verify `nvidia-smi` |
| First likelihood call takes 10–15s | JIT compilation | Expected; keep batch size modest while debugging |
| `ResourceExhaustedError` | GPU memory | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, reduce `batch_size` |
| NaNs in likelihood | Parameters out of bounds or bad data | Check prior ranges, ensure FITS inputs have no NaN/Inf, enable masks |
| `ValueError` on `array or default` | Ambiguous JAX array truthiness | Use an explicit `None` check: `array if array is not None else default` |

### Native JAX array truthiness

Earlier releases patched `jax.Array.__bool__` at import so `optional_array or
default` worked. The patch is removed: importing TinyLensGpu no longer changes
JAX boolean behavior for other code in the process. Select optional arrays
with explicit `None` checks (`x if x is not None else default`). Multi-element
arrays still raise `ValueError` on `bool(...)`; scalar arrays keep native
truthiness. There is no switch that reinstalls the patch.

---

## 7. Resources

- **README**: Project overview and citation instructions.
- **examples/**: End-to-end runnable examples (`lens_only`, `lens_src`, `lens_src_mge`, etc.).
- **tests/**: Core library behavior and regression tests.
- **examples/tests/**: Automated example and pipeline regressions.
- **benchmarks/**: Explicit timing and scaling benchmarks.
- **Point-source guide**: `docs/guides/point-source-model.md`

Contributions should update the relevant guide when behavior or workflows change.

---

**Maintainer**: TinyLensGpu development team  
**Contact**: Open an issue or pull request on GitHub.
