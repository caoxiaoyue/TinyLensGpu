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
  - Use `make_likelihood(..., vectorized=True)` to accept the sampler's
    full proposal batch. Overlap evaluates one proposal at a time by default
    to bound GPU workspace. Set `vectorized_chunk_size` to choose how many
    proposals run together. The sampler's `n_batch` and the returned score
    array keep their original size, including when the last chunk is smaller.
    [Matched GPU timings and memory](../validation/adaptive-image-overlap-gpu.md)

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
    likelihood points before sampling. Failed convergence returns a finite
    evidence penalty.
  - Both pixelized models accept `adapt_image`, a finite PSF-convolved source
    arc on the native observation grid with the same 2D shape as `image_data`.
    Pass the producer's square `fixed_source_bbox` with the reference and set
    `adaptive_reg_rho` on `PixelizedSourceModel`. Fixed rho below `1e-10`
    uses uniform regularization and permits an inferred grid. A positive fixed
    rho at or above the cutoff or a rho with dynamic or unresolved dependencies
    requires both inputs, including a sampled rho initialized at zero.
  - The observation mask selects pixels after PSF convolution. Both backends
    ray-trace any native pixel in the cutout whose light can reach the fitted
    pixels through a nonzero PSF tap. The cutout is zero-extended and the
    kernel uses the same crop as `fftconvolve(..., mode="same")`. Neither the
    RMS map nor the source seed and adaptive projection regions expand. Use
    fresh pixelized stage outputs when migrating from the old mask-before-PSF
    convention; regenerate the reference arc and calibrate regularization.
    The simulators expose this shared geometry through `psf_response`:
    `fit_stencil`, `expand_fit_values`, `integrate_fit_rays`, and
    `convolved_columns` keep fitted-ray selection and column assembly consistent
    across dense and operator paths.
    Prepared operator data use `n_render` and `render_parents` for the PSF-expanded
    rendering domain; `n_active` continues to count fitted native pixels.
    The unused `TinyLensGpu.utils.lensing.psf` module and its public helpers
    `apply_psf_to_mapping_matrix`, `build_psf_matrix_dense`, and
    `build_psf_matrix_sparse` have been removed. Their shared input/output
    unmasked domain cannot represent the render-to-fit response. The simulator's
    `psf_response.forward` maps native render brightness to fitted pixels;
    `adjoint` maps fitted values back to render pixels. Use `convolved_columns`
    for render-subpixel source stencils; it returns `(n_columns, n_fit)`.
  - The models clip negative reference brightness to zero and normalize by
    its 99.5th percentile over active native pixels, with a denominator floor
    of `1e-10` and normalized values capped at one. For each proposal, active
    subpixel rays project their native parent brightness to the fixed source
    nodes. Both signal and coverage include the subpixel averaging factor.
    Bilinear projection divides signal by coverage above `1e-8` and assigns
    zero brightness at or below that cutoff. Hermite squares each ray's stencil
    before accumulating and divides by coverage plus the internal offset
    `1e-4`. Image prediction and data curvature use the signed Hermite
    weights. The projected brightness `u` weights the finite-difference
    prior with node scale `exp(rho * (1 - u))`. Each evaluation applies
    `exp(log_lambda_reg)` as a separate strength.
  - Set parameter modes before constructing the model. Fixed mass and grid
    allow both backends to reuse the mapping and projected brightness. They
    also reuse weighted prior terms when rho is fixed. With moving mass, the
    mapping and adaptive prior follow the proposal even when rho is fixed.
    The dense joint design and its data-only normal equations also depend on
    the unit-amplitude lens-light bases. The fixedness check includes
    registered pointer dependencies. Rebuild the model and compiled
    likelihood after changing a fixed value, parameter mode, observation,
    reference or grid setting.
  - Operator fits default to `evidence_logdet="overlap"`. Overlap combines
    exact principal-window determinant ratios with signed corrections for
    their intersections. The evidence tile width `evidence_block_size`
    defaults to 10 and is independent of the solver's `block_size`. Joint
    Overlap also uses the physical source–lens-light cross curvature and
    full source curvature for its lens-light Schur term. Explicit
    `evidence_logdet="block"` uses the historical preconditioner-based
    score. Both choices share the same reconstruction solve and use
    unconstrained curvature for the determinants, including with PNPG.
    Dense fits use complete determinants. [Matched mock score comparison](../validation/adaptive-image-overlap-euclid-mock.md)
  - For a tile width of `b`, each Overlap window spans at most `2*b`
    nodes per axis. A window of `k` nodes uses `k × k` local Gram, prior
    and posterior matrices for its determinant calculation. Fixed geometry
    can cache the individual window Grams. When a single window spans the
    source grid, the exact window matrices are full source size and the
    Gram is assembled by applying the normal operator to small batches of
    source basis vectors. The image workspace uses the column batch size
    in that path. The GPU resource measurements used tile width 10 on an
    80×80 source grid. [Workspace audit](../validation/adaptive-image-overlap-gpu.md)
  - The retired `fixed_reg_template` and `fixed_reg_scale` observation-model
    keywords raise `TypeError`. Generate a native reference arc from the
    producer reconstruction and use `adapt_image` with its fitted grid to
    migrate an adaptive fit.

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
    adapt_image = diagnostics.source_image  # native PSF-convolved source component
    fixed_source_bbox = result.source_bbox
    ```

    The single-parameter host-side diagnostic entry point returns the same
    `InversionResult` plus an `InversionDiagnostics` from one reconstruction.
    It is not the JIT/vmap interface; `evaluate`, `forward_model`, and the
    likelihood are unchanged. `include_effective_dof` and
    `include_components` default to ``False`` and skip the optional
    statistics/component work; unrequested fields stay `None`. Basic
    diagnostics always include `normalized_residuals`, `chi2`, `lambda_reg`,
    `adaptive_reg_rho`, the applied `regularization_scale`, and an enabled
    position record. Both backends report `evidence_logdet`,
    `evidence_valid` and `evidence_invalid_reason` separately from
    reconstruction validity. An auxiliary evidence failure retains a valid
    primary reconstruction and returns the finite sampling penalty. The dense
    `effective_dof_method` is `"dense_curvature"` and keeps
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
    the finite iteration sequence. Overlap score gradients include the
    adaptive projection and determinant terms. PNPG requires a locally stable
    active set with strict complementarity. Invalid active sets and primary
    or auxiliary solve failures produce **NaN gradients**. Primary forward
    failure returns the evidence penalty and zero reconstruction. Check
    `solver_info` and the diagnostic evidence status
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
    The complete Hermite Overlap score converged from three bounded L-BFGS-B
    starts in the qualified Float64 Euclid mock. [Optimizer qualification](../validation/adaptive-image-overlap-m3-optimizer.md)

---


### Shared pixelized example support

Maintained dense and operator pixelized examples use `TinyLensGpu.example_support`
for sampling, stage storage, source-package transfer, and image panels. Model
construction, stage order, masks, grid and solver settings, paths, and rendering
variants remain in each case's entry point. Launch and resume commands are
unchanged; for example, from `examples/pix_src_demo_operator/pipe/slacs_examples/`,
use `python model.py J0737+3216 --skip-done`.

Fresh pixelized fits default to `bilinear`, with zero brightness outside the
fitted bounding box. Select `interpolation="cubic_hermite"` on
`PixelizedSourceModel` to interpolate with the cubic Hermite basis and fixed
zero exterior nodes. Brightness and its first derivatives go continuously to
zero two grid spacings beyond the fitted box. NNLS and PNPG constrain the
fitted node intensities. The signed Hermite weights can produce negative
brightness between those nodes.

The operator SLACS example accepts an explicit mode. Run these commands from
the same `slacs_examples` directory to start and then resume a Hermite run.

```bash
python model.py J0737+3216 --out-dir output_hermite --source-interpolation cubic_hermite
python model.py J0737+3216 --out-dir output_hermite --skip-done
```

On resume, the pipeline inherits the recorded mode and treats legacy
checkpoints as bilinear. An explicit conflicting mode requires a fresh run.
To initialize the fresh M0 solver from saved S1 node intensities, add
`--initial-source-from output/stage_m1.pkl`, with the path relative to the
lens directory. Use a package with the same grid size. M0 uses the saved
fitted grid and recomputes its evidence using the new run's selected
interpolation. Current validation covers fixed-parameter reconstructions
and gradient consistency.

The other staged pixelized demos accept the same `--source-interpolation`
option. In the basic fit and inversion scripts, select the mode with
`TINYLENS_SOURCE_INTERPOLATION=cubic_hermite`.

Pass sampling choices explicitly. For a case using 300 live points, batches of
100 and vectorized likelihood evaluation, the adapter is:

```python
from TinyLensGpu.example_support import SamplingConfig, run_sampling

stage = run_sampling(
    likelihood,
    SamplingConfig(n_live=300, n_eff=600, vectorized=True, n_batch=100),
    tag="stage-M1",
)
medians = stage.medians()
```

`run_sampling` returns `StagePosterior`, including parameter ordering and prior
metadata. Keep the originating case's `n_batch`, `vectorized_chunk_size`, weight
normalization, sampler and run options; some dense examples use `n_batch=n_live`,
and the dense matrix case uses a smaller batch for its pixelized stage. Optional
before/after hooks support the existing warmup and metrics output in specialized
cases. Importing the shared package adds no working-directory, environment, JAX
precision, or pyplot initialization changes; runtime policy stays with entry
points.

`StageStorageConfig(output_dir=..., include_schema=...)`, `save_stage`, and
`load_stage` preserve `stage_{tag}.pkl` payloads. Select `include_schema=False`
for legacy cases and `True` for cases already writing `stage_schema`.
`stage_from_payload` reads both formats. Existing case adapters retain their
filenames, fields and numerical array representations. B1938's M4/M5 schema
validation and protected-M5 overwrite checks still run at the case layer.
In interpolation-aware workflows, `StageStorageConfig` records the selected
mode and boundary in the existing `extra` dictionary.

Use `export_source_package(..., record_adapt_image=True)` to package the
producer's PSF-convolved native source component as `adapt_image`. The same
package stores `source_image` as the source-plane raster, `source_pixels`,
the fitted square grid, and reference provenance. In Euclid, M1 exports this
package, M2 fixes the M1 mass to calibrate `log_lambda_reg` and
`adaptive_reg_rho`, and M3 fixes those strengths while varying mass on the
same grid and reference. Workflows with an M0 reference producer export the
arc there. [Maintained Euclid VIS qualification](../validation/adaptive-image-overlap-euclid-vis.md)

Use `--skip-done` to resume matching stage products. The objective record
identifies the observation, masks, PSF, subpixel sampling, source grid and
interpolation, adaptation recipe, reference and upstream stage identities,
fixed strengths, and evidence and solver settings. Changes in these inputs
trigger the applicable refit. A verified historical producer package can
regenerate its arc through the component renderer if the original source
pixels, mass point, grid, interpolation and observation provenance are
available. Rebuild the producer when those inputs cannot be verified.
Recalibrate M2 and refit M3 when migrating old fixed-template adaptive
posteriors. For controlled Block comparisons, use the same observation,
reference, grid, prior and solver settings with explicit Block evidence.

J0946 paired M4 and M5 scores require matching adaptation and evidence
provenance. A new macro handoff includes the actual scale and evidence
method at the selected M3 draw. Fixed-scale source–potential reconstruction
consumes the saved scale through its `source_scale` input. Archived handoffs
continue to use the scale and objective recorded with their original fit.

Formal source export and diagnostic plotting stop with
`InvalidReconstructionError` on explicit operator non-convergence or nonfinite
reconstruction values; saved stage posteriors remain available. Unsupported
operator joint effective-parameter statistics are distinct from failure:
valid component images and raw chi-square remain usable. Shared image panels
retain the case's masking, crop, color, labels and overlays.

---

### Source–potential reconstruction

Use `PotentialCorrectionRenderer` with the separate
`SingleStepPotentialCorrection` or `IterativePotentialCorrection` entry point
for one lens-light-subtracted band with a fixed macro and source geometry.
These signed coupled fits select `backend="dense"` or `"operator"` and use
direct positive source/D4 strengths. Their public result distinguishes
conditional, local-Gauss–Newton and principal-block evidence, and reports
numerical/nonlinear failures explicitly. The dedicated
[potential-correction guide](potential-correction.md) covers imports, complete
small use, truth-derived initialization, support/priors, natural-log DIRECT
searches, scientific exports/reload and measured precision/resource limits.

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
- **Potential-correction guide**: `docs/guides/potential-correction.md`

Contributions should update the relevant guide when behavior or workflows change.

---

**Maintainer**: TinyLensGpu development team  
**Contact**: Open an issue or pull request on GitHub.
