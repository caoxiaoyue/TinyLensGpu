# Potential-correction examples

`generate_mock.py` creates single-band, lens-light-subtracted Pseudo-Jaffe
subhalo observations and a matched no-subhalo control. It does not fit a
correction. Use the `tinylens_gpu` environment; PyAuto packages are unnecessary.

For the small fixed-strength nonlinear algorithm demonstration, run
`iterate_fixed_strength.py --backend dense` or `--backend operator`; see the
[dense validation](../../docs/validation/potential-correction-dense-iteration.md)
and [operator validation](../../docs/validation/potential-correction-operator-iteration.md).

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/generate_mock.py \
  --output output/potential_correction_mock
```

The default image is 200×200 at 0.02 arcsec/pixel, with two midpoint integration
samples per axis, a fixed true SIS+external-shear macro, two Gaussian source
clumps, and `PseudoJaffeSubhalo(b_sub=0.01, theta_E_host=1.0, center=(0.9,0.1))`.
The component's adopted tidal-radius relation is `rt=sqrt(b_sub*theta_E_host)`;
it is separate from the cored `PseudoJaffe` profile.

A small invocation suitable for inspection is:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/generate_mock.py \
  --npix 16 --dpix 0.18 --nsub 2 --psf-size 3 --seed 91 \
  --output output/potential_correction_small
```

`--config` accepts JSON fields from `MockConfig`, including nested `source`
parameters, or the saved `configuration.json`. Command-line fields override
that configuration. Image sampling, source brightness, macro and subhalo
physical parameters, noise, PSF, mask radii, dtype, and output directory can
therefore all be configured. `--nsub` controls simulated-image integration;
subsequent inversions configure their own integration independently.

`subhalo.npz` and `control.npz` contain `image_data`, `noise_map`,
`noise_realization`, `model_image`, `psf_kernel`, `mask`, coordinate grids,
`true_potential`, `true_deflection_x/y`, `true_convergence`, and
`true_convergence_valid`. Arrays require no pickle. `configuration.json`
records physical values, noise seed/generator, units, integration, initialization
provenance, and truth conventions. Regenerate the identical inputs with:

```bash
python examples/potential_correction/generate_mock.py \
  --config output/potential_correction_small/configuration.json \
  --output output/potential_correction_small_repeated
```

The observations share the same independent Gaussian noise draw. The default
annular observation mask has `True` for excluded pixels. The generator follows
the current pixelized imaging convention: excluded native pixels contribute
zero brightness before native-grid circular FFT PSF convolution. It does not
add a PSF halo. No lens light is included. Source brightness is evaluated
analytically at subpixels before averaging; it is not first sampled on the
inversion source grid.

The Python interface supports truth-derived initialization directly:

```python
from examples.potential_correction.mock import MockConfig, generate_mock_pair

pair = generate_mock_pair(MockConfig())
observation = pair.subhalo
# observation.macro is the fixed true macro PhysicalModel.
# observation.sim_config is the corresponding SimulatorConfig.
# gx, gy = observation.source_truth.gradient(beta_x, beta_y)
# initial_source = observation.source_truth.brightness(source_x, source_y)
# true_potential = observation.subhalo_truth.potential(potential_x, potential_y)
```

`SubhaloTruth` uses the analytic singular-profile potential with zero at the
subhalo center. It reports physical point-sampled convergence, with `NaN` for
radii below the mass component's 1e-8 arcsec numerical radius convention.
Deflection follows that existing component convention, including a zero vector
at the exact center; no new physical core is introduced. Potential derivatives
and the analytic convergence/deflection relations should be compared outside
that tiny numerical domain. At zero strength every product is zero, including
at the center. Stored truth maps are point samples, not pixel averages. Later
coarse-grid convergence comparisons must separately state their finite-
difference/sampling convention.

Generation scopes the requested float32/float64 setting and preserves the
process's prior JAX precision configuration. Callers doing additional JAX
arithmetic on float64 outputs must keep an appropriate x64 context enabled;
otherwise JAX itself may downcast subsequent operations. These mock products
establish reproducible inputs and analytic truth, with no claim of recovery,
formal detection significance, mass precision, or exact correspondence to the
reference repository's simulation.

## Fixed-strength operator reconstruction

The small runnable slice uses a 12×12 observation, a 5×5 source, the analytic
mock-truth source gradient at fixed macro rays, and explicit fixed strengths:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/single_step_operator.py \
  --output output/potential_correction_operator
```

Add `--control` for the matched no-subhalo observation. `--source-n`,
`--block-size`, the three `--lambda-*` strengths, and `--pcg-max-iter` configure
the fit. This deliberately small assembly example has no tuning or scientific
recovery acceptance; its amplitude value of 100 is explicit and has not been
established as a scientifically weak prior.

`reconstruction.npz` saves the public source/potential fields, coordinates,
valid derivative maps, prior-window coverage, both linear and nonlinear model
images, and residuals. `configuration.json` records the observation, strengths,
initialization, exact block partition, dtype, actual coupled-system residual,
PCG controls and evidence approximation. Numerical failures write a diagnostic
`failure.json` and terminate before publishing reconstruction arrays.

The reusable library interface is
`problem.fit(lambda_source, lambda_potential, backend="operator", block_size=64)`.
It retains the complete source–potential coupling in matrix-free PCG. Bounded
principal blocks approximate its preconditioner and the two determinants;
each prior block contains the independent amplitude prior on every potential
support coefficient. Optional `block_indices` specifies a complete disjoint
joint partition, including mixed source/potential blocks. Results identify
`determinant_method="principal_block_cholesky"` and
`evidence_is_exact=False`. Use `backend="dense"` for the complete numerical
reference. Calibration of strength selection and recovery belongs to ticket 07.

## Dense strength tuning

`tune_single_step.py` runs a small dense single-step reconstruction with the
fixed true macro and frozen analytic mock gradient. Supply the independent
amplitude strength and both natural-log search bounds explicitly:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/tune_single_step.py \
  --lambda-amplitude 0.4 --log-source-bounds -2 2 \
  --log-potential-bounds -4 2 --maxfun 80 \
  --output output/potential_correction_tuning
```

The example uses a 12x12 Pseudo-Jaffe image at 0.25 arcsec/pixel, nsub=1,
3x3 PSF, and 5x5 source in `[-0.6,0.6]^2`. `--source-n`,
`--potential-spacing`, `--dtype`, and DIRECT controls configure the run.
`tuning.json` saves bounds, controls, selected strengths, all scalar candidate
statuses/reasons, initialization and evidence provenance. A successful best
fit also saves public fields and both renderings in `reconstruction.npz`.
The Python `run_tuning(...)` function returns the search object, including
`result.strengths` and `result.reconstruction`.

`has_valid_fit` and `optimizer_success` are separate: reaching an approximate
evaluation budget can leave a useful successful reconstruction while DIRECT
reports incomplete optimization. Boundary flags identify a selected strength
within `--boundary-rtol` (default 0.01) of a configured natural-log interval
edge. This command's prior and bounds are numerical example inputs;
scientific weak-prior selection and recovery require the later sensitivity
and backend-calibration checks.

For a prepared `SingleStepPotentialCorrection` problem, use
`TinyLensGpu.Inference.optimize_potential_correction(problem, log_bounds=...)`.
`PotentialCorrectionEvidence(problem)` is the reusable negative-evidence
callable over `[log_lambda_source, log_lambda_potential]`. Every candidate
performs a complete fit with fixed initialization; failed numerical candidates
return positive infinity and retain their reasons. See the
[API and verification record](../../.scratch/potential-correction/research/dense-strength-optimization.md).

## Small backend calibration

The calibration driver scans fixed strengths and independently runs bounded
DIRECT with both backends on matched Pseudo-Jaffe/control images:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/calibrate_single_step.py \
  --config examples/potential_correction/calibration_configuration.json \
  --output output/potential_correction_calibration
```

The explicit input fixes a 64x64 image, 49x49 source in `[-0.4,0.4]^2`,
0.075-arcsec potential spacing, nsub=2, and the analytic truth source gradient.
It varies the independently fixed amplitude precision between complete runs.
Scan strengths are physical positive numbers; DIRECT bounds are natural logs.
The driver saves `calibration.json`, `inputs.npz`, and `reconstructions.npz`.
Scalar records distinguish complete determinants from principal-block
approximations, compare fixed-strength solves, retain candidate failures and
DIRECT budget/boundary status, and repeat candidates in reversed order.
Raw signed fields, valid derivative masks and both renderings are unchanged.
The separately named affine diagnostics change no fitted image or evidence.
The record also separates source interpolation, potential discretization,
analytic-source Taylor remainder, and the fitted conditional/nonlinear image
discrepancy. Actual macro, truth-corrected and fitted ray margins identify
source-box truncation. The small test configurations exercise the workflow;
their coarse grids do not establish scientific recovery.

The example-level partition helper is reusable with explicit initial arrays:

```python
from examples.potential_correction.calibrate_single_step import source_incidence_blocks

mapping = renderer.source_mapping()
# Evaluate the chosen initial-source derivative once, at mapping.beta_x/y.
# For iterative reconstruction use the initial pixelized source's derivative.
blocks = source_incidence_blocks(
    renderer, mapping, initial_gradient_x, initial_gradient_y,
    source_tile=24, block_size=1024,
)
fit_options = {"block_size": 1024, "block_indices": blocks,
               "pcg_max_iter": 20000, "pcg_rtol": 1e-12}
```

The helper assigns potential nodes to fixed source tiles using sparse query
and derivative-stencil incidence, attaches support-only nodes spatially,
interleaves fields, and splits at the declared cap. It constructs no dense
source-by-potential matrix and retains no truth callable. Keep this partition
fixed during a run; refreshing an iterative Jacobian does not require retaining
the truth gradient. Saved records contain its method, actual block sizes and
complete indices. A source grid, mask, integration, initialization or potential
spacing change requires rebuilding the partition and checking its approximation.

The completed [64x64 calibration record](../../docs/validation/potential-correction-single-step-calibration.md)
adopts independently fixed `lambda_amplitude=10000`, source-strength bounds
`[1e-5,1e-3]`, D4 bounds `[1e5,1e11]`, and the declared incidence partition
for that exact configuration. All six subhalo/control sensitivity runs are
retained, including raw fields, low-mode/affine diagnostics, peak suppression,
control artifacts and the dense upper-bound flag. DIRECT reaches its configured
budget; successful inner fits and incomplete optimizer termination remain
separate. The chosen block evidence reproduces a similar localized correction,
while its absolute evidence differs from the complete calculation. This
calibration does not establish precise mass recovery or transfer automatically
to the 200x200 scientific example.

## Scientific single-step pair

`single_step_scientific.py` runs both the Pseudo-Jaffe observation and its
matched no-subhalo control, with the fixed true macro and frozen analytic
truth gradient. It supports complete dense evidence and the calibrated sparse
source-incidence block approximation:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/single_step_scientific.py \
  --config examples/potential_correction/scientific_configuration.json \
  --backend both --output output/potential_correction_scientific
```

The explicit scientific configuration uses a 200x200 image at 0.012
arcsec/pixel, retaining the small calibration's 2.4-arcsec field of view. It
adopts a deliberately recorded physical scenario with `b_sub=0.04`,
`theta_E=0.8`, and subhalo center `(0.75,0.1)`; the mock generator's original
200x200/`b_sub=0.01` defaults remain available. Source widths `(0.16,0.12)`
and clump scale 0.7 are smooth enough for the fixed 49x49 source in
`[-0.4,0.4]^2`. Potential spacing is 0.075 arcsec. The Gaussian PSF has physical
sigma 0.05 arcsec and 27 samples across its approximately three-sigma radius.
This changes its finite truncation relative to the 64x64 calibration's 5x5
kernel; validation numbers must be attributed to the actual configuration.

JSON configures every mock parameter, `inversion_nsub`, source geometry,
potential spacing, independent `lambda_amplitude`, source/D4 strengths,
backend selection, PCG controls, output plots, and optional affine diagnostics.
Mock and inversion integration are stored separately. CLI overrides include
`--npix`, `--dpix`, `--mock-nsub`, `--inversion-nsub`, `--source-n`,
`--potential-spacing`, `--dtype`, `--mode`, `--backend`, and `--no-plots`.
Neither source nor potential geometry changes during a fit or search.

Use `--mode tune` for bounded DIRECT in the configured natural-log
`log_bounds`; `direct_maxfun`, `direct_maxiter`, `direct_len_tol`, and
`boundary_rtol` control that search. Every candidate starts from the same
frozen gradient and zero potential. Fixed mode uses `fixed_strengths` keyed
by `subhalo` and `control`, optionally with additional `dense` and `operator`
keys for independently selected strengths. A successful fit and successful
optimizer termination remain separate metadata. A budget-limited best fit
is labeled accordingly, and all-invalid searches retain failure reasons.

`run.json` saves the complete input, actual data/source/core/support/query
counts, full block partition, software/revision and implementation hashes,
search records, and recovery summaries. Each `subhalo|control/dense|operator`
directory contains:

- `reconstruction.npz`: raw signed source/potential fields, source axes/box,
  all potential node/core/derivative/interpolation/query geometry, derivative
  validity and regularization-window coverage, both images and their residuals,
  observation/noise/PSF/mask, and sampled analytic and discrete truth maps.
- `result.json`: solver status and actual residual, all three strengths,
  conditional evidence and determinant methods, initialization provenance,
  units, integration settings, truth conventions, and scientific summaries.
- `images.png/.pdf`: explicitly separate linear prediction and nonlinear
  re-render, both noise-normalized residuals and their image difference.
- `source.png/.pdf` and `fields.png/.pdf`: source, potential, both deflection
  components and convergence, with raw truth comparisons and differences.

Support-only derivative values stay undefined/NaN and are blank in plots.
Convergence errors are reported on all valid core nodes, a predeclared
illuminated subset and a matched disk around the injected position. Analytic
convergence remains a point sample with the singular-center convention;
discrete truth applies the reconstruction's centered differences to analytic
nodal potential. Localization is a convergence-peak location, without a
statistical detection interpretation. `supplied_truth` separately records
source interpolation, potential discretization, integration differences and
the analytic-source Taylor remainder. These are separate from the fitted
linear/nonlinear mismatch, which also reflects a fitted source gradient that
differs from the frozen analytic gradient.

Raw fitted fields never undergo gauge alignment or clipping. Optional
`diagnostic_affine_potential_values` and `affine_diagnostic_only` metrics remove
a least-squares constant+x+y difference only for a separately named diagnostic;
they do not replace fields, images or evidence. Failure writes `failure.json`
and removes this exporter's previous success arrays/figures from that run's
directory. Plotting and persistence accept only the public
`PotentialCorrectionResult` plus observation/truth context through
`result_products.export_result`; no numerical solve is reconstructed.
`result_products.load_result(output)` restores the saved public result from
NPZ/JSON for later diagnostics or replotting with explicit observation/truth
context. Failed records cannot be loaded as a successful result.

Scientific float64 validation and its remaining peak/low-mode limitations are
recorded in `docs/validation/potential-correction-single-step-scientific.md`.
This workflow makes no detection-significance, mass-interval, or unmeasured
performance claim. Float32 is configurable; scientific precision/performance
comparisons are a separate validation task.

## Fixed-strength nonlinear reconstruction

The small nodal example starts from truth-derived source nodes and zero
potential, then updates the source and potential together using current
bilinear mappings and gradients:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/iterate_fixed_strength.py \
  --backend operator --block-size 16 --pcg-max-iter 800 --pcg-rtol 1e-12
```

Use `--backend dense` for the complete reference. Both backends share actual
nonlinear-objective acceptance, damping, stopping and failure handling. The
operator retains all cross-field terms in PCG; bounded principal blocks
approximate only preconditioning and determinants. `--pcg-atol` controls its
recursive stopping threshold; the library separately certifies the true
coupled damped-system residual using constructor `solve_rtol`/`solve_atol`.

The Python `main(...)` returns the common public result, including scalar
accepted/rejected history, final nonlinear images, valid-domain diagnostics,
and separate local-Gauss–Newton and determinant-method labels. The command
prints a scalar summary and raises on numerical failure. It is a representable
12×12 nodal algorithm demonstration, without a regularization search or a
scientific recovery claim. The amplitude value 100 is an explicit numerical
setting. For larger models, reuse one explicit source-incidence partition per
run/candidate as described above; a refreshed Jacobian does not rebuild that
partition. Block-evidence calibration is specific to geometry and initialization.

## Nested nonlinear strength tuning

The small numerical nodal example runs a complete fixed-strength nonlinear
reconstruction for every two-dimensional DIRECT candidate:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/tune_iterative.py \
  --backend operator --output output/potential_correction_iterative_tuning
```

Use `--backend dense` for complete local determinants. The explicit defaults
are a 4x4 image, source2 in [-0.5,0.5]^2, 33 potential nodes, float64,
zero-order source prior, fixed amplitude precision20, source strength bounds
[0.1,2], and D4 bounds[0.001,1]. The source is truth-derived bilinear node
intensity; the Gaussian nodal potential and deterministic image perturbation
define a controlled numerical case. Every candidate resets the source, zero
correction and initial damping. The fields, geometry, amplitude precision and
bounded explicit block partition remain fixed through the outer search.

`--log-source-bounds` and `--log-potential-bounds` take natural logarithms;
`--maxfun`, `--maxiter`, and `--max-iter` configure DIRECT and inner iteration
budgets. The accepted tiny partition groups all four source nodes with16
potential nodes, plus a second17-potential block. Its frozen indices came from
initial response calibration; runtime selection requires no dense Jacobian.
This layout is validated only for the stated tiny model and bounds. Changing
the model, geometry or prior requires a new approximation check.

The Python `main(...)` returns the search object and its common public result
at `.reconstruction`. `search.json` retains scalar candidates, both evidence
labels, exact termination reasons, bounds, selected strengths and actual
budget/boundary status. Successful retained fields, nonlinear images, history,
geometry and supplied nodal truth are handed to compact `nested_result.json`
and `nested_result.npz` diagnostics without another fit. The example-specific
numeric format is separate from scientific exports. All-invalid searches
write `failure.json` and remove stale nested success products.

The recorded default searches retain converged inner fits but exceed the
approximate100-evaluation budget at115 and favor the upper D4 boundary.
`has_valid_fit` is separate from `optimizer_success`. Local Gauss–Newton
evidence is approximate even with dense determinants; operator block
determinants add a separate approximation. The independent scan, actual
selected strengths/recovery changes, and a rejected partition remain in the
[nested validation report](../../docs/validation/potential-correction-iterative-inference.md).
This case makes no scientific subhalo-recovery, detection or performance claim.
