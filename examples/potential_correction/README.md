# Potential-correction examples

Start with the [complete user guide](../../docs/guides/potential-correction.md)
for public APIs, algorithm/initialization differences, geometry and priors,
status/evidence interpretation, exports, scientific reproduction and precision
limits. The [coverage record](../../docs/validation/potential-correction-coverage.md)
connects the approved requirements to implementation and verification.

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
provenance, and truth conventions. Regenerate inputs from a configuration saved
with the same PSF convention with:

```bash
python examples/potential_correction/generate_mock.py \
  --config output/potential_correction_small/configuration.json \
  --output output/potential_correction_small_repeated
```

The observations share the same independent Gaussian noise draw. The default
annular observation mask has `True` for pixels excluded from the fit. The
generator evaluates analytic source brightness across the cutout, averages
subpixels and convolves with zero extension. Light emitted outside the
observation mask can therefore contribute to the fitted pixels through the
PSF. No lens light is included.

Archived runs that masked brightness before convolution used a different
observation model. Regenerate both control and subhalo runs before comparing
them with the current renderer.

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
interleaves fields, and splits at the declared cap. Each source neighbor votes
with the magnitude of its weight. The forward model and its transpose use the
signed weights. Keep this partition fixed during a run while the iterative
Jacobian refreshes. Saved records contain its method, actual block sizes, and
complete indices.

Single-step, iterative, and calibration JSON accept
`"source_interpolation": "cubic_hermite"`. Omitting it selects hard-box
bilinear interpolation. Hermite extends source support by two grid spacings
on each side using fixed zero exterior nodes. The saved source box and field
arrays still describe the fitted nodes.

Set `partition_file` to a saved run or calibration JSON, or to its extracted
`block_partition` object. Automatic reuse checks the geometry hash,
interpolation, boundary convention, vote rule, method, tile size, and block
cap. Source-incidence reuse also checks the traced macro rays and the
initial gradients used for the votes. The drivers regenerate a partition
when provenance differs and record the rejection in
`block_partition.reuse_rejected`.
For a controlled comparison, copy the saved `indices` into `block_indices`
in the input configuration. This takes precedence over automatic reuse,
validates the partition against the joint coefficients and cap, and records
its origin as `explicit_fixed`.

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

## Iterative scientific reconstruction and comparison

`iterative_scientific.py` uses the same analytic 200x200 physical mock,
fixed macro, noise, PSF, source box and potential geometry as the scientific
single-step configuration above. Every fit/candidate samples truth brightness
at the source **nodes** and starts with zero potential. All subsequent source
derivatives come from the current bilinear source at corrected rays.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/iterative_scientific.py \
  --config examples/potential_correction/iterative_scientific_configuration.json \
  --backend both --output output/potential_correction_iterative_scientific \
  --single-step-reference output/potential_correction_scientific
```

The optional reference directory is a saved `single_step_scientific.py` run;
comparison loads its public results without refitting. Dense single-step versus
dense iterative is the primary algorithm comparison. Each algorithm's
operator-minus-dense comparison reports determinant contributions separately.
The default scientific configuration requests the bundled historical
single-step partition. A byte-identical copy is reusable with matching
bilinear geometry, traced macro rays, tile size, and cap. The pinned
gradient fingerprints come from the analytic gradient in
`scientific_configuration.json` and the paired initial nodal derivative in
`iterative_scientific_configuration.json`. Floating-point coordinates can
produce different hashes across devices. The iterative example
builds a partition from the initial pixelized-source gradient when automatic
reuse fails. With `partition_source="initial_nodal"` and no `partition_file`,
it directly builds that partition. Both paths keep the selected partition
fixed through the fit and strength search.

Use `--mode tune` for DIRECT around **complete** inner reconstructions, with
the configured natural-log bounds and budget. Candidates reset source,
potential and damping, hold all three strengths fixed internally, and expose
evidence only after nonlinear success. Search termination/boundary/budget
metadata remain separate from inner convergence. The retained 200 scientific
results use fixed strengths. The bounded nested assembly is exercised in
small example tests; no converged 200-pixel outer search is claimed.

JSON configures mock/inversion integration separately, both geometries,
independent amplitude precision, fixed strengths or search bounds, static
partition, linear-solve controls, and `iteration_options`.
`case_iteration_options` explicitly overrides the common controller options
for a case; operator PCG options remain separate. The recorded float64
scientific controls use subhalo/control `gradient_atol=5e-4/3e-5`,
`gradient_rtol=1e-8`, `solve_atol=5e-5`, and `solve_rtol=1e-8`.
These example settings were measured against strict failed endpoints,
successful stopping controls, and stricter independently converged operator
control fields. They do not alter library defaults or automatically validate
another configuration.

The common NPZ/JSON contract contains final nonlinear image/residuals, raw
signed fields, geometry/validity, truth maps, strengths and initialization.
Iterative linear products are `None` and omitted from NPZ. Evidence is labeled
`local_gauss_newton`; its full versus principal-block determinant method is
recorded independently, and even the dense nonlinear evidence is approximate.
`comparisons.json` preserves the single-step linear inference objective and
reports a separate nonlinear objective for its recovered fields.

`history.png/.pdf` shows before/trial/retained objectives, accepted/rejected
trials, damping, pre-step gradients, the separately refreshed final gradient,
and inner true residuals with their thresholds. The initial PCG recursive
residual curve is labeled separately. Failure writes metadata and diagnostic
`failure-history` figures when plots are enabled, while removing previous
successful reconstruction arrays/figures. Failed candidates never supply
successful scientific evidence. For operator steps, a single separately
accumulated residual correction may use the **remaining total PCG budget**;
actual full-system certification remains mandatory and phase diagnostics are
saved. The legacy recursive residual is the initial phase on the original RHS.

The [scientific validation report](../../docs/validation/potential-correction-iterative-scientific.md)
records actual 200 both-backend fits, matched single-step comparisons, raw
versus optional affine diagnostics, the measured 10000-to-30000 amplitude-prior
sensitivity, and the failed lower 3000 probes. Localized convergence recovery
is useful with substantial peak suppression. Potential/deflection low modes
remain strongly prior-sensitive, and lower-side convergence is unverified.
The prior choice is specific to this model. The results do not establish a
globally weak prior or broad robustness, detection, mass intervals or performance.

## Fixed-strength L-BFGS-B mock

Run the matched Pseudo-Jaffe injection and no-subhalo control from the repository
root. The driver uses the same noise draw, mask, PSF, fixed SIS plus shear macro,
geometry, and prior strengths for both observations. Choose a new output
directory for each run.

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
  NVIDIA_TF32_OVERRIDE=0 MPLBACKEND=Agg python -m examples.potential_correction.lbfgs_mock \
  --backend both --output output/potential_correction_lbfgs_mock
```

The [default configuration](lbfgs_mock_configuration.json) records the 32 × 32
observation with seed 17 and 2 × 2 subpixel integration, the 17 × 17 cubic
Hermite source, and the 0.15 arcsec potential grid. Each fit starts at
analytic mock brightness sampled on the fixed source nodes and zero
potential. The same Hermite source initializes the whitening metric. The
first-order source, fourth-difference, and all-node amplitude strengths are
`0.001`, `10000`, and `10000`. Select `--backend dense` (the default) or
`--backend operator` for one backend. The operator configuration uses a
field-separated contiguous partition capped at 64 coefficients, a PCG limit
of 2000 iterations and a recursive relative tolerance of `1e-10`.
Set `operator_options.block_indices` in JSON to supply an explicit partition.

The mock validation configuration explicitly sets `gn_stationarity_tol=0.001`.
All four runs on the RTX 5080 selected 568 image pixels and qualified at that
cutoff. Convergence metrics use all 200 derivative-valid nodes, and the
residual RMS is in noise units on selected image pixels.

| Observation | Backend | Objective | Local GN log evidence | Convergence RMSE vs discrete truth | Positive convergence peak | Residual RMS / noise |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Injection | Dense | 175.20734 | 1239.17317 | 0.05137 | 0.26035 | 0.72976 |
| Injection | Operator | 175.20734 | 1152.05308 | 0.05137 | 0.26035 | 0.72976 |
| Control | Dense | 156.24661 | 1261.08349 | 0.03568 | 0.09435 | 0.70969 |
| Control | Operator | 156.24661 | 1177.98926 | 0.03568 | 0.09435 | 0.70969 |

The largest backend difference in convergence is `1.81e-7` for the injection
and `2.34e-7` for the control over the full valid fields. The objective
differences were below `3e-11`. The nonlinear images differ by `3.24e-7` and
`2.42e-7` noise units RMS on the selected pixels. Both recover the injection
peak at `(0.75, 0.15)` arcsec, 0.05 arcsec from the injected position, and the
control peak at `(-1.05, 0.6)` arcsec. Operator endpoint PCG took 340 and 349
iterations, with true residuals of `1.78e-13` and `7.79e-14` against thresholds
of `1.92e-11` and `9.96e-12`, respectively.
[Retained comparison](../../.scratch/lbfgs-potential-correction/output/mock-comparison-final/run.json)

The output contains reusable `configuration.json` and per-run summaries and
backend differences in `run.json`. Load any successful result directory
such as `subhalo/operator` with
`examples.potential_correction.result_products.load_result` to inspect the
raw fields, nonlinear images, and `optimizer_info`. Both local GN evidence
methods carry `evidence_is_exact=False`. Dense uses `full_data_space`
determinants and operator uses `principal_block_cholesky`. An unqualified
run writes `failure.json` with the available optimizer diagnostics and null
evidence.
