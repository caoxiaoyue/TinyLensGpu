# Potential-correction mock inputs

`generate_mock.py` creates single-band, lens-light-subtracted Pseudo-Jaffe
subhalo observations and a matched no-subhalo control. It does not fit a
correction. Use the `tinylens_gpu` environment; PyAuto packages are unnecessary.

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
