# Dense single-step reconstruction: ticket 04

This stage implements the conditional linear Gaussian reconstruction in the
approved potential-correction spec. It uses the ticket 03 renderer's fixed
geometry, macro rays, physical potential derivatives, source interpolation,
subpixel integration, PSF and observation mask. It does not yet optimize
strengths, provide an operator backend, or establish scientific recovery.

## Public entry point

```python
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection

problem = SingleStepPotentialCorrection(
    renderer, image_data, noise_map, truth_gradient,
    lambda_amplitude=0.4,  # explicit example value, not a calibrated weak prior
    source_regularization="second-order",
    initialization_provenance={"source": "mock truth analytic derivative"},
)
result = problem.fit(lambda_source=1.3, lambda_potential=0.3, backend="dense")
# Equivalent callable boundary: result = problem(1.3, 0.3).
```

The renderer supplies the fixed macro, source geometry and potential geometry.
Data and noise are native image arrays; only selected observations require
finite data and finite positive noise. `truth_gradient(beta_x, beta_y)` returns
two arrays at macro image-subpixel rays. Those gradients are evaluated once,
masked to the fixed source domain and frozen. The full signed source is solved,
rather than fixed to a truth intensity or treated as an increment. The signed
potential response includes the lens-equation minus sign.

`lambda_potential` is the fourth-difference strength. `lambda_amplitude` is
independent, positive, fixed on the problem and applied to every potential
unknown. No affine constraint, clipping, numerical prior jitter or factorization
rescue is applied. Every complete horizontal/vertical five-node window in U is
included, including wholly support-only windows; gaps are never bridged.
Window coverage is retained per unknown. Source regularization retains all
three existing nonadaptive choices and their existing boundary behavior.

The dense JAX kernel forms and solves the full coupled system by Cholesky. The
conditional evidence includes the complete prior and posterior log determinants,
the prior quadratic penalty, and selected-observation noise normalization.
It uses the fitted linear image. The separate nonlinear re-render never replaces
that image in the evidence calculation.

`PotentialCorrectionResult`, `PotentialCorrectionStrengths` and
`PotentialSolverInfo` are re-exported by `TinyLensGpu.ObservationModel`.
The result preserves raw fields, source/potential geometry, core derivative
validity, fourth-window coverage, both images and residual diagnostics,
strengths, solver residuals, configuration and initialization provenance.
Physical maps are NaN at support-only nodes. Full residual images are NaN at
excluded observations. Dense method annotations are
`evidence_method="conditional_linear_gaussian"`,
`determinant_method="full_cholesky"`, and `evidence_is_exact=True`; exactness
refers to this fixed-gradient conditional model.

Invalid configuration raises a validation error. Numerical failure returns
`success=False`, `status="failed"`, a reason and available diagnostics, and
`log_evidence=None`. Factorization is never repaired by changing the declared
amplitude prior. The dense relative-system-residual limit is configurable as
`solve_rtol`, defaulting to 1e-8 for float64 and 1e-3 for float32.

## Numerical and backend handoff

- `forward(r)` and `adjoint(y)` are the exact coupled response pair.
- `forward_image(r)` returns the full native linear prediction.
- `prior_matvec(r, lambda_source, lambda_potential)` applies the complete prior
  without creating square matrices.
- `dense_mapping()` and `dense_prior(...)` explicitly request dense storage;
  construction otherwise keeps source/potential regularization compact.
- `result_from_solution(...)` assembles the shared result and rejects invalid
  fields, images, evidence or solver status for a numerical backend.
- `PotentialRegularization` in `utils/inversion/potential_regularization.py`
  exposes compact complete-window forward/adjoint and prior actions, plus
  explicit `to_dense(...)` materialization.

New macro rays can be obtained from `renderer.prepare_macro(new_macro)` and
passed as `macro_mapping=` to a new problem. This reuses geometry while
rebuilding source mapping and initial gradients; it does not mutate old fits.
Public numerical actions compose inside JAX. Callers compiling an outer
float64 kernel must use a float64 context, as for the renderer actions.
Each fit scopes precision internally and preserves process defaults. The dense
kernel requests full-precision matrix products locally. Linear-image JIT
wrapping is cached across strength evaluations. Render validation transfers
only a finite-status scalar instead of complete fitted fields to NumPy.

## Verification

Environment: `tinylens_gpu`, Python 3.11.14, JAX 0.7.1, NVIDIA GeForce RTX 5080.
From this worktree's `tests/` directory:

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest test_potential_correction_dense.py test_potential_correction_rendering.py -q
```

Final result: **31 passed in 28.19 s**, comprising 14 dense reconstruction
cases and 17 rendering regressions. `git diff --check` also passed.

The independent 3x3 dense problem uses a 2x2 source, 0.1 image-pixel spacing,
default 0.2 potential spacing, source strength 1.3, fourth-difference strengths
0.03 and 2.0, fixed amplitude strength 0.4, gradient (1.2, -0.8), and noise
standard deviations from 0.2 to 0.4. A host construction of bilinear source
weights, physical potential derivatives and center-enumerated D4 windows
provides an independent coupled solution. A separate data-covariance Gaussian
calculation verifies evidence, including support-only normalization and both
potential strengths. Field tolerances are 2e-11 relative / 2e-12 absolute;
evidence tolerances are 2e-12 relative and absolute. The case has nonzero
cross-curvature and signed source/potential coefficients, so a decoupled or
nonnegative solve cannot pass.

A 5x5 nsub=3 case uses an asymmetric 3x3 PSF, a mask hole, a bilinear source
with analytic derivatives and a quadratic potential. It checks the independent
analytic negative image response, frozen gradient, coupled transpose identity,
and dense/action agreement. Halving the perturbation from 0.1 to 0.05 reduces
the nonlinear-minus-linear norm by four (relative tolerance 2e-7), consistent
with a second-order remainder. A separate irregular four-pixel nsub=3 fit
checks selected data/noise, both full images and residual masks.

All three existing source-prior types are compared with worked 2x2 source
precision matrices in both dtypes while the surrounding precision context
requests the opposite dtype. Float64 field/prior tolerance is 3e-12; float32
is 3e-5. These tests initially exposed GPU reduced-mantissa Gram products:
coefficient error was about 1.56e-3 on the zero-order case despite a small
reported system residual. Local full-precision products reduced the measured
error against the host assembly to about 1.26e-6. Both process x64 and matmul
precision defaults remain unchanged.

Failure checks exercise finite-strength overflow, a numerically singular prior
at amplitude strength 1e-30, and a deliberately unattainable 1e-30 solve
tolerance. All return explicit failed results without valid evidence or an
altered amplitude strength. Invalid strengths/backend remain validation errors.

These are small numerical checks. There is no dense/operator calibration,
amplitude-prior sensitivity conclusion, 200x200 reconstruction, optimizer,
scientific recovery assessment or GPU timing/memory claim at this stage.

## Preliminary physical-mock check after integration

The coordinator additionally ran nine dense candidates on a 48-by-48,
float64 Pseudo-Jaffe mock with nsub=2, dpix=0.05 arcsec, a 41-by-41 source
over [-0.5,0.5] squared, and potential spacing 0.1 arcsec. The amplitude
precision was independently fixed at 1000. Source strengths were
{1e-6,1e-4,1e-2}; fourth-difference strengths were {1e4,1e6,1e8}.
The [complete configuration and scalar results](dense-preflight.json) and
[script](../experiments/dense_preflight.py) retain this preliminary check.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \
  python .scratch/potential-correction/experiments/dense_preflight.py
```

All nine candidates solved, with relative system residuals below 1e-13. At
source strength 1e-2 and fourth-difference strength 1e4, normalized residual
RMS was 1.03 for the linear fit and 2.24 for the nonlinear re-render. This
illustrates a consequential linearization discrepancy that a finite evidence
would not expose. The largest evidence on this coarse grid occurs at its
upper potential-strength boundary. This check establishes neither an optimum
nor scientific recovery and does not replace tickets 07 or 11.
