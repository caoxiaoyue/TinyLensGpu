# Dense fixed-strength nonlinear source–potential reconstruction

Ticket 08 adds `IterativePotentialCorrection`, independently of the existing
frozen-gradient single-step algorithm. It minimizes the actual nonlinear image
misfit plus priors on the accumulated signed source and potential. The macro,
source box/grid, potential support, data mask, PSF and integration stay fixed.
This record concerns small numerical algorithm checks. It is not an outer
regularization search, scientific field-recovery calibration, detection result
or mass-precision study.

## Public use and numerical policy

```python
from TinyLensGpu.ObservationModel import IterativePotentialCorrection

problem = IterativePotentialCorrection(
    renderer, image_data, noise_map, initial_source,
    lambda_amplitude=100.0,
    initialization_provenance={"source": "truth-derived fixed-grid nodes"},
)
result = problem.fit(lambda_source=1e-4, lambda_potential=100.0)
```

`initial_source` uses the renderer's source-node ordering. Each fit resets to
those node intensities and zero potential. Current analytical derivatives of
the bilinear source replace any initial truth-gradient function; no such
function is retained. Interior knots use the established right/up branch;
clipped upper-endpoint derivatives and out-of-bounds behavior are unchanged.
The renderer exposes `source_spatial_gradient(source, mapping=None)`;
`linearize`, `jacobian_forward` and `jacobian_adjoint` expose the current
coupled numerical operator for forward/adjoint checks.

Dense increments solve `(J.T Cinv J + Q + mu*diag(H)) delta =
J.T Cinv(data-f) - Q*state`, with Cholesky equilibration and highest matmul
precision in JAX. Both fields update together. Acceptance requires a finite
actual nonlinear decrease and a positive model reduction. Rejected proposals
cannot replace accepted fields or caches. The backend's narrow numerical
operations are initialization, current preparation, solve, step measures and
undamped determinant evaluation; one Python controller owns all policies for
later operator reuse. There is no operator adapter in ticket 08.

The explicit controls include damping bounds/factors, maximum iterations,
additional rejection retries, scaled objective/update/gradient tolerances,
stalling count and maximum damping-energy fraction for objective/update
convergence. A small physical gradient suffices; it is not a universal
condition at a bilinear kink. Tiny damping-dominated steps cannot manufacture
success. Iteration limits, rejection/damping limits, stalling, inner-solve
failure and nonfinite states/proposals remain distinct. The last finite
accepted fields are retained on failure, with no evidence.

Successful evidence refreshes the final Jacobian and uses complete dense
prior/posterior determinants without LM damping. Its label is
`evidence_method="local_gauss_newton"`, `determinant_method="full_cholesky"`,
`evidence_is_exact=False`. This is a local linearization approximation, not
an exact nonlinear integral. The complete independent amplitude prior enters
normalization on every potential coefficient. No jitter or gauge rescue is
added. Common result linear products are None; nonlinear products define chi2
and objective. Inner linear-solver status is separate from nonlinear success.

## Independent numerical checks

The maintained core tests exercise:

- A non-affine 3×3 source with independent within-cell brightness differences,
  lower/upper endpoints, right/up knots, coordinate orientation and invalid
  rays. This detects interpolation of source-node finite differences.
- Current Jacobian actions at nonzero potential and changed source values,
  including changed source-cell validity, nsub=3 and the adjoint identity.
- A tiny coupled nonlinear problem compared with an independent NumPy
  coordinate/stencil forward model and SciPy finite-difference least-squares
  optimization. Final fields use absolute tolerance 4e-7/relative 3e-6; final
  nonlinear image/objective use 2e-13/2e-12 absolute tolerance. Independent
  finite-difference-Jacobian evidence uses 3e-8 absolute tolerance.
- The controlled source-only ridge limit with a fixed amplitude precision
  1e8, multiple damped updates, signed source values, and explicit float64 and
  float32 contracts. Source tolerances are 2e-7 and 2e-4 respectively.
- Reset-per-fit reproducibility under interleaved strengths and different
  initial damping, with final state/evidence comparisons at declared 2e-6
  tolerances. Numerical values change without recompiling stable kernels.
- Actual accepted-objective monotonicity, honest kink rejection, iteration,
  rejection and stall limits, huge-damping tiny steps, unsuccessful linear
  solves, and nonfinite initial/proposed states. A finite extreme-scale
  constant-deflection fixture makes proposed gradients overflow while its
  initial state and absolute-tolerance linear solve remain finite; that
  proposal is rejected without replacing accepted fields.

The signed float32 ridge fixture initially used amplitude precision 2, which
was not a source-only quadratic model: roundoff could escape its symmetric
stationary point and reach a lower coupled objective before stalling near a
kink. Both ridge-limit fixtures now use the explicit fixed precision 1e8.
Production defaults and stopping policies were unchanged. The
[retained audit](../../.scratch/potential-correction/research/dense-iterative-ridge-limit.json)
and [reproduction script](../../.scratch/potential-correction/experiments/iterative_ridge_limit.py)
record 15 repeats per configuration: the weak case had 11 gradient stops,
three damping limits and one stall, with escaped objective 11.65286 versus
source-only ridge 13.71429; the controlled limit had 15 gradient stops. Weak
counts vary with GPU reduction order and are not expected frequencies.

## Small runnable example

From the repository root, in `tinylens_gpu`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python \
  examples/potential_correction/mock/iterative/demos/fixed_strength.py
```

The 12×12 image has 0.12 arcsec pixels, delta PSF, nsub=1, and a selected
annulus 0.18–0.65 arcsec. The fixed macro is SIS theta_E=0.4 at zero center
plus shear (0.03,0.01). The source is a two-clump Gaussian sampled on 5×5
nodes spanning [-0.6,0.6]^2. Source widths are (0.22,0.17) arcsec and clump
scale 0.8. Nodal Pseudo-Jaffe truth has b=0.01, host theta_E=0.4 and position
(0.38,0.07) arcsec; its nodal mean is explicitly removed as a truth-reference
constant only. The noiseless image is rendered with the same nodal grids;
noise sigma 0.01 sets the objective scale. This representable discretization
isolates the iterative algorithm from source/potential sampling mismatch.

At fixed (lambda_source,lambda4,lambda0)=(1e-4,100,100), float64, default
controller settings, there are 84 data, 25 source and 97 potential unknowns.
The run converged by scaled gradient after six accepted updates: objective
2.4989361109 -> 0.04609580215, chi2=0.00013947416,
local-GN log evidence=169.4653712. The fitted potential is nonzero. These
numbers do not measure independently generated analytic-image recovery or
calibrate regularization choice. Full scientific plots/exports and outer
optimization belong to subsequent tickets.

## Verification record

All commands use `tinylens_gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and
`PYTHONPATH=/tmp/tinylens-pc-08` to select the isolated worktree. Core files
run from `tests/`; example files run from `examples/tests/` in a separate
process. Core/example imports were not mixed.

The initial combined core run found 161 passed/one failed because of the
incorrect weak-prior ridge fixture described above. After that fixture was
corrected, the expanded command ran 194 passed/one failed:

```bash
pytest -q test_potential_correction_iterative.py test_potential_correction_dense.py \
  test_potential_correction_operator.py test_potential_correction_rendering.py \
  test_pixelized_source_utils.py test_regularization.py test_operator_autodiff.py \
  test_solver_autodiff.py test_curvature_operator.py
```

All 21 new iterative cases and all source/potential dense/operator, rendering,
source-utils, regularization, solver-autodiff and curvature checks passed.
The remaining failure was the previously documented float32 legacy node
`test_operator_reconstruction_and_evidence_jacobian[True-True-pcg]`: one
finite-difference source-derivative entry was 0.83434 versus 0.92554, beyond
its unchanged rtol=0.05/atol=0.03. That node passed on an isolated untouched
35d0318 worktree. Its baseline variability and tighter fixture controls are
already recorded in [operator-single-step.md](../../.scratch/potential-correction/research/operator-single-step.md#existing-float32-jacobian-fixture-variability).
No legacy assertion or solver control was relaxed here.

The shared cell-selection extraction retains the original brightness-mapping
JAX primitive order; an explicit `make_jaxpr` comparison with untouched
35d0318 was identical. Final focused commands passed:

- From `tests/`, `pytest -q test_operator_autodiff.py
  test_potential_correction_iterative.py`: 33 passed in 85.39s, including the
  previously variable legacy case and all new iterative checks.
- From `tests/`, `pytest -q test_potential_correction_inference.py`:
  13 passed in 13.48s.
- In a separate process from `examples/tests/`, `pytest -q
  test_potential_correction_iterative_example.py
  test_potential_correction_operator_example.py
  test_potential_correction_calibration.py`: four passed in 37.65s.

Environment: Python 3.11.14, JAX 0.7.1, NumPy 2.4.6, SciPy 1.17.1,
CUDA device 0 on the available RTX 5080. No broad benchmark or new
autodifferentiation support is claimed.

After merging integration `329cdb8` (ticket 11 included) without conflicts,
`pytest -q test_potential_correction_scientific.py
 test_potential_correction_iterative_example.py` from `examples/tests/`
passed all four cases in 25.80s. These are small compatibility cases, not a
production 200×200 scientific run.
