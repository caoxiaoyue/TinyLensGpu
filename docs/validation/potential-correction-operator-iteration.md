# Operator fixed-strength nonlinear source–potential reconstruction

Ticket 09 adds `backend="operator"` to the same `IterativePotentialCorrection`
controller used by the dense reference. This record validates small numerical
algorithm cases and bounded allocation. It does not calibrate nonlinear
block-evidence strength selection, scientific recovery, the amplitude prior,
or a production image size; those remain subsequent stages.

Implementation commit: `7c642bd`. Integration `f26cc49` was merged as `1597ca9`
without conflicts; it changed only the feature README's progress record.

## Numerical contract

```python
result = problem.fit(
    lambda_source=1e-4, lambda_potential=100.0,
    backend="operator", block_size=16,
    pcg_max_iter=800, pcg_rtol=1e-12, pcg_atol=0.0,
)
```

Both backends use the same current nonlinear image, physical accumulated-field
prior, right-hand side, objective acceptance, damping, convergence and failure
policy. Each fit resets to the supplied source nodes and zero potential.
Only the numerical adapter differs. It passes the current immutable
`PotentialLinearization(mapping, gradient_x, gradient_y)` as dynamic JAX data
into bounded-block preparation, the exact forward/adjoint curvature actions,
PCG, and the undamped step-prediction calculation. It never substitutes the
frozen single-step nonlinear forward or cached truth gradients.

Geometry, the padded disjoint partition, and unit source/fourth-difference
principal blocks are prepared once per topology. Fixed-strength prior blocks
and their determinant are combined once per fit. Every current point rebuilds
bounded likelihood blocks from current Jacobian columns inside `lax.scan`;
rejected damping retries reuse those blocks. No complete identity, design,
prior, potential curvature, joint curvature or cross-field matrix is needed.
One current block set is retained; histories store only scalar diagnostics.
The public optional `block_indices` accepts the same bounded, possibly mixed
partition as single-step reconstruction. Its complete indices, sizes and
SHA256 digest are persisted in result configuration.

The true operator is `J.T(noise**-2 * J(v)) + Q(v) + mu*diag(H)*v`, including
all cross-field terms and cross-block prior windows. The bounded Cholesky
preconditioner factors the matching damped principal blocks. Optional inverse
lower factors in the existing principal-block representation are computed
once per damping trial and applied repeatedly within PCG. Padding is identity
outside the physical model; it receives no damping or physical normalization.
The existing source–lens-light block-Schur implementation is unchanged.

The final step is certified using the recomputed full damped-system residual:
`norm <= solve_atol + solve_rtol * norm(rhs)`. Constructor defaults remain
unchanged. `pcg_atol`/`pcg_rtol` govern recursive PCG stopping separately;
recursive convergence alone never establishes a successful step. Both
absolute and relative true residuals remain in each history solver record.
The adapter records the last recursive residual separately in configuration.

Final evidence refreshes the current Jacobian at converged fields and factors
its **undamped** bounded physical posterior blocks. Prior normalization uses
`lambda4*D4.T@D4 + lambda0*I` on every potential unknown, including support-only
nodes. Step damping affects neither physical objective nor prior/evidence.
Results keep `evidence_method="local_gauss_newton"`,
`determinant_method="principal_block_cholesky"`, `evidence_is_exact=False`.
The local linearization and determinant approximation are separate statements.
Successful final physical scaled gradient and the initial baseline are exposed
as `final_scaled_gradient` and `initial_scaled_gradient`; objective/update
convergence does not assert that the branch gradient must vanish.

## Maintained numerical checks

`tests/test_potential_correction_iterative_operator.py` compares complete public
fits and independently assembled small reference systems:

- A 4×4 delta-PSF, nsub=1, zero-macro case uses source2 in [-0.5,0.5]^2,
  non-affine source values (0.4,1.1,0.2,1.4), a modest nodal Gaussian potential,
  deterministic image perturbation, noise sigma 0.07 and strengths
  (lambda_source,lambda4,lambda0)=(0.3,0.02,20). Both fields change and the
  source–potential cross curvature is nonzero. Dense/operator fields, nonlinear
  images, objective and valid maps agree at atol=2e-8/rtol=2e-7.
- A 12×12, 0.12 arcsec/pixel, nsub=2 annular case uses a fixed SIS theta_E=0.4
  plus shear (0.03,0.01), source5 in [-0.6,0.6]^2, a smooth two-clump nodal
  source, Gaussian nodal potential amplitude 0.02 centered at (0.35,0.07),
  noise sigma 0.01 and strengths (1e-4,100,100). Both source gradients and
  source-cell assignments change. Dense/operator fields/images/objective/maps
  use the same 2e-8/2e-7 tolerances. Repeated and fresh fits match after
  interleaved strengths (4e-4,70) and damping changes, with no post-warm-up
  JAX compilation entries at fixed topology. Independent public re-rendering
  of fitted fields agrees at 2e-13.
- The one-step small case independently assembles full current J/H/Q. Its
  step agrees with the full damped solve at 2e-12, the recomputed true residual
  at 3e-13, and exact **undamped coupled** predicted decrease/damping fraction
  at 2e-13. An uncoupled solve or block-only prediction cannot satisfy it.
- Mixed, irregular, padded cap7 partitions compare final posterior and complete
  prior principal determinants to a separate coordinate/stencil nonlinear
  model with finite-difference Jacobian. Two fourth-difference strengths,
  fixed lambda0=20, and initial damping 0.01/10 test the complete prior sum and
  refreshed undamped final determinant. Prior logdet tolerance is 3e-12,
  posterior 3e-8, evidence 2e-8; damping-schedule evidence agreement is 2e-6.
- Float32 coupled fields/images/objective match the float64 dense baseline at
  atol/rtol=2e-4. Signed ridge-limit checks run both dtypes and both backends.
  Outputs, determinants and evidence preserve the requested dtype and scoped
  x64/matmul settings preserve process defaults.
- Real PCG iteration limits, loose recursive relative/absolute tolerances,
  nonfinite current/trial states, actual-objective rejection at a known kink,
  high-damping tiny steps, stalling and outer limits retain the common explicit
  failure semantics and no evidence. Failed/rejected attempts cannot replace
  accepted fields. Invalid controls raise configuration errors.

The [retained comparison script](../../.scratch/potential-correction/experiments/iterative_operator_comparison.py)
reuses these maintained public-fit fixtures and writes the
[compact numerical record](../../.scratch/potential-correction/research/operator-iterative-comparison.json):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python \
  .scratch/potential-correction/experiments/iterative_operator_comparison.py \
  --output .scratch/potential-correction/research/operator-iterative-comparison.json
```

The retained float64 run reports:

| Case | data/source/potential | accepted updates | final objective | max source error | max potential error | max nonlinear image error | final true relative residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bilinear | 16/4/33 | 16 | 0.446379594762 | 4.44e-16 | 7.29e-17 | 3.33e-16 | 6.69e-14 |
| Changed cells, nsub=2 | 84/25/97 | 6 | 0.412900227946 | 8.49e-15 | 7.87e-16 | 7.77e-16 | 9.98e-13 |

The second case changes 11 source-cell assignments and reduces objective
241.1011285841 to 0.412900227946. Its final physical scaled gradient is
6.03e-9. Float32 errors against the bilinear float64 dense reference are
3.10e-5 source, 4.23e-6 potential, 2.35e-7 nonlinear image and 1.42e-7
objective; its final true relative residual is 8.48e-7. These observed errors
are below the declared assertions, without requiring reproducible iteration
counts across GPU reduction orders.

Matching fields do not make the determinant approximation exact. Dense versus
operator local log evidence is 8.97916 versus -0.50449 in the bilinear case,
and 174.07276 versus 166.34698 in the changed-cell case. These small default
partitions are numerical checks, not a calibrated nonlinear evidence-search
layout. Future strength selection must validate its fixed partition at the
actual geometry and initialization; the single-step calibration does not
establish that property for a nonlinear workflow.

## Observable allocation

The isolated allocation probe runs without a dense reference or old dense
buffers. Source96 and 33 local plus 9216 isolated support-only potential
unknowns yield 18,465 total coefficients with nine observations, cap8 and
2309 blocks. The iterative branch starts at zero source with nonzero data,
then evaluates two current points and performs real PCG steps/retries. It
intentionally ends at `iteration_limit`, with no evidence; this is a storage
stress case, not a converged astrophysical fit.

On CUDA device 0, two current-point iterations used nine completed linear
solves. Peak JAX buffers were **14,563,840 bytes**, retained buffers
**3,673,088 bytes**, and the largest retained array **147,776 elements**.
There were no full source/potential/joint squares, source-by-potential arrays,
or full joint design shapes in the live-array scan. The unchanged single-step
probe measured 13,099,264 peak bytes and the same largest retained array.
One dense float64 source square alone would require 679,477,248 bytes.
Allocation-path inspection confirms bounded basis/image panels inside scans.
JAX statistics exclude CUDA context/driver memory; they do not measure total
process memory. This check establishes bounded preparation/execution storage,
without a scientific timing or speedup claim.

## Runnable example and verification

Use `tinylens_gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`PYTHONPATH=/tmp/tinylens-pc-09` for this worktree. Specific core files run from
`tests/`; example and benchmark suites run in separate processes from their
own directories.

```bash
pytest -q test_potential_correction_iterative_operator.py \
  test_potential_correction_iterative.py test_potential_correction_dense.py \
  test_potential_correction_operator.py test_potential_correction_rendering.py \
  test_potential_correction_inference.py
```

Result: **108 passed in 95.23s**. The initial test-first comparison failed on
missing operator controls; the final-gradient diagnostic test failed on missing
metadata; operator example selection and the iterative allocation CLI each
failed before their implementations. An intermediate core run passed all ten
then-current operator cases but exposed seven test-plumbing `NameError`s in
an accidentally edited pre-existing validation test; that edit was corrected.
No production tolerance or stopping rule was loosened. A new diagnostic test
was also corrected to respect existing objective/update convergence instead
of incorrectly imposing a universal small-gradient gate.

Separate processes:

- From `examples/tests/`, `pytest -q test_potential_correction_iterative_example.py`:
  **2 passed in 8.95s**.
- From `benchmarks/`, `pytest -q -s test_potential_correction_memory.py`:
  **2 passed in 11.10s** (single-step and iterative isolated probes).

The standalone small example is:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python \
  examples/potential_correction/mock/iterative/demos/fixed_strength.py \
  --backend operator --block-size 16 --pcg-max-iter 800 --pcg-rtol 1e-12
```

It returns the same structured result and scalar-history semantics as dense
iteration, and prints a summary with actual solver status and final gradient.
Its representable nodal Pseudo-Jaffe mock differs from the Gaussian-potential
core fixture above. These small fixed-strength checks do not transfer the
single-step calibration to nonlinear evidence optimization or establish
scientific recovery accuracy at another geometry or prior.

The unchanged source–lens-light operator contract also passes from `tests/`:

```bash
pytest -q test_pixelized_operator.py \
  -k 'pcg or schur or joint_curvature or ragged_source_blocks or preconditioner_is_spd'
```

Result: **23 passed, 173 deselected in 34.95s**. No CG internals or existing
Schur code were changed. The environment is Python 3.11.14, JAX 0.7.1,
NumPy 2.4.6 on the available NVIDIA GeForce RTX 5080.

The standalone command also completed successfully: six accepted updates,
objective 2.4989361108943 -> 0.0460958021481, chi2=0.000139474156,
local block log evidence=160.23787048, final gradient=1.00e-9. The last PCG
step used 141 iterations and true relative residual 3.06e-12. Public default
controls and failure semantics were unchanged throughout this work.

The final separate example compatibility command from `examples/tests/`,
`pytest -q test_potential_correction_iterative_example.py
 test_potential_correction_operator_example.py
 test_potential_correction_scientific.py`, passed **6 in 30.65s**. These include
small pre-existing scientific assembly cases, not new production scientific
fits. `black --check` on all nine changed Python files and `git diff --check`
also pass.
