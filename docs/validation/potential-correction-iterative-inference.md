# Nested iterative strength selection

Ticket 10 reuses `PotentialCorrectionEvidence` and bounded SciPy DIRECT for
both `IterativePotentialCorrection` backends. It adds scalar candidate
diagnostics and a small reproducible nonlinear validation. No reconstruction
equation, controller, solver default, prior, geometry or interpolation changes.
The implementation starts at integration `2675c9b`; implementation commits are
`057e84a` and `6729301` (independent source-cell locality certification).

## Public contract

```python
from TinyLensGpu.Inference import (
    PotentialCorrectionEvidence, optimize_potential_correction,
)

objective = PotentialCorrectionEvidence(
    problem, backend="operator",
    fit_options={"block_size": 20, "block_indices": blocks,
                 "pcg_rtol": 1e-12, "max_iter": 240},
)
negative_log_evidence = objective([log_lambda_source, log_lambda_potential])

search = optimize_potential_correction(
    problem, backend="operator", log_bounds=explicit_natural_log_bounds,
    fit_options=objective.fit_options,
    maxfun=100, maxiter=100, len_tol=1e-3,
)
result = search.reconstruction  # retained successful public result, or None
```

Each candidate performs a complete inner run from the problem's immutable
truth-derived source nodes and zero potential, with fresh initial damping.
One source/D4 strength pair stays fixed inside that run; the separately
configured amplitude precision and geometry stay fixed throughout the search.
Neither previous fields nor damping nor evidence become the next initializer.
The selected result is handed to diagnostics without a further fit.

Candidate records append optional scalar `algorithm`, `termination_reason`,
`evidence_method`, `determinant_method`, `evidence_is_exact` and `solver_reason`.
The original positional fields remain valid. Nonlinear termination and the
last inner linear-solver reason are separate: a stalled fit can have a
converged linear solve; `linear_solve_failed` can retain `pcg_iteration_limit`
or `linear_residual_exceeds_tolerance`. Before-fit unrepresentable strengths
have `strength_out_of_range` and no invented algorithm/evidence method.
Numerical failures return infinity and cannot supply evidence or best fields;
configuration exceptions still propagate. An all-invalid search has no
reconstruction or selected strengths. Candidate histories/full fields are not
stored for every outer evaluation.

Both successful backends label `evidence_method="local_gauss_newton"` and
`evidence_is_exact=False`. Dense uses `full_cholesky`; operator uses
`principal_block_cholesky`. Evidence comes only from the converged refreshed
nonlinear state and its undamped local curvature. The determinant and local
linearization approximations remain independently visible in each candidate.

## Controlled numerical model and independent reference

The maintained case has 16 observations, four source nodes and 33 potential
unknowns. A 4×4 image, dpix=0.1 arcsec, nsub=1 and a delta PSF use a fixed
zero-shear macro. The 2×2 source box is [-0.5,0.5]^2, with truth nodes
(0.4,1.1,0.2,1.4), equivalently samples of
`0.775+0.95*x+0.05*y+0.5*x*y`. The potential truth is the raw nodal field
`0.003*exp(-((x-.07)^2+(y+.03)^2)/.02)`, without mean subtraction/alignment.
Data are the nonlinear nodal rendering plus `0.002*sin(pixel_index)`, with
nominal sigma 0.07. The source prior is zero-order, amplitude precision is
independently fixed at 20, and all arithmetic validation is float64. This value
is a numerical fixture input; it is not a scientifically calibrated weak prior.

The explicit strength bounds are source [0.1,2] and D4 [0.001,1]. A 5×5 log
scan includes both endpoints. At every point an independent coordinate and
stencil NumPy model assembles the complete prior, starts from the same supplied
source/zero-potential state, and uses SciPy finite-difference least-squares plus
a stationarity polish. No library Jacobian, prior matrix or evidence supplies
its expected fit. Centered image derivatives at steps 1e-5 and 5e-6 agree within
1e-8. Independent scaled stationarity must be below 1e-8. Corrected queries and
both finite-difference probes stay inside the same source cell; the retained
minimum source-cell margin is 0.10149052 arcsec. This validation does
not exercise a nonsmooth optimum or establish global nonlinear uniqueness.

Independent full and matching block log determinants use the complete
amplitude-augmented Q and final J.T Cinv J+Q. The two-dimensional scan increases
with D4 at every sampled source strength, so a separate independent bounded
source-strength refinement on the upper-D4 edge provides a sharper local
reference. This is an independently scanned/refined branch, not a proof of a
global nonlinear integral or an unbounded regularization optimum.

The explicit inner budget is 240 iterations; stopping controls remain public
defaults. DIRECT uses maxfun=100, maxiter=100, len_tol=1e-3. The 0.005
log-evidence budget/resolution guard is separate from fixed-strength numerical
accuracy: fields atol=4e-7/rtol=3e-6, physical objective/penalty atol=2e-12,
same-state independent evidence formula atol=3e-8, independent-MAP evidence
atol=2e-6, and same-strength dense/operator fields/images/objective
atol=2e-8/rtol=2e-7. The independent-MAP evidence allowance accounts for different
terminal field criteria; the actual accepted fields still receive the tighter
formula check. No assertion was loosened after a failure.

## Partition calibration and selected recovery

The initial proposed 19/18 layout split source nodes 2+2 and potential nodes by
row-major halves. It failed the predeclared 10% recovery-error increase guard:
its selected source strength 1.63860 increased raw potential RMSE from 0.0197747
to 0.0237348 (20.0%) and deflection RMSE from 0.0903132 to 0.104519 (15.7%).
Full log evidence at that selected fit was 12.4484739, compared with 12.5357856
at the dense selection. Its finite block evidence did not establish acceptable
strength selection. The rejected layout, strengths, search status and measured
effects remain in the numerical record and reproduction script.

The accepted cap20 layout groups all four *small* source nodes with the 16
potential nodes having strongest initial independent image-response squared
norms; the remaining 17 potential nodes form the second block. Complete frozen
indices and initial scores are retained. Runtime uses those explicit indices,
with no dense Jacobian to select blocks and no candidate-dependent partition.
Neither block contains the entire potential or joint field. All true solves
remain fully coupled; only preconditioning and determinants use the partition.
This calibration applies to this exact tiny model/initialization/bounds.

Actual DIRECT results:

| Quantity | Dense selection | Calibrated operator selection |
| --- | ---: | ---: |
| source strength | 1.24422731 | 1.25966123 |
| D4 strength | 0.99842197 | 0.99842197 |
| independent full local log evidence | 12.53578556 | 12.53564077 |
| independent block local log evidence | 12.77036001 | 12.77047149 |
| source RMSE | 0.20414769 | 0.20458225 |
| raw potential RMSE | 0.01977469 | 0.01993297 |
| core deflection RMSE | 0.09031324 | 0.09089624 |
| core convergence RMSE | 0.40220322 | 0.40370008 |
| image RMSE | 0.00262119 | 0.00264299 |

The independent upper-edge optima are source strength 1.24490944/D4=1 with full
logZ=12.53724503, and source strength 1.25741142/D4=1 with block logZ=12.77230495.
Operator-selected full-evidence regret versus that full reference is 0.00160426;
the full-evidence difference between the two DIRECT-selected fits is 0.00014479.
Raw recovery-error changes are below 1%, and the selected nonlinear-image RMS
difference is below 0.1 nominal noise sigma. Absolute block evidence has an
offset of about 0.235 at the selected fits; matching recovery does not make it
exact or justify transferring the layout to scientific grids.

Across all 25 scan points both backends converged. The largest coefficient
error against independent stationary solutions was 4.75e-8; the largest
same-state evidence-formula error was 5.02e-11, and independent-MAP evidence
error was 4.27e-7. The selected image RMS difference was 2.50e-5, or 0.000358 of
the nominal noise sigma. The largest selected recovery-error increase was
0.801% for raw potential, below the unchanged 10% guard.

The raw potential error exceeds this mock's small injected potential. This
underconstrained numerical case validates coupled MAP/evidence-search behavior
and its approximation effect, not accurate scientific perturbation recovery,
detectability or mass calibration. No affine alignment or clipping hides the
low-mode error. Derivative errors use only the valid core.

Both searches have 115 completed evaluations and 13 DIRECT iterations, status 1,
`optimizer_success=False` with the exact maxfun message retained. SciPy's
maxfun is approximate and was exceeded; the requested100 is not reported as
an exact cap. All 115 inner candidates converged in each final recorded search.
The D4 selection is flagged at the upper log-range boundary (fractional
distance 0.000228624). Successful retained fits, incomplete optimizer
termination, and the configured boundary are distinct facts.

## Reproduction, examples and verification

Use `tinylens_gpu` with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and an explicit
checkout `PYTHONPATH`. Run the retained scan, actual searches and rejected
partition comparison from the repository root:

```bash
python .scratch/potential-correction/experiments/iterative_strength_search.py \
  --output .scratch/potential-correction/research/iterative-strength-search.json
```

The [versioned summary](../../.scratch/potential-correction/research/iterative-strength-search.json)
contains bounds, controls, all scan points, selected strengths and recovery,
independent references, both determinant formulas, rejected calibration,
actual statuses, source hashes and software versions. This experiment uses
Python 3.11.14, JAX 0.7.1, NumPy 2.4.6, SciPy 1.17.1 and CUDA device 0 on RTX 5080.
It makes no timing/performance claim.

The runnable numerical example uses the same fixed truth source, geometry and
calibrated frozen layout:

```bash
python examples/potential_correction/tune_iterative.py --backend operator \
  --output output/potential_correction_iterative_tuning
```

`main(...)` returns the search object with the common public reconstruction
at `.reconstruction`. `search.json` preserves candidate scalar diagnostics,
controls/bounds, fixed amplitude, budget and boundary status. `nested_result.json`
and safe `nested_result.npz` contain the retained public nonlinear fields,
history, method labels, geometry and supplied nodal truth. This compact numeric
format is specific to the example, separate from the scientific result format.
No extra fit is made for persistence. All-invalid runs retain `failure.json`
and remove known stale nested success products; nonfinite JSON scalars use
null. Scientific Pseudo-Jaffe/control plots and exports remain ticket 12.

TDD checks first failed on absent scalar candidate fields (2 cases), then
passed. The new example tests first failed on its absent module (3 cases),
then passed. The initial partition comparison failed its recovery guard;
calibration corrected the partition with the same guard. Repeated/reordered
and cross-backend candidates compare with fresh cold problems and independently
check initial objective, damping, complete final prior, fixed data/noise/source
box/potential topology. Real iteration limits, high-damping stalling/rejection,
PCG iteration failure, finite-input overflow and unrepresentable strengths
remain invalid. Explicit float64/float32 ridge candidates preserve process
precision and matmul defaults; scientific float32 calibration is separate.

An intermediate combined core command passed 75 and failed one unchanged
float32 operator test in 104.47s:
`test_operator_float32_matches_float64_coupled_reference_without_changing_defaults`.
It ended at `rejection_limit`; rounded source values differed from the float64
reference by about 1.27e-4, with the last inner relative residual 1.10e-7.
An isolated unchanged integration check passed in 8.38s; an isolated ticket 10
check failed in 8.05s with the same termination and source error, relative
residual 9.35e-8. All numerical source files are identical across the checkouts;
the only production edit is scalar inference metadata. This unresolved
float32 stopping difference is handed to ticket 13, with no solver/default or
assertion edits and no claim that the combined suite passed.

Final focused checks, with specific files run from their own directories and
core/example suites in separate processes:

```bash
# tests/
pytest -q test_potential_correction_iterative_inference.py \
  test_potential_correction_inference.py
# examples/tests/ (separate process)
pytest -q test_potential_correction_iterative_tuning.py \
  test_potential_correction_tuning.py
```

Results: **31 passed in 59.89s**, then **4 passed in 13.64s**. The complete
retained reproduction also passed all independent scan/selection/formula/
recovery assertions after source-cell margin certification. The latest
integration `efbb3ba` was merged as `3a3c332`; it changes only the progress
README. `black --check` on six new/changed Python files and `git diff --check`
pass. No scientific-size fits or performance benchmarks were run.

The standalone operator CLI also completed with `--maxfun 5`: seven completed
evaluations, status 1, a converged retained fit and complete scalar/search and
nonlinear-array products. Its requested small budget was kept separate from
the 100-evaluation calibration above.
