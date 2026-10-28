# Dense single-step strength optimization: ticket 05

`TinyLensGpu.Inference` exports `PotentialCorrectionEvidence` and
`optimize_potential_correction`, with structured evaluation and search results.
They consume a prepared reconstruction problem's public `fit` boundary.

```python
from TinyLensGpu.Inference import (
    PotentialCorrectionEvidence,
    optimize_potential_correction,
)

# Natural logarithms, in source then potential-fourth-difference order.
objective = PotentialCorrectionEvidence(problem)
negative_log_evidence = objective([0.0, -2.0])
fit = objective.last_reconstruction
records = objective.evaluations

search = optimize_potential_correction(
    problem,
    log_bounds=[[-2.0, 2.0], [-4.0, 2.0]],
    maxfun=80,
    len_tol=1e-3,
)
best_fit = search.reconstruction
```

The problem retains its frozen gradient and geometry and its independent
positive amplitude-prior strength. Every representable candidate calls `fit`
afresh; no fitted fields are passed to another candidate. Both float64 and
float32 problems keep their requested numerical dtype without changing the
surrounding JAX precision configuration. Log coordinates and exponentiation
use host float64, avoiding the generic optimizer's float32 parameter path.

The callable returns negative evidence for a successful fit and positive
infinity for a failed numerical candidate. Scalar evaluation records retain
the attempted strengths, dtype/backend, status and failure reason. Finite log
coordinates whose exponentials cannot be represented in the problem dtype
are rejected as `strength_out_of_range`. Malformed coordinates, invalid
search bounds/controls, or validation errors raised by `fit` remain
configuration exceptions. The wrapper does not catch and disguise them.
`fit_options` forwards backend/algorithm-specific controls unchanged but
cannot override searched strengths, the fixed amplitude prior, or `backend`.
This is the extension boundary for tickets 07 and 10; no operator calibration
or iterative inference is established by this ticket.

Search results preserve all scalar candidate records, the best successful
public reconstruction, explicit bounds and controls, and SciPy's separate
termination status. They retain only the best full reconstruction rather than
all candidate fields. The callable additionally exposes the last result for
inspection. A budget-limited search may have `has_valid_fit=True` and
`success=False`; `optimizer_success` reports DIRECT's termination. No valid
candidate leaves the reconstruction, strengths and selected logs absent.
SciPy treats `maxfun` as an approximate budget and can exceed it during a
rectangle subdivision; `nfev` records the actual count.

DIRECT samples rectangle centers. Boundary reporting therefore uses explicit
`boundary_rtol`, default 0.01, as a fraction of each configured natural-log
interval. `boundary_sides` identifies the nearby edge and
`boundary_distances` records the normalized nearest-edge distance. A flag
reports proximity to the configured boundary, not proof of a global optimum.
The tolerance must be finite and satisfy `0 <= boundary_rtol < 0.5`.

## Independent numerical comparison

The retained experiment is
[`verify_dense_strength_search.py`](../experiments/verify_dense_strength_search.py),
with the small result in
[`dense-strength-optimization-summary.json`](dense-strength-optimization-summary.json).
It reuses the core regression's independent host construction of bilinear
source weights, physical potential differences, complete D4 windows and the
Gaussian data covariance. It does not call the library's matrix or evidence
builders for the independent calculation.

Environment: Python 3.11.14, JAX 0.7.1, SciPy 1.17.1, RTX 5080. The problem has
a 3x3 image at 0.1 spacing, a 2x2 zero-order source in `[-0.3,0.3]^2`, zero
macro shear, fixed gradient `(1.2,-0.8)`, noise from 0.2 to 0.4 and fixed
amplitude strength 0.4. These are numerical check settings without a weak-prior
calibration claim. Source bounds are `[0.05,30]`, D4 bounds `[0.001,30]` in
physical strength units; DIRECT receives their natural logarithms.

With `maxfun=180`, `maxiter=100`, `len_tol=1e-3`, `vol_tol=1e-12`, DIRECT used
195 evaluations. It selected source strength 0.40896133176233385 and D4
strength 29.78862977155536. Its negative evidence was 11.03025782614915,
versus 11.030011085504283 at the best point of the independent 31x31 scan.
The independent calculation at the selected point differs by 6.03e-13.
The scan comparison allows 1e-3 objective error; the independent evidence
comparison allows 2e-10. The potential strength is flagged at the upper edge
(relative distance 0.0006858711). DIRECT ended at the approximate evaluation
budget, so the result preserves the valid fit while reporting unsuccessful
optimizer termination.

Reproduce from the repository root:

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false python .scratch/potential-correction/experiments/verify_dense_strength_search.py
```

## Runnable small mock

`examples/potential_correction/tune_single_step.py` uses the fixed true macro
and analytic mock gradient on a 12x12 Pseudo-Jaffe image, 5x5 source in
`[-0.6,0.6]^2`, 0.5 potential spacing, nsub=1, and a 3x3 PSF. It requires the
amplitude strength and both natural-log bounds explicitly. It returns the
search object and saves `tuning.json` plus `reconstruction.npz` from public
result fields when a successful fit exists.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/tune_single_step.py \
  --lambda-amplitude 0.4 --log-source-bounds -2 2 \
  --log-potential-bounds -4 2 --maxfun 80 \
  --output output/ticket05-dense-tuning
```

This command returned strengths 0.13570708323176198 and 0.01839116718971317,
with fixed amplitude strength 0.4. Both searched strengths were flagged near
the lower bounds and DIRECT reported budget termination. The artifact files
are generated outputs and can be recreated by the command. These settings
exercise orchestration and persistence; ticket 07 owns prior/bound sensitivity
and backend calibration, and ticket 11 owns scientific recovery/export work.

## Regression verification

Core tests exercise independent bounded scanning, repeated/reordered real
fits, fixed amplitude/gradient/geometry, requested precision in opposite JAX
contexts, boundary flags, invalid configuration, numerical solve failure,
strength under/overflow, and the all-invalid result. The example test checks
saved bounds, candidate statuses and both public renderings in a separate
process. Exact final verification commands and outcomes are recorded in the
[ticket](../issues/05-optimize-dense-single-step-strengths.md).
