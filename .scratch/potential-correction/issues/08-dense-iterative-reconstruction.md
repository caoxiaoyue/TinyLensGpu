# 08: Dense iterative reconstruction at fixed strengths

**What to build:** A caller can jointly update the pixelized source and potential with a fixed-strength dense Gauss–Newton/LM reconstruction, receiving the final nonlinear image, explicit termination status, iteration history, and local-linearization evidence.

**Blocked by:** 07 — Operator tuning and single-step backend calibration.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] A separate iterative entry point starts from truth-derived source-node intensities and zero potential correction with fixed source/potential geometry and macro parameters.
- [x] Current pixelized-source spatial gradients analytically differentiate the existing bilinear brightness formula and retain its cell/boundary/out-of-bounds conventions.
- [x] Each required linearization updates corrected ray positions, source mapping, and current source gradients; the initial truth gradient is not reused as the ongoing gradient.
- [x] Source and potential increments solve the coupled damped Gauss–Newton/LM problem, including the negative prior-gradient term for the accumulated fields.
- [x] Acceptance is based on the actual nonlinear data misfit plus the accumulated source/potential prior, and accepted updates reduce that objective within justified numerical tolerances.
- [x] Damping, objective-change, scaled-update, gradient, and iteration-limit controls are explicit and recorded.
- [x] Convergence, iteration-limit termination, repeated rejection/stalling, failed linear solves, and nonfinite states produce distinguishable outcomes.
- [x] Successful final evidence uses complete dense determinants of the local Gauss–Newton system without LM damping and is labeled as a local-linearization approximation.
- [x] The common result includes the final nonlinear image, residuals, valid-domain maps, and useful accepted/rejected iteration history.
- [x] Python manages control flow while JAX performs expensive numerical work; large systems are not solved with NumPy.
- [x] Tests cover within-cell source gradients, objective acceptance, cumulative-prior behavior, cache refresh, damping-independent final evidence, and failure paths.
- [x] A small real fixed-strength reconstruction demonstrates iterative recovery independently of outer optimization or full scientific plotting.

## Scope and handoff

The dependency on ticket 07 is the explicitly approved stage gate. This ticket
does not repeatedly apply and sum the single-step solution. Operator iterations
are ticket 09 and nested hyperparameter search is ticket 10.

## Comments

- 2026-10-29: Claimed by the ticket-08 implementer on branch `codex/pc-08`,
  worktree `/tmp/tinylens-pc-08`, from integration `35d0318`. Ticket 07 is
  resolved with its calibration report and retained summary. The approved
  spec already confirms the public result/operator testing seams; development
  uses focused red/green numerical slices. Operator iteration stays in 09.


- 2026-10-29: Implemented on `codex/pc-08` in
  `d1ab91f856e35dcdb335df976c7de4dd8b670d9d`. The independent public
  `IterativePotentialCorrection` reuses shared observation/prior/result
  preparation, exact current bilinear derivatives and one Python LM
  controller. The dense numerical adapter is the sole adapter in this ticket;
  its initialize/prepare/solve/measures/logdets seam is ready for ticket 09.
  Initial source nodes and zero potential reset for every fit. Accepted steps
  reduce the actual accumulated-prior nonlinear objective. Failed/limited/
  stalled fits retain diagnostics and no evidence. Successful local-GN
  evidence refreshes the final Jacobian and excludes all LM damping.
- Verification in `tinylens_gpu`, with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  and `PYTHONPATH=/tmp/tinylens-pc-08`:
  - From `tests/`: `pytest -q test_potential_correction_iterative.py
    test_potential_correction_dense.py test_potential_correction_operator.py
    test_potential_correction_rendering.py test_pixelized_source_utils.py
    test_regularization.py test_operator_autodiff.py test_solver_autodiff.py
    test_curvature_operator.py`: 194 passed/one pre-existing variable float32
    joint-adaptive PCG finite-difference case failed. The exact node passed
    on untouched `35d0318`; its baseline variability is retained in the
    earlier operator record. No existing assertion or solver control was
    relaxed. The brightness mapping's JAX primitive sequence was explicitly
    identical to the untouched baseline after shared cell extraction.
  - Final focused run from `tests/`: `pytest -q test_operator_autodiff.py
    test_potential_correction_iterative.py`: 33 passed in 85.39s, including
    all 21 new iterative cases and the earlier variable case.
  - From `tests/`: `pytest -q test_potential_correction_inference.py`:
    13 passed in 13.48s.
  - Separate process from `examples/tests/`: `pytest -q
    test_potential_correction_iterative_example.py
    test_potential_correction_operator_example.py
    test_potential_correction_calibration.py`: four passed in 37.65s.
- The small runnable `examples/potential_correction/iterate_fixed_strength.py`
  uses a representable nodal source/Pseudo-Jaffe mock, without outer search.
  At 84 data/25 source/97 potential unknowns, float64, fixed strengths
  (1e-4,100,100), it converged in six accepted updates, objective
  2.4989361109 -> 0.04609580215. This does not establish analytic-image
  scientific recovery or mass/detection calibration.
- Full numerical policy, input configuration, tolerances, failure coverage,
  limitations and commands are recorded in
  [dense-iteration validation](../../../docs/validation/potential-correction-dense-iteration.md).
  The earlier float32 ridge fixture was corrected to an explicit strong
  potential-prior limit after finding a real lower coupled objective under
  its weak prior; production defaults were unchanged. The retained audit is
  [dense-iterative-ridge-limit.json](../research/dense-iterative-ridge-limit.json).
  Unresolved bilinear kinks may honestly stall or reach rejection/damping
  limits; they receive no evidence. Operator iteration, outer search and full
  scientific plotting/export remain in later tickets.
- Final integration check: merged integration `329cdb8` (including ticket 11)
  in `d0be226`, without conflicts. From `examples/tests/`, in the same
  documented environment: `pytest -q test_potential_correction_scientific.py
  test_potential_correction_iterative_example.py`: four passed in 25.80s.
  This verifies the merged scientific result/export consumers and the small
  dense iteration together; no production 200×200 run was performed here.
