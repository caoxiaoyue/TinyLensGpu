# 10: Optimize strengths around complete iterative reconstructions

**What to build:** A researcher can select source and potential smoothing strengths for either iterative backend through an outer DIRECT search whose candidates each perform a complete, fixed-strength inner reconstruction.

**Blocked by:** 09 — Operator iterative reconstruction at fixed strengths.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] Both dense and operator iterative fits are callable by the bounded two-dimensional log-strength optimizer.
- [x] Every candidate resets to the same truth-derived source initialization and zero correction; no source, potential, or damping state leaks from the preceding candidate.
- [x] Strengths remain fixed throughout each inner iteration and the amplitude-prior strength remains fixed throughout the outer search.
- [x] Candidate evidence is evaluated only for successfully converged inner results using the proper backend-specific final local curvature.
- [x] Failed, nonfinite, stalled, or iteration-limited inner runs use the optimizer's invalid-objective convention and retain their termination reasons.
- [x] Result metadata identifies local-linearization and determinant approximations independently, as well as search bounds, boundary optima, precision, and selected strengths.
- [x] Repeated and reordered candidate evaluations are reproducible at the declared numerical tolerance.
- [x] A small controlled real-problem comparison checks the nested optimum against an independent scan and checks the effect of backend approximation on recovery.
- [x] Tests cover initial-state reset, fixed inner strengths, invalid candidates, explicit precision, and successful-result handoff to diagnostics.
- [x] A runnable small example demonstrates the complete outer-search/inner-reconstruction workflow without a posterior-sampling requirement.

## Scope and handoff

Reuse the established optimizer and result boundaries. Do not update strengths
inside LM, reinterpret failed terminal iterates as successful fits, or change
the search to a path-dependent warm-start procedure.

## Comments

- Claimed by `/root/iterative_exploration` on `codex/pc-10` in
  `/tmp/tinylens-pc-10`, based exactly on integration
  `2675c9b3107b47d3d2c7ac2af4ef2dad28dcbc72`. Ticket 09 is resolved and merged.
  Implementation follows the approved public fit/evidence/result and artifact
  seams, with independent small nonlinear validation; no scientific-200 work.

- Implemented in `057e84a`, with independent source-cell locality certification
  in `6729301`. Latest integration `efbb3ba` was merged as `3a3c332` without
  conflicts; that integration change is progress documentation only. The only
  production change appends scalar candidate method/termination/solver
  diagnostics to the established inference boundary. The numerical controller,
  prior, geometry, signed fields and public defaults are unchanged.
- The [validation report](../../../docs/validation/potential-correction-iterative-inference.md)
  and [versioned scan/search summary](../research/iterative-strength-search.json)
  retain the 16-data/37-unknown real nonlinear case, independent cold-start
  coordinate/stencil fits, 5×5 scan, actual two-backend DIRECT runs, edge
  refinements, complete versus block evidence, source/raw-potential/core-map
  recovery, full indices/rule, and rejected 19/18 partition. The calibrated
  20/17 layout passes the unchanged 10% recovery-error guard, with maximum
  selected error increase 0.801% and full-evidence regret 0.00160426. Runtime
  uses frozen indices without a dense Jacobian or candidate-dependent layout.
- Both final searches use fixed amplitude precision 20 and explicit source/D4
  bounds [0.1,2]/[0.001,1]. They retain successful inner fits, but DIRECT reports
  status 1 with 115 evaluations for requested approximate maxfun=100 and an
  upper-D4 boundary. No optimizer-success or interior-optimum claim is made.
- Verification in `tinylens_gpu`, preallocation disabled, explicit worktree
  `PYTHONPATH`, specific files from their own directories, separate processes:
  from `tests/`, `pytest -q test_potential_correction_iterative_inference.py
  test_potential_correction_inference.py` passed **31 in 59.89s**; from
  `examples/tests/`, `pytest -q test_potential_correction_iterative_tuning.py
  test_potential_correction_tuning.py` passed **4 in 13.64s**. The retained
  `experiments/iterative_strength_search.py` command also passed all numerical
  assertions. Six-file `black --check` and `git diff --check` pass.
- The standalone `tune_iterative.py --backend operator --maxfun 5` CLI completed
  with seven evaluations/status 1 and a converged retained public result. Its
  compact numerical diagnostics preserve the common result, scalar search
  records, absent linear products, history and safe arrays; all-invalid runs
  retain reasons and remove stale success products without refitting.
- An intermediate compatibility run was **75 passed, 1 failed**: unchanged
  `test_operator_float32_matches_float64_coupled_reference_without_changing_defaults`
  ended at `rejection_limit`. Untouched integration passed its isolated node;
  this branch failed an isolated node with approximately 1.27e-4 source error.
  Numerical files are identical across the checkouts. The report retains this
  unresolved float32 stopping difference for precision-stage ticket 13; no
  assertion/default change or passing-combined-suite claim hides it. All 18
  new nested checks, including controlled explicit-dtype/reset/failure cases,
  pass. Scientific-size recovery/exports/performance remain tickets 12–13.

