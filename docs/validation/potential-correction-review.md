# Potential-correction mandatory review fixes

Review ownership: `codex/pc-review-fix`, worktree
`/tmp/tinylens-pc-review-fix`, based on integration commit
`e14ea508f53eec3ebe4c6f57f887aa6716a725fc`. The implementation agent owns all
three frozen findings. Status: fixes implemented; broader verification pending.
The 14 implementation tickets remain resolved; the parent spec remains open
for the coordinator's final review and closure.

The independently frozen reports are retained verbatim and separately:

- [Standards axis](../../.scratch/potential-correction/research/review/frozen-standards.md):
  zero hard violations, two nonblocking duplication judgments.
- [Spec axis](../../.scratch/potential-correction/research/review/frozen-spec.md):
  one P2 stale-success-artifact finding.
- [Original public probe evidence](../../.scratch/potential-correction/research/review/frozen-spec-probes.json).

## Spec finding: rerun artifacts

Public seams: `single_step_operator.main()` and `tune_single_step.run_tuning()`.
The approved spec's testing decisions already confirm saved example artifacts
as test surfaces. Maintained regressions use real numerical fits through these
entry points, including PCG's one-iteration limit and an all-invalid search
over representable source strengths. Each exercises success, failure, and
success in the same output directory. Cleanup is restricted to known files
owned by each exporter; unrelated user files are preserved. Optimizer budget
exhaustion remains separate from validity of the selected reconstruction.

Resolution: fixed failures remove `configuration.json` and `reconstruction.npz`;
successful fixed fits remove `failure.json`. All-invalid tuning removes its
`reconstruction.npz`, retaining current `tuning.json` candidate reasons. Both
maintained public cycle tests first failed on the stale-file assertions, then
the two affected example files passed **4 tests** (13.94 s). The one-iteration
failure remains `pcg_iteration_limit`; invalid finite strength bounds produce
`nonfinite_system`. No solve is mocked.

## Standards finding 1: shared operator invariants

Share invariant padded partitions, per-block unit priors, equilibrated
factorization/validity, and PCG control validation through narrow internal
helpers. Keep fixed single-step preparation and current iterative Jacobian
preparation/cache policy separate. Preserve coupled actions, damping,
one-correction total budget, all-node amplitude normalization, identity padding,
inverse-factor application, precision scope, and independent test references.

Resolution: `_potential_correction_operator_common.py` owns these invariant
helpers. Unit-prior and basis helpers are invoked inside each original bounded
scan. The fixed adapter retains its combined data/prior preparation; the
iterative adapter still refreshes current-J data separately from unit priors.
Both use the original iterative factor-validity checks (finite matrix, factor,
positive diagonal and finite log determinant), without adding jitter. The first
operator checkpoint passed **36 tests** (51.83 s), including coupled dense
agreement, complete-prior determinants, failure status, precision and caching.

## Standards finding 2: shared consumer diagnostics

Share consumer truth-domain and recovery-metric calculations while preserving
the calibration and scientific export schemas. Retain the 5% macro-source
illumination threshold, all/illuminated core domains, the nominal configured
subhalo disk in controls, raw and aligned potential comparisons, signed fields,
and derivative NaN support. Independent analytic truth and numerical references
remain separate.

Resolution: `_recovery_diagnostics.py` owns truth sampling/domains, field
metrics, affine calculations and source-box ray preparation. Export adapters
retain their prior schemas and residual definitions. The
[saved public fixtures](../../examples/tests/fixtures/potential_correction_consumers/README.md)
and [capture script](../../.scratch/potential-correction/experiments/capture_review_consumers.py)
anchor parity to the original consumers at `e14ea508`, rather than deriving
expected metrics from the new helpers. A small paired 12-pixel calibration
uses nominal `b_sub=0.4`, so its local disk exceeds `2h`: both cases retain
radius `1.058300524425836` arcsec and domain counts 37/15/18 for all/illuminated/local
core. This is a diagnostic fixture, not a new scientific acceptance setting.
The focused consumer suite passed **4 tests** (29.75 s), checking both layouts,
saved public-result restoration, original schemas/values, signed fields, NaN
support and unchanged image/evidence products.

## Verification and provenance

New commands, outcomes, source hashes and their scope will be recorded here and
in a compact research summary. Existing 957-case broad core, 67-case compatibility,
286-case example and 14-case benchmark evidence remains historical at its
original source/configuration; it is not a new measurement of these fixes.
Large generated output and logs are retained under the integration worktree's
ignored `.scratch/potential-correction/output/review-fix/`.
