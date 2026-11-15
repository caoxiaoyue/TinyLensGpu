# Potential-correction mandatory review fixes

Review ownership: `codex/pc-review-fix`, worktree
`/tmp/tinylens-pc-review-fix`, based on integration commit
`e14ea508f53eec3ebe4c6f57f887aa6716a725fc`. The implementation agent owns all
three frozen findings. Status: complete. Both original reviewers independently
cleared the merged fixes at `58b4a9b` with zero outstanding findings. All 14
implementation tickets and the parent spec are resolved.

The independently frozen reports are retained verbatim and separately:

- [Standards axis](../../.scratch/potential-correction/research/review/frozen-standards.md):
  zero hard violations, two nonblocking duplication judgments.
- [Spec axis](../../.scratch/potential-correction/research/review/frozen-spec.md):
  one P2 stale-success-artifact finding.
- [Original public probe evidence](../../.scratch/potential-correction/research/review/frozen-spec-probes.json).

## Spec finding: rerun artifacts

Public seams: `mock/single_step/demos/operator.py:main()` and `mock/single_step/tune.py:run_tuning()`.
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

Resolution: `shared/recovery_diagnostics.py` owns truth sampling/domains, field
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

The [compact post-review record](../../.scratch/potential-correction/research/review/post-review-verification.json)
retains exact commands, working directories, source/log/artifact hashes,
environment, red/green observations and bounded performance measurements.
Broad checks ran against committed runtime and tests at
`7ea87e4e6f999cd9ea8cbd09c16f6e8549aa8111`. Subsequent changes only update this
record and the feature index; the recorded runtime hashes match the final code.

All commands used `tinylens_gpu`, disabled JAX preallocation and ran the core,
example and benchmark suites in separate serial processes. Specific tests ran
from their file directories.

| Post-review scope | Passed | Time |
| --- | ---: | ---: |
| All potential-correction core files, including both slow 64-pixel refinement/budget cases; curvature, source precision/prior, legacy pixelized PCG/Schur and solver/operator autodiff | 434 | 423.81 s |
| All potential-correction example files and source compatibility; artifact cycles, calibration, scientific consumers/plots and tuning | 73 | 122.19 s |
| Isolated bounded-memory, principal preconditioner application and workflow checks | 5 | 37.37 s |

The core run also checks the native bounded-map path and the legacy-signature
fallback, complete amplitude/prior normalization, current-J cache behavior,
precision and process defaults. The example run repeats the real numerical
artifact failure cycles after the refactor. No numerical solve was mocked and
no error guard or tolerance was relaxed. Black defaults check all 12 changed
Python files; `git diff --check` and local review/fixture links pass.

Both large allocation probes use 9,216 source and 9,249 potential coefficients
(18,465 joint), nine data pixels and blocks of eight. Their largest live array
has 147,776 elements; allocator peaks are 13,099,264 and 14,567,424 bytes for
single-step and iterative preparation/execution respectively. The iterative
probe exercises current steps and retains its expected iteration-limit failure
with null evidence. The physical 64-pixel/image worker with a 65-by-65 source
has 2,252 data, 4,225 source and 592 support coefficients, no prohibited live
square/cross shapes, and a 228,778,752-byte cumulative allocator peak. Its
operator fit succeeds. The small 37-coefficient timing worker also succeeds;
its 4-by-4 image coincides with the source-square shape, so that small worker
does not serve as the no-square-shape check. The synthetic inverse-factor
application comparison meets its existing numerical and timing assertions.

These are representative post-review checks at the stated snapshot. The
historical 45-worker performance campaign and 200-pixel scientific campaigns
were not rerun; their timing, recovery, amplitude-sensitivity, float32 failure
and determinant-approximation records remain attached to their original
configurations and source snapshots. Existing 957-case broad core, 67-case
compatibility, 286-case example and 14-case benchmark evidence remains
historical. No whole-workflow speedup or new scientific acceptance claim follows
from these checks.

Large generated output and logs are retained under the integration worktree's
ignored `.scratch/potential-correction/output/review-fix/`. Both frozen reports
were checked byte-for-byte against their supplied originals. The review fixes
preserved approved requirements, ticket states, generic CG/Schur code and
independent numerical references. Final lifecycle closure follows the
independent verification below.

## Independent final verification

The reviewers separately inspected the fix diff at `58b4a9b`, keeping their
axes independent. Their frozen reports precede the final lifecycle-only
closure, so references there to the parent being open describe that snapshot.

| Axis | Original findings | Resolved | Outstanding |
| --- | --- | ---: | ---: |
| [Standards](../../.scratch/potential-correction/research/review/verified-standards.md) | Zero hard violations; two duplication judgments | 2 | 0 |
| [Spec](../../.scratch/potential-correction/research/review/verified-spec.md) | One P2 stale-success-artifact bug | 1 | 0 |

Standards found no new violation or meaningful smell. Spec independently ran
four artifact/consumer example tests (14.45 s) and five core determinant and
current-mapping tests (20.14 s); all passed, with no new spec finding.
The [final review receipt](../../.scratch/potential-correction/research/review/final-review-verification.json)
retains both report hashes and exact counts. No numerical source changed
after the verified implementation snapshot.

The [cleanup record](../../.scratch/potential-correction/research/implementation-worktree-cleanup.json)
records removal of 17 clean, integrated task-owned implementation/baseline/fix
worktrees. Local branches and the integration worktree remain. Two historical
ignored tuning outputs were copied and SHA256-verified in the root evidence
directory before removal; all scientific, performance and verification output
in the root checkout was preserved. User worktrees were retained.
