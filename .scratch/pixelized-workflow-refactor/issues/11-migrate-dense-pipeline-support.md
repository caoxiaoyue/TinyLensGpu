# 11: Migrate dense pipelines to shared run support

**What to build:** Dense pipelines use reusable configuration, sampling, persistence, source-transfer, and plotting support while keeping their existing scientific stages and legacy artifacts usable.

**Blocked by:** 10: Transfer a source template and render it through shared support

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Migrate affected dense pipelines, including SLACS, the pre-subtraction family, dense simultaneous joint MGE, and the dense matrix variant located among operator examples.
- [x] Replace private tuple-returning sampling consumption with StagePosterior while retaining each case's prior semantics, parameter ordering, and numerical sampling settings.
- [x] Preserve intentional sampler differences, including cases whose batch size follows the live-point setting rather than another case's fixed value.
- [x] Preserve legacy cache filenames, formats, fields without a stage schema where applicable, numerical representations, and recovery behavior despite the unified in-memory posterior type.
- [x] Use explicit configuration and shared source-transfer and plotting support wherever applicable while retaining visible model construction, stage order, and existing launch methods.
- [x] Verify representative dense stage transitions, saved-artifact compatibility, restoration, and plot data; keep core numerical behavior unchanged and record focused results.

## Comments

- Claimed by `/root/dense_support` on `codex/pwr-11-dense-support`, based on integration commit `83f6002`.

- Migrated the six dense pipeline implementations: SLACS, Bspline/Sersic/MGE
  pre-subtraction, simultaneous joint MGE, and the dense matrix variant under
  operator/no_lens_light. Sampling returns `StagePosterior`; callers consume
  named attributes while the existing stage function returns and schema-free
  `stage_*.pkl` payloads remain compatible. Sampling batch policies remain
  SLACS=100, pre-subtraction/joint=n_live, and matrix=M2's explicit batch of 1.
- Also migrated six maintained dense simple sampling scripts and both inversion
  demo renderers. Simple CSV/gzip filenames and fields remain unchanged;
  stabilized-weight summary semantics are retained. Rendering uses shared
  panels and reconstruction validity checks, preserving case layouts/overlays.
- Focused verification, conda `tinylens_gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `MPLBACKEND=Agg`, `PYTHONPATH=/tmp/tinylens-pwr-11`, from `examples/tests`:
  `pytest test_dense_shared_support.py -q` -> **9 passed** (24.19 s).
  Covers all six real stage-A assembly/persistence paths with controlled external
  sampling, legacy restoration, SLACS A-to-M numerical/rendered-data agreement,
  matrix A-to-M2 batch=1 transition, and a complete simple-script launch with tiny
  FITS data, the existing 40x40 source grid, and CSV/gzip/figure outputs.
- `pytest test_dense_shared_support.py test_joint_fit_plotting.py
  test_source_compatibility.py -q` (before the final two added direct tests)
  -> 48 passed, 31 failed. All 39 source compatibility checks passed; all 31
  failures were stale AST harness namespaces missing newly imported shared
  support names. Central harness maintenance is coordinated under ticket 15;
  the directly imported stage tests pass. No numerical core was changed.
- `git diff --check` passed. Long full-data sampling is intentionally outside
  acceptance scope.
