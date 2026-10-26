# 13: Migrate operator survey workflows to shared run support

**What to build:** SLACS, Euclid, and lens-light pre-subtraction workflows consistently use the shared example support while remaining recognizable, runnable, and resumable in their existing forms.

**Blocked by:** 10: Transfer a source template and render it through shared support

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Complete shared-support migration for the remaining operator SLACS stages, Euclid, and standard, Sersic, MGE, and split pre-subtraction variants.
- [x] Preserve case-specific explicit configuration, stage model construction, priors, parameter transfer, sampling settings, and input/output location behavior.
- [x] Preserve target-specific launch entry points, stage cache formats, filenames, and recovery behavior while replacing repeated sampling and storage helpers.
- [x] Reuse source-package validation and transfer without altering mask, bounding-box, template, or regularization semantics.
- [x] Retain layout, zoom, mask display, colors, and overlays through shared plotting support and case-specific render choices.
- [x] Verify representative target entry points, stage recovery and transition, artifact compatibility, and output data, including differences between survey families; record focused results.

## Comments

- Claimed by Codex on `integration/pixelized-workflow-refactor`; migrating operator survey support (ticket 13).

- Completed shared sampling/storage/source/plot migration for Euclid and the
  standard, Sersic, MGE and split pre-subtraction workflows; SLACS already
  uses the same helpers from tickets 09–10. Split MGE now consumes
  `StagePosterior` internally and continues writing legacy schema-less payloads.
  Survey batch size stays 100; metadata, paths, explicit stage assembly, source
  grids, templates, masks, crops, labels and overlays retain case behavior.
- Verification (conda `tinylens_gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `MPLBACKEND=Agg`, from `examples/tests`):
  `pytest test_survey_shared_support.py test_source_compatibility.py -q`
  → 53 passed. Direct imports exercise six stage-A sampling/save/recovery paths
  and real SLACS/Euclid/MGE export→adaptive-stage→five-panel data products.
  Existing survey checks additionally compare masks, colors, limits, labels,
  position overlays and failure propagation. No full-data sampling was needed.
