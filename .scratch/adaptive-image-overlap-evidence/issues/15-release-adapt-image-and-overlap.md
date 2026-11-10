# 15: Release the adapt-image interface and default Overlap together

**What to build:** The library release exposes reference-image adaptation on both ordinary pixelized backends and uses Overlap by default in operator fits. Maintained callers and the guide use the released configuration, with a clear path to regenerate artifacts for the changed objective.

**Blocked by:** [07 — Migrate SLACS, no-lens-light and Galán workflows](07-migrate-survey-and-source-workflows.md), [08 — Migrate lens-light-subtracted pixelized examples](08-migrate-lens-light-subtracted-examples.md), [09 — Migrate J0946 paired subhalo evidence](09-j0946-paired-subhalo-evidence.md), [10 — Export new J0946 macro handoffs with the actual source scale](10-j0946-macro-handoffs.md), [13 — Qualify Overlap GPU workspace and chunk width](13-overlap-gpu-qualification.md), [14 — Qualify the Euclid M1 through M3 workflow on real VIS data](14-euclid-vis-qualification.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Remove `fixed_reg_template` from both constructors after migrating the in-scope callers. Public input tests confirm acceptance of `adapt_image` and rejection of the removed fixed-template and fixed-scale keywords.
- [x] The omitted operator evidence choice agrees with explicit `overlap` in source-only and joint fits. Explicit `block` reproduces the matched calculation. Apply the qualified Hermite offset and automatic chunk policy.
- [x] Verify the maintained Euclid stage regression and in-scope migration regressions against the released defaults. Check the real VIS artifacts and retained qualification reports against the released objective and rerun affected validation if it changed.
- [x] Update the guide with the reference-image inputs, scoring defaults, coverage and batching behavior, and migration and recovery rules. Link the matched mock, optimizer, GPU and VIS qualification records.
- [x] Run the required core, example and benchmark checks in separate processes under the repository rules. Use targeted in-scope regressions and apply the B1938+666 scope exclusion.

## Comments

- Claimed on `codex/overlap-15-release`.
- Released in `da29187`, with CUDA comparison assertions corrected in `9e972de`. The [guide](../../../docs/guides/guide.md) documents both public constructors, Overlap and Block, coverage and batching, and artifact recovery.
- CUDA core verification in `tinylens_gpu` passed 1,025 cases with 5 deselected in the broad run. The corrected `test_overlap_evidence.py::test_joint_operator_default_matches_explicit_overlap` passed in a focused rerun. The separately run in-scope examples passed 250 cases in the broad run. The 22-case affected rerun passed and covered all 14 superseded assertions. B1938+666 was excluded.
- The separately run benchmark suite passed 14 cases. The transient `test_performance.py::TestScalability::test_component_count_scaling` failure passed in an exact rerun on the idle device. The run logs are under `/tmp/tinylens-overlap.EIfsZz/15-*.log`.
- The release VIS check with omitted flags preserved scores and pickle hashes. Four released mock GPU replay checks agreed within `1.55e-11` and confirmed bounded workspace. See the [mock replay](../../../docs/validation/adaptive-image-overlap-euclid-mock.md), [optimizer qualification](../../../docs/validation/adaptive-image-overlap-m3-optimizer.md), [GPU qualification](../../../docs/validation/adaptive-image-overlap-gpu.md), and [real VIS qualification](../../../docs/validation/adaptive-image-overlap-euclid-vis.md).
