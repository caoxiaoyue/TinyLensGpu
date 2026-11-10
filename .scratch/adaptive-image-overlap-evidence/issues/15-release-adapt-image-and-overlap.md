# 15: Release the adapt-image interface and default Overlap together

**What to build:** The library release exposes reference-image adaptation on both ordinary pixelized backends and uses Overlap by default in operator fits. Maintained callers and the guide use the released configuration, with a clear path to regenerate artifacts for the changed objective.

**Blocked by:** [07 — Migrate SLACS, no-lens-light and Galán workflows](07-migrate-survey-and-source-workflows.md), [08 — Migrate lens-light-subtracted pixelized examples](08-migrate-lens-light-subtracted-examples.md), [09 — Migrate J0946 paired subhalo evidence](09-j0946-paired-subhalo-evidence.md), [10 — Export new J0946 macro handoffs with the actual source scale](10-j0946-macro-handoffs.md), [13 — Qualify Overlap GPU workspace and chunk width](13-overlap-gpu-qualification.md), [14 — Qualify the Euclid M1 through M3 workflow on real VIS data](14-euclid-vis-qualification.md).

**Status:** ready-for-agent
**State:** claimed

**Spec:** [Parent specification](../spec.md)

- [ ] Remove `fixed_reg_template` from both constructors after migrating the in-scope callers. Public input tests confirm acceptance of `adapt_image` and rejection of the removed fixed-template and fixed-scale keywords.
- [ ] The omitted operator evidence choice agrees with explicit `overlap` in source-only and joint fits. Explicit `block` reproduces the matched calculation. Apply the qualified Hermite offset and automatic chunk policy.
- [ ] Verify the maintained Euclid stage regression and in-scope migration regressions against the released defaults. Check the real VIS artifacts and retained qualification reports against the released objective and rerun affected validation if it changed.
- [ ] Update the guide with the reference-image inputs, scoring defaults, coverage and batching behavior, and migration and recovery rules. Link the matched mock, optimizer, GPU and VIS qualification records.
- [ ] Run the required core, example and benchmark checks in separate processes under the repository rules. Use targeted in-scope regressions and apply the B1938+666 scope exclusion.

## Comments

- Claimed on `codex/overlap-15-release`.
