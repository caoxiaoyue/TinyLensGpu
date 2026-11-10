# 07: Migrate SLACS, no-lens-light and Galán workflows

**What to build:** Modelers can run the maintained SLACS, no-lens-light and Galán pixelized workflows with reference-image adaptation. Each workflow exports the reference at its existing producer stage and consumes it in the adaptive stages.

**Blocked by:** [06 — Run Euclid M1 through M3 with a shared reference arc](06-euclid-adapt-image-stages.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Migrate SLACS, both no-lens-light variants and both Galán variants from the fixed-template constructor input to `adapt_image`.
- [x] Use the PSF-convolved native source component from the workflow's existing M0 or M1 producer. Keep its fitted grid, mask, mass point and interpolation with the reference.
- [x] Run focused stage and export regressions for each workflow family. Verify the consumers use the exported reference and recover matching artifacts using the shared provenance checks.
- [x] Old adaptive stage outputs with a different prior or score require new fits. A compatible upstream source package can generate the reference through the established recovery path.

## Comments

- Claimed on `codex/overlap-07-survey-migrations`.
- Implemented in `47d00cd`. From `examples/tests/` in `tinylens_gpu` on CPU, the in-scope selections from `test_basic_shared_support.py`, `test_adaptive_demo_s0_package.py`, `test_survey_shared_support.py`, `test_pixelized_support.py`, `test_pipeline_source_export.py`, and `test_source_compatibility.py` passed 73 cases, with 106 deselected. The final focused `test_basic_shared_support.py` provenance run passed 9 cases. Checks cover matching and stale resume, archived M0 S0 replay and rebuild, tampered arc rejection, and retention of the fixed Galán grid.
