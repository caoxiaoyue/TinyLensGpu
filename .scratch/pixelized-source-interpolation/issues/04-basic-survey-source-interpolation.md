# 04: Carry source interpolation through basic and survey workflows

**What to build:** Users of the remaining maintained basic, survey, and lens-light pre-subtraction workflows can select, save, and resume the same source interpolation using the established staged workflow rules.

**Blocked by:** 03 — Save and resume a staged fit with its source interpolation.

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Apply the shared mode and provenance contract to the remaining basic, no-lens-light, adaptive-template, Galan, SLACS, Euclid, and pre-subtraction workflows, including their dense and operator variants where applicable.
- [ ] Confirm configured mode reaches the source model before fitting and remains consistent across all pixelized stages and source-template transfers. Persist it in stage settings and exported source packages.
- [ ] Verify known older bilinear checkpoints restore correctly, recorded Hermite checkpoints reconstruct the same model, and conflicting explicit choices stop before saved results are reused.
- [ ] Exercise representative default and explicit-mode round trips through the existing basic and survey regression entry points. Preserve each workflow's fitted-grid and source-template validation rules.

