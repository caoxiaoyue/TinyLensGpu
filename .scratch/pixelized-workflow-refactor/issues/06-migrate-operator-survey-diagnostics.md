# 06: Migrate operator survey and lens-subtracted diagnostics

**What to build:** SLACS, Euclid, and lens-light pre-subtraction workflows produce their existing plots and source exports from the common public numerical interfaces.

**Blocked by:** 03: Deliver operator diagnostics through a complete example

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Migrate operator SLACS and Euclid pixelized stages and the standard, Sersic, MGE, and split variants of the pre-subtraction workflows.
- [ ] Obtain reconstructions, regularization displays, enabled position diagnostics, and the existing source-only effective-parameter approximation through public results rather than private solver assembly.
- [ ] Preserve each case's masks, crop or zoom behavior, color choices, overlays, position constraints, and source-package conventions.
- [ ] Reject explicitly unconverged or nonfinite reconstruction before formal plotting or export; unavailable optional quantities alone do not indicate failure.
- [ ] Preserve stage scientific choices and numerical results, including parameter ordering, source-grid configuration, and regularization settings.
- [ ] Extend appropriate source-export and example regression coverage for the materially different display and source-package variants, recording focused verification results.
