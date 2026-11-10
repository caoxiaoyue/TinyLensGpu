# 09: Migrate J0946 paired subhalo evidence

**What to build:** A J0946 researcher can run the M4 and M5 comparison with reference-image adaptation and verify that the baseline and candidate use compatible scoring configurations before interpreting their Bayes factor.

**Blocked by:** [06 — Run Euclid M1 through M3 with a shared reference arc](06-euclid-adapt-image-stages.md).

**Status:** ready-for-agent
**State:** claimed

**Spec:** [Parent specification](../spec.md)

- [ ] Both M4 and M5 construct their adaptive fits using `adapt_image`, with a common reference identity, fixed grid, preparation rule and calibrated strengths.
- [ ] The comparison fingerprints the relevant observation, interpolation, adaptation and evidence settings and records the actual scoring method for each fit.
- [ ] A saved M4 can serve as a comparison denominator when its provenance matches. Detect an old fixed-template M4 or a different score and require the applicable matched refit.
- [ ] Focused J0946 regressions exercise both matched comparison and mismatched recovery through the maintained entry point.

## Comments

- Claimed on `codex/overlap-09-j0946-evidence`.
