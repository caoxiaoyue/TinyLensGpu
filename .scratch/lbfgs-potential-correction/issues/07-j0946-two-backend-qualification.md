# 07: Qualify the J0946 two-backend refit

**What to build:** A researcher can qualify the operator J0946 reconstruction against the validated dense reconstruction at the same fixed strengths and inspect both numerical and resource measurements.

**Blocked by:** 05 — Operator mock fitting and result round trip, 06 — Verify bounded memory for operator fits.

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] The maintained J0946 entry point runs the operator fit with the same archived observation, geometry, actual start, whitening reference, and strengths as the dense refit from ticket 03. Retain the exact operator partition and controls used for qualification.
- [ ] Both fits qualify at the common \(10^{-3}\) GN stationarity threshold, including the operator true-residual certificate. Check matching potential topology and derivative-valid domains before comparing fields.
- [ ] The operator objective differs from the dense objective by at most \(0.01\). The maximum absolute \(\delta\kappa\) difference over the full valid domain is at most \(0.005\). The positive peak positions differ by at most one potential grid spacing.
- [ ] Retain both local GN evidence values, determinant methods, and the measured difference. Use the small explicit principal-block check from ticket 04 as the operator evidence verification.
- [ ] The validation report links the input identity and fit products, records the comparison gates and measured differences, and reports stage timings and peak device memory for both backends. Include the commands and measurement configuration needed to repeat the qualification.
