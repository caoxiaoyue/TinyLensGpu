# 14: Qualify the Euclid M1 through M3 workflow on real VIS data

**What to build:** The maintained Euclid workflow completes M1, M2 and M3 on real VIS observations and produces separately attributable stage artifacts and diagnostics.

**Blocked by:** [06 — Run Euclid M1 through M3 with a shared reference arc](06-euclid-adapt-image-stages.md).

**Status:** ready-for-agent
**State:** claimed

**Spec:** [Parent specification](../spec.md)

- [ ] Use verified real VIS inputs and the maintained stage definitions with default bilinear source interpolation and explicit Overlap scoring.
- [ ] Complete M1 reference export, M2 calibration and M3 mass fitting with valid stage products. Rebuild upstream producers when old inputs lack the required provenance.
- [ ] Verify the two adaptive stages use the same reference identity and fixed source grid, and that M3 records the selected M2 strengths.
- [ ] Retain the run command, input identities, actual scoring and batching settings, fit diagnostics and outputs separately from the archived mock qualification.

## Comments

- Claimed on `codex/overlap-14-vis-qualification`.
