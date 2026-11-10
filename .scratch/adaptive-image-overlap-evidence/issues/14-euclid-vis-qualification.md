# 14: Qualify the Euclid M1 through M3 workflow on real VIS data

**What to build:** The maintained Euclid workflow completes M1, M2 and M3 on real VIS observations and produces separately attributable stage artifacts and diagnostics.

**Blocked by:** [06 — Run Euclid M1 through M3 with a shared reference arc](06-euclid-adapt-image-stages.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Use verified real VIS inputs and the maintained stage definitions with default bilinear source interpolation and explicit Overlap scoring.
- [x] Complete M1 reference export, M2 calibration and M3 mass fitting with valid stage products. Rebuild upstream producers when old inputs lack the required provenance.
- [x] Verify the two adaptive stages use the same reference identity and fixed source grid, and that M3 records the selected M2 strengths.
- [x] Retain the run command, input identities, actual scoring and batching settings, fit diagnostics and outputs separately from the archived mock qualification.

## Comments

- Claimed on `codex/overlap-14-vis-qualification`.
- Implemented in `d7211ee`. The maintained Euclid CLI completed fresh M0–M3 with explicit Overlap on CUDA in `tinylens_gpu`. Input identities matched the original VIS files and the re-rendered M1 arc agreed to a maximum absolute difference of `2.79e-9`. M2 and M3 used matching reference and fixed grid identities, with M3 applying the selected M2 strengths at Float32 precision. All reporting-point diagnostics were valid. A matching `--skip-done` run reused the stage products with identical pickle hashes. See the [validation report](../../../docs/validation/adaptive-image-overlap-euclid-vis.md) and [retained results](../experiments/euclid_vis_results.json).
