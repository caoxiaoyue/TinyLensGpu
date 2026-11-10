# 06: Run Euclid M1 through M3 with a shared reference arc

**What to build:** The maintained Euclid workflow exports its reference arc from M1, recalibrates regularization in M2 and uses that calibration in M3. It can resume a matching fit or regenerate the reference from a compatible historical M1.

**Blocked by:** [05 — Differentiate and batch the full Overlap score](05-overlap-gradients-and-batching.md).

**Status:** ready-for-agent
**State:** claimed

**Spec:** [Parent specification](../spec.md)

- [ ] From a valid M1 inversion, export its PSF-convolved native source component as `adapt_image` alongside the distinct source-plane raster, pixels, fitted grid and parent model point.
- [ ] M2 fixes the M1 mass and fits lambda and rho. M3 fixes those calibrated strengths and uses the same reference arc and grid for changing mass proposals. Fitting and plotting consume the same recorded objective.
- [ ] Record actual evidence and solver settings at every operator-scored stage, including uniform stages. Fingerprint reference, observation, mask, PSF, sampling, grid, interpolation, adaptation and upstream stage identity as applicable.
- [ ] Verify matching resume and detection of changed provenance. A verified historical M1 can produce a reference through the component renderer. A missing or unverifiable source package requires rebuilding its producer, and old adaptive M2 and M3 outputs require new fits.
- [ ] A regenerable example regression exercises M1 export, M2 calibration assembly, M3 consumption and both matched and incompatible recovery. The reference export and provenance behavior are reusable by the other maintained workflows.

## Comments

- Claimed on `codex/overlap-06-euclid-stages`.
