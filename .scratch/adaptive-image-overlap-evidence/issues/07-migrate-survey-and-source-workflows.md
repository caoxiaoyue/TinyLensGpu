# 07: Migrate SLACS, no-lens-light and Galán workflows

**What to build:** Modelers can run the maintained SLACS, no-lens-light and Galán pixelized workflows with reference-image adaptation. Each workflow exports the reference at its existing producer stage and consumes it in the adaptive stages.

**Blocked by:** [06 — Run Euclid M1 through M3 with a shared reference arc](06-euclid-adapt-image-stages.md).

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Migrate SLACS, both no-lens-light variants and both Galán variants from the fixed-template constructor input to `adapt_image`.
- [ ] Use the PSF-convolved native source component from the workflow's existing M0 or M1 producer. Keep its fitted grid, mask, mass point and interpolation with the reference.
- [ ] Run focused stage and export regressions for each workflow family. Verify the consumers use the exported reference and recover matching artifacts using the shared provenance checks.
- [ ] Old adaptive stage outputs with a different prior or score require new fits. A compatible upstream source package can generate the reference through the established recovery path.
