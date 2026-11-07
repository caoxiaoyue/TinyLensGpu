# 07: Generate and reuse scientific partitions for Hermite sources

**What to build:** The maintained single-step and iterative scientific examples and their calibration flow can run with an explicit Hermite source and prepare, reuse, or explicitly specify a compatible block partition with recorded provenance.

**Blocked by:** 06 — Run single-step and iterative potential correction with Hermite.

**Status:** ready-for-agent
**State:** claimed

**Spec:** [Parent specification](../spec.md)

- [ ] Both example flows carry their configured interpolation into reconstruction and exported metadata. Omitted settings and known historical bilinear calibrations retain the bilinear behavior.
- [ ] Partition preparation supports sixteen source neighbors with four potential neighbors and validates the common ray count independently of the stencil widths.
- [ ] Source tile votes sum the magnitudes of individual weights. A test with cancelling signed contributions verifies the intended incidence. Forward and transpose evaluations keep the signed interpolation weights.
- [ ] Automatic partition provenance checks include geometry, interpolation, and vote rule. Verify reuse for matching cases and regeneration on mismatch, including the treatment of known old bilinear partitions.
- [ ] A valid explicitly supplied fixed partition can be used for a controlled comparison and its origin is recorded. Exercise small example runs and their result records for both potential-correction methods.

## Comments

Claimed for implementation on `codex/psi-07-scientific-partitions`.
