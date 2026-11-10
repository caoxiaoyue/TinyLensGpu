# 04: Score joint semi-linear inversions with Overlap

**What to build:** A modeler fitting a pixelized source and unit-amplitude lens-light bases together can use Overlap evidence. The score accounts for the physical cross curvature and reports an auxiliary evidence failure separately from a failed primary reconstruction.

**Blocked by:** [03 — Score source-only operator fits with Overlap](03-source-only-overlap-evidence.md).

**Status:** ready-for-agent
**State:** claimed

**Spec:** [Parent specification](../spec.md)

- [ ] Use the established Block–Schur preconditioner for the primary solve. The Overlap score uses the physical cross block, the full source curvature and the augmented-residual QR Schur calculation.
- [ ] Independent dense joint comparisons verify the Schur determinant and its lens-light prior normalization on the same solved coefficients. Exercise PCG and PNPG under their existing score convention.
- [ ] One primary observation-data solve supplies the image, source, lens-light intensities and score. A reconstruction-only call skips the score-only determinants and auxiliary Schur solves.
- [ ] A controlled auxiliary failure returns the sampling penalty and exposes the score-specific cause through diagnostics while retaining a valid primary reconstruction. A primary-solve failure follows its established output convention.

## Comments

- Claimed on `codex/overlap-04-joint-overlap`.
