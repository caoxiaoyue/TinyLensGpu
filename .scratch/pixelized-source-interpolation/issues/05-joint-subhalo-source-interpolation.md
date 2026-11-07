# 05: Carry source interpolation through joint and subhalo workflows

**What to build:** Users of the maintained joint semi-linear inversion and subhalo example workflows can carry a chosen source interpolation through their staged fits, including source-template transfer and saved evidence reuse.

**Blocked by:** 03 — Save and resume a staged fit with its source interpolation.

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] The maintained dense and operator joint examples and the J0946 and B1938 subhalo example pipelines pass one source interpolation through their pixelized stages and persist it in the relevant packages and checkpoints.
- [ ] Exercise the shared default, explicit Hermite, and known legacy bilinear behavior through representative joint and subhalo workflows, including restoration before and after a source package has been produced.
- [ ] Reject conflicting interpolation before reuse of cached source templates, stage fits, or saved evidence, and verify that a compatible continuation retains its existing fitted grid and adaptive-template behavior.
- [ ] Verify that the joint forward and diagnostic results preserve the existing lens-light intensity contract. The active J0946 research handoff and archived replay are covered by ticket 08.

