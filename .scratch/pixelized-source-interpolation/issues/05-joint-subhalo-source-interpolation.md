# 05: Carry source interpolation through joint and subhalo workflows

**What to build:** Users of the maintained joint semi-linear inversion examples and the J0946 subhalo workflow can carry a chosen source interpolation through their staged fits, including source-template transfer and saved evidence reuse.

**Blocked by:** 03 — Save and resume a staged fit with its source interpolation.

**Status:** ready-for-agent
**State:** claimed

**Spec:** [Parent specification](../spec.md)

B1938+666 is outside the implementation and acceptance scope. In shared example test modules, select only the applicable joint and J0946 cases.

- [ ] The maintained dense and operator joint examples and the J0946 subhalo example pipeline pass one source interpolation through their pixelized stages and persist it in the relevant packages and checkpoints.
- [ ] Exercise the shared default, explicit Hermite, and known legacy bilinear behavior through representative joint workflows and the J0946 workflow, including restoration before and after a source package has been produced.
- [ ] Reject conflicting interpolation before reuse of cached source templates, stage fits, or saved evidence, and verify that a compatible continuation retains its existing fitted grid and adaptive-template behavior.
- [ ] Verify that the joint forward and diagnostic results preserve the existing lens-light intensity contract. The active J0946 research handoff and archived replay are covered by ticket 08.

## Comments

Scope clarified on 2026-11-07. This ticket covers the maintained joint examples and J0946. This clarification supersedes the earlier reference to B1938.

Claimed for implementation on `codex/psi-05-joint-j0946`.
