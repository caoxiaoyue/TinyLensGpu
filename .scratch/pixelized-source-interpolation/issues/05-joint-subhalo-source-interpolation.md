# 05: Carry source interpolation through joint and subhalo workflows

**What to build:** Users of the maintained joint semi-linear inversion examples and the J0946 subhalo workflow can carry a chosen source interpolation through their staged fits, including source-template transfer and saved evidence reuse.

**Blocked by:** 03 — Save and resume a staged fit with its source interpolation.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

B1938+666 is outside the implementation and acceptance scope. In shared example test modules, select only the applicable joint and J0946 cases.

- [x] The maintained dense and operator joint examples and the J0946 subhalo example pipeline pass one source interpolation through their pixelized stages and persist it in the relevant packages and checkpoints.
- [x] Exercise the shared default, explicit Hermite, and known legacy bilinear behavior through representative joint workflows and the J0946 workflow, including restoration before and after a source package has been produced.
- [x] Reject conflicting interpolation before reuse of cached source templates, stage fits, or saved evidence, and verify that a compatible continuation retains its existing fitted grid and adaptive-template behavior.
- [x] Verify that the joint forward and diagnostic results preserve the existing lens-light intensity contract. The active J0946 research handoff and archived replay are covered by ticket 08.

## Comments

Scope clarified on 2026-11-07. This ticket covers the maintained joint examples and J0946. This clarification supersedes the earlier reference to B1938.

Claimed for implementation on `codex/psi-05-joint-j0946`.

Implemented in `f922e3b`, merged as `9bb0ab7`. From `examples/tests/` in
`tinylens_gpu` with CPU, preallocation disabled and `MPLBACKEND=Agg`,
`pytest -xq` passed 70 tests on both the implementation and merged ticket
checkouts. Selection was `test_joint_fit_plotting.py`, the applicable joint
and J0946 node IDs in `test_joint_shared_support.py`, and the joint and
J0946 diagnostic/export node IDs in `test_operator_diagnostics_plotting.py`.
B1938 cases were excluded by node ID. CLI help checks passed for both joint
drivers and J0946.

Tests cover fresh and resumed dense and operator joint fits, the
unit-amplitude basis and fitted intensity contract, pre-S1 and post-S1
J0946 resume, incompatible M0/M1/M4/M5 state, and M5-only continuation.
The comparison schema preserves compatible legacy bilinear caches and
distinguishes Hermite caches.
