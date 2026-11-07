# 03: Save and resume a staged fit with its source interpolation

**What to build:** A representative maintained staged workflow, using the operator SLACS source-template transition, can run with an explicit Hermite choice, save it, and resume with the same forward model. Reusable source-package and checkpoint rules make the workflow a reference for the remaining migrations.

**Blocked by:** 02 — Reconstruct the same Hermite source with the operator backend.

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] New source packages record the actual `source_interpolation` and `source_boundary`, retaining the fitted-node `source_bbox`, axes, and single grid size. The support is derivable from this metadata and the compiled inversion result keeps its existing structure.
- [ ] Checkpoint configuration records the interpolation before a source package exists and carries it through the source-template transition to the next stage's model.
- [ ] Verify a fresh default run, an explicit Hermite run, a Hermite save-and-resume round trip, and a known legacy checkpoint without the fields that resumes as bilinear.
- [ ] A conflicting explicit mode fails before any completed stage or its evidence is reused. A fresh run can explicitly use old node intensities as initialization.
- [ ] The guide and a maintained runnable example show how to select Hermite explicitly, how resume chooses the mode, and how to interpret nodal non-negativity with signed weights. Exercise the workflow through its existing configuration, export, and restore entry points.

