# 01: Restore native JAX array truthiness

**What to build:** Users can pass optional arrays and run supported modeling and inference workflows while JAX retains its native boolean behavior throughout the process.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Correct confirmed non-scalar optional-array boolean selections before removing the compatibility patch; valid PSF arrays and omitted PSFs retain their intended behavior.
- [ ] Importing TinyLensGpu no longer patches JAX array boolean methods, including the concrete array implementation; scalar and ambiguous non-scalar boolean behavior remains native.
- [ ] Preserve required Caskade backend initialization and existing public imports and model calls.
- [ ] Verify real supported model and inference dependency paths as well as isolated-process import behavior; do not treat static search alone as compatibility evidence.
- [ ] Extend appropriate existing regressions for the repaired optional-array paths and add the focused native-truthiness regression at the import seam.
- [ ] Document the compatibility change without adding a switch that reinstalls the global patch; record focused verification results.
