# 01: Restore native JAX array truthiness

**What to build:** Users can pass optional arrays and run supported modeling and inference workflows while JAX retains its native boolean behavior throughout the process.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Correct confirmed non-scalar optional-array boolean selections before removing the compatibility patch; valid PSF arrays and omitted PSFs retain their intended behavior.
- [x] Importing TinyLensGpu no longer patches JAX array boolean methods, including the concrete array implementation; scalar and ambiguous non-scalar boolean behavior remains native.
- [x] Preserve required Caskade backend initialization and existing public imports and model calls.
- [x] Verify real supported model and inference dependency paths as well as isolated-process import behavior; do not treat static search alone as compatibility evidence.
- [x] Extend appropriate existing regressions for the repaired optional-array paths and add the focused native-truthiness regression at the import seam.
- [x] Document the compatibility change without adding a switch that reinstalls the global patch; record focused verification results.

## Comments

- Claimed by Sidekick on branch `pwr/01-truthiness` (worktree /tmp/tinylens-pwr-01).
- Resolved: shim removed from `TinyLensGpu/__init__.py` (CASKADE_BACKEND init kept);
  `psf_kernel or _delta_psf()` callsites in `tests/test_pixelized_inversion.py` now use
  explicit `is not None` selection; subprocess import regression added to
  `tests/test_integration.py`; migration note added to `docs/guides/guide.md`; AGENTS.md updated.
  Verification (conda tinylens_gpu, from `tests/`):
  `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest test_integration.py -k 'truthiness or PSFConvolution' -q` -> 3 passed.
  `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest test_pixelized_inversion.py -k 'non_delta_psf or delta_psf or prior_transformation or vectorized_likelihood or simulator_returns' -q` -> 5 passed.
  New truthiness test confirmed red before shim removal, green after.
