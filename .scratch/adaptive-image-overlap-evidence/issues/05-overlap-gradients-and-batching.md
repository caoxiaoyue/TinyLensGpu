# 05: Differentiate and batch the full Overlap score

**What to build:** A modeler can differentiate the full fixed-grid Overlap score through the public array-returning model and submit proposal batches to the normal likelihood builder with bounded work.

**Blocked by:** [04 — Score joint semi-linear inversions with Overlap](04-joint-overlap-evidence.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Compare Float64 automatic derivatives of the reconstruction and full score against independent numerical derivatives for valid moving-mass and dynamic-regularization cases. Include source-only and joint evidence with the Hermite adaptive projection.
- [x] Check the PCG route and the locally stable PNPG active-set regime. Control or record the position-penalty regime in derivative comparisons, and verify invalid gradients for the applicable primary and auxiliary solve failures.
- [x] Scalar and batched results agree for identical proposals, including batches whose length is not a multiple of the explicit inner chunk width. Keep the sampler's input and output batch shape and honor `vectorized_chunk_size`.
- [x] Demonstrate bounded forward and reverse scoring workspace. Use a conservative initial automatic chunk policy that can be qualified against same-host GPU measurements later.

## Comments

- Claimed on `codex/overlap-05-grad-batch`.
- Implemented in `3ff5f74`. From `tests/` in `tinylens_gpu` on CPU, the three `test_hermite_overlap_derivatives_follow_mass_and_adaptive_strengths` cases and four compiled workspace and sampler-chunk cases in `test_overlap_evidence.py` all passed. The primary and auxiliary invalid-gradient regression also passed. `pytest -q test_multiband_parametric.py -k make_likelihood` passed 10 cases.
