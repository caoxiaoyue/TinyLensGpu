## Context

The current adaptive-regularization implementation can build a scale map by ray-tracing image-plane seed pixels through the current mass model, accumulating a source-plane brightness proxy, smoothing it, normalizing it, and applying the adaptive scale formula. `adaptive_reg_freeze` exists to stop that scale map from drifting during final mass inference, but the workflow remains tied to an initially selected mass-model mapping.

For the `model_adpt_reg.py` demo, we want to replace that older mass-dependent adaptive scale path with a statistically cleaner empirical-Bayes flow: first reconstruct a fixed source-plane brightness template with uniform regularization (`S0`), then derive all later adaptive regularization from that fixed source template and its fixed source grid. This affects the PhysicalModel source configuration, the ObservationModel operator likelihood, and the example pipeline orchestration.

## Goals / Non-Goals

**Goals:**
- Add a stage-m0 that reconstructs `S0` using fixed stage-A mass parameters and uniform regularization.
- Store `S0` with the source grid metadata needed to preserve pixel-to-coordinate meaning across m0/m1/m2.
- Let operator likelihoods use a fixed source bbox and an externally supplied adaptive scale map.
- Build scale from `S0` without additional Gaussian smoothing.
- Retire the older mass-dependent adaptive scale path and `freeze_scale()` workflow globally for adaptive regularization.
- Preserve JAX/GPU performance by treating fixed scale and bbox as concrete arrays/scalars captured before JIT tracing.

**Non-Goals:**
- Dynamically moving the source bbox in the new `S0` path.
- Changing the regularization scale formula or finite-difference scale application.
- Supporting GP-style regularization kernels in the operator backend.
- Preserving backwards compatibility for callers that rely on dynamic seed-ray scale construction or `freeze_scale()`.

## Decisions

### D1: Represent `S0` as a package, not just an array

Stage-m0 will save a package containing:

- `source_pixels`: flat source solution with shape `(nx * ny,)`
- `source_image`: optional 2D view with shape `(ny, nx)` for diagnostics
- `source_bbox`: `(xmin, xmax, ymin, ymax)`
- `source_x_axis`, `source_y_axis`
- `nx`, `ny`
- `lambda_best`, `log_lambda_best`
- stage-A mass medians used to construct it

**Rationale:** A scale array alone is ambiguous unless the source grid is also fixed. Keeping bbox and axes with `S0` makes the empirical-Bayes conditioning explicit and makes plot/replot/cache validation straightforward.

**Alternative considered:** Save only `scale_map`. Rejected because it prevents later verification that m1/m2 are using the same source grid as m0.

### D2: Fix the source bbox/grid for m1 and m2

The new path will use the stage-m0 source bbox in m1 and m2. `PixelizedImageProbModelOperator._get_bbox()` should support an optional fixed bbox; when configured, it still computes the sub-grid betas needed for the lensing operator but returns the fixed bbox instead of inferring one from seed betas.

**Rationale:** If bbox varies with mass parameters, the same source pixel index maps to different physical coordinates and `S0[i]` no longer aligns with `scale[i]`. Fixing the grid gives a fixed prior structure for final mass inference.

**Alternative considered:** Keep bbox dynamic while using fixed `S0` scale values. Rejected because it weakens the statistical interpretation of a fixed source-template prior.

### D3: Build scale directly from `S0`

The source-template path will compute:

```text
S0_pos = max(S0, 0)
b_norm = S0_pos / max(mean(S0_pos), eps)
scale = floor + (1 - floor) / (1 + alpha * b_norm)
```

No extra Gaussian smoothing is applied.

**Rationale:** `S0` is already the MAP source under a regularization prior, so it is a denoised source estimate. Additional smoothing would erase structure twice. The global-mean normalization and scale formula remain consistent with the existing adaptive-regularization spec.

**Alternative considered:** Reuse `adaptive_reg_smooth_sigma` on `S0`. Rejected for the first implementation to keep the behavior minimal and avoid over-smoothing.

### D4: Put fixed bbox/scale support at the ObservationModel boundary

`PixelizedImageProbModelOperator` should accept fixed adaptive data, either directly as constructor arguments or through a small helper called by the example before JIT compilation:

```python
fixed_source_bbox=(xmin, xmax, ymin, ymax)
fixed_reg_scale=scale
```

When adaptive regularization is enabled, the likelihood should consume `fixed_reg_scale` directly rather than rebuilding scale from current seed-ray betas. The older dynamic scale construction and `adaptive_reg_freeze` branch should be removed. Backends that do not yet accept fixed template inputs should fail clearly when adaptive regularization is requested rather than preserving the old mass-dependent behavior.

**Rationale:** The fixed scale is an inference/evidence-model input, not a physical light component. Keeping it on the likelihood avoids turning a large source-grid array into a Caskade/ParamU model parameter.

**Alternative considered:** Store `S0` or `scale_map` on `PixelizedSourceModel`. Acceptable if implementation ergonomics require it, but less clean because the physical model layer should not own empirical-Bayes cache state.

### D5: Keep stage-m1's lambda optimization semantics

Stage-m1 should still grid-search `log_lambda_reg`, but now with the fixed `S0`-derived adaptive scale. Stage-m2 should continue to fix its source regularization strength to the stage-m1 optimum while sampling EPL+shear mass parameters.

**Rationale:** This preserves the existing pipeline's hyperparameter handoff while only changing how the adaptive spatial weighting is seeded.

## Risks / Trade-offs

- **Initial mass bias can propagate through `S0`.** Mitigation: store diagnostics for `S0`, keep `adaptive_reg_floor` configurable, and leave bbox padding as an explicit constant for later tuning.
- **Fixed bbox may exclude source-plane rays for later mass proposals.** Mitigation: expose bbox padding in the demo and include residual/source diagnostics for m2.
- **Fixed arrays must be available before JIT tracing.** Mitigation: construct/load `S0` and scale before calling `make_likelihood`; treat them as concrete JAX arrays captured by the compiled likelihood.
- **Removing the old path can break external callers.** Mitigation: update tests and examples to use the `S0` path, and fail clearly when adaptive regularization is requested without a fixed scale source.
- **Changing demo cache format can break stale outputs.** Mitigation: validate required `stage_m0.pkl` fields and recompute m0 when missing.

## Migration Plan

1. Add operator support for fixed source bbox and fixed adaptive scale.
2. Add source-template scale builder tests.
3. Add stage-m0 to the demo and write `stage_m0.pkl`.
4. Update m1/m2 construction and replot paths to require/load `S0`.
5. Remove or bypass the older seed-ray adaptive scale and `freeze_scale()` code paths in the operator workflow.
6. Keep old stage-m1/m2 output names, but invalidate or recompute when required `S0` metadata is absent.
7. Remove the old seed-ray/freeze adaptive implementation from the dense backend and physical source configuration, or fail clearly where fixed-template adaptive inputs are not yet supported.

Rollback is straightforward while the code is under development: restore the seed-ray adaptive scale branch and remove the fixed `S0` arguments from m1/m2 construction.

## Open Questions

- What default bbox padding should stage-m0 use for this dataset? Initial implementation can keep the current inferred bbox and make padding a visible constant.
- Should fixed source bbox support also be added to the dense `PixelizedImageProbModel` now, or only after the operator demo path is stable?
