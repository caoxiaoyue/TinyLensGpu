## Context

The current adaptive regularization in `PixelizedImageProbModel` and `PixelizedImageProbModelOperator` computes per-pixel scale factors as `scale = f(brightness × ray-count)`. The ray-count term (how many image pixels map to a source pixel) is proportional to lensing magnification. In the semi-linear inversion framework, the Hessian curvature `MᵀC⁻¹M` already has larger diagonal entries in high-magnification regions, naturally reducing regularization's relative contribution there. The adaptive scale further reduces regularization in those same regions — double counting magnification.

This design refactors the brightness estimation pipeline to produce a pure source-brightness proxy, adds inverse-variance weighting throughout, and standardizes normalization and the scale formula.

## Goals / Non-Goals

**Goals:**
- Eliminate magnification double-counting via a `brightness_only` mode using normalized convolution (`N/C`)
- Upgrade the legacy `brightness_weighted` mode with inverse-variance weighting
- Unify normalization (global mean, no hard thresholds) and scale formula (soft asymptote, everywhere differentiable)
- Add configurable smoothing scale and empirical-Bayes freeze
- Keep `alpha=0` fast-path unchanged (returns `None` for uniform regularization)

**Non-Goals:**
- GP kernel regularization (Matérn, etc.) with adaptive scaling — finite-difference only
- Auto-tuning of `alpha` or `floor`
- Changing block-diagonal preconditioner structure (scale flows through existing `RegData`)

## Decisions

### D1: Dual-mode architecture with shared downstream

```
                        brightness_only          brightness_weighted
                        ═══════════════          ═══════════════════
inv-var weighting        ✓ (1/σ²)               ✓ (1/σ²)
numerator               bright × w / σ²         bright × w / σ²
denominator             w / σ² (C)              none
b_raw                   N_sm / C_sm             smooth(q)
                        ↓                       ↓
                  ┌───── shared downstream ─────┐
                  │ b_norm = b_raw / mean(b_raw)│
                  │ scale = floor+(1-floor)/    │
                  │         (1+α·b_norm)        │
                  │ freeze: optional            │
                  └─────────────────────────────┘
```

**Rationale:** The only difference between modes is whether `C` normalizes the brightness estimate to cancel magnification. Inv-var weighting, normalization, and scale formula are identical concerns. Sharing minimizes code duplication (~80% of the pipeline is common).

**Alternative considered:** Completely separate functions for each mode. Rejected — leads to normalization/formula drift and harder maintenance.

### D2: Inverse-variance weighting for both modes

Image pixel `k` contributes with weight `1/σ_k²` to both `N` (numerator) and `C` (denominator, `brightness_only` only). The noise map is accessed at seed-mask pixels via `self.noise_map.ravel()[self.sim_obj.seed_flat_indices]`.

**Rationale:** Without inv-var weighting, a single very noisy pixel can drive the brightness estimate as strongly as a clean pixel. This is particularly problematic in the `brightness_weighted` mode where there is no `C` denominator to normalize away such artifacts. Making inv-var universal eliminates this class of error.

**Alternative considered:** Inv-var only in `brightness_only` mode. Rejected — `brightness_weighted` without inv-var is vulnerable to noisy outliers, especially at mask boundaries.

### D3: Global-mean normalization (Option B)

`b_norm = b_smooth / max(mean(b_smooth), ε)`, computed over ALL source pixels (not just positive ones).

**Rationale:**
- Continuously differentiable everywhere (no `b_smooth > 0` hard threshold)
- Naturally produces high contrast: sparse bright regions get `b_norm ≫ 1`, dark regions get `b_norm ≈ 0`
- Dark-region behavior is correct: `b_norm → 0` means `scale → 1` (strong regularization)
- Single `hot pixel` doesn't contaminate the global normalization

**Alternative considered:** Positive-only mean (`mean(b_smooth[b_smooth > 0])`). Rejected — introduces a non-differentiable threshold, and a single hot pixel can inflate the mean, suppressing scale reduction everywhere else.

### D4: Soft-asymptote scale formula

`scale_i = floor + (1 - floor) / (1 + α × b_norm_i)`

**Rationale:**
- Continuously differentiable (no `max(·, floor)` clamp)
- Natural endpoints: `b_norm=0 → scale=1`, `b_norm→∞ → scale→floor`
- Smoother transition in the mid-range compared to `max(1/(1+α·b), floor)`
- Compatible with JAX autodiff for gradient-based optimization and HMC/NUTS sampling

**Alternative considered:** Keep `max(1/(1+α·b), floor)`. Rejected — hard clamp creates a gradient discontinuity at the floor boundary, problematic for gradient-based samplers and optimizers.

### D5: Freeze via explicit eager API on evidence model

The frozen scale map is stored as `self._frozen_scale: Array | None` on the evidence model instance (`PixelizedImageProbModel` / `PixelizedImageProbModelOperator`). It is a plain JAX array, not a `caskade.Param`. The user MUST call `freeze_scale()` eagerly **before** JIT tracing (i.e. before `make_likelihood` / sampler startup). At trace time, `_compute_reg_scale_from_betas` checks `getattr(self, '_frozen_scale', None)` and, if set, returns the cached array immediately; the JIT compiler then captures the concrete array as a closure constant and the traced betas become dead args. The core brightness→scale math is extracted into `_compute_scale_core` so `freeze_scale()` can compute eagerly without re-entering the freeze cache check (which would spuriously warn).

**Rationale:** The scale map is not a model parameter — it's a cached computation. Storing it on the evidence model instance (not on `PixelizedSourceModel`) keeps the physical model layer clean of inference-workflow state. JAX JIT captures the frozen array by value in closures, which is correct behavior (no silent staleness). Computing the scale eagerly outside JIT is the only way to make the freeze observable to the compiled graph — storing a traced array inside JIT (the original design) does not work because the cache-write happens after trace and the Python-level `is not None` branch is already baked into the computation graph.

**Alternative considered:** Store on `PixelizedSourceModel` as a `ParamU`. Rejected — the scale map is large (Ns floats), not a scalar parameter, and should not participate in prior/posterior transformation. Auto-cache on first JIT call. Rejected — `object.__setattr__` inside a JIT-traced function stores a stale traced array and the trace-time branch cannot observe it.

### D6: Kernel size auto-adaptation

`ksize = max(5, 2 × ceil(3 × sigma) + 1)` ensures the Gaussian kernel spans ±3σ in each direction.

**Rationale:** At `sigma=1` (default), `ksize=5` matches legacy behavior. At `sigma=3`, `ksize=19` avoids truncating the Gaussian tails, preventing artifacts in heavily smoothed brightness maps.

## Risks / Trade-offs

- **`brightness_only` mode requires two convolutions** (N and C) vs one in the legacy path. Risk: ~2× smoothing cost. Mitigation: 5×5 convolution on an 80×80 grid is negligible (~0.01ms) relative to deflection-angle computation (~10ms) and PCG (~100ms).
- **Global-mean normalization shifts with grid resolution.** Doubling source pixels (40×40 → 80×80) adds mostly dark pixels, lowering `b_mean` and amplifying `b_norm` for bright pixels. Risk: `alpha` may need retuning when resolution changes. Mitigation: document this behavior; the effect is predictable and monotonic.
- **Freeze mechanism introduces mutable state in a JAX pipeline.** Risk: JIT captures the frozen array by value; if the user forgets to freeze before starting sampling, each evidence call recomputes the scale. Mitigation: log a warning if `freeze=True` but no scale is stored and the function is called > once.
