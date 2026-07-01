## Context

Adaptive regularization currently derives a fixed S0 source-template scale map with a mean-normalized brightness formula:

```text
scale = floor + (1 - floor) / (1 + alpha * S0_pos / mean(S0_pos))
```

That formulation treats dark pixels as baseline regularization and weakens bright pixels toward `floor`. The new behavior should follow the Galan et al. luminosity-weighting intuition for high-dynamic-range sources: bright regions stay at baseline precision scale, while faint regions receive stronger regularization. The code already has the right integration boundary: `PixelizedImageProbModelOperator` accepts `fixed_reg_template`, and finite-difference regularization consumes a positive per-pixel scale through geometric-mean edge weights.

## Goals / Non-Goals

**Goals:**

- Replace the S0-derived scale-map formula with a Galan-style precision scale:
  `scale = exp(rho * (1 - u))`.
- Use the 99.5 percentile of clipped positive S0 brightness as the reference brightness.
- Replace active adaptive hyperparameters `adaptive_reg_alpha` and `adaptive_reg_floor` with `adaptive_reg_rho`.
- Keep dynamic `ParamU` support for `rho` so samplers can fit it with `log_lambda_reg`.
- Preserve fixed S0 template, fixed source bbox, and existing operator backend matrix-free regularization consumption.

**Non-Goals:**

- Do not implement full covariance-kernel luminosity weighting for dense GP kernels.
- Do not restore seed-ray adaptive scale construction or scale freezing.
- Do not change source-grid bbox inference or the finite-difference edge-weight regularization operator.

## Decisions

### D1: Treat `scale` as a precision scale

The new scale map will be:

```text
s_pos = max(S0, 0)
s_ref = percentile(s_pos, 99.5)
u = clip(s_pos / max(s_ref, eps), 0, 1)
scale = exp(rho * (1 - u))
```

Bright pixels with `u = 1` get `scale = 1`; dark pixels with `u = 0` get `scale = exp(rho)`. This preserves the existing downstream contract that larger `scale` means stronger finite-difference regularization.

Alternative considered: `scale = exp(-rho * u)`, which keeps dark pixels at 1 and weakens bright pixels. Rejected because it keeps the old baseline semantics rather than the Galan-style dark-region strengthening.

### D2: Use a fixed 99.5 percentile brightness reference

The reference brightness will be the 99.5 percentile of `s_pos`, not the maximum. This limits sensitivity to a single noisy S0 spike while still making the brightest source structure the baseline-regularization region. Values above the percentile are clipped to `u = 1`.

Alternative considered: maximum brightness. Rejected because high-dynamic-range reconstructions can contain isolated artifacts that would compress all other `u` values.

### D3: Replace active alpha/floor parameters with rho

`PixelizedSourceModel` should expose `adaptive_reg_rho` as a scalar or `ParamU`, constrained to `rho >= 0`. `rho = 0` is uniform regularization and should preserve the existing `None` fast path when statically known. Dynamic `rho` should be traversed by Caskade and consumable by `make_prior_transformation`.

Existing `adaptive_reg_alpha` and `adaptive_reg_floor` are no longer active adaptive scale controls. Because this is a breaking proposal, implementation can remove them or reject non-default use with clear errors; the spec should require callers to use `adaptive_reg_rho`.

### D4: Keep the ObservationModel boundary unchanged where possible

`PixelizedImageProbModelOperator._get_reg_scale()` should still be the place that turns `fixed_reg_template` into a scale map. `fixed_reg_scale` remains accepted for callers that precompute a scale externally, provided it is finite and positive. This avoids storing large source-grid arrays in the PhysicalModel layer and keeps S0 empirical-Bayes state at the likelihood boundary.

### D5: Do not alter finite-difference regularization assembly

The existing `DenseRegularizationBuilder` matrix and matrix-free paths already accept positive scale arrays and convert them to edge weights:

```text
w_edge(i,j) = sqrt(scale_i * scale_j)
```

The new scale range `[1, exp(rho)]` remains valid for this operator. Tests should focus on changed values and numerical stability, not on rewriting regularization assembly.

## Risks / Trade-offs

- Large `rho` can make the regularized system poorly conditioned -> constrain example priors to modest ranges, such as `rho in [0, 3]`, and keep scale validation finite-positive.
- Percentile calculation under JIT can be awkward if the percentile is treated as dynamic -> make the percentile value static configuration, defaulting to `99.5`; only `rho` needs to be traced dynamically.
- Breaking alpha/floor can invalidate existing demos or cached artifacts -> update examples and cache metadata to expect `adaptive_reg_rho`.
- All-dark S0 templates make `s_ref` effectively zero -> use `max(s_ref, eps)` so `u = 0` and scale is finite `exp(rho)`.

## Migration Plan

1. Update `PixelizedSourceModel` adaptive hyperparameter API to use `adaptive_reg_rho`.
2. Update `source_template_scale_map()` to compute the percentile-normalized Galan-style precision scale.
3. Update `PixelizedImageProbModelOperator._get_reg_scale()` to pass `rho` instead of alpha/floor.
4. Update tests for scale-map values, dynamic prior extraction, static rho-zero fast path, and vectorized likelihood behavior.
5. Update examples and pipeline artifacts that refer to `adaptive_reg_alpha` or `adaptive_reg_floor`.
6. Rollback, if needed, is to restore the previous source-template formula and alpha/floor parameters without changing the regularization operator.
