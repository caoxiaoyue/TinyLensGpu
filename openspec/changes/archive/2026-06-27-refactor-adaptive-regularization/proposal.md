## Why

The current adaptive regularization scheme computes per-pixel scale factors from `brightness × ray-count` — a product that confounds source brightness with lensing magnification. In the semi-linear inversion framework, the Hessian curvature term `MᵀC⁻¹M` already encodes magnification: highly magnified source pixels naturally receive stronger data constraints and proportionally weaker regularization. Reducing regularization again via the adaptive scale therefore double-counts magnification, over-weakening the prior in bright caustic regions while under-serving bright but weakly lensed regions. We need a scheme where the adaptive scale reflects only source brightness — the one quantity the curvature cannot encode.

## What Changes

- **New default: `brightness_only` mode.** Computes brightness via normalized convolution (`N/C`), where numerator `N` and denominator `C` both carry inverse-variance weighting (`1/σ²`). Magnification cancels in the ratio, yielding a pure brightness estimate on the source plane.
- **Retained legacy: `brightness_weighted` mode.** Keeps the brightness×ray-count path as an option, upgraded with inverse-variance weighting so data quality is respected regardless of mode.
- **Unified downstream pipeline.** Both modes share the same normalization (global mean, Option B), the same continuously-differentiable scale formula (`floor + (1-floor)/(1+α·b̃)`), and the same smoothing path — eliminating hard thresholds (positive-only masking, `max(·, floor)` clamp) for stable autodiff gradients.
- **Configurable smooth scale.** `adaptive_reg_smooth_sigma` replaces the hardcoded σ=1 kernel width. Kernel size auto-adapts to `2·ceil(3σ)+1` to avoid truncation.
- **Empirical-Bayes freeze.** `adaptive_reg_freeze` flag pins the scale map after initial computation, preventing adaptive-prior drift during lens-parameter sampling or evidence comparison.
- **Reduced hyperparameter surface.** Only `alpha`, `floor`, and `smooth_sigma` are exposed as model hyperparameters. Internal normalization weights default to all-ones (fixed source-domain mask).

## Capabilities

### New Capabilities

- `adaptive-regularization`: Per-pixel adaptive regularization for pixelized source inversion. Supports two brightness-estimation modes (`brightness_only` and `brightness_weighted`), inverse-variance-weighted accumulation, normalized-convolution brightness estimation, a continuously-differentiable scale formula, configurable smoothing, and an empirical-Bayes freeze mechanism for unbiased lens-parameter inference.

### Modified Capabilities

None. The existing `adaptive_reg_alpha` and `adaptive_reg_floor` parameters on `PixelizedSourceModel` are extended with new options, but the default behavior (`alpha=0` → uniform regularization) is unchanged and no spec-level requirements are altered.

## Non-goals

- Adding gates, thresholds, or multi-tier regularization logic beyond the single continuous scale formula.
- Supporting GP-type regularization kernels (exponential, Matérn, etc.) with adaptive scaling — finite-difference types only, consistent with the current operator backend limitation.
- Auto-tuning `alpha` or `floor` from data.
- Changing the block-diagonal preconditioner or PCG solver logic except where necessary to consume the new scale format.

## Impact

| Area | Files | Notes |
|------|-------|-------|
| `PixelizedSourceModel` | `PhysicalModel/.../Pixelized/Light/pixelized_source.py` | New params: `mode`, `smooth_sigma`, `freeze` |
| `DenseRegularizationBuilder` | `utils/inversion/regularization.py` | `make_reg_data` already accepts `scale`; no structural changes needed |
| Dense evidence model | `ObservationModel/.../pixelized_image_model.py` | Replace `_compute_reg_scale_from_betas` with dual-mode implementation |
| Operator evidence model | `ObservationModel/.../pixelized_image_model_operator.py` | Same replacement; `_get_bbox()` already returns seed betas needed for brightness mapping |
| Example scripts | `examples/pix_src_demo_operator/pipe/no_lens_light/model_adpt_reg.py` | Update parameter names and defaults |
| Tests | `tests/test_regularization.py` | Add coverage for new modes, normalization, freeze |
