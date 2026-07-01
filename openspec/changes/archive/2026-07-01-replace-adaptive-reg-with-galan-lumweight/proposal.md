## Why

The current S0-based adaptive regularization scale map uses mean-normalized brightness and a floor formula that weakens regularization in bright source pixels. For high-dynamic-range sources, this can saturate bright regions toward the same floor and does not match the Galan et al. (2024) luminosity-weighting intuition of keeping bright regions at baseline regularization while strengthening dark outskirts.

## What Changes

- **BREAKING**: Replace the current source-template adaptive scale formula with a Galan-style luminosity-weighted precision scale.
- Build the scale map from a fixed S0 template as:
  - `s_pos = max(S0, 0)`
  - `s_ref = percentile(s_pos, 99.5)`
  - `u = clip(s_pos / max(s_ref, eps), 0, 1)`
  - `scale = exp(rho * (1 - u))`
- Replace `adaptive_reg_alpha` and `adaptive_reg_floor` as the active adaptive scale-map hyperparameters with `adaptive_reg_rho`.
- Constrain `adaptive_reg_rho >= 0`; `rho = 0` gives uniform regularization, while larger values strengthen regularization in faint source regions up to `exp(rho)`.
- Preserve the existing fixed S0 template and fixed source-bbox workflow for operator likelihoods.
- Preserve the existing finite-difference regularization matrix/operator application: scale remains a positive per-pixel precision scale consumed through geometric-mean edge weights.
- Keep JAX-compatible traced evaluation for dynamic `rho` values in vectorized likelihoods.

## Non-goals

- Do not implement the full Galan et al. covariance-kernel transformation `C_lum = D(W) C D(W)` for dense GP/Matern regularization in this change.
- Do not reintroduce image-plane seed-ray adaptive scale construction, smoothing, or freeze/unfreeze cache behavior.
- Do not change the operator backend's edge-weighted finite-difference regularization construction except as needed to consume the new scale values.
- Do not make the source bbox dynamic during adaptive S0-based inference.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `adaptive-regularization`: Replace the source-template scale-map formula and adaptive hyperparameter contract with the Galan-style luminosity-weighted precision scale.

## Impact

- Affected APIs: `PixelizedSourceModel` adaptive hyperparameters, `source_template_scale_map()`, and `PixelizedImageProbModelOperator` scale generation from `fixed_reg_template`.
- Affected tests: adaptive scale-map formula tests, dynamic hyperparameter prior extraction, vectorized operator likelihood tests, and any examples or cached pipeline metadata that expect `adaptive_reg_alpha` or `adaptive_reg_floor`.
- Performance impact should be small: scale generation remains O(Ns), JAX-friendly, and occurs before the existing matrix-free regularization operator consumes the per-pixel scale.
