## Why

The current adaptive regularization path can derive its scale map from image-plane arc pixels ray-traced through the active mass model, so the adaptive prior can drift with lens parameters unless explicitly frozen. A stage-m0 source template (`S0`) provides a clearer empirical-Bayes reference: estimate a fixed source-plane brightness map once with uniform regularization, then condition later adaptive regularization and mass inference on that fixed source grid and scale map.

## What Changes

- Add a stage-m0 step to `examples/pix_src_demo_operator/pipe/no_lens_light/model_adpt_reg.py` before the current stage-m1.
- Stage-m0 will fix the SIE+shear mass model at the stage-A median parameters, disable adaptive regularization (`adaptive_reg_alpha = 0`), grid-search the uniform regularization strength, reconstruct the MAP pixelized source, and save the result as `S0`.
- Save the source grid metadata used by `S0`, including `nx`, `ny`, source bbox, and grid axes, so downstream stages use the same pixel-to-source-plane coordinate mapping.
- Build adaptive regularization scale maps directly from `S0` instead of using the current `lensed arc -> mass-model ray trace -> smoothed source-plane proxy` path.
- Use the minimal `S0 -> scale` transformation: clip negative source brightness to zero, normalize by the global mean over all source pixels, and apply the existing scale formula.
- Do not apply additional Gaussian smoothing when deriving scale from `S0`; `S0` is already a regularized source reconstruction.
- Update stage-m1 to accept the fixed `S0` package and use its derived scale map while re-optimizing the adaptive regularization strength.
- Update stage-m2 to accept the same `S0` package, inherit/fix the stage-m1 source hyperparameter value as before, and sample the mass model using the same fixed source grid and scale map.
- Retire the older mass-dependent adaptive-scale path and its `freeze_scale()` workflow globally in favor of the fixed `S0` source-template path.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `adaptive-regularization`: Add support for fixed source-template-derived adaptive scale maps, where the scale is generated from a saved pixelized source reconstruction and reused on a fixed source grid during later regularization optimization and mass inference.

## Non-goals

- Changing the mathematical scale formula used after brightness normalization.
- Auto-tuning `adaptive_reg_alpha`, `adaptive_reg_floor`, or the source bbox padding strategy.
- Making source bbox dynamically follow the mass model in the new `S0` path.
- Extending adaptive scaling to GP-style regularization kernels.

## Impact

- `examples/pix_src_demo_operator/pipe/no_lens_light/model_adpt_reg.py`: add stage-m0 orchestration, caching, plotting, and downstream `S0` loading.
- `TinyLensGpu/ObservationModel/LensImage/pixelized_image_model_operator.py`: use a fixed source bbox and externally supplied adaptive scale map for adaptive regularization, replacing the older seed-ray scale construction.
- `TinyLensGpu/ObservationModel/LensImage/pixelized_image_model.py`: reject adaptive regularization in the dense backend until it can consume explicit fixed template inputs; do not keep the old seed-ray/freeze implementation available.
- `TinyLensGpu/PhysicalModel/LensImage/Pixelized/Light/pixelized_source.py`: remove retired adaptive seed-ray/freeze configuration fields.
- Tests should cover `S0 -> scale` normalization, fixed-scale evidence evaluation, fixed source bbox behavior, and the stage-m0/m1/m2 cache contract where practical.
- GPU/JAX performance should improve or remain stable in M1/M2 because the scale map is precomputed from `S0` and no longer requires per-call source-plane brightness accumulation from seed rays.
