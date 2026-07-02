## Why

The operator pixelized-source backend currently solves the source reconstruction with unconstrained PCG and can only discourage negative source pixels through a soft evidence penalty. Pixelized source reconstruction with non-negative brightness is a constrained quadratic problem, so the operator backend should provide a matrix-free hard non-negative solver that preserves its GPU and memory advantages.

## What Changes

- Add a matrix-free FISTA solver path for `PixelizedImageProbModelOperator` that enforces `source_pixels >= 0`.
- Add an explicit operator-backend solver selection, with the existing unconstrained PCG path retained and a new FISTA path available for hard source non-negativity.
- Reuse the existing operator `A(s) = M^T C^-1 M s + lambda R s` and RHS machinery so the FISTA path avoids dense design or curvature matrices.
- Keep the current operator evidence log-determinant approximation for the FISTA path: `logdet(A) ~= logdet(P)` and `logdet(R)` from the existing matrix-free regularization approximation.
- **BREAKING** Remove `source_nonnegativity_sigma` and the old soft non-negativity evidence penalty from the operator backend.
- Update tests and affected examples to use the FISTA solver when hard non-negative source reconstruction is required.

## Capabilities

### New Capabilities

- `operator-fista-nnls`: Matrix-free FISTA source solver for the operator pixelized-source backend with hard non-negative source pixels and existing approximate evidence terms.

### Modified Capabilities

- None.

## Non-goals

- Do not implement active-set Laplace corrections or exact truncated-Gaussian evidence in this change.
- Do not add stochastic log-determinant estimation.
- Do not add operator-backend support for lens-light joint inversion.
- Do not add GP regularization support to the operator backend.

## Impact

- Affects `TinyLensGpu/ObservationModel/LensImage/pixelized_image_model_operator.py`, `TinyLensGpu/ForwardSimulation/LensImage/pixelized_operator.py` integration points, and solver utilities under `TinyLensGpu/utils/`.
- Removes the public `source_nonnegativity_sigma` constructor argument for `PixelizedImageProbModelOperator`.
- Adds tests comparing operator FISTA against dense NNLS on small problems and checking that `forward_model(return_source=True)` returns non-negative source pixels.
- Keeps GPU memory usage low by using matrix-vector products, FFTs, bilinear scatter/gather, finite-difference regularization matvecs, and O(Ns) FISTA state vectors.
