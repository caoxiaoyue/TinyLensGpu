## Why

The pixelized-source core now enforces square `n x n` grids, but remaining demo, cache, docstring, and test code still describes source-plane grids with legacy `nx`/`ny` metadata. This keeps obsolete rectangular-grid assumptions visible in the codebase and makes the square-grid contract harder to reason about.

## What Changes

- **BREAKING**: S0 source-template packages SHALL use a single `n` field instead of `nx`/`ny`.
- **BREAKING**: Legacy S0 packages containing only `nx`/`ny` SHALL be rejected with a clear regeneration message instead of being silently adapted.
- S0 package validation, fingerprinting, scale-map construction, and package creation SHALL derive source size from `n`.
- Source-plane reshapes in pixelized-source examples and plotting helpers SHALL use `reshape(n, n)` and source-grid variable names such as `n` or `source_n`.
- Library docstrings and error text SHALL describe source vectors as `(n * n,)` and 2D source images as `(n, n)`.
- Tests SHALL cover the new S0 package schema and stale legacy package rejection.

### Non-goals

- No changes to pixelized-source interpolation, regularization mathematics, PCG, dense Cholesky evidence, or JAX compilation behavior.
- No changes to image-plane shape conventions such as `ny, nx = image.shape`; those are standard array semantics and are not source-grid metadata.
- No compatibility shim for old S0 cache/package files; users should regenerate them under the single-`n` schema.
- No changes to parametric source models or non-pixelized examples.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `square-pixelized-source-grid`: Strengthen the square-grid contract so persisted S0 source-template metadata, source-plane helper code, and public-facing documentation use a single source-grid dimension `n` instead of legacy `nx`/`ny`.

## Impact

- Affects `examples/pix_src_demo_operator/pipe/no_lens_light/model_adpt_reg.py` and related pixelized-source demo plotting helpers that reshape reconstructed sources.
- Affects `tests/test_adaptive_demo_s0_package.py` and any S0 fixture data built with `nx`/`ny`.
- Affects docstrings in `PixelizedSourceModel`, `PixelizedImageProbModelOperator`, and any nearby source-plane helper text.
- Existing S0 cache/package files using `nx`/`ny` will be invalid and must be regenerated. This is CPU/GPU neutral and does not add runtime work to JAX-traced likelihood paths.
