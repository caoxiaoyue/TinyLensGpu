## MODIFIED Requirements

### Requirement: S0 source-template packages use square grid metadata

Pixelized source demo pipelines that persist or consume S0 source-template packages SHALL represent the source grid with a single integer `n`.

An S0 package SHALL contain source pixels compatible with `(n * n,)`, optional 2D source images compatible with `(n, n)`, source axes compatible with `(n,)`, and a square `source_bbox`. The package schema SHALL NOT require or write source-grid `nx` or `ny` fields. Cached or loaded S0 packages that only provide legacy `nx`/`ny` source-grid metadata SHALL be rejected before being used to build adaptive regularization scale maps or fixed source-grid likelihoods.

#### Scenario: Single-n S0 package is reusable

- **WHEN** a saved S0 package has `n == N`, `source_pixels.shape == (N * N,)`, source axes with shape `(N,)`, and a square `source_bbox`
- **THEN** the pipeline SHALL allow the package to be reused

#### Scenario: Legacy nx-ny S0 package is rejected

- **WHEN** a saved S0 package provides `nx` and `ny` source-grid metadata but does not provide `n`
- **THEN** the pipeline SHALL fail with a clear validation error requiring regeneration under the single-`n` source-grid schema

#### Scenario: Invalid S0 source vector shape is rejected

- **WHEN** a saved S0 package has `n == N` but `source_pixels.shape` is not `(N * N,)`
- **THEN** the pipeline SHALL fail with a clear validation error describing the expected `(n * n,)` shape

#### Scenario: Rectangular S0 source bbox is rejected

- **WHEN** a saved S0 package has `n == N` and valid source pixels but a non-square `source_bbox`
- **THEN** the pipeline SHALL fail with a clear validation error requiring a square pixelized source bbox
