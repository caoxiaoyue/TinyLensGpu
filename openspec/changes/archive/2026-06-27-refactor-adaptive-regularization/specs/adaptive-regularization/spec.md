## ADDED Requirements

### Requirement: Brightness-only adaptive regularization mode

The system SHALL provide a `brightness_only` mode that estimates source brightness independently of lensing magnification, using normalized convolution with inverse-variance weighting.

- Numerator: `N_i = Σ_k w_{ki} × d_k⁺ / σ_k²`
- Denominator: `C_i = Σ_k w_{ki} / σ_k²`
- Raw brightness: `b_raw = smooth(N) / (smooth(C) + ε)`

Both `smooth(N)` and `smooth(C)` SHALL use the same normalized Gaussian kernel. The magnification dependence cancels in the `N/C` ratio.

#### Scenario: Bright source pixel in low-magnification region

- **WHEN** a source pixel is intrinsically bright (high surface brightness) but weakly magnified (few image pixels map to it)
- **THEN** `N_i` is moderate (few contributing pixels, each bright), `C_i` is small (few contributing pixels), and `b_raw_i` SHALL reflect the high intrinsic brightness

#### Scenario: Moderately bright source pixel near caustic

- **WHEN** a source pixel is moderately bright but highly magnified (many image pixels map to it)
- **THEN** `N_i` is large (many contributing pixels), `C_i` is proportionally large, and `b_raw_i` SHALL reflect only the moderate intrinsic brightness, NOT the magnification

#### Scenario: Uncovered source pixel

- **WHEN** no seed image pixel maps to a given source pixel
- **THEN** `C_i^sm` SHALL be near-zero, `b_raw_i` SHALL be near-zero via the `+ε` protection, and the final `scale_i` SHALL approach 1.0 (strongest regularization)

### Requirement: Brightness-weighted legacy mode

The system SHALL provide a `brightness_weighted` mode that retains the brightness×ray-count brightness estimator, upgraded with inverse-variance weighting.

- Quantity: `q_i = Σ_k w_{ki} × d_k⁺ / σ_k²` (no `C` denominator)
- `b_raw = smooth(q)`

This mode preserves the magnification dependence in the scale map for comparison and legacy use.

#### Scenario: Same brightness, different magnification

- **WHEN** two source pixels have identical intrinsic brightness but different magnifications
- **THEN** `q_i` SHALL be larger for the more highly magnified pixel, and its `scale_i` SHALL be correspondingly lower

### Requirement: Inverse-variance weighting in both modes

Both `brightness_only` and `brightness_weighted` modes SHALL weight image-plane pixel contributions by inverse variance `1/σ_k²`.

The inverse variance `1/σ_k²` SHALL be computed from the per-pixel noise map `self.noise_map` at the seed-mask pixel locations.

#### Scenario: Noisy pixel contribution

- **WHEN** an image pixel has σ_k = 100 (very noisy) and another has σ_k = 1 (clean), both with identical brightness
- **THEN** the clean pixel SHALL contribute 10,000× more weight to the brightness estimate than the noisy pixel

#### Scenario: Uniform noise

- **WHEN** all image pixels have identical noise σ
- **THEN** the inverse-variance weights reduce to a constant scaling factor and both modes SHALL produce results equivalent to unweighted accumulation up to a global scale

### Requirement: Unified downstream normalization

Both modes SHALL share a single normalization pipeline:

1. `b_smooth = GaussianSmooth(b_raw, sigma)` applied per source-plane pixel
2. `b_mean = mean(b_smooth)` computed over ALL source pixels (global mean)
3. `b_norm = b_smooth / max(b_mean, ε)`

No hard threshold (e.g., `b_smooth > 0` filtering) SHALL be used in the normalization step.

#### Scenario: Sparse source coverage

- **WHEN** only 10% of source pixels have non-zero brightness estimate
- **THEN** `b_mean` SHALL include the zero-valued pixels (lowering the global mean), bright pixels SHALL have `b_norm ≫ 1`, and dark pixels SHALL have `b_norm ≈ 0`

#### Scenario: All pixels dark

- **WHEN** all `b_raw_i` are near-zero
- **THEN** `b_mean` SHALL be near-zero, `b_norm` SHALL remain finite via the `max(b_mean, ε)` protection, and all `scale_i` SHALL approach 1.0

### Requirement: Continuously-differentiable scale formula

The final per-pixel regularization scale SHALL be computed as:

`scale_i = floor + (1 - floor) / (1 + α × b_norm_i)`

where `α ≥ 0`, `0 < floor ≤ 1`, and `b_norm_i ≥ 0`.

This formula SHALL be continuously differentiable with respect to `b_norm_i` for all finite inputs.

#### Scenario: Darkest pixel (b_norm = 0)

- **WHEN** `b_norm_i = 0`
- **THEN** `scale_i = 1.0` regardless of `α` and `floor`

#### Scenario: Brightest pixel (b_norm → ∞)

- **WHEN** `b_norm_i` is very large
- **THEN** `scale_i → floor` asymptotically

#### Scenario: Mean-brightness pixel (b_norm = 1) with default α=1, floor=0.1

- **WHEN** `b_norm_i = 1`, `α = 1`, `floor = 0.1`
- **THEN** `scale_i = 0.1 + 0.9 / 2 = 0.55`

### Requirement: Uniform regularization at α=0

When `adaptive_reg_alpha == 0`, the system SHALL bypass all brightness estimation and return `scale = None`, producing uniform regularization identical to the pre-adaptive behavior.

#### Scenario: Alpha is zero

- **WHEN** `adaptive_reg_alpha` is 0.0 (or within floating-point tolerance 1e-10)
- **THEN** `_compute_reg_scale_from_betas` SHALL return `None` without computing brightness, weights, or segment_sum

### Requirement: Configurable smoothing scale

The system SHALL expose `adaptive_reg_smooth_sigma` (default 1.0 source pixel) to control the Gaussian kernel width for brightness-map smoothing.

The kernel size SHALL auto-adapt: `ksize = max(5, 2 × ceil(3 × sigma) + 1)`.

#### Scenario: Default sigma

- **WHEN** `smooth_sigma = 1.0`
- **THEN** kernel size SHALL be 5 (matching legacy hardcoded behavior)

#### Scenario: Larger sigma

- **WHEN** `smooth_sigma = 2.0`
- **THEN** kernel size SHALL be at least `2 × ceil(6) + 1 = 13` to avoid Gaussian truncation

### Requirement: Empirical-Bayes freeze

The system SHALL support an `adaptive_reg_freeze` flag. When `True` and a frozen scale map has been stored by an explicit eager `freeze_scale()` call, subsequent calls to `_compute_reg_scale_from_betas` SHALL return the stored scale map without recomputation.

When `True` and no frozen scale exists, `_compute_reg_scale_from_betas` SHALL warn and recompute the scale for that call without storing it. Implementations SHALL NOT create the frozen cache implicitly inside JIT-traced likelihood evaluation.

When `False`, the scale map SHALL be recomputed on every call.

#### Scenario: Freeze during nested sampling

- **WHEN** `adaptive_reg_freeze = True` and a frozen scale map exists
- **THEN** all evidence evaluations SHALL use the same scale map regardless of lens parameter proposals

#### Scenario: Freeze without explicit eager computation

- **WHEN** `adaptive_reg_freeze = True` but no frozen scale map has been stored
- **THEN** `_compute_reg_scale_from_betas` SHALL warn, compute, and return a per-call scale map without storing it

### Requirement: Scale application to regularization matrix

The per-pixel `scale` array SHALL be applied to the finite-difference regularization matrix through edge weights, using geometric-mean interpolation of adjacent pixel scales:

`w_edge(i,j) = sqrt(scale_i × scale_j)`

This preserves symmetry and positive semi-definiteness of the regularizer.

#### Scenario: Uniform scale

- **WHEN** all `scale_i = 1`
- **THEN** all edge weights SHALL be 1, recovering the uniform regularizer

#### Scenario: Adjacent bright and dark pixel

- **WHEN** `scale_i = floor` (bright pixel adjacent to dark pixel where `scale_j = 1`
- **THEN** the shared edge weight SHALL be `sqrt(floor × 1) = sqrt(floor)`, intermediate between the two extremes
