# Supplied-field potential rendering: ticket 03

Implemented against the approved potential-correction spec, with no changes to
existing source interpolation, PSF kernels, source priors, or source–lens-light
inversion. This record verifies rendering and physical operators; it does not
claim an inversion or scientific recovery.

## Public entry point

```python
from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer

renderer = PotentialCorrectionRenderer(
    macro_model, sim_config, source_n=31,
    source_bbox=(-0.5, 0.5, -0.5, 0.5),
    potential_spacing=2 * sim_config.dpix,
    dtype="float64",
)
# source_values: source_n**2 signed intensities, row-major (iy*n + ix).
# potential_values: one value per renderer.potential_grid unknown.
result = renderer.render(source_values, potential_values)
```

The full native image is `result.model_image`; the selected observation vector
is `result.image_active`. The result retains source axes/bounds/values, raw
potential values and geometry, corrected source coordinates, query deflections,
and physical deflection/convergence maps on the potential unknowns. The latter
contain NaN outside the core and have an explicit `derivative_valid` mask.

`PotentialGrid.from_queries` is available from
`TinyLensGpu.utils.lensing.potential`. Its integer `node_indices` are `(ix, iy)`
pairs sorted by increasing y, then x. Physical node coordinates are `x`, `y`.
`core_indices` addresses the unknown vector, whereas `interpolation_indices`
addresses the ordered core vector. `derivative_indices` stores center, x−, x+,
y−, y+ unknown indices. All support-only coefficients remain unknowns.

The default grid is the exact four-corner union C for every actual selected
image subpixel, enlarged by the union of axial ±1 and ±2 offsets. It never
cleans the observation mask or clips support at an image edge. An isolated
single-cell query has four core nodes and 20 unknowns, rather than a 6×6
rectangle. Explicit `node_indices` must include the required support; additional
explicit nodes remain unknowns with no claimed derivative coverage. Explicit
renderer geometry must match the actual query coordinates and dtype.

Physical first differences and the centered Laplacian use angular spacing.
Fourth-difference regularization is the next ticket's responsibility.

## Numerical handoff

- `renderer.prepare_macro(model=None)` returns `MacroMapping(beta_x, beta_y)`
  without rebuilding geometry or replacing the renderer's fixed default rays.
- `renderer.source_mapping(p=None, macro_mapping=None)` returns current
  `SourceMapping(weights, indices, valid, beta_x, beta_y)` after subtracting the
  correction deflection. No current-source or corrected-ray cache is hidden.
- `renderer.source_forward(s, mapping)` and
  `renderer.source_adjoint(y, mapping)` are the current source action pair.
- `renderer.image_forward(subpixel_values)` and `renderer.image_adjoint(y)`
  are the integration/PSF/masking action pair for arbitrary subpixel responses.
- `grid.deflection(p)` and `grid.deflection_adjoint(vx, vy)` are the identical
  physical deflection/interpolation action pair for rendering and linear fits.
- `grid.derivatives(p)` returns core-only `PotentialDerivatives` with
  `deflection_x`, `deflection_y`, and `convergence`.

These actions compose inside JAX kernels. A reconstruction creates/fixes
macro-dependent rays separately from topology, then supplies fresh dynamic
source mappings as needed. `render` also validates finite supplied fields and
assembles a Python result; numerical fitting should use the numerical actions.

The imaging operations reuse the actual pixelized `_apply_L_jit`,
`_apply_Lt_jit`, and `_psf_convolve_full_jit` kernels. Active subpixel brightness
is averaged into native pixels, masked native input light is zero, and the
same circular native FFT PSF is used, with conjugation for its adjoint. No PSF
halo, alternate convolution utility, or expanded source box is introduced.

Precision contexts are scoped to each preparation/public call. Input
coordinates already sampled in float32 retain that original sampling precision
when promoted; construct input configurations in a float64 context for the
float64 reference. Callers performing further JAX arithmetic or compiling an
outer float64 kernel should likewise use a float64 context. The process
precision default is unchanged after these calls.

## Verification

Environment: `tinylens_gpu`, JAX 0.7.1, default CUDA device (RTX 5080).

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
cd tests
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest test_potential_correction_rendering.py -q
```

Result after final formatting and public re-exports: **17 passed in 12.54 s**.
Red→green checks initially failed for the absent public geometry, physical
derivative, transpose, and renderer capabilities; the index-range validation
test initially exposed a raw overflow and now raises an informative error.

The tests cover quadratic physical deflection/convergence and interpolation;
potential/source/image inner-product identities; exact zero-correction parity
with `PixelizedLensOperator` at nsub=1 and 3 with an asymmetric 3×3 PSF; a
nonzero quadratic field with an independently evaluated shear lens equation and
linear source; narrow arcs, holes, disconnected selections and an isolated
image-edge pixel at nsub=3; source-grid endpoint/out-of-bounds behavior; signed
source values; explicit float32/float64 and unchanged defaults; explicit
geometry failure; and rebuilding macro rays without replacing geometry.

Float64 derivative and adjoint absolute tolerances are 2–3×10⁻¹⁴; existing-image
relative tolerance is 2×10⁻¹³. The independent nonzero rendering check uses an
absolute image tolerance of 2×10⁻¹⁴. Float32 endpoint/output checks use 10⁻⁶.
These are small numerical systems, not a recovery or performance calibration.

An additional supplied-truth check at the 200-by-200 mock size is retained in
[supplied-truth-rendering.json](supplied-truth-rendering.json). Reproduce it
from the repository root in `tinylens_gpu` with:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \
  python .scratch/potential-correction/experiments/supplied_truth_rendering.py
```

It samples analytic source brightness and potential onto the fixed grids and
compares the nonlinear rendering to the noiseless analytic mock, with no fit.
The record includes the complete mock configuration. Source sides 41 and 61
give approximately 0.48 and 0.21 noise units RMS, respectively, for both mock
and control. The nearly matched control errors identify source interpolation
as the dominant contribution in this configuration. This result informs later
grid calibration; it does not establish reconstruction accuracy.

The coordinator already recorded the 810-test unchanged core baseline and
ticket 01's 372 source/operator/autodiff regressions. This ticket adds new
modules and re-exports only; it does not repeat those broad checks.
