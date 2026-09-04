# Fixed pixelized-source grids and source-bbox boundary behavior

## Scope and conclusion

This note records the design discussion around source-grid construction in a
parametric-to-pixelized strong-lens modeling pipeline. It covers the current
implementation, the case for freezing the pixelized-source coordinate system,
the scientific prior introduced by a fixed source bounding box, out-of-bounds
lens-mapping behavior, and the limited role of zero-valued ghost pixels.

The main conclusions are:

1. The current default infers a square source-plane bounding box from
   image-plane seed pixels ray-traced through the current lens mass model.
2. A fixed source grid is preferable while optimizing or sampling lens mass:
   every proposal then uses the same source basis, physical source-pixel scale,
   and regularization geometry.
3. A bbox derived from a preliminary parametric source fit is a reasonable way
   to initialize that fixed grid, but rejecting models that leave the bbox
   turns the bbox into an informative source-support prior. It is not merely a
   numerical choice.
4. The lens-mapping operator should remain finite and linear in source-pixel
   amplitudes. Out-of-domain source brightness should be zero; clamping,
   extrapolating, renormalizing partial weights, and inserting NaNs are all
   undesirable.
5. A zero-valued ghost ring only changes behavior within one source-grid
   spacing of the bbox. It can replace a jump in model brightness with a
   continuous linear taper, but it does not recover gradients once rays lie
   beyond the halo. It is optional and is not a substitute for an explicit
   source-coverage constraint.
6. Model admissibility belongs in the likelihood, using fixed arc/seed pixels
   and an explicit coverage diagnostic or signed-distance penalty. It should
   not be an accidental consequence of zero rows in the mapping matrix.

## Current source-grid construction

[`SimulatorConfig`](../../TinyLensGpu/ForwardSimulation/LensImage/config.py#L131-L190)
contains two image-plane masks:

- `mask` selects pixels used by the data likelihood;
- `source_seed_mask` selects pixels used to infer the source-plane bbox.

Both use `True` to mean excluded. If `source_seed_mask` is omitted, it defaults
to `mask`. A separately supplied seed mask must have an active region contained
within the active data region and should isolate the lensed arc.

The dense simulator and matrix-free operator ray-trace active seed pixels
through the current mass model. With the default settings,
[`infer_source_bbox`](../../TinyLensGpu/utils/lensing/mapping.py#L142-L190):

1. takes the absolute minimum and maximum of the seed-ray source coordinates
   (`source_bbox_outlier_frac=0`);
2. adds 5% of each coordinate span on each side
   (`source_bbox_padding=0.05`);
3. enforces a small nonzero span for point-like inputs; and
4. expands the shorter dimension to produce a square bbox.

[`build_source_grid`](../../TinyLensGpu/utils/lensing/mapping.py#L86-L94)
then constructs `n` nodes per axis with

\[
    x_j = \operatorname{linspace}(x_{\min}, x_{\max}, n),\qquad
    y_k = \operatorname{linspace}(y_{\min}, y_{\max}, n).
\]

The nodal spacing is therefore

\[
    \Delta x_s = \frac{x_{\max}-x_{\min}}{n-1},\qquad
    \Delta y_s = \frac{y_{\max}-y_{\min}}{n-1}.
\]

The bbox is inferred from seed pixels, but the final lens mapping is built for
all active image pixels or subpixels. In the dense path this happens in
[`PixelizedLensSimulator.design_matrix`](../../TinyLensGpu/ForwardSimulation/LensImage/pixelized.py#L347-L419).

### Dynamic bbox does not mean a differentiable bbox

The dense [`PixelizedImageProbModel`](../../TinyLensGpu/ObservationModel/LensImage/pixelized_image_model.py#L189-L203)
re-infers the bbox during every likelihood evaluation. The simulator applies
`jax.lax.stop_gradient` to inferred bbox bounds by default. This avoids trying
to differentiate through `min`, `max`, or quantiles, but it does not freeze the
source coordinate system: bbox values still change between mass proposals.
Consequently, the objective value changes with the bbox while its local
gradient omits that dependence.

The matrix-free
[`PixelizedImageProbModelOperator`](../../TinyLensGpu/ObservationModel/LensImage/pixelized_image_model_operator.py#L110-L195)
already accepts `fixed_source_bbox`. Its public
[`infer_source_bbox`](../../TinyLensGpu/ObservationModel/LensImage/pixelized_image_model_operator.py#L335-L343)
helper is intended to obtain a reference bbox for constructing a second model.
Current operator pipelines also pass a bbox frozen from an S0 reconstruction
into later stages. The dense probability-model constructor does not currently
expose an equivalent fixed-bbox option, although lower-level dense simulator
methods accept an explicit `source_bbox`.

## Why freeze the source grid during mass inference

With a fixed number of source pixels, a lens-dependent bbox changes all of the
following between mass proposals:

- source-grid center and extent;
- physical source-pixel spacing;
- the location and shape of every bilinear source basis function;
- which four source nodes neighbor each ray-traced image position;
- finite-difference regularization scaling; and
- evidence terms that depend on the regularization and curvature matrices.

Different mass proposals are therefore evaluated with different source
discretizations and, in a practical sense, different discretized source
priors. Detaching bbox gradients does not remove that inconsistency.

A fixed source grid makes the source coordinate system an invariant of the
pixelized stage:

```text
fixed source frame
    +-- fixed source-node positions
    +-- fixed source-pixel scale
    +-- fixed regularization geometry
    `-- lens proposals change only ray positions and interpolation weights
```

This improves automatic-differentiation consistency, but the statistical
benefit is at least as important: likelihood or evidence values from different
mass proposals become comparable under one source basis and one regularization
geometry.

## Building a fixed bbox from a parametric source fit

A preliminary parametric fit jointly constrains the lens mass and approximate
source center, scale, ellipticity, and orientation. These source results can be
used to construct the fixed coordinate system for the later pixelized source.

The repository already provides
[`source_bbox_from_center_reff`](../../TinyLensGpu/utils/lensing/mapping.py#L115-L139):

\[
    h = f_R R_e,
\]

\[
    B = [c_x-h,c_x+h]\times[c_y-h,c_y+h].
\]

Here `radius_factor=f_R` specifies the **half-width** in units of effective
radius. `radius_factor=3` gives a box with half-width \(3R_e\) and full side
length \(6R_e\); it does not give a full side length of \(3R_e\).

### Prefer a posterior envelope over a point estimate

A box formed from posterior-median center and radius ignores uncertainty in
both. A more robust handoff uses samples from the joint parametric posterior:

1. compute a source footprint for each posterior sample;
2. propagate center, radius, axis ratio, and orientation uncertainty;
3. take a high credible envelope, such as 99% or 99.5%;
4. take the union over multiple source components;
5. expand the result to the square required by the current grid; and
6. add a numerical guard margin.

A posterior-predictive isophote tied to the data's detectable surface-brightness
threshold is more meaningful than a universal multiple of \(R_e\). A Sersic
profile has infinite support, and the fraction of light enclosed by a fixed
multiple of \(R_e\) depends strongly on Sersic index.

Ellipticity also matters. TinyLensGpu uses an area-preserving circularized
coordinate transform in
[`ellipse2circle_transform`](../../TinyLensGpu/utils/geometry/transforms.py#L155-L176).
For an isophote at circularized radius \(kR_e\), the approximate semi-major and
semi-minor axes are

\[
    a = \frac{kR_e}{\sqrt q},\qquad b = kR_e\sqrt q.
\]

A simple conservative square can therefore use \(kR_e/\sqrt q\) rather than
\(kR_e\) as its geometric half-extent. A rotated-ellipse or posterior-predictive
footprint avoids the extra area of that worst-case square.

### A larger bbox reduces resolution at fixed `n`

For square half-width \(h\),

\[
    \Delta\beta = \frac{2h}{n-1}.
\]

Increasing the safety factor while holding `n` fixed lowers source-plane
resolution and changes regularization scaling. If a wider posterior envelope
is required, `n` may also need to increase to preserve the intended
\(\Delta\beta\). A bbox cannot be made arbitrarily conservative at zero cost.

## A fixed bbox is either numerical support or a physical prior

Two uses of a fixed bbox should be distinguished.

### Numerical coordinate frame

The bbox is deliberately wide enough that every scientifically plausible mass
proposal keeps relevant arc rays away from its edge. The bbox then acts mainly
as a fixed discretization choice and should have negligible influence on the
mass posterior.

### Source-support prior

A mass proposal is penalized or rejected when relevant rays leave the bbox.
The stage-one parametric source estimate is then an informative prior on source
position and extent:

\[
    p(\theta_{\rm mass})=0
    \quad\text{when the implied source violates the adopted support.}
\]

This may be intentional, but it should not be described as a purely numerical
optimization. Mass-sheet and related source-position transformations can map
nearly equivalent image configurations to sources with different positions or
scales. A fixed support can suppress those degeneracies and narrow or bias the
mass posterior. Parametric model mismatch, clumpy sources, multiple source
components, extended low-surface-brightness emission, and wavelength-dependent
morphology create additional failure modes.

The precise interpretation of an out-of-bbox proposal is therefore:

> The proposal is inconsistent with the source-support prior inherited from
> the preliminary parametric fit.

It is not, without that prior, a proof that the mass model is physically wrong.

## Current out-of-bounds lens mapping

[`lens_mapping_operator_bilinear_from`](../../TinyLensGpu/utils/lensing/mapping.py#L40-L83)
computes normalized coordinates

\[
    u_x=\frac{\beta_x-x_{\min}}{\Delta x_s},\qquad
    u_y=\frac{\beta_y-y_{\min}}{\Delta y_s},
\]

and considers a point valid when

\[
    0\le u_x\le n-1,\qquad 0\le u_y\le n-1.
\]

Valid points receive the usual four bilinear weights. Invalid points receive
four zero weights. Neighbor indices are clipped only to keep gathers and
scatters in bounds; zero weights ensure that invalid points do not contribute
to edge source pixels. The resulting dense mapping row is zero, as covered by
[`test_out_of_bounds_rays_produce_zero_rows`](../../tests/test_pixelized_source_utils.py#L85-L100).

A zero row means zero **pre-PSF source contribution** for that image-plane
sample. The final convolved image pixel can still receive flux from neighboring
image pixels through the PSF.

### Why a zero row is reasonable but insufficient

Zero is the appropriate terminal source-brightness condition if the finite
source grid represents compact support. The mapping matrix remains finite and
linear in source amplitudes. However, a high-S/N arc ray outside the grid then
has no direct mapping derivative with respect to lens parameters. Its residual
can be large without providing a useful direction for returning to the grid.
PSF mixing, lens light, noise, and other in-bounds pixels can also partially
absorb the discrepancy. A zero row is therefore forward-model semantics, not a
complete mass-model rejection policy.

## Rejected boundary behaviors

### Coordinate clamping

Clamping \(\beta\) into the bbox assigns every outside ray to the nearest edge
source nodes. This piles arbitrary outside area onto the source boundary,
encourages bright edge pixels, and lets an incorrect mass proposal continue to
fit data. It should not be used.

### Linear extrapolation

The grid contains no justified morphology outside its support. Extrapolation
can create negative or unbounded surface brightness and makes predictions
depend strongly on edge values. It should not be used.

### Renormalizing partial weights

If only some interpolation neighbors are represented, their remaining weights
should not be rescaled to sum to one. Renormalization redistributes missing
outside flux into edge pixels and is another form of clamping.

### NaN or exceptions in the mapping matrix

NaNs propagate into \(L^T C^{-1}L\), Cholesky factors, iterative solvers, and
evidence values. JIT-compiled code also cannot use ordinary exceptions as a
per-proposal control path. The mapping should always return finite arrays;
likelihood validity is a separate concern.

## Zero-valued ghost pixels

A ghost construction extends the regular grid by one knot spacing and fixes
all added source values to zero. The ghost values are not linear parameters and
do not require new mapping-matrix columns.

For the current nodal grid, the natural ghost width is one source spacing:

\[
    h_{\rm ghost,x}=\Delta x_s,\qquad
    h_{\rm ghost,y}=\Delta y_s.
\]

Choosing a different width defines a separate tapering convention rather than
ordinary equally spaced bilinear interpolation.

### One-dimensional behavior

Let the final solved source node have value \(s_e\) at \(x_{\max}\). With the
current hard cutoff,

\[
    I(x_{\max})=s_e,\qquad I(x_{\max}+\epsilon)=0,
\]

so the model value is discontinuous whenever \(s_e\ne0\).

Place a fixed zero ghost node at

\[
    x_g=x_{\max}+\Delta x_s.
\]

For \(t=(x-x_{\max})/\Delta x_s\) in \([0,1]\), interpolation against that ghost
node gives

\[
    I(x)=(1-t)s_e.
\]

The corresponding rows over the solved source vector change continuously:

```text
x = xmax          -> [..., 1.0]
x = xmax + 0.2 dx -> [..., 0.8]
x = xmax + 0.9 dx -> [..., 0.1]
x >= xmax + dx    -> [..., 0.0]
```

Weights associated with out-of-range ghost indices are omitted, not clipped to
an active source column. Consequently, row sums are one in the original grid,
between zero and one in the ghost halo, and zero beyond the halo. They must not
be renormalized.

### What the ghost ring improves

The ghost ring changes only the one-spacing interval immediately outside the
original grid. It replaces a jump in model brightness with a continuous taper
and retains a direct interpolation gradient within that interval:

\[
    \frac{\partial I}{\partial x}=-\frac{s_e}{\Delta x_s}.
\]

This changes a discontinuous function into a continuous, piecewise-linear one.
That is a genuine numerical improvement when rays cross a boundary whose edge
source value is nonzero.

### What the ghost ring does not improve

At the outside ghost boundary,

\[
    \lim_{x\to x_g^-}\frac{\partial I}{\partial x}
      =-\frac{s_e}{\Delta x_s},\qquad
    \lim_{x\to x_g^+}\frac{\partial I}{\partial x}=0.
\]

The gradient is still discontinuous. Beyond the halo, the mapping row and its
direct position gradient are both zero. A ghost ring therefore cannot recover
an optimizer from a proposal whose relevant rays lie far outside the grid.

This remaining derivative kink is usually less severe than a jump in the model
value. Standard bilinear interpolation is already continuous but only
piecewise differentiable at every internal cell boundary. Nevertheless, the
far-outside zero-gradient plateau remains and must be handled separately.

### Smooth tapers and zero boundary rings

A cubic smoothstep such as

\[
    w(t)=1-3t^2+2t^3
\]

can make the taper derivative vanish at its endpoints. More elaborate Hermite
or spline constructions can match an interior slope as well. These approaches
change the source basis near the edge and still produce a zero-gradient region
once the compact-support taper ends. The extra complexity is not justified
unless measured boundary artifacts require it.

An alternative is to fix the outermost source-node ring itself to zero and
solve only for the interior nodes. Standard interpolation then approaches zero
before reaching the bbox edge, so a hard zero outside is value-continuous. This
uses up one source node on each side and may require increasing `n` to preserve
interior resolution.

If the bbox is conservative and reconstructed edge brightness is already
negligible, the hard-cutoff discontinuity is also negligible. In that common
case, neither external ghost nodes nor a fixed zero boundary ring adds much
value.

## Separate mapping from source-coverage validity

The mapping module should have a narrow responsibility:

- build finite interpolation weights and indices;
- return zero source contribution outside the adopted support; and
- expose coverage information without deciding whether a lens model is valid.

A useful result would retain information already computed at the interpolation
stage, conceptually:

```python
MappingResult(
    weights=weights,
    indices=indices,
    inside=inside,
    coverage=coverage,
)
```

For hard-zero interpolation, `coverage` may be Boolean. With a zero ghost or
taper, a useful scalar is

\[
    c_i=\sum_j L_{ij},
\]

which is one inside, between zero and one in the taper, and zero outside.

The likelihood should consume this diagnostic through an explicit coverage
policy. It should not inspect all active data pixels: background pixels can
legitimately map outside a compact source grid. Coverage should be evaluated
on a fixed arc/seed selection, possibly weighted by preliminary parametric
source flux or observed source signal-to-noise.

One aggregate diagnostic is

\[
    C=\frac{\sum_i w_i c_i}{\sum_i w_i}.
\]

A stricter geometric diagnostic uses the signed distance from each seed ray to
the nearest bbox edge:

\[
    m_i=\min\left(
      \beta_{x,i}-x_{\min},
      x_{\max}-\beta_{x,i},
      \beta_{y,i}-y_{\min},
      y_{\max}-\beta_{y,i}
    \right).
\]

Here \(m_i>0\) is inside, \(m_i=0\) is on the boundary, and \(m_i<0\) is
outside. Expressing \(m_i\) in units of \(\Delta\beta\) makes guard margins
portable across grid resolutions.

### Gradient-based inference

For optimization or HMC, a smooth penalty should begin before important rays
reach the numerical edge. For example,

\[
    \log p_{\rm cov}
      =-\lambda_{\rm cov}\sum_i w_i
        \operatorname{softplus}\!\left(
          \frac{m_{\rm guard}-m_i}{\tau}
        \right)^2.
\]

Unlike compact-support interpolation, this signed-distance penalty can retain a
useful gradient when a ray lies well outside the bbox. A guard margin of one or
two source spacings is a reasonable starting point, subject to validation on
representative systems.

### Non-gradient samplers

Nested sampling can use a hard rejection or an extremely low likelihood once
weighted coverage crosses a declared threshold. A single-pixel `any(outside)`
rule is fragile because masks can contain contamination, PSF wings, or isolated
low-significance pixels.

In both cases the coverage term is an explicit source-support prior and should
be reported as such.

## Subpixel integration

When `nsub > 1`, some subpixels of a native image pixel may map inside the
source support and others outside. If outside source brightness is zero, the
native mapping remains the average over **all** subpixels:

\[
    L_{\rm native}
      =\frac{1}{n_{\rm sub}^2}\sum_k L_{{\rm sub},k}.
\]

The denominator must not be changed to the number of valid subpixels; doing so
would amplify the in-bounds contribution and violate image-pixel area
integration. The current dense aggregation uses the fixed
`nsub ** 2` denominator in
[`_aggregate_mapping_to_native`](../../TinyLensGpu/ForwardSimulation/LensImage/pixelized.py#L325-L345).

## Recommended staged design

A practical parametric-to-pixelized pipeline should use the following
separation of concerns:

1. Fit a parametric lens plus source model and retain the joint posterior, not
   only its best fit.
2. Build a conservative square bbox from a high posterior-predictive source
   footprint, accounting for center uncertainty, source ellipticity, multiple
   components, and detectable surface brightness.
3. Choose `n` jointly with the bbox so that the source-pixel spacing remains
   scientifically adequate.
4. Freeze bbox, `n`, source-node coordinates, and regularization geometry for
   the pixelized mass-inference stage.
5. Keep the mapping finite and linear. Use bilinear interpolation inside and
   zero source contribution outside. Do not clamp, extrapolate, renormalize, or
   insert NaNs.
6. Initially omit ghost pixels unless nonzero reconstructed edge brightness
   produces observable boundary jumps. A one-spacing zero ghost ring is an
   optional continuity improvement, not a validity mechanism.
7. Evaluate a fixed arc/seed coverage diagnostic separately. Use a smooth
   signed-distance penalty for gradient-based inference or a documented hard
   coverage threshold for non-gradient sampling.
8. Record minimum seed-ray margin, weighted outside fraction, and reconstructed
   flux in the outer one or two source-node rings.

## Validation and sensitivity checks

A fixed bbox is safe only if its scientific effect is measured. Recommended
checks include:

- repeat representative fits with posterior envelopes comparable to
  \(2.5R_e\), \(3R_e\), and \(4R_e\), increasing `n` as needed to keep
  \(\Delta\beta\) approximately fixed;
- compare lens-mass posteriors rather than only maximum likelihood;
- verify that significant reconstructed source flux does not accumulate in the
  outer one or two source-node rings;
- monitor the minimum signed margin and weighted coverage of fixed arc/seed
  pixels over posterior samples;
- test elongated, multi-component, and high-Sersic-index sources;
- test mass proposals near known source-scale degeneracies; and
- for gradient-based paths, compare automatic gradients against finite
  differences away from interpolation cell boundaries and inspect optimizer
  recovery from deliberately displaced proposals.

If the lens posterior moves materially when only the conservative bbox extent
changes at fixed source resolution, the bbox is functioning as an active prior
rather than a neutral numerical coordinate frame. That may be acceptable, but
it must be an explicit modeling decision.
