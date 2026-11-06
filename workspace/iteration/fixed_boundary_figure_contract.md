# Fixed-boundary J0946 diagnostic figure

- Question: does the fixed zero ring plus a single-step initialization make
  the fixed-lambda nonlinear solve sufficiently converged and repeatable?
- Claim: the candidate correction is reproducible in broad morphology, but
  all 12 endpoints miss the unchanged stationarity gate and their diagnostic
  evidence is still sensitive to technical repeats.
- Archetype: image plate + quantitative validation; Python/matplotlib, reusing
  the established backend. New layout, no external/private plotting template.
- Panels: (a) zero-ring/single-step-start repeat 0, declared before inspecting
  maps, supplies spatial context; (b) all 12 stationarity diagnostics plus the
  actual gate tests convergence; (c) all 12 diagnostic evidence values tests
  score repeatability. No best-looking or passing endpoint is selected.
- Units: one fixed observed lens, no injected truth. Three technical repeats
  per condition, identical data and intended starts, re-created GPU coordinate
  factors per repeat. Dots are individual runs, no statistical intervals or
  significance tests. Paired raw rows are exported to CSV.
- Integrity: all 802 finite/core kappa nodes retained, unsupported nodes masked
  rather than zero-filled; original angular coordinates; nearest-cell display,
  symmetric untransformed color limits; no smoothing, local contrast adjustment,
  or feature selection. Axes show the complete potential-node bounding box.
- Export: report figure, 11 x 4.7 inches, editable SVG/PDF and 300 dpi PNG;
  fonts >=8 pt, PDF glyph floor >=5 pt. Preserve source CSV, alignment JSON,
  final PDF text/collision audits and rendered visual inspection.
- Scope: diagnostic local-GN evidence at unqualified endpoints must not be
  presented as validated marginal evidence, Bayes factors, or a detection.
