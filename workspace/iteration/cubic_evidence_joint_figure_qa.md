# Joint source/lens evidence figure QA

Python/matplotlib structurally adapts the previous image plate. Baseline is
trial1 of this run: remeasured C1 solution at the previous two-dimensional
optimum with M3 lambda_s fixed. Selected is the original retained trial497;
the figure never selects a preferred cold restart or replaces the state with
the separate local confirmation result.

Panels map lens-subtracted data, selected nonlinear model and normalized
residual; then baseline convergence, selected convergence and their difference.
All2293 likelihood pixels and802 finite potential-core nodes are included.
The view guard verifies no selected data pixel is cropped; only unused blank
native-image border is omitted. Gray denotes outside the mask or finite core.
No interpolation/smoothing of map display, positivity clipping or scale change
between the two convergence maps. Difference and residual use labelled full
symmetric ranges; data/model share their full brightness range.

All555 main-run trial records, including8 rejected exploration points, are
in CSV. The raw plotted arrays, extents and view are saved to NPZ. The separate
confirmation trace is independently archived, not mixed into the figure count.

Final QA:

- Six-panel alignment PASS,7 comparisons, no failures/warnings/exemptions,
  physical tolerance1.5pt.
- PDF text minimum8pt,97 text runs, no undersized glyphs.
- Final PDF collision audit PASS, no failures or warnings.
- Source static preflight18 PASS,0 FAIL. Three accepted exploratory-export
  WARN:254mm canvas,300dpi PNG preview,no journal-production TIFF. Editable
  PDF/SVG text are provided; there is no journal submission target.
- Every panel and full figure visually inspected. Spatial coordinates, peak
  labels, masks, colorbars and field difference match immutable data. Captions
  explicitly distinguish fixed source lambda_s from jointly optimized strengths.
  There are no clipped titles, overlaps or missing negative convergence values.

6/6 denotes original endpoint plus five same-data cold restarts that passed
unchanged numerical checks; it is not a posterior confidence. The reduction
from.20057 to.17433 is shown rather than described as identical recovery. This
figure establishes a conditional numerical reconstruction, not an injected
subhalo truth or global-maximum certificate.
