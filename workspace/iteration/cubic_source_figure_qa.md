# C1 source figure QA

Python/matplotlib produced the final PNG, PDF and SVG from the immutable
`cubic_source_v1` outputs. All 24 trial records are in `figure/source_data.csv`.
Six map panels display the predeclared repeat 0 at the prior selected and
single-step lens pairs; no trial is selected by outcome. The other two lens
pairs remain in the complete numerical report and source CSV.

The same finite 802-node potential core is displayed for each map without
display smoothing or cropping. Two field columns share one symmetric scale;
the difference column uses a separately labelled common scale. Pale fields
in the strongly regularized first row are expected under the common scale.
Gray regions are outside the finite core, not discarded fit residuals.

Final rendered checks:

- Panel alignment: PASS, 7 comparisons, 0 failures/warnings/exemptions, 1.5 pt.
- PDF text: all 87 text runs at least 8 pt, 0 glyph-floor violations.
- Final PDF collision audit: PASS, 0 failures and 0 warnings.
- Visual inspection of all six maps and full figure: labels, signs, scales,
  coordinates, counts and row ordering match the saved inputs; no clipped
  labels, overlapping titles, missing colorbars or cropped field core.
- Source preflight: 18 PASS, 0 FAIL. Three documented WARN: exploratory 228.6 mm
  figure width, 300 dpi PNG, no TIFF. PDF/SVG retain editable text; there is no
  journal production requirement for this test artifact.

An initial layout had title/x-label collisions between rows. The final export
uses a taller canvas and larger row gutters; all audits were rerun on that
final PDF. This was a layout change only, with no data or scientific changes.

This is a numerical repeat experiment on real data, not a posterior or noise
uncertainty visualization. Counts use the unchanged native-success/GN gate.
There is no injected convergence truth and no outer hyperparameter search.
