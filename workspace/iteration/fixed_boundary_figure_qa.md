# Fixed-boundary figure QA

Source: `plot_fixed_boundary.py`; outputs: `fixed_boundary_v2/figure/`.
The contract is recorded in `fixed_boundary_figure_contract.md`.

- Render-time alignment: passed the unchanged 1.5 pt gate on all three final
  plot-area rectangles; measurement JSON preserved beside exports.
- Source preflight: 17 pass, 0 fail. Four warnings reviewed: 279.4 mm width,
  300 dpi PNG and no TIFF are intentional for this internal report figure,
  not a journal-production submission; the log-guard warning is conservative,
  since the source explicitly rejects nonpositive stationarity values before
  drawing the logarithmic axis.
- Final PDF glyph audit: minimum 5.6 pt, 0 below the 5 pt floor; editable text.
- Final PDF collision audit: 0 fail, 0 warn. JSON and overlay PDF preserved.
- Inspected the final PNG panel by panel and as an assembled figure. No crop,
  annotation overlap, mismatched axes geometry or unsupported smoothing.

| Panel | Inference / evidence | Center and variability | Replicate unit | Labels / scale | Render review |
|---|---|---|---|---|---|
| a | Spatial context of the predeclared zero-ring/warm candidate, without claiming convergence | First repeat, no aggregate or intervals | One saved endpoint, 802 valid nodes | Complete potential-node footprint; equal angular x/y scale; symmetric δκ colors, invalid nodes gray | Pass; gate failure stated below colorbar |
| b | All four conditions miss the strict stationarity criterion | All raw values; no means/intervals | Three technical repeats per condition, all 12 retained | Same logarithmic scale, actual .001 gate drawn; condition labels explicit | Pass; all points visible above threshold |
| c | Diagnostic evidence varies across repeats even where fields look similar | All raw values; no means/intervals | Same 12 endpoints as b | Common offset 6305 explicitly shown; same conditions and dot order as b | Pass; diagnostic qualification also stated in footer |

Lines connect repeat-order dots within a condition solely to expose their
variation; no fit, statistical significance or independent-noise sampling is
implied. Quantitative values are exported in `source_data.csv`. All figures
are diagnostics of unqualified endpoints; none is labeled a final detection.
