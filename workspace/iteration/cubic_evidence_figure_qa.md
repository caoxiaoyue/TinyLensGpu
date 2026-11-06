# C1 evidence-search figure QA

Final figure uses Python/matplotlib and the retained original trial194, with
trial3 as the remeasured C1 baseline at single-step lens weights. The original
quantitative layout is new; only the existing potential-grid image helper and
typography were reused. Both lower maps are iterative C1 solutions, not one
single-step and one iterative solution. No new data were generated.

Data mapping:

- a/b: saved lens-subtracted image and selected nonlinear source model image;
  both use the same full-range brightness normalization and the same mask.
- c: saved (data-model)/noise; the full finite symmetric range is shown.
- d/e: convergence on all 802 finite core nodes, one common symmetric scale.
- f: selected minus baseline convergence, its own labelled symmetric scale.
- all 2293 likelihood pixels are retained, including the pre-existing high-noise
  pixels. Gray is outside the likelihood mask or finite potential core. View
  bounds are explicitly asserted to include every selected image pixel and
  every finite potential-core value; only unused blank image border is cropped.

No spatial smoothing, positivity clipping, histogram equalization, source
reconstruction reselection or cherry picking of a repeat. The source CSV
contains all345 actual fits, including10 rejected points. Plotted data arrays,
spatial extents and view limits are saved verbatim in `plotted_arrays.npz`.

Final checks:

- Rendered six-panel alignment: PASS, no exemptions, 1.5pt tolerance.
- Final PDF text audit: PASS, minimum8pt, no undersized text.
- Final PDF collision audit: PASS, no failures or warnings.
- Source static preflight:18 PASS,0 FAIL;3 documented WARN for exploratory
  254mm width,300dpi preview and lack of a journal TIFF production file.
  Editable PDF/SVG text are retained; no journal submission was requested.
- All six panels and full composition visually inspected: no clipped fields,
  titles or colorbars; captions identify the C1 baseline and the repeat count.
  A first-export overlap between the repeat header and panelb was corrected
  by increasing header separation, followed by rerunning all final audits.

The6/6 count means the selected original endpoint and five same-data cold
restarts passed unchanged numerical checks. It is not a posterior confidence
or a proof of physical subhalo recovery. The main peak stays near.2, but the
complete baseline-to-selected convergence difference is substantial and is
shown without suppressing it. DIRECT's finite budget and the approximate
GN-evidence definition are recorded in the numerical report.
