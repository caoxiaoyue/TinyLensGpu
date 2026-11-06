# J0946 C1 source comparison: figure contract

The question is whether removing source grid-line derivative jumps improves
L-BFGS convergence and repeatable potential-correction fields at fixed priors.
The field figure shows the effect on the recovered field, bounded by each
method's strict convergence status; it cannot establish a true subhalo mass.

Python/matplotlib is the established backend in this experiment series. Use
an original quantitative image grid, not an external plotting template.
Export PNG/PDF/SVG with editable vector text and at least 8 pt plot typography.
This is an exploratory research artifact, without a journal submission target.

The 2x3 panel map has the old fixed-M3 selected lens strengths on the first
row and the single-step lens strengths on the second. These are pre-existing
failure cases, selected by name before inspecting new cubic outputs. Columns
are bilinear, C1 cubic and their field difference. Show repeat 0 for both
methods, never select the best repeat. The first two columns share one
symmetric color scale across both rows; the difference column shares a
separate symmetric scale. Show all 802 finite core values with no smoothing,
interpolation, clipping to positive convergence or display crop. Blank cells
are support-only or absent potential nodes. Units of convergence are dimensionless.

All 24 fits, including strong-prior and previous-iteration controls not shown
as image panels, go into the source-data CSV and numerical report. Report all
three technical repeats, fixed source hyperparameters, signed coefficients,
and the fact that these are fixed-lambda comparisons rather than an outer
evidence search. They are repeated numerical solves of the same observations,
not independent datasets or uncertainty samples. State no injected truth.

The final figure title records the unchanged native-success/GN gate outcome;
the figure does not imply that a larger peak is more physically accurate.
Each panel's title identifies the method/difference and each image row records
the fixed lens pair in its label. Gate counts are shown for each method/row.
Panel alignment must pass at 1.5 pt; final PDF text and collision audits plus
visual inspection are required. Preserve QA records alongside exports.
