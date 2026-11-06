# DIRECT / L-BFGS experiment figure contract

Question: how does evidence-based regularization selection change the recovered
potential correction relative to fixed weights, and what search behaviour and
matched-control behaviour accompany that change? The figures describe the saved
experiment; they do not select hyperparameters or establish a global maximum.

Python/matplotlib is the established workflow. Export editable PDF/SVG and PNG,
with 180 mm width, main text of 7–9 pt and a 5 pt rendered-glyph floor. Run rendered panel-alignment, PDF
font and collision audits, followed by visual inspection. Source NPZ and JSONL
files are the quantitative source data; CSV exports preserve candidate history.

The correction figure is an image plate with paired fixed/selected maps for the
subhalo image and its matched no-subhalo control. Within each row use the same
symmetric colour range, coordinates and 724 derivative-valid core nodes. Each
row has its own explicitly labelled colour bar so the much smaller control
field remains visible. All 956 potential coefficients remain in the fit;
support-only nodes have undefined derivatives and are not drawn. Do not align,
centre, clip or interpolate the fitted field. Mark injection position only in
the injected case. These four views establish the change and its control.

A separate search-history figure shows both cases' local-GN evidence changes
relative to their own fixed-weight baseline, all finite candidate scores and
the cumulative best score. Rejected candidates are marked in an axis-coordinate
bottom strip, not assigned an invented evidence. Use a labelled symmetric-log
axis if needed to retain low-evidence candidates. No evidence difference between
the two different data images is interpreted as a detection Bayes factor.

This is one fixed mock and one control, with no between-seed uncertainty bars.
Budget termination, invalid candidates, initialization, local-GN approximation
and repeat discrepancies remain in the report even if the maps look plausible.

## Requested six-panel recovery diagnostic

Question: does the frozen evidence-selected result reproduce the observed image,
and where does its convergence correction differ from the injected perturbation?
Use an image-plate layout with two rows and three equal columns: data, nonlinear
model, (data-model)/noise; injected perturbation kappa, recovered perturbation
kappa, recovered-minus-injected kappa. These six roles distinguish image-fit
adequacy from recovery fidelity. This remains one seed and one point estimate.

Read the saved fit arrays, not a new reconstruction. Obtain injected truth from
the existing scientific mock archive after checking exact data, noise, mask,
node coordinates and integer topology. By default compare discrete kappa from
the same centered potential stencil on the same 0.075 arcsec grid. Preserve the
analytic truth alongside it in source data, and state the derivative convention.

The image row displays all 22016 fitted pixels; the other 17984 pixels are
outside the fitting mask and remain blank in all three panels. The kappa row
shows all 724 derivative-valid core nodes; 232 support-only nodes are blank.
Data/model share intensity limits, injected/recovered kappa share symmetric
limits. Residual and kappa-difference maps use separately labelled symmetric
limits that contain every displayed value. Do not clip, align, smooth or
subtract an offset from the fields. The difference sign is recovered-minus-
injected; normalized image residual is data-minus-model divided by noise.

Python/matplotlib, 183 mm width, 7–9 pt main text and 5 pt rendered-glyph floor;
editable PDF/SVG, 300 dpi PNG preview and 600 dpi TIFF. Audit the final six image
rectangles, exclude only colourbar axes from comparison, run PDF font/collision
checks, then inspect the complete figure. Retain exact six-panel source arrays,
hashes, masks, scalar checks and scientific conventions in an output manifest.
