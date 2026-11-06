# C1 evidence-search figure contract

Question: after maximizing the same coupled GN evidence over lens lambda4 and
lambda0 at fixed M3 source hyperparameters, what perturbation field and fit
does the retained candidate produce, and how does it differ from the previous
fixed single-step lens-weight case?

Use Python/matplotlib, the established backend. Build a new 2x3 image plate;
reuse only the checked potential-node-to-image helper and typography from the
earlier workspace figure. The scientific inputs are this search's own best
accepted trial and its remeasured C1 single-step-lens-weight baseline. Both are
iterative C1 results; the baseline is not a new single-step reconstruction.

Panels: (a) lens-subtracted observed image, (b) selected C1 forward model,
(c) normalized residual, (d) baseline delta-kappa, (e) selected delta-kappa,
(f) selected minus baseline delta-kappa. The upper row establishes the image
fit and residual structure; the lower row answers the field-amplitude question.
Use the exact retained original state, not the most attractive repetition.

All 2293 likelihood pixels, including the existing high-noise pixels, are
included. Pixels outside the likelihood mask are gray. Spatial view bounds
must include every selected image pixel and all 802 finite potential-core
values. Any unused blank image border outside this view is only a display
crop; assert no selected likelihood pixel is omitted. No map smoothing or
clipping to positive brightness/convergence. A/B share full-range brightness
normalization; D/E share full-range symmetric convergence normalization. C/F
have separately labelled full-range symmetric scales. No injected truth.

All actual search/repeat fits, including rejected points, go to source CSV.
Export the exact plotted arrays to NPZ. Repeats are technical cold starts of
the same observation, not data uncertainty. Record the actual outer native
exit statuses separately; finite-budget DIRECT does not certify a global
optimum. Field peaks and evidence are conditional on the fixed macro/priors.

Exploratory PNG (300 dpi), editable PDF/SVG, minimum 8 pt text; no target
journal production size. Final rendered panel alignment at 1.5 pt, PDF glyph
floor and collision checks plus every-panel visual inspection are required.
