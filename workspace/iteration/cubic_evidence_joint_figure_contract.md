# Three-hyperparameter C1 evidence comparison

Question: does freeing source lambda_s while reoptimizing lambda4/lambda0
preserve the near-.2 perturbation peak, and what changes in the entire field?

Use the established Python/matplotlib backend. Structurally adapt the previous
C1 evidence-search image plate, preserving the mapping and spatial integrity
guards while changing all fixed-source labels and the baseline definition.

Baseline = this run's cold remeasurement of the previous best with M3 lambda_s
fixed. Selected = the original highest qualified evidence trial of this joint
three-dimensional search. Do not choose a favorable repetition. Source adaptive
rho, template and source_scale remain fixed; lambda_s itself is now optimized.
Both maps use the same C1 source, joint L-BFGS and input data.

Panels: lens-subtracted data, selected forward image, normalized residual;
then baseline convergence, selected convergence, selected minus baseline.
Use all2293 likelihood pixels and802 finite potential-core nodes; assert view
bounds include them all. Unused blank image border may be cropped for display.
No map smoothing, positivity clipping or hidden negative corrections. Data/model
share a full-range brightness normalization, baseline/selected share symmetric
full-range kappa normalization. Residual and difference have separately labelled
symmetric scales. All actual fit records, including rejected points and cold
repeats, go to CSV, and plotted arrays go to NPZ.

Figures show numerical solves of the same observed data, not independent data
or posterior samples. Counts are numerical qualification counts, not physical
confidence. Record source lambda_s in the header and the native outer exits in
the report; finite DIRECT exploration is not a global-optimum certificate.

Exploratory PNG300dpi, editable PDF/SVG, minimum8pt text. Require final rendered
alignment1.5pt, PDF text/collision checks and inspection of every panel.
