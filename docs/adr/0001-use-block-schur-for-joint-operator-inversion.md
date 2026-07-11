# Use block-Schur preconditioning for joint operator inversion

The operator backend will retain the source–lens-light cross-curvature block and use a block-Schur preconditioner for joint semi-linear inversion. This costs additional storage and operator applications, but preserves the dominant source–lens-light covariance in both iterative convergence and the preconditioner log determinant used to approximate the evidence; an uncoupled block-diagonal joint preconditioner will not be provided.
