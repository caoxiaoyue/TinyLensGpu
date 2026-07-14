# Stabilizing FISTA for joint pixel-source and MGE lens-light inversion

## Context

The operator backend jointly solves the linear parameters

\[
x = (s, a_l),
\]

where `s` contains pixelized source intensities and `a_l` contains lens-light
basis amplitudes. Both groups must remain non-negative when the lens light is
represented by an MGE, so this example uses projected FISTA rather than the
unconstrained PCG solution as its final linear estimate.

This note records the reusable lessons from the joint MGE mock experiment. The
corresponding measured results are kept in
[`docs/validation/pix-src-lens-light-mge-operator-mock.md`](../validation/pix-src-lens-light-mge-operator-mock.md).

## What failed

A zero-initialized FISTA solve was not reliable for the original 20-component
MGE. It remained outside the projected-gradient convergence gate after 5000
iterations. Adding a PCG warm start alone did not cure that case.

The underlying difficulty was not just the number of unknowns. Many nearby,
fixed-width Gaussian profiles described nearly the same image structure, which
created strongly correlated and poorly conditioned lens-light directions. Weak
zero-order regularization stabilizes those amplitudes, but does not make an
over-redundant basis cheap for a first-order constrained solver.

## Working strategy

The stable production configuration combines two measures:

1. Solve the same regularized joint curvature system approximately with
   block-Schur-preconditioned PCG, project that estimate onto the non-negative
   orthant, and use it as the FISTA initial point.
2. Use the smallest MGE basis that can represent the expected lens light. Ten
   Gaussian components were sufficient for the single-Sersic mock; twenty were
   unnecessarily redundant.

The PCG result is only a warm start. FISTA still performs the final constrained
optimization and enforces non-negativity for both source pixels and lens-light
amplitudes. The block-Schur preconditioner retains the source--lens cross block,
so its starting estimate includes the dominant covariance between the two
linear components.

## Convergence and evidence

Do not treat a finite iterate as a successful solve merely because FISTA
exhausted its iteration budget. Use the projected-gradient convergence metric;
if it fails its tolerance, gate the likelihood rather than allowing an
under-converged linear solution into sampling.

For the successful mock configuration, 500, 1000, and 5000 FISTA iterations
gave truth-point log-evidence approximations of 2864.832, 2864.878, and
2864.890. Thus 500 iterations changed the value by about 0.06 relative to the
5000-iteration reference and was an acceptable sampling compromise for that
specific problem. This comparison should be repeated when the image size,
source grid, regularization range, or light basis changes.

The non-negative MAP estimate and the evidence curvature approximation are
deliberately different concepts here. FISTA supplies the constrained MAP
solution, while the determinant term continues to use the unconstrained
regularized curvature approximation. This is an approximation to the evidence,
not the exact normalization of a truncated Gaussian posterior.

## Practical tuning order

When FISTA convergence is poor:

1. Inspect the physical basis first. Remove Gaussian components that are
   redundant at the data resolution instead of immediately increasing the
   iteration count.
2. Check that source and lens-light regularization are both present and that the
   joint block-Schur system is positive definite.
3. Use the projected block-Schur PCG solution as the initial point.
4. Compare the projected-gradient metric and log likelihood at successively
   larger iteration budgets at representative parameter points.
5. Only then choose the smallest iteration budget whose likelihood error is
   negligible for the intended sampler.

These observations are problem-dependent. Ten MGE components and 500 FISTA
iterations are validated settings for the current single-Sersic mock, not
universal defaults for arbitrary galaxies.

## PNPG replacement

The 20-component MGE remains too ill-conditioned for scalar-step FISTA: its
lens-light Gram block spans more dynamic range than float32 can resolve after
adding the default weak amplitude regularization. The operator backend now
offers `solver_type="pnpg"`, which combines:

1. an unconstrained block-Schur PCG estimate, projected and minimized along
   its feasible ray to obtain a safe warm start;
2. the non-negative variable transformation `x = diag(P)^(-1/2) y`, using the
   existing preconditioner diagonal to equilibrate source-pixel and MGE
   amplitude scales;
3. projected Nesterov iterations with quadratic backtracking and gradient
   restart in the equilibrated coordinates; and
4. a componentwise-scaled KKT gate, so large RHS entries cannot hide an
   unconverged low-curvature variable.

The small dense lens and Schur preconditioner blocks are also diagonally
equilibrated before Cholesky factorization. A precision-relative eigenvalue
floor repairs float32 roundoff indefiniteness in the preconditioner only; it
does not change the physical joint curvature operator.

The corresponding 20-MGE end-to-end measurement is recorded in
[`docs/validation/pix-src-lens-light-mge-pnpg-operator-mock.md`](../validation/pix-src-lens-light-mge-pnpg-operator-mock.md).
