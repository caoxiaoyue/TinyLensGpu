## ADDED Requirements

### Requirement: Operator backend FISTA solver selection
`PixelizedImageProbModelOperator` SHALL accept `solver_type="pcg"` and `solver_type="fista"` constructor values. `solver_type="pcg"` SHALL preserve the existing unconstrained PCG source solve. `solver_type="fista"` SHALL solve the source reconstruction with a matrix-free FISTA algorithm that enforces non-negative source pixels.

#### Scenario: FISTA solver is selected
- **WHEN** a `PixelizedImageProbModelOperator` is constructed with `solver_type="fista"`
- **THEN** its evidence and forward-model source solves use the FISTA solver path

#### Scenario: Invalid solver is rejected
- **WHEN** a `PixelizedImageProbModelOperator` is constructed with an unsupported `solver_type`
- **THEN** construction fails with a clear `ValueError`

### Requirement: Hard non-negative source reconstruction
The FISTA solver SHALL return source pixel values with shape `(n * n,)` and dtype compatible with the operator inputs. Every returned source pixel SHALL be greater than or equal to zero up to numerical tolerance.

#### Scenario: Forward model returns non-negative source
- **WHEN** `forward_model(return_source=True)` is called on a model configured with `solver_type="fista"`
- **THEN** the returned source vector has shape `(n * n,)`
- **THEN** all returned source pixels are non-negative within numerical tolerance

#### Scenario: Evidence solve uses non-negative source
- **WHEN** `__call__()` evaluates evidence for a model configured with `solver_type="fista"`
- **THEN** the internal source reconstruction used for the data and regularization energies is the FISTA non-negative solution

### Requirement: Matrix-free JAX-compatible FISTA
The FISTA implementation SHALL be JAX `jit` compatible, side-effect free, and SHALL use existing matrix-free operator data and matvec callbacks rather than constructing dense design matrices or dense curvature matrices.

#### Scenario: FISTA runs through JIT-compatible loops
- **WHEN** the FISTA solver is traced by JAX
- **THEN** iteration is represented with fixed-shape JAX control flow such as `lax.scan` or an equivalent JIT-compatible loop

#### Scenario: FISTA avoids dense operator materialization
- **WHEN** the FISTA solver computes gradient steps
- **THEN** it applies `A(s)` through the existing operator matvec path
- **THEN** it does not construct an explicit `(Nd, Ns)` design matrix or `(Ns, Ns)` curvature matrix

### Requirement: FISTA convergence diagnostics and evidence gating
The FISTA solver SHALL return diagnostics that include at least iteration count, a finite convergence metric, and a boolean convergence status. `PixelizedImageProbModelOperator` SHALL penalize non-converged FISTA solves in evidence consistently with the existing non-converged PCG behavior.

#### Scenario: Converged FISTA evidence is finite
- **WHEN** the FISTA solver converges for a valid pixelized-source operator problem
- **THEN** `PixelizedImageProbModelOperator.__call__()` returns a finite scalar evidence value

#### Scenario: Non-converged FISTA evidence is penalized
- **WHEN** the FISTA solver reports non-convergence
- **THEN** `PixelizedImageProbModelOperator.__call__()` applies a large negative evidence penalty

### Requirement: FISTA evidence uses existing logdet approximation
For `solver_type="fista"`, the operator backend SHALL continue to use the existing operator evidence log-determinant approximation: the preconditioner log determinant for the curvature term and `reg_builder.logdet_free(...)` for the regularization term.

#### Scenario: FISTA evidence does not require active-set logdet
- **WHEN** evidence is evaluated with `solver_type="fista"`
- **THEN** the implementation does not compute active-set Laplace corrections
- **THEN** the implementation reuses the existing block-preconditioner and regularization logdet approximations

### Requirement: Soft non-negativity prior is removed
`PixelizedImageProbModelOperator` SHALL NOT support `source_nonnegativity_sigma`. Hard source non-negativity SHALL be requested through `solver_type="fista"`.

#### Scenario: Removed soft prior argument is rejected
- **WHEN** a caller passes `source_nonnegativity_sigma` to `PixelizedImageProbModelOperator`
- **THEN** construction fails because the argument is no longer part of the supported API

#### Scenario: Existing soft prior penalty is absent
- **WHEN** evidence is evaluated by `PixelizedImageProbModelOperator`
- **THEN** the evidence calculation does not add a half-Gaussian penalty for negative source pixels

### Requirement: Dense NNLS parity on small operator problems
For small source grids where dense matrices are practical, the operator FISTA source solution SHALL be numerically consistent with the dense pixelized NNLS solution for the same image, noise, PSF, source grid, lens parameters, and finite-difference regularization.

#### Scenario: FISTA source matches dense NNLS reference
- **WHEN** a small source-only problem is solved with dense `PixelizedImageProbModel(..., solver_type="nnls")`
- **WHEN** the same problem is solved with `PixelizedImageProbModelOperator(..., solver_type="fista")`
- **THEN** the operator FISTA source vector is non-negative
- **THEN** the operator FISTA objective value is close to the dense NNLS objective within a documented tolerance
