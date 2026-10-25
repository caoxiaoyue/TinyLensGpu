# PNPG joint inversion and implicit derivatives

The operator backend supports two solvers: unconstrained PCG and non-negative
PNPG. PNPG solves source pixels and optional unit-amplitude lens-light bases
in one system, including their cross-curvature. Its forward algorithm uses
diagonal equilibration, a projected PCG warm start, quadratic backtracking,
and accelerated iterations with restart. The curvature operator, physical
regularization, and approximate evidence determinants are shared with PCG.

## Implicit differentiation

With a fixed source bbox, PNPG supports first-order reverse-mode derivatives
of reconstructions and evidence. The numerical forward iterates are those of
`pnpg_nnls_solve`; `pnpg_nnls_solve_with_implicit_grad` adds diagnostics and a
custom VJP without differentiating the iteration history or backtracking loop.

For the physical quadratic `0.5 x^T A x - b^T x`, let `F` denote the free
variables of the converged non-negative solution. For an output cotangent `g`,
the adjoint solves `A_FF v_F = g_F`, with zero entries on active variables.
The pullback is `v^T db - v^T (dA) x`. The masked adjoint uses PCG and the
existing preconditioner; it operates in the physical coordinates, not PNPG's
rescaled coordinates. Initial guesses and numerical preconditioners receive
no derivatives. Integer operator indices retain JAX's `float0` cotangents.

Derivatives require a locally stable active set, componentwise KKT accuracy,
strict complementarity, and a converged adjoint. Failure of any of these
checks yields NaN gradients. It does not turn a successful forward solve into
a failure. A failed forward solve retains the finite evidence penalty and
zero reconstruction arrays. `solver_info.kkt_violation` reports the largest
normalized componentwise residual; `min_active_residual` reports the smallest
active-variable dual residual, or infinity for an empty active set. Raw PNPG
leaves those two extra diagnostics as NaN; the implicit wrapper fills them.

```python
model = PixelizedImageProbModelOperator(
    image_data, noise_map, psf_kernel, dpix, phys_model,
    solver_type="pnpg",
    fixed_source_bbox=(-0.5, 0.5, -0.5, 0.5),
    pnpg_max_iter=1000,
    pnpg_rtol=1e-5,
)
theta = model.get_values()
value, gradient = jax.jit(jax.value_and_grad(model))(theta)
```

The shown tolerance is a starting point, not a convergence guarantee. Check
forward diagnostics and finite gradients on representative parameters, and
increase the iteration budget as needed. The default forward tolerance 0.02
can be insufficient for the stricter derivative checks. Changing active sets,
dynamic-bbox derivatives, and higher-order autodiff are outside this contract.

## Migration from FISTA

FISTA and its constructor arguments were removed on 2026-10-24 after migrating
the fixed-active-set derivative to PNPG. Use `solver_type="pnpg"` and the
`pnpg_max_iter`, `pnpg_rtol`, and `pnpg_power_iter` controls. Retune tolerances:
PNPG's componentwise KKT gate differs from the old global projected-gradient
gate. No compatibility alias or silent tolerance conversion is provided.

`pnpg_comp_rel_tol` exposes the implicit wrapper's relative complementarity
margin, with the existing default of `1e-4`. It does not change PNPG's forward
iterates or forward convergence gate. Use a smaller margin only when the
positive dual residuals are resolved accurately and derivative checks support
it. A large relative margin can reject a mathematically stable active set in
a bright correlated MGE system even after the forward residual is small.

The existing default solver remains PCG. Examples expose PNPG controls for
their optional non-negative path; their PCG configuration stays available.
Historical FISTA measurements are preserved as dated evidence, rather than
relabelled as PNPG results:

- [Original scalar-step/MGE experiments](fista-joint-pixel-source-mge.md)
- [20-MGE solver comparison](../validation/pix-src-lens-light-mge-fista-comparison.md)
- [Original PNPG sampling validation](../validation/pix-src-lens-light-mge-pnpg-operator-mock.md)

Validation covers analytic reduced-system derivatives, independently varied
matrix/RHS data, dynamic numerical preconditioners and PCG initial guesses,
JIT/vmap, finite differences of source/lens/evidence outputs, and rejection
of invalid forward solves, line searches, KKT conditions, active sets, and
adjoint solves. Tests live in `tests/test_solver_autodiff.py`,
`tests/test_operator_autodiff.py`, and `tests/test_pixelized_operator.py`.

## Validation on 2026-10-24

- `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest -m "not slow" -q`:
  717 passed, 2 deselected in 311.46 seconds in `tinylens_gpu`.
- The targeted solver run also passed the slow float64 active-set stress
  comparison against an analytic reduced-system derivative.
- Finite-difference checks cover source-only and joint lens-light models,
  including a dynamic adaptive-regularization strength on a fixed template.
- The unchanged 20-MGE example at the true nonlinear values and
  `log_lambda_reg=-8`, source grid 80 x 80 and `nsub=4`, passes with PNPG
  at 1000 iterations and `rtol=0.02`. Using the modified checkout on RTX 5080,
  one warmed evaluation took 0.157 seconds, with log evidence 2685.994385,
  chi-square per fitted datum 0.927852, and non-negative source/lens amplitudes.
  This is a conditional forward check, not a new nested-sampling validation
  or a gradient check on the example's dynamic bbox.

The local validation harness at
`.scratch/fista-mge-validation/compare_solvers.py` now accepts PNPG only,
pins imports to its own repository, and records the actual package path.
When running an example against this checkout while the environment's editable
installation points elsewhere, run from the repository root with
`PYTHONPATH="$PWD" python examples/.../fit_joint.py`, or install this checkout
with `pip install -e .` first.

The [fixed-bbox 20-MGE gradient validation](../validation/pnpg-mge-fixed-bbox-gradients.md)
extends these checks to the large model. It also exposed and fixed a NaN VJP
in the shared position constraint at zero pair distances, and a float32
downcast of double-precision ray coordinates in bilinear mapping.
The reference-point gradient comparisons pass in float64 with tighter solver
settings, but the full ten-parameter optimization rejected a gradient at its
configured complementarity margin after 22 accepted steps even with 300000
PNPG iterations. This large example
therefore does not yet establish stable convergence of gradient optimization.
