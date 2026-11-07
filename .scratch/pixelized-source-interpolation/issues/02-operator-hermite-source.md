# 02: Reconstruct the same Hermite source with the operator backend

**What to build:** A lens modeler can use the same source configuration for matrix-free inversion. At the same fixed parameters, the operator backend applies the same source model as the dense backend and prepares a preconditioner for the selected interpolation.

**Blocked by:** 01 — Reconstruct an opt-in Hermite source with the dense backend.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] The matrix-free forward and transpose use the selected shared weights and satisfy the adjoint identity. Forward images agree with the dense backend for both modes with a mask, asymmetric PSF, and subpixel integration.
- [x] The native-resolution mapping used to prepare the preconditioner follows the same interpolation choice. Fixed-cache and freshly prepared paths agree for the same inputs.
- [x] At fixed geometry and regularization strengths, compare PCG with Cholesky and PNPG with NNLS on small converged problems. Check reconstructed nodes, images, objectives, solve status, and the applicable residual or KKT conditions.
- [x] Include joint semi-linear inversion with unit-amplitude lens-light bases. Retain the coupled curvature and block-Schur preconditioner. In the constrained case the non-negativity constraint applies to node intensities and the forward uses the signed Hermite weights.
- [x] Run the applicable default bilinear and supported differentiation regressions. Keep the full operator action matrix-free and retain the existing determinant approximation contract.

## Comments

Claimed for implementation on `codex/psi-02-operator`.

Implemented in `cfceac4`, merged as `980265f`. From `tests/` in
`tinylens_gpu` with preallocation disabled and `OPENBLAS_NUM_THREADS=1`,
the CUDA regression command passed 229 tests.

```bash
pytest -q test_pixelized_operator.py test_operator_autodiff.py \
  test_solver_autodiff.py test_pixelized_source_precision.py
```

On CPU, `pytest -q test_operator_source_interpolation.py` passed 7 tests
and `pytest -q test_operator_autodiff.py` passed 14, including two added
Hermite joint/adaptive finite-difference cases. The operator matched the
independent forward/adjoint and native block references at `1e-12`.
Fixed- and freshly prepared Cholesky/PCG and NNLS/PNPG fits, including
joint inversion, agreed within `2e-8` in coefficients and images and
`5e-9` relative in objective. The PCG normal residual and PNPG KKT
residual were within `2e-11` and `2e-9` of the RHS norm, respectively.
