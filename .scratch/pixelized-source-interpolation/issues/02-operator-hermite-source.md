# 02: Reconstruct the same Hermite source with the operator backend

**What to build:** A lens modeler can use the same source configuration for matrix-free inversion. At the same fixed parameters, the operator backend applies the same source model as the dense backend and prepares a preconditioner for the selected interpolation.

**Blocked by:** 01 — Reconstruct an opt-in Hermite source with the dense backend.

**Status:** ready-for-agent
**State:** claimed

**Spec:** [Parent specification](../spec.md)

- [ ] The matrix-free forward and transpose use the selected shared weights and satisfy the adjoint identity. Forward images agree with the dense backend for both modes with a mask, asymmetric PSF, and subpixel integration.
- [ ] The native-resolution mapping used to prepare the preconditioner follows the same interpolation choice. Fixed-cache and freshly prepared paths agree for the same inputs.
- [ ] At fixed geometry and regularization strengths, compare PCG with Cholesky and PNPG with NNLS on small converged problems. Check reconstructed nodes, images, objectives, solve status, and the applicable residual or KKT conditions.
- [ ] Include joint semi-linear inversion with unit-amplitude lens-light bases. Retain the coupled curvature and block-Schur preconditioner. In the constrained case the non-negativity constraint applies to node intensities and the forward uses the signed Hermite weights.
- [ ] Run the applicable default bilinear and supported differentiation regressions. Keep the full operator action matrix-free and retain the existing determinant approximation contract.

## Comments

Claimed for implementation on `codex/psi-02-operator`.
