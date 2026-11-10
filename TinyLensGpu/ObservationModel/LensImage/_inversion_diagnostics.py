"""Host-side diagnostic assembly shared by dense and operator inversions."""

# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from .inversion_result import InversionDiagnostics, PositionDiagnostics


def reconstruction_diagnostics(model, result, scale) -> InversionDiagnostics:
    """Assemble validity, residual, and regularization diagnostics.

    Runs inside the model's current parameter context. Invalid
    reconstructions do not raise; validity is reported so consumers can
    reject them before producing formal output.
    """
    finite = all(
        bool(np.all(np.isfinite(np.asarray(value))))
        for value in (result.model_image, result.source_pixels, result.lens_amplitudes)
        if value is not None
    )
    reason = None
    if result.solver_info is not None and not bool(result.solver_info.converged):
        reason = f"Operator reconstruction did not converge: {result.solver_info}"
    if not finite:
        detail = "Reconstruction contains nonfinite model, source, or lens-light values"
        reason = detail if reason is None else f"{reason}; {detail}"
    residuals = (model.image_data - result.model_image) / model.noise_map
    chi2 = float(jnp.sum(residuals[model.unmask] ** 2))
    n = int(model.source_model.n)
    scale = (
        jnp.ones((n, n), dtype=result.source_pixels.dtype)
        if scale is None
        else jnp.asarray(scale).reshape(n, n)
    )
    position = None
    if model._has_pos_penalty:
        bx, by = model.phys_model.deflection(model._pos_px, model._pos_py)
        distances = jnp.sqrt(
            (bx[:, None] - bx[None, :]) ** 2 + (by[:, None] - by[None, :]) ** 2
        )
        position = PositionDiagnostics(
            penalty=float(model._position_likelihood_penalty_jax()),
            max_separation=float(jnp.max(distances)),
            image_positions=jnp.stack((model._pos_px, model._pos_py), axis=-1),
            source_positions=jnp.stack((bx, by), axis=-1),
        )
    return InversionDiagnostics(
        valid=reason is None,
        invalid_reason=reason,
        normalized_residuals=residuals,
        chi2=chi2,
        lambda_reg=float(jnp.exp(jnp.asarray(model.source_model.log_lambda_reg.value))),
        adaptive_reg_rho=float(
            jnp.asarray(getattr(model.source_model.adaptive_reg_rho, "value",
                                model.source_model.adaptive_reg_rho))
        ),
        regularization_scale=scale,
        position=position,
    )


def evidence_diagnostics(
    diagnostics: InversionDiagnostics, method: str, valid, *, schur_solved=None,
) -> InversionDiagnostics:
    """Attach the score status while retaining the primary reconstruction status."""
    valid = bool(valid)
    reason = None
    if not valid:
        if not diagnostics.valid:
            reason = "Evidence requires a valid primary reconstruction."
        elif schur_solved is not None and not bool(schur_solved):
            reason = "Overlap auxiliary source-curvature solve failed."
        else:
            reason = "Evidence calculation returned a nonfinite value."
    return replace(
        diagnostics, evidence_logdet=method, evidence_valid=valid,
        evidence_invalid_reason=reason,
    )
