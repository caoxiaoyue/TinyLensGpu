"""Shared outputs of dense and operator pixelized-source inversions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from jax import Array

from TinyLensGpu.utils.cg_solver import PCGInfo
from TinyLensGpu.utils.pnpg_solver import PNPGInfo


class InvalidReconstructionError(RuntimeError):
    """A consumer rejected an unconverged or nonfinite reconstruction."""


class InversionResult(NamedTuple):
    """One inversion's evidence and reconstruction on the returned source grid.

    ``solver_info`` contains PCG/PNPG diagnostics for the operator backend;
    it is ``None`` for the dense backend, whose current solvers expose no
    convergence status. Operator solve failures zero the reconstruction;
    dense reconstruction retains its existing numerical-failure behavior.
    ``log_evidence`` is ``None`` when only reconstruction was requested.
    """

    log_evidence: Array | None
    model_image: Array
    source_pixels: Array
    lens_amplitudes: Array | None
    source_bbox: tuple[Array, Array, Array, Array]
    solver_info: PCGInfo | PNPGInfo | None


@dataclass(frozen=True)
class PositionDiagnostics:
    """Observed and ray-traced image positions for a constrained inversion."""

    penalty: float
    max_separation: float
    image_positions: Array  # (N, 2)
    source_positions: Array  # (N, 2)


@dataclass(frozen=True)
class InversionDiagnostics:
    """Host-side diagnostics from one pixelized-source inversion.

    ``valid`` is False when the operator solve did not converge or the
    reconstruction contains nonfinite values; ``invalid_reason`` explains it.
    ``chi2`` is the chi-square over active pixels only, while
    ``normalized_residuals`` is the full image in the established plot
    convention. ``regularization_scale`` is the (n, n) S0-template scale map,
    ones for uniform regularization. Optional effective-degrees-of-freedom and
    component-image fields stay ``None`` unless requested; an invalid
    reconstruction keeps the three statistics ``None`` and explains why in
    ``effective_dof_reason``.
    """

    valid: bool
    invalid_reason: str | None
    normalized_residuals: Array
    chi2: float
    lambda_reg: float
    adaptive_reg_rho: float
    regularization_scale: Array
    position: PositionDiagnostics | None = None
    effective_parameters: float | None = None
    residual_dof: float | None = None
    reduced_chi2: float | None = None
    effective_dof_method: str | None = None
    effective_dof_reason: str | None = None
    source_image: Array | None = None
    lens_light_image: Array | None = None


__all__ = [
    "InvalidReconstructionError", "InversionResult",
    "InversionDiagnostics", "PositionDiagnostics",
]
