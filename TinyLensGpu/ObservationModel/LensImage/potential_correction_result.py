"""Shared scientific output contract for source–potential reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field

from jax import Array

from TinyLensGpu.utils.lensing.potential import PotentialGrid


@dataclass(frozen=True)
class PotentialCorrectionStrengths:
    """Independent strengths of the source, D4, and all-node amplitude priors."""

    lambda_source: float
    lambda_potential: float
    lambda_amplitude: float


@dataclass(frozen=True)
class PotentialSolverInfo:
    """Status of the actual inner linear system, including LM step damping.

    Nonlinear convergence is recorded separately in the reconstruction result.
    """

    method: str
    converged: bool
    failed: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    reason: str | None = None


@dataclass(frozen=True)
class PotentialCorrectionResult:
    """Raw fitted fields, their valid domains, images, and inference metadata.

    No affine alignment or clipping is applied to the fields. Physical maps
    have one entry per potential unknown, with NaN outside ``derivative_valid``.
    Residuals use data minus model; full residual images are NaN at excluded
    observation pixels. Single-step evidence uses only the linear prediction.
    ``log_evidence`` is None on failure. Evidence and determinant annotations
    are separate so local linearization and block determinant approximations
    can be identified independently by later backends and algorithms.

    Iterative results set linear products to None and use the nonlinear image
    for chi2/objective. History contains scalar dictionaries for initialization
    and accepted/rejected attempts. ``success`` describes nonlinear convergence;
    ``solver_info`` describes the last inner solve and may be converged even
    when the nonlinear reconstruction stalls.
    """

    source_values: Array
    source_x_axis: Array
    source_y_axis: Array
    source_bbox: tuple
    potential_values: Array
    potential_grid: PotentialGrid
    deflection_x: Array
    deflection_y: Array
    convergence: Array
    derivative_valid: Array
    potential_window_coverage: Array
    linear_model_image: Array | None
    nonlinear_model_image: Array
    linear_image_active: Array | None
    nonlinear_image_active: Array
    linear_residual: Array | None
    nonlinear_residual: Array
    linear_normalized_residual: Array | None
    nonlinear_normalized_residual: Array
    data_active: Array
    noise_active: Array
    strengths: PotentialCorrectionStrengths
    log_evidence: Array | None
    chi2: Array
    regularization_penalty: Array
    objective: Array
    prior_logdet: Array
    posterior_logdet: Array
    noise_logdet: Array
    dtype: str
    backend: str
    algorithm: str
    status: str
    success: bool
    failure_reason: str | None
    termination_reason: str
    solver_info: PotentialSolverInfo
    evidence_method: str
    determinant_method: str
    evidence_is_exact: bool
    initialization_provenance: dict = field(default_factory=dict)
    configuration: dict = field(default_factory=dict)
    history: tuple = ()
