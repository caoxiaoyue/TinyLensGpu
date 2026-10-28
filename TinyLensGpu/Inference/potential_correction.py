"""Bounded log-strength inference for prepared potential-correction problems.

The reconstruction problem owns geometry, initialization, precision, and the
fixed amplitude prior. This layer calls its public ``fit`` boundary for every
candidate and keeps scalar evaluation records plus the best reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import direct

if TYPE_CHECKING:
    from TinyLensGpu.ObservationModel import (
        PotentialCorrectionResult,
        PotentialCorrectionStrengths,
    )


@dataclass(frozen=True)
class PotentialCorrectionEvaluation:
    """A scalar candidate record; full fields are retained only for the best fit."""

    log_strengths: tuple[float, float]
    strengths: PotentialCorrectionStrengths
    objective: float
    log_evidence: float | None
    success: bool
    failure_reason: str | None
    dtype: str
    backend: str


class PotentialCorrectionEvidence:
    """Callable negative evidence in natural-log source/D4 strength coordinates.

    ``objective([log_lambda_source, log_lambda_potential])`` performs a complete
    fresh fit with the problem's fixed initialization. No result is fed into a
    later candidate. Failed fits return positive infinity and retain their
    reason; configuration exceptions propagate from ``problem.fit``.

    ``evaluations`` contains small records rather than every fitted field.
    ``last_reconstruction`` and ``best_reconstruction`` expose the corresponding
    public results for other optimizers, samplers, and diagnostic consumers.
    ``backend`` and ``fit_options`` are forwarded on every candidate, allowing
    the same boundary to consume an operator or iterative problem.
    """

    def __init__(self, problem, *, backend="dense", fit_options=None):
        self.problem = problem
        self.backend = backend
        self.fit_options = dict(fit_options or {})
        reserved = {"lambda_source", "lambda_potential", "lambda_amplitude", "backend"}
        overridden = reserved.intersection(self.fit_options)
        if overridden:
            raise ValueError(
                "fit_options cannot override fixed or searched settings: "
                + ", ".join(sorted(overridden))
            )
        self.dtype = np.dtype(problem.dtype)
        self._evaluations = []
        self.last_reconstruction: PotentialCorrectionResult | None = None
        self.best_reconstruction: PotentialCorrectionResult | None = None
        self.best_evaluation: PotentialCorrectionEvaluation | None = None

    @property
    def evaluations(self):
        """All completed numerical candidates in evaluation order."""
        return tuple(self._evaluations)

    def __call__(self, log_strengths):
        coordinates = np.asarray(log_strengths, dtype=np.float64)
        if coordinates.shape != (2,) or not np.all(np.isfinite(coordinates)):
            raise ValueError("log_strengths must contain two finite natural logarithms")
        self.last_reconstruction = None
        with np.errstate(over="ignore", under="ignore"):
            strengths = np.exp(coordinates)
            represented = strengths.astype(self.dtype)
        if not np.all(np.isfinite(represented) & (represented > 0)):
            # PhysicalModel imports Inference.ParamU while ForwardSimulation
            # is initializing. Defer the observation-layer import until a
            # prepared problem is actually evaluated to preserve that order.
            from TinyLensGpu.ObservationModel import PotentialCorrectionStrengths

            self._evaluations.append(
                PotentialCorrectionEvaluation(
                    tuple(float(value) for value in coordinates),
                    PotentialCorrectionStrengths(
                        float(strengths[0]),
                        float(strengths[1]),
                        self.problem.lambda_amplitude,
                    ),
                    np.inf,
                    None,
                    False,
                    "strength_out_of_range",
                    self.dtype.name,
                    self.backend,
                )
            )
            return np.inf
        lambda_source, lambda_potential = strengths
        result = self.problem.fit(
            float(lambda_source),
            float(lambda_potential),
            backend=self.backend,
            **self.fit_options,
        )
        self.last_reconstruction = result
        log_evidence = float(result.log_evidence) if result.success else None
        evaluation = PotentialCorrectionEvaluation(
            tuple(float(value) for value in coordinates),
            result.strengths,
            -log_evidence if result.success else np.inf,
            log_evidence,
            result.success,
            result.failure_reason,
            result.dtype,
            result.backend,
        )
        self._evaluations.append(evaluation)
        if evaluation.success and (
            self.best_evaluation is None
            or evaluation.objective < self.best_evaluation.objective
        ):
            self.best_evaluation = evaluation
            self.best_reconstruction = result
        return evaluation.objective


@dataclass(frozen=True)
class PotentialCorrectionOptimizationResult:
    """DIRECT termination, retained candidates, and the best successful fit.

    A budget-limited search can have a usable ``reconstruction`` while
    ``success`` is false. Check ``has_valid_fit`` separately from optimizer
    convergence. Boundary flags mean the selected log strength lies within
    ``boundary_rtol`` of that edge as a fraction of its configured log range;
    DIRECT samples rectangle centers rather than exact endpoints.
    """

    reconstruction: PotentialCorrectionResult | None
    log_strengths: tuple[float, float] | None
    objective: float
    log_bounds: tuple[tuple[float, float], tuple[float, float]]
    evaluations: tuple[PotentialCorrectionEvaluation, ...]
    optimizer_success: bool
    optimizer_status: int
    message: str
    nfev: int
    nit: int
    boundary_sides: tuple[str | None, str | None]
    boundary_distances: tuple[float, float] | None
    configuration: dict

    @property
    def has_valid_fit(self):
        return self.reconstruction is not None

    @property
    def success(self):
        return self.optimizer_success and self.has_valid_fit

    @property
    def strengths(self):
        return None if self.reconstruction is None else self.reconstruction.strengths

    @property
    def at_boundary(self):
        return any(side is not None for side in self.boundary_sides)


def optimize_potential_correction(
    problem,
    *,
    log_bounds,
    backend="dense",
    fit_options=None,
    maxfun=200,
    maxiter=1000,
    eps=1e-4,
    locally_biased=True,
    vol_tol=1e-16,
    len_tol=1e-6,
    boundary_rtol=1e-2,
):
    """Select source and D4 strengths with bounded SciPy DIRECT.

    ``log_bounds`` is the required pair of finite increasing natural-log
    intervals, ordered source then potential. The independent amplitude
    strength remains fixed on ``problem``. All DIRECT controls and the
    boundary-reporting tolerance are retained in the returned configuration.
    The best public reconstruction is handed off without an additional fit.
    """
    bounds = np.asarray(log_bounds, dtype=np.float64)
    if (
        bounds.shape != (2, 2)
        or not np.all(np.isfinite(bounds))
        or not np.all(bounds[:, 0] < bounds[:, 1])
    ):
        raise ValueError(
            "log_bounds must contain two finite increasing (lower, upper) pairs"
        )
    for name, value in (("maxfun", maxfun), ("maxiter", maxiter)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("eps", eps),
        ("vol_tol", vol_tol),
        ("len_tol", len_tol),
        ("boundary_rtol", boundary_rtol),
    ):
        try:
            valid = (
                np.ndim(value) == 0
                and not isinstance(value, (bool, np.bool_))
                and np.isfinite(value)
                and value >= 0
            )
        except TypeError:
            valid = False
        if name in ("vol_tol", "len_tol"):
            valid = valid and value <= 1
        elif name == "boundary_rtol":
            valid = valid and value < 0.5
        if not valid:
            interval = (
                "[0, 0.5)"
                if name == "boundary_rtol"
                else "[0, 1]" if name in ("vol_tol", "len_tol") else "[0, infinity)"
            )
            raise ValueError(f"{name} must be a finite scalar in {interval}")
    if not isinstance(locally_biased, (bool, np.bool_)):
        raise ValueError("locally_biased must be True or False")
    objective = PotentialCorrectionEvidence(
        problem, backend=backend, fit_options=fit_options
    )
    with np.errstate(over="ignore", under="ignore"):
        represented = np.exp(bounds).astype(objective.dtype)
    if not np.all(np.isfinite(represented) & (represented > 0)):
        raise ValueError(
            "log_bounds must give positive strengths representable in the problem dtype"
        )
    controls = {
        "maxfun": int(maxfun),
        "maxiter": int(maxiter),
        "eps": float(eps),
        "locally_biased": bool(locally_biased),
        "vol_tol": float(vol_tol),
        "len_tol": float(len_tol),
    }
    optimized = direct(objective, bounds.tolist(), **controls)
    best = objective.best_evaluation
    sides = (None, None)
    distances = None
    if best is not None:
        lower = (np.asarray(best.log_strengths) - bounds[:, 0]) / np.diff(bounds)[:, 0]
        upper = (bounds[:, 1] - np.asarray(best.log_strengths)) / np.diff(bounds)[:, 0]
        distances = tuple(float(value) for value in np.minimum(lower, upper))
        sides = tuple(
            "lower" if lo <= boundary_rtol else "upper" if hi <= boundary_rtol else None
            for lo, hi in zip(lower, upper)
        )
    return PotentialCorrectionOptimizationResult(
        reconstruction=objective.best_reconstruction,
        log_strengths=None if best is None else best.log_strengths,
        objective=np.inf if best is None else best.objective,
        log_bounds=tuple(tuple(float(value) for value in pair) for pair in bounds),
        evaluations=objective.evaluations,
        optimizer_success=bool(optimized.success),
        optimizer_status=int(optimized.status),
        message=str(optimized.message),
        nfev=int(optimized.nfev),
        nit=int(optimized.nit),
        boundary_sides=sides,
        boundary_distances=distances,
        configuration={
            "optimizer": "scipy.optimize.direct",
            "coordinate_system": "natural_log_strengths",
            "strength_order": ("lambda_source", "lambda_potential"),
            "backend": backend,
            "dtype": objective.dtype.name,
            "lambda_amplitude": problem.lambda_amplitude,
            "fit_options": dict(objective.fit_options),
            "boundary_rtol": boundary_rtol,
            **controls,
        },
    )
