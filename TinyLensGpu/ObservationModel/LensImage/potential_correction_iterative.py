"""Fixed-strength nonlinear source–potential reconstruction with coupled LM steps.

Python controls accepted points, damping and termination. Numerical points are
immutable JAX PyTrees; every current mapping and source derivative is explicit
numerical data rather than a mutable value captured by a compiled single step.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import SourceMapping
from ._potential_correction_common import (
    _PotentialCorrectionProblem,
    _dense_response_columns,
    _integer,
    _nonnegative,
    _positive,
)
from .potential_correction_result import PotentialCorrectionResult, PotentialSolverInfo


class PotentialLinearization(NamedTuple):
    """Current corrected interpolation and analytical pixelized-source gradients."""

    mapping: SourceMapping
    gradient_x: jax.Array
    gradient_y: jax.Array


class _CurrentPoint(NamedTuple):
    values: jax.Array
    linearization: PotentialLinearization
    image: jax.Array
    residual: jax.Array
    negative_gradient: jax.Array
    chi2: jax.Array
    penalty: jax.Array
    objective: jax.Array
    finite: jax.Array


class _DenseSystem(NamedTuple):
    curvature: jax.Array
    metric: jax.Array
    finite: jax.Array


@jax.jit
def _logdet_spd(matrix):
    with jax.default_matmul_precision("highest"):
        scale = jnp.sqrt(jnp.diag(matrix))
        chol = jnp.linalg.cholesky(matrix / (scale[:, None] * scale[None, :]))
        diagonal = jnp.diag(chol)
        value = 2 * jnp.sum(jnp.log(diagonal)) + 2 * jnp.sum(jnp.log(scale))
        valid = (
            jnp.all(jnp.isfinite(matrix))
            & jnp.all(jnp.isfinite(chol))
            & jnp.all(diagonal > 0)
            & jnp.isfinite(value)
        )
        return value, valid


@jax.jit
def _dense_increment(system, rhs, damping):
    with jax.default_matmul_precision("highest"):
        scale = jnp.sqrt(system.metric)
        normalized = system.curvature / (scale[:, None] * scale[None, :])
        matrix = normalized + damping * jnp.eye(rhs.size, dtype=rhs.dtype)
        chol = jnp.linalg.cholesky(matrix)
        delta = jsp.linalg.cho_solve((chol, True), rhs / scale) / scale
        residual = system.curvature @ delta + damping * system.metric * delta - rhs
        norm = jnp.linalg.norm(residual)
        relative = norm / jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(rhs.dtype).tiny)
        positive = jnp.all(jnp.isfinite(chol)) & jnp.all(jnp.diag(chol) > 0)
        finite = (
            jnp.all(jnp.isfinite(delta)) & jnp.isfinite(norm) & jnp.isfinite(relative)
        )
        return delta, norm, relative, positive, finite


@jax.jit
def _step_measures(system, rhs, values, delta, damping):
    with jax.default_matmul_precision("highest"):
        physical = jnp.vdot(delta, system.curvature @ delta)
        scale = jnp.sqrt(system.metric)
        damped_step = jnp.sqrt(damping) * scale * delta
        damped = jnp.vdot(damped_step, damped_step)
        predicted = jnp.vdot(rhs, delta) - 0.5 * physical
        fraction = damped / jnp.maximum(physical + damped, jnp.finfo(values.dtype).tiny)
        update = jnp.max(jnp.abs(scale * delta)) / jnp.maximum(
            1.0, jnp.max(jnp.abs(scale * values))
        )
        return predicted, fraction, update


class _DenseIterationBackend:
    """Dense numerical adapter; the shared controller owns every LM policy.

    The operator adapter implements the same operations and physical step
    measures with current bounded principal blocks. Both share one controller.
    """

    name = "dense"
    determinant_method = "full_cholesky"
    configuration = {}

    def __init__(self, problem):
        self.problem = problem

        @jax.jit
        def prepare(linearization, prior):
            with jax.default_matmul_precision("highest"):
                design = _dense_response_columns(
                    lambda vector: problem.jacobian_forward(vector, linearization),
                    problem.n_joint,
                    problem.dtype,
                )
                weighted = design / problem.noise_active[:, None]
                curvature = weighted.T @ weighted + prior
                metric = jnp.diag(curvature)
                finite = jnp.all(jnp.isfinite(curvature)) & jnp.all(metric > 0)
                return _DenseSystem(curvature, metric, finite)

        self.prepare = prepare

    def initialize(self, strengths):
        prior = self.problem.dense_prior(*strengths)
        logdet, positive = _logdet_spd(prior)
        return prior, logdet, positive

    def solve(self, system, rhs, damping):
        delta, norm, relative, positive, finite = _dense_increment(system, rhs, damping)
        reason = None
        if not bool(positive):
            reason = "damped_factorization_failed"
        elif not bool(finite):
            reason = "nonfinite_increment"
        elif float(norm) > self.problem.solve_atol + self.problem.solve_rtol * float(
            jnp.linalg.norm(rhs)
        ):
            reason = "linear_residual_exceeds_tolerance"
        return delta, PotentialSolverInfo(
            "cholesky",
            reason is None,
            reason is not None,
            1,
            float(norm),
            float(relative),
            reason,
        )

    def measures(self, system, point, delta, damping):
        return _step_measures(
            system, point.negative_gradient, point.values, delta, damping
        )

    def logdets(self, system):
        return _logdet_spd(system.curvature)


class IterativePotentialCorrection(_PotentialCorrectionProblem):
    """Joint signed source/potential reconstruction at fixed prior strengths.

    ``initial_source`` contains truth-derived intensities at the renderer's
    fixed source nodes, in row-major order. Each fit resets to these values
    and zero potential. Subsequent derivatives use only the current pixelized
    source, with no truth gradient callable. Invalid input raises ValueError;
    numerical nonconvergence returns the common result with no evidence.

    ``forward`` is the nonlinear image action. ``linearize``,
    ``jacobian_forward`` and ``jacobian_adjoint`` provide the current numerical
    operator contract; a linearization is immutable and must be refreshed when
    either field changes. Geometry and the macro rays stay fixed.
    """

    def __init__(
        self,
        renderer,
        image_data,
        noise_map,
        initial_source,
        *,
        lambda_amplitude,
        source_regularization="second-order",
        initialization_provenance=None,
        solve_rtol=None,
        solve_atol=0.0,
    ):
        super().__init__(
            renderer,
            image_data,
            noise_map,
            lambda_amplitude=lambda_amplitude,
            source_regularization=source_regularization,
            solve_rtol=solve_rtol,
            solve_atol=solve_atol,
        )
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            source = renderer._source_values(initial_source)
            if not bool(jnp.all(jnp.isfinite(source))):
                raise ValueError(
                    "initial_source must contain finite source-node intensities"
                )
            self.initial_values = jnp.concatenate(
                (source, jnp.zeros(self.n_potential, dtype=self.dtype))
            )
        self.initialization_provenance = {
            **dict(initialization_provenance or {}),
            "source_initialization": "supplied_truth_derived_source_node_intensities",
            "potential_initialization": "zero",
            "gradient_policy": (
                "current_analytic_cubic_hermite"
                if renderer.source_interpolation == "cubic_hermite"
                else "current_analytic_bilinear_branch"
            ),
            "reset_each_fit": True,
        }
        self._evaluate_kernel = jax.jit(self._evaluate)
        self._dense_backend = None
        self._operator_backends = {}

    def linearize(self, joint_values):
        """Refresh corrected rays and spatial gradients from the supplied fields."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            source, potential = self._joint(joint_values)
            mapping = self.renderer.source_mapping(
                potential, macro_mapping=self.macro_mapping
            )
            gx, gy = self.renderer.source_spatial_gradient(source, mapping)
            return PotentialLinearization(mapping, gx, gy)

    def forward(self, joint_values):
        """Nonlinear selected image with the accumulated source and potential."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            source, potential = self._joint(joint_values)
            mapping = self.renderer.source_mapping(
                potential, macro_mapping=self.macro_mapping
            )
            return self.renderer.source_forward(source, mapping)

    def jacobian_forward(self, increment, linearization):
        """Current coupled image derivative, including the lens-equation sign."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            source, potential = self._joint(increment)
            mapping, gx, gy = linearization
            brightness = jnp.sum(mapping.weights * source[mapping.indices], axis=1)
            ax, ay = self.renderer.potential_grid.deflection(potential)
            return self.renderer.image_forward(brightness - gx * ax - gy * ay)

    def jacobian_adjoint(self, image_active, linearization):
        """Exact transpose of the current coupled image derivative."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            mapping, gx, gy = linearization
            subpixel = self.renderer.image_adjoint(image_active)
            potential = self.renderer.potential_grid.deflection_adjoint(
                -gx * subpixel, -gy * subpixel
            )
            return jnp.concatenate(
                (self.renderer.source_adjoint(image_active, mapping), potential)
            )

    def _evaluate(self, values, lambda_source, lambda_potential):
        with jax.default_matmul_precision("highest"):
            linearization = self.linearize(values)
            image = self.renderer.source_forward(
                values[: self.n_source], linearization.mapping
            )
            residual = self.data_active - image
            prior_values = self.prior_matvec(values, lambda_source, lambda_potential)
            negative_gradient = (
                self.jacobian_adjoint(residual / self.noise_active**2, linearization)
                - prior_values
            )
            chi2 = jnp.vdot(residual / self.noise_active, residual / self.noise_active)
            penalty = jnp.vdot(values, prior_values)
            finite = (
                jnp.all(jnp.isfinite(values))
                & jnp.all(jnp.isfinite(image))
                & jnp.all(jnp.isfinite(linearization.mapping.beta_x))
                & jnp.all(jnp.isfinite(linearization.mapping.beta_y))
                & jnp.all(jnp.isfinite(linearization.gradient_x))
                & jnp.all(jnp.isfinite(linearization.gradient_y))
                & jnp.all(jnp.isfinite(negative_gradient))
                & jnp.isfinite(chi2 + penalty)
            )
            return _CurrentPoint(
                values,
                linearization,
                image,
                residual,
                negative_gradient,
                chi2,
                penalty,
                0.5 * (chi2 + penalty),
                finite,
            )

    def fit(
        self,
        lambda_source,
        lambda_potential,
        *,
        backend="dense",
        max_iter=80,
        initial_damping=1e-3,
        damping_min=1e-12,
        damping_max=1e12,
        damping_increase=5.0,
        damping_decrease=3.0,
        max_rejections=12,
        objective_rtol=None,
        step_rtol=None,
        gradient_atol=None,
        gradient_rtol=None,
        stall_max_iter=4,
        damping_fraction_max=0.5,
        acceptance_ratio=1e-4,
        block_size=64,
        block_indices=None,
        pcg_max_iter=400,
        pcg_rtol=None,
        pcg_atol=0.0,
    ):
        """Minimize the actual nonlinear objective with recorded LM controls.

        Damping is mu*diag(H), where H is the current undamped Gauss–Newton
        curvature. A small physical scaled gradient suffices for convergence.
        Alternatively both objective and scaled update must be small and the
        step must not be dominated by damping. Rejected/tiny high-damping steps
        never establish convergence; unresolved bilinear kinks return stalling.
        Final evidence uses refreshed H without damping and is a local Gaussian
        approximation. The operator backend retains exact coupled actions;
        bounded principal blocks approximate preconditioning and determinants.
        ``block_size``/``block_indices`` fix its partition; PCG controls govern
        recursive convergence. The independent ``solve_atol``/``solve_rtol``
        constructor controls certify the recomputed damped-system residual.

        ``max_iter`` bounds current-point iterations. ``max_rejections`` allows
        that many additional damping retries after the first trial at a point;
        zero therefore permits one attempt. All effective controls and scalar
        diagnostics are returned in configuration/history. A stalled result
        does not assert that a nonsmooth point is nonoptimal.
        """
        if backend not in ("dense", "operator"):
            raise ValueError("backend must be 'dense' or 'operator'")
        ls = _positive(lambda_source, "lambda_source")
        lp = _positive(lambda_potential, "lambda_potential")
        max_iter = _integer(max_iter, "max_iter", minimum=0)
        max_rejections = _integer(max_rejections, "max_rejections", minimum=0)
        stall_max_iter = _integer(stall_max_iter, "stall_max_iter")
        mu = _positive(initial_damping, "initial_damping")
        damping_min = _positive(damping_min, "damping_min")
        damping_max = _positive(damping_max, "damping_max")
        damping_increase = _positive(damping_increase, "damping_increase")
        damping_decrease = _positive(damping_decrease, "damping_decrease")
        if not damping_min <= mu <= damping_max:
            raise ValueError(
                "damping_min <= initial_damping <= damping_max is required"
            )
        if damping_increase <= 1 or damping_decrease <= 1:
            raise ValueError("damping increase/decrease factors must exceed one")
        double = self.dtype == jnp.float64
        objective_rtol = _nonnegative(
            (1e-10 if double else 1e-6) if objective_rtol is None else objective_rtol,
            "objective_rtol",
        )
        step_rtol = _nonnegative(
            (1e-7 if double else 1e-4) if step_rtol is None else step_rtol, "step_rtol"
        )
        gradient_atol = _nonnegative(
            (1e-9 if double else 1e-5) if gradient_atol is None else gradient_atol,
            "gradient_atol",
        )
        gradient_rtol = _nonnegative(
            (1e-8 if double else 1e-5) if gradient_rtol is None else gradient_rtol,
            "gradient_rtol",
        )
        damping_fraction_max = _nonnegative(
            damping_fraction_max, "damping_fraction_max"
        )
        acceptance_ratio = _nonnegative(acceptance_ratio, "acceptance_ratio")
        if damping_fraction_max >= 1 or acceptance_ratio >= 1:
            raise ValueError(
                "damping_fraction_max and acceptance_ratio must be below one"
            )
        controls = dict(
            max_iter=max_iter,
            initial_damping=mu,
            damping_min=damping_min,
            damping_max=damping_max,
            damping_increase=damping_increase,
            damping_decrease=damping_decrease,
            max_rejections=max_rejections,
            objective_rtol=objective_rtol,
            step_rtol=step_rtol,
            gradient_atol=gradient_atol,
            gradient_rtol=gradient_rtol,
            stall_max_iter=stall_max_iter,
            damping_fraction_max=damping_fraction_max,
            acceptance_ratio=acceptance_ratio,
            damping_metric="physical_curvature_diagonal",
            solve_atol=self.solve_atol,
        )
        with jax.experimental.enable_x64(double):
            if backend == "dense":
                if self._dense_backend is None:
                    self._dense_backend = _DenseIterationBackend(self)
                numerical = self._dense_backend
            else:
                from .potential_correction_iterative_operator import (
                    iteration_operator_backend,
                )

                numerical = iteration_operator_backend(
                    self,
                    block_size=block_size,
                    block_indices=block_indices,
                    pcg_max_iter=pcg_max_iter,
                    pcg_rtol=pcg_rtol,
                    pcg_atol=pcg_atol,
                )
            strengths = jnp.asarray([ls, lp], dtype=self.dtype)
            point = self._evaluate_kernel(self.initial_values, *strengths)
            prior, prior_logdet, prior_ok = numerical.initialize(strengths)
            info = PotentialSolverInfo("not_run", False, False, 0, 0.0, 0.0, None)
            history = [
                dict(
                    iteration=0,
                    trial=0,
                    accepted=False,
                    reason="initial",
                    objective_before=float(point.objective),
                    objective_trial=float(point.objective),
                    objective_after=float(point.objective),
                    damping=mu,
                )
            ]
            reason = "iteration_limit"
            initial_gradient = None
            stalls = 0
            accepted_count = 0
            system = None
            if not bool(point.finite):
                reason = "nonfinite_current_state"
            elif not bool(prior_ok):
                reason = "prior_factorization_failed"
            else:
                for iteration in range(max_iter):
                    system = numerical.prepare(point.linearization, prior)
                    if not bool(system.finite):
                        reason = "nonfinite_linearization"
                        break
                    gradient = float(
                        jnp.max(
                            jnp.abs(point.negative_gradient) / jnp.sqrt(system.metric)
                        )
                    )
                    if not np.isfinite(gradient):
                        reason = "nonfinite_gradient"
                        break
                    if initial_gradient is None:
                        initial_gradient = gradient
                    if gradient <= gradient_atol + gradient_rtol * initial_gradient:
                        reason = "converged_gradient"
                        if accepted_count == 0:
                            info = PotentialSolverInfo(
                                "not_required", True, False, 0, 0.0, 0.0, None
                            )
                        break
                    accepted = False
                    nonfinite_trials = 0
                    for trial in range(max_rejections + 1):
                        damping = jnp.asarray(mu, dtype=self.dtype)
                        delta, info = numerical.solve(
                            system, point.negative_gradient, damping
                        )
                        if not info.converged or info.failed:
                            history.append(
                                dict(
                                    iteration=iteration + 1,
                                    trial=trial + 1,
                                    accepted=False,
                                    reason=info.reason,
                                    objective_before=float(point.objective),
                                    objective_trial=None,
                                    objective_after=float(point.objective),
                                    damping=mu,
                                    solver_info=info.__dict__,
                                )
                            )
                            reason = "linear_solve_failed"
                            break
                        predicted, fraction, update = map(
                            float, numerical.measures(system, point, delta, damping)
                        )
                        candidate = self._evaluate_kernel(
                            point.values + delta, *strengths
                        )
                        finite = bool(candidate.finite)
                        actual = (
                            float(point.objective - candidate.objective)
                            if finite
                            else float("-inf")
                        )
                        ratio = (
                            actual / predicted
                            if predicted > 0 and np.isfinite(predicted)
                            else float("-inf")
                        )
                        measures_finite = (
                            np.isfinite(predicted)
                            and np.isfinite(fraction)
                            and np.isfinite(update)
                        )
                        accepted = (
                            finite
                            and measures_finite
                            and actual > 0
                            and ratio >= acceptance_ratio
                        )
                        trial_reason = (
                            "accepted"
                            if accepted
                            else (
                                "nonfinite_candidate"
                                if not finite
                                else "objective_rejected"
                            )
                        )
                        record = dict(
                            iteration=iteration + 1,
                            trial=trial + 1,
                            accepted=accepted,
                            reason=trial_reason,
                            objective_before=float(point.objective),
                            objective_trial=(
                                float(candidate.objective) if finite else None
                            ),
                            objective_after=float(
                                candidate.objective if accepted else point.objective
                            ),
                            chi2=float(candidate.chi2) if finite else None,
                            regularization_penalty=(
                                float(candidate.penalty) if finite else None
                            ),
                            damping=mu,
                            actual_reduction=actual if finite else None,
                            predicted_reduction=(
                                predicted if np.isfinite(predicted) else None
                            ),
                            gain_ratio=ratio if np.isfinite(ratio) else None,
                            scaled_update=update if np.isfinite(update) else None,
                            scaled_gradient=gradient,
                            damping_fraction=(
                                fraction if np.isfinite(fraction) else None
                            ),
                            solver_info=info.__dict__,
                        )
                        history.append(record)
                        if accepted:
                            change = actual / max(1.0, abs(float(point.objective)))
                            point = candidate
                            accepted_count += 1
                            tiny = change <= objective_rtol and update <= step_rtol
                            stalls = stalls + 1 if tiny else 0
                            mu = (
                                max(damping_min, mu / damping_decrease)
                                if ratio > 0.25
                                else mu
                            )
                            if tiny and fraction <= damping_fraction_max:
                                reason = "converged_objective_update"
                            elif stalls >= stall_max_iter:
                                reason = "stalled"
                            break
                        nonfinite_trials += not finite
                        if trial == max_rejections:
                            reason = (
                                "nonfinite_candidate_limit"
                                if nonfinite_trials == trial + 1
                                else "rejection_limit"
                            )
                            break
                        if mu >= damping_max:
                            reason = "damping_limit"
                            break
                        mu = min(damping_max, mu * damping_increase)
                    if reason in (
                        "converged_objective_update",
                        "stalled",
                        "linear_solve_failed",
                        "nonfinite_candidate_limit",
                        "rejection_limit",
                        "damping_limit",
                    ):
                        break
                    if not accepted:
                        break
            success = reason.startswith("converged_")
            log_evidence = None
            final_gradient = None
            posterior_logdet = jnp.asarray(jnp.nan, dtype=self.dtype)
            if success:
                # Refresh at the accepted FINAL fields; no step damping enters this system.
                point = self._evaluate_kernel(point.values, *strengths)
                system = numerical.prepare(point.linearization, prior)
                if bool(point.finite & system.finite):
                    final_gradient = float(
                        jnp.max(
                            jnp.abs(point.negative_gradient) / jnp.sqrt(system.metric)
                        )
                    )
                posterior_logdet, posterior_ok = numerical.logdets(system)
                evidence = -0.5 * (
                    point.chi2
                    + point.penalty
                    + posterior_logdet
                    - prior_logdet
                    + self.noise_logdet
                    + self.data_active.size * jnp.log(2 * jnp.pi)
                )
                if not bool(
                    point.finite & system.finite & posterior_ok & jnp.isfinite(evidence)
                ):
                    success = False
                    reason = "final_evidence_failed"
                else:
                    log_evidence = evidence
            products = self._result_products(point.values, ls, lp)
            return PotentialCorrectionResult(
                **products,
                log_evidence=log_evidence,
                prior_logdet=prior_logdet,
                posterior_logdet=posterior_logdet,
                backend=numerical.name,
                algorithm="iterative",
                status="success" if success else "failed",
                success=success,
                failure_reason=None if success else reason,
                termination_reason=reason,
                solver_info=info,
                evidence_method="local_gauss_newton",
                determinant_method=numerical.determinant_method,
                evidence_is_exact=False,
                initialization_provenance=dict(self.initialization_provenance),
                configuration={
                    **self._configuration(),
                    **controls,
                    **numerical.configuration,
                    "accepted_iterations": accepted_count,
                    "final_damping": mu,
                    "initial_scaled_gradient": initial_gradient,
                    "final_scaled_gradient": final_gradient,
                },
                history=tuple(history),
            )

    def __call__(self, lambda_source, lambda_potential, **kwargs):
        return self.fit(lambda_source, lambda_potential, **kwargs)
