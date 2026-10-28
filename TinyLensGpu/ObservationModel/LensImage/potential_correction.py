"""Fixed-gradient single-step signed source–potential reconstruction.

Geometry and the macro ray mapping belong to the forward renderer. This
module owns observation weighting, the coupled solve, and Gaussian evidence.
Construction prepares compact stencils only; dense matrices are lazy explicit
requests so the same problem can be consumed by an operator backend.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .potential_correction_result import (
    PotentialCorrectionResult,
    PotentialCorrectionStrengths as PotentialCorrectionStrengths,
    PotentialSolverInfo,
)
from ._potential_correction_common import (
    _PotentialCorrectionProblem,
    _dense_response_columns,
    _positive,
)


class _DenseSolution(NamedTuple):
    values: jax.Array
    log_evidence: jax.Array
    prior_logdet: jax.Array
    posterior_logdet: jax.Array
    residual_norm: jax.Array
    relative_residual: jax.Array
    finite_system: jax.Array
    prior_positive: jax.Array
    posterior_positive: jax.Array
    finite_solution: jax.Array


@jax.jit
def _solve_dense(design, prior, data, noise):
    # GPU defaults can use reduced-mantissa products for float32. Preserve
    # the requested dtype's accuracy in the coupled reference calculation
    # without changing the surrounding process's matmul precision policy.
    with jax.default_matmul_precision("highest"):
        weighted = design / noise[:, None]
        curvature = weighted.T @ weighted + prior
        rhs = weighted.T @ (data / noise)
        prior_chol = jnp.linalg.cholesky(prior)
        chol = jnp.linalg.cholesky(curvature)
        values = jsp.linalg.cho_solve((chol, True), rhs)
        residual = data - design @ values
        prior_logdet = 2 * jnp.sum(jnp.log(jnp.diag(prior_chol)))
        posterior_logdet = 2 * jnp.sum(jnp.log(jnp.diag(chol)))
        log_evidence = -0.5 * (
            jnp.vdot(residual / noise, residual / noise)
            + values @ prior @ values
            + posterior_logdet
            - prior_logdet
            + 2 * jnp.sum(jnp.log(noise))
            + data.size * jnp.log(2 * jnp.pi)
        )
        residual_norm = jnp.linalg.norm(curvature @ values - rhs)
        relative_residual = residual_norm / jnp.maximum(
            jnp.linalg.norm(rhs), jnp.finfo(data.dtype).tiny
        )
        return _DenseSolution(
            values,
            log_evidence,
            prior_logdet,
            posterior_logdet,
            residual_norm,
            relative_residual,
            jnp.all(jnp.isfinite(curvature))
            & jnp.all(jnp.isfinite(rhs))
            & jnp.all(jnp.isfinite(prior)),
            jnp.all(jnp.isfinite(prior_chol)) & jnp.all(jnp.diag(prior_chol) > 0),
            jnp.all(jnp.isfinite(chol)) & jnp.all(jnp.diag(chol) > 0),
            jnp.all(jnp.isfinite(values))
            & jnp.isfinite(log_evidence)
            & jnp.isfinite(relative_residual),
        )


class SingleStepPotentialCorrection(_PotentialCorrectionProblem):
    """A reusable fixed-strength conditional Gaussian reconstruction problem.

    Parameters
    ----------
    renderer : PotentialCorrectionRenderer
        Fixed macro, source geometry, potential geometry, mask, PSF and nsub.
    image_data, noise_map : array_like
        Full native images. Only selected pixels must be finite; selected
        noise standard deviations must be positive.
    source_gradient : callable
        Truth-derived provider ``gradient(beta_x, beta_y) -> (gx, gy)`` at
        macro image-subpixel rays. Evaluated once and frozen for this problem.
        Gradients outside the fixed source box are zero. No truth intensity
        is fixed: the full signed source brightness is solved along with p.
    lambda_amplitude : float
        Explicit positive independent potential amplitude precision, fixed
        for all evaluations. It is never adjusted to rescue factorization.
    source_regularization : str
        Existing nonadaptive source prior, default ``"second-order"``.
    macro_mapping : MacroMapping, optional
        Explicit fresh macro rays from ``renderer.prepare_macro(new_macro)``.
        Creating a new problem with these rays reevaluates the gradient and
        source mapping while reusing fixed renderer geometry.
    solve_rtol : float, optional
        Maximum relative residual of the actual coupled system for either
        backend, recomputed after the linear solve.

    ``fit(lambda_source, lambda_potential)`` and the equivalent callable
    return :class:`PotentialCorrectionResult`. Invalid configuration raises
    ValueError; valid numerical candidates that fail return ``success=False``
    and no evidence. Public forward/adjoint/prior actions compose inside JAX.
    """

    def __init__(
        self,
        renderer,
        image_data,
        noise_map,
        source_gradient,
        *,
        lambda_amplitude,
        source_regularization="second-order",
        initialization_provenance=None,
        macro_mapping=None,
        solve_rtol=None,
    ):
        if not callable(source_gradient):
            raise TypeError("source_gradient must be callable")
        super().__init__(
            renderer,
            image_data,
            noise_map,
            lambda_amplitude=lambda_amplitude,
            source_regularization=source_regularization,
            macro_mapping=macro_mapping,
            solve_rtol=solve_rtol,
        )
        self.source_gradient = source_gradient
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            self.mapping = renderer.source_mapping(macro_mapping=self.macro_mapping)
            gradient = source_gradient(*self.macro_mapping)
            if not isinstance(gradient, (tuple, list)) or len(gradient) != 2:
                raise ValueError("source_gradient must return (gradient_x, gradient_y)")
            gx, gy = (jnp.asarray(value, dtype=self.dtype) for value in gradient)
            if gx.shape != renderer.query_x.shape or gy.shape != renderer.query_y.shape:
                raise ValueError(
                    "Source gradients must match image-subpixel ray shapes"
                )
            if not bool(jnp.all(jnp.isfinite(gx)) & jnp.all(jnp.isfinite(gy))):
                raise ValueError("Initial source gradients must be finite")
            self.gradient_x = jnp.where(self.mapping.valid, gx, 0)
            self.gradient_y = jnp.where(self.mapping.valid, gy, 0)
            self.noise_logdet = 2 * jnp.sum(jnp.log(self.noise_active))
        self.initialization_provenance = {
            "gradient_coordinates": "fixed_macro_image_subpixel_rays",
            "gradient_provider": getattr(
                source_gradient, "__qualname__", type(source_gradient).__name__
            ),
            "potential_linearization": "zero_potential",
            **dict(initialization_provenance or {}),
        }
        self._dense_design = None
        self._dense_source_prior = None
        self._dense_potential_roughness = None
        self._linear_image_kernel = jax.jit(self.forward_image)
        self._operator_backends = {}

    def _subpixel_forward(self, joint_values):
        source, potential = self._joint(joint_values)
        brightness = jnp.sum(
            self.mapping.weights * source[self.mapping.indices], axis=1
        )
        ax, ay = self.renderer.potential_grid.deflection(potential)
        return brightness - self.gradient_x * ax - self.gradient_y * ay

    def forward(self, joint_values):
        """Apply the full signed linear image response M=[F0,G0]."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            return self.renderer.image_forward(self._subpixel_forward(joint_values))

    def forward_image(self, joint_values):
        """Full native linear image, retaining the established PSF outside mask."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            return self.renderer.image_forward(
                self._subpixel_forward(joint_values),
                full_image=True,
            )

    def adjoint(self, image_active):
        """Apply the exact transpose M.T to selected native data pixels."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            subpixel = self.renderer.image_adjoint(image_active)
            potential = self.renderer.potential_grid.deflection_adjoint(
                -self.gradient_x * subpixel,
                -self.gradient_y * subpixel,
            )
            return jnp.concatenate(
                (
                    self.renderer.source_adjoint(image_active, self.mapping),
                    potential,
                )
            )

    def dense_mapping(self):
        """Explicitly materialize the complete coupled image design matrix."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            if self._dense_design is None:
                self._dense_design = jax.jit(
                    lambda: _dense_response_columns(
                        self.forward, self.n_joint, self.dtype
                    )
                )()
            return self._dense_design

    def fit(
        self,
        lambda_source,
        lambda_potential,
        *,
        backend="dense",
        block_size=64,
        block_indices=None,
        pcg_max_iter=400,
        pcg_rtol=None,
        pcg_atol=0.0,
    ):
        """Solve one complete fixed-strength single-step candidate.

        ``backend="dense"`` forms the reference system and complete Cholesky
        determinants. ``backend="operator"`` uses fully coupled matrix-free
        PCG and explicitly approximate principal-block determinants.

        Operator controls are ``block_size`` (maximum coefficients per block),
        optional ``block_indices`` (a complete, disjoint joint partition),
        ``pcg_max_iter``, ``pcg_rtol`` and ``pcg_atol``. Default blocks separate
        the fields in existing node order, splitting each field at least once.
        Explicit blocks may mix fields, but cannot contain a complete joint or
        potential field. All potential support coefficients belong to blocks.
        The default PCG rtol is min(solve_rtol/10, 1e-10) for float64 or
        min(solve_rtol/10, 1e-5) for float32; the independent final acceptance
        uses ``solve_rtol``. Controls and the exact partition are returned in
        result configuration. Geometry/block preparation and compiled kernels
        are reused for changed strengths with fixed controls and topology.
        """
        if backend not in ("dense", "operator"):
            raise ValueError("backend must be 'dense' or 'operator'")
        lambda_source = _positive(lambda_source, "lambda_source")
        lambda_potential = _positive(lambda_potential, "lambda_potential")
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            if backend == "operator":
                from .potential_correction_operator import fit_operator

                return fit_operator(
                    self,
                    lambda_source,
                    lambda_potential,
                    block_size=block_size,
                    block_indices=block_indices,
                    pcg_max_iter=pcg_max_iter,
                    pcg_rtol=pcg_rtol,
                    pcg_atol=pcg_atol,
                )
            solved = _solve_dense(
                self.dense_mapping(),
                self.dense_prior(lambda_source, lambda_potential),
                self.data_active,
                self.noise_active,
            )
            reason = None
            if not bool(solved.finite_system):
                reason = "nonfinite_system"
            elif not bool(solved.prior_positive):
                reason = "prior_factorization_failed"
            elif not bool(solved.posterior_positive):
                reason = "posterior_factorization_failed"
            elif not bool(solved.finite_solution):
                reason = "nonfinite_solution_or_evidence"
            elif float(solved.relative_residual) > self.solve_rtol:
                reason = "linear_residual_exceeds_tolerance"
            info = PotentialSolverInfo(
                "cholesky",
                reason is None,
                reason is not None,
                1,
                float(solved.residual_norm),
                float(solved.relative_residual),
                reason,
            )
            return self.result_from_solution(
                solved.values,
                lambda_source,
                lambda_potential,
                log_evidence=solved.log_evidence,
                prior_logdet=solved.prior_logdet,
                posterior_logdet=solved.posterior_logdet,
                solver_info=info,
                backend="dense",
                determinant_method="full_cholesky",
                evidence_is_exact=True,
                failure_reason=reason,
            )

    def __call__(self, lambda_source, lambda_potential, **kwargs):
        return self.fit(lambda_source, lambda_potential, **kwargs)

    def result_from_solution(
        self,
        joint_values,
        lambda_source,
        lambda_potential,
        *,
        log_evidence,
        prior_logdet,
        posterior_logdet,
        solver_info,
        backend,
        determinant_method,
        evidence_is_exact,
        failure_reason=None,
        solver_configuration=None,
    ):
        """Assemble the common single-step result for a numerical backend.

        Backends provide their solve status and determinant annotations; this
        boundary additionally rejects nonfinite fields, images or evidence.
        It never changes the raw fields or their physical prior.
        """
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            linear_image = self._linear_image_kernel(joint_values)
            products = self._result_products(
                joint_values, lambda_source, lambda_potential, linear_image=linear_image
            )
            reason = failure_reason or solver_info.reason
            if not bool(jnp.all(jnp.isfinite(joint_values))):
                reason = reason or "nonfinite_fields"
            if not bool(
                jnp.all(jnp.isfinite(linear_image))
                & jnp.all(jnp.isfinite(products["nonlinear_model_image"]))
            ):
                reason = reason or "nonfinite_model_image"
            if log_evidence is None or not bool(jnp.isfinite(log_evidence)):
                reason = reason or "nonfinite_evidence"
            if not solver_info.converged or solver_info.failed:
                reason = reason or "linear_solve_failed"
            success = reason is None
            return PotentialCorrectionResult(
                **products,
                log_evidence=log_evidence if success else None,
                prior_logdet=prior_logdet,
                posterior_logdet=posterior_logdet,
                backend=backend,
                algorithm="single_step",
                status="success" if success else "failed",
                success=success,
                failure_reason=reason,
                termination_reason="solved" if success else reason,
                solver_info=solver_info,
                evidence_method="conditional_linear_gaussian",
                determinant_method=determinant_method,
                evidence_is_exact=evidence_is_exact,
                initialization_provenance=dict(self.initialization_provenance),
                configuration={
                    **self._configuration(),
                    **dict(solver_configuration or {}),
                },
            )
