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
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.utils.inversion.potential_regularization import PotentialRegularization
from TinyLensGpu.utils.inversion.regularization import FiniteDifferenceRegularization

from .potential_correction_result import (
    PotentialCorrectionResult,
    PotentialCorrectionStrengths,
    PotentialSolverInfo,
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


def _positive(value, name):
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return scalar


class SingleStepPotentialCorrection:
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
        Maximum relative residual of the actual coupled dense system.

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
        if not isinstance(renderer, PotentialCorrectionRenderer):
            raise TypeError("renderer must be a PotentialCorrectionRenderer")
        if not callable(source_gradient):
            raise TypeError("source_gradient must be callable")
        self.renderer = renderer
        self.dtype = renderer.dtype
        self.lambda_amplitude = _positive(lambda_amplitude, "lambda_amplitude")
        self.solve_rtol = _positive(
            (
                (1e-8 if self.dtype == jnp.float64 else 1e-3)
                if solve_rtol is None
                else solve_rtol
            ),
            "solve_rtol",
        )
        self.n_source = renderer.n_source
        self.n_potential = renderer.potential_grid.n_unknown
        self.n_joint = self.n_source + self.n_potential
        self.source_regularizer = FiniteDifferenceRegularization(
            renderer.source_n,
            source_regularization,
        )
        self.potential_regularizer = PotentialRegularization(renderer.potential_grid)
        self.source_gradient = source_gradient
        self.macro_mapping = (
            renderer.macro_mapping if macro_mapping is None else macro_mapping
        )
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            self.mapping = renderer.source_mapping(macro_mapping=self.macro_mapping)
            self.image_data = jnp.asarray(image_data, dtype=self.dtype)
            self.noise_map = jnp.asarray(noise_map, dtype=self.dtype)
            if (
                self.image_data.shape != renderer.image_shape
                or self.noise_map.shape != renderer.image_shape
            ):
                raise ValueError(
                    "image_data and noise_map must match the native image shape"
                )
            self.data_active = self.image_data.ravel()[renderer.flat_indices]
            self.noise_active = self.noise_map.ravel()[renderer.flat_indices]
            if not bool(jnp.all(jnp.isfinite(self.data_active))):
                raise ValueError("Selected image_data must be finite")
            if not bool(
                jnp.all(jnp.isfinite(self.noise_active) & (self.noise_active > 0))
            ):
                raise ValueError("Selected noise_map must be finite and positive")
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

    def _joint(self, joint_values):
        values = jnp.asarray(joint_values, dtype=self.dtype)
        if values.shape != (self.n_joint,):
            raise ValueError(f"joint_values must have shape ({self.n_joint},)")
        return values[: self.n_source], values[self.n_source :]

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

    def prior_matvec(self, joint_values, lambda_source, lambda_potential):
        """Apply the complete block prior without creating square matrices."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            source, potential = self._joint(joint_values)
            return jnp.concatenate(
                (
                    jnp.asarray(lambda_source, dtype=self.dtype)
                    * self.source_regularizer.matvec_free(
                        source, *self.renderer.source_bbox
                    ),
                    self.potential_regularizer.matvec(
                        potential,
                        lambda_potential,
                        self.lambda_amplitude,
                    ),
                )
            )

    def dense_mapping(self):
        """Explicitly materialize the complete coupled image design matrix."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            if self._dense_design is None:
                self._dense_design = jax.jit(jax.vmap(self.forward))(
                    jnp.eye(self.n_joint, dtype=self.dtype)
                ).T
            return self._dense_design

    def dense_prior(self, lambda_source, lambda_potential):
        """Explicitly materialize Q, including the independent all-node prior."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            if self._dense_source_prior is None:
                self._dense_source_prior = self.source_regularizer.matrix(
                    *self.renderer.source_bbox,
                ).astype(self.dtype)
                self._dense_potential_roughness = self.potential_regularizer.to_dense(
                    1, 0
                )
            prior = jnp.zeros((self.n_joint, self.n_joint), dtype=self.dtype)
            prior = prior.at[: self.n_source, : self.n_source].set(
                jnp.asarray(lambda_source, dtype=self.dtype) * self._dense_source_prior
            )
            return prior.at[self.n_source :, self.n_source :].set(
                jnp.asarray(lambda_potential, dtype=self.dtype)
                * self._dense_potential_roughness
                + jnp.asarray(self.lambda_amplitude, dtype=self.dtype)
                * jnp.eye(self.n_potential, dtype=self.dtype)
            )

    def fit(self, lambda_source, lambda_potential, *, backend="dense"):
        """Solve one complete fixed-strength single-step candidate."""
        if backend != "dense":
            raise ValueError("backend must be 'dense'")
        lambda_source = _positive(lambda_source, "lambda_source")
        lambda_potential = _positive(lambda_potential, "lambda_potential")
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
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
    ):
        """Assemble the common single-step result for a numerical backend.

        Backends provide their solve status and determinant annotations; this
        boundary additionally rejects nonfinite fields, images or evidence.
        It never changes the raw fields or their physical prior.
        """
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            source, potential = self._joint(joint_values)
            reason = failure_reason or solver_info.reason
            finite_fields = bool(
                jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(potential))
            )
            linear_image = self._linear_image_kernel(joint_values)
            linear_active = linear_image.ravel()[self.renderer.flat_indices]
            if finite_fields:
                rendered = self.renderer.render(
                    source,
                    potential,
                    macro_mapping=self.macro_mapping,
                )
                nonlinear_image, nonlinear_active = (
                    rendered.model_image,
                    rendered.image_active,
                )
                deflection_x, deflection_y, convergence = (
                    rendered.deflection_x,
                    rendered.deflection_y,
                    rendered.convergence,
                )
            else:
                reason = reason or "nonfinite_fields"
                nonlinear_image = jnp.full(
                    self.renderer.image_shape, jnp.nan, dtype=self.dtype
                )
                nonlinear_active = jnp.full_like(self.data_active, jnp.nan)
                deflection_x = deflection_y = convergence = jnp.full_like(
                    potential, jnp.nan
                )
            residual = self.data_active - linear_active
            nonlinear_residual = self.data_active - nonlinear_active
            chi2 = jnp.vdot(residual / self.noise_active, residual / self.noise_active)
            penalty = jnp.vdot(
                jnp.asarray(joint_values, dtype=self.dtype),
                self.prior_matvec(joint_values, lambda_source, lambda_potential),
            )
            if not bool(
                jnp.all(jnp.isfinite(linear_image))
                & jnp.all(jnp.isfinite(nonlinear_image))
            ):
                reason = reason or "nonfinite_model_image"
            if log_evidence is None or not bool(jnp.isfinite(log_evidence)):
                reason = reason or "nonfinite_evidence"
            if not solver_info.converged or solver_info.failed:
                reason = reason or "linear_solve_failed"
            success = reason is None

            def full(values):
                return (
                    jnp.full(
                        self.renderer.image_shape[0] * self.renderer.image_shape[1],
                        jnp.nan,
                        dtype=self.dtype,
                    )
                    .at[self.renderer.flat_indices]
                    .set(values)
                    .reshape(self.renderer.image_shape)
                )

            return PotentialCorrectionResult(
                source_values=source,
                source_x_axis=self.renderer.source_x_axis,
                source_y_axis=self.renderer.source_y_axis,
                source_bbox=self.renderer.source_bbox,
                potential_values=potential,
                potential_grid=self.renderer.potential_grid,
                deflection_x=deflection_x,
                deflection_y=deflection_y,
                convergence=convergence,
                derivative_valid=self.renderer.potential_grid.derivative_valid,
                potential_window_coverage=self.potential_regularizer.window_coverage,
                linear_model_image=linear_image,
                nonlinear_model_image=nonlinear_image,
                linear_image_active=linear_active,
                nonlinear_image_active=nonlinear_active,
                linear_residual=full(residual),
                nonlinear_residual=full(nonlinear_residual),
                linear_normalized_residual=full(residual / self.noise_active),
                nonlinear_normalized_residual=full(
                    nonlinear_residual / self.noise_active
                ),
                data_active=self.data_active,
                noise_active=self.noise_active,
                strengths=PotentialCorrectionStrengths(
                    lambda_source, lambda_potential, self.lambda_amplitude
                ),
                log_evidence=log_evidence if success else None,
                chi2=chi2,
                regularization_penalty=penalty,
                objective=0.5 * (chi2 + penalty),
                prior_logdet=prior_logdet,
                posterior_logdet=posterior_logdet,
                noise_logdet=self.noise_logdet,
                dtype=self.dtype.name,
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
                    "source_n": self.renderer.source_n,
                    "source_regularization": self.source_regularizer.regularization_type,
                    "potential_spacing": self.renderer.potential_grid.spacing,
                    "potential_regularization": "complete_pixel_unit_fourth_differences",
                    "potential_amplitude_prior": "independent_all_unknowns",
                    "nsub": self.renderer.nsub,
                    "image_shape": self.renderer.image_shape,
                    "n_data": self.renderer.n_active,
                    "n_source": self.n_source,
                    "n_potential": self.n_potential,
                    "n_core": self.renderer.potential_grid.n_core,
                    "solve_rtol": self.solve_rtol,
                    "dtype": self.dtype.name,
                },
            )
