"""Fixed-strength L-BFGS-B reconstruction of signed source and potential."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from scipy.optimize import minimize

from TinyLensGpu.utils.lensing.mapping import (
    source_boundary_for_interpolation,
    validate_source_interpolation,
)

from ._potential_correction_common import (
    _PotentialCorrectionObservation,
    _dense_response_columns,
    _integer,
    _nonnegative,
    _positive,
)
from .potential_correction_result import (
    PotentialCorrectionResult,
    PotentialCorrectionStrengths,
    PotentialOptimizerInfo,
)

# Keep the archived optimizer controls fixed across backends and strength calls.
_LBFGSB_OPTIONS = {
    "maxfun": 120000,
    "maxcor": 30,
    "maxls": 80,
    "ftol": 1e-15,
    "gtol": 1e-8,
}


@dataclass(frozen=True)
class PotentialCorrectionWhiteningReference:
    """Reference source on the fitted grid, evaluated at zero potential."""

    source_values: jax.Array
    source_interpolation: str


class _ScaledFactor(NamedTuple):
    scale: jax.Array
    cholesky: jax.Array
    logdet: jax.Array


class _DenseReference(NamedTuple):
    source_factor: _ScaledFactor
    source_response: jax.Array
    source_covariance: jax.Array
    potential_jacobian: jax.Array
    potential_roughness: jax.Array


class _DenseCoordinates(NamedTuple):
    source_factor: _ScaledFactor
    source_response: jax.Array
    potential_factor: _ScaledFactor
    potential_response: jax.Array
    eigenvectors: jax.Array
    coefficients: jax.Array


@dataclass(frozen=True)
class _Endpoint:
    prior_logdet: jax.Array
    posterior_logdet: jax.Array
    logdet_ratio: jax.Array
    gn_stationarity: float
    diagnostics: dict = field(default_factory=dict)
    failure_reason: str | None = None


@jax.jit
def _scaled_cholesky(matrix):
    scale = jnp.sqrt(jnp.diag(matrix))
    cholesky = jnp.linalg.cholesky(matrix / scale[:, None] / scale[None, :])
    logdet = 2 * (jnp.log(jnp.diag(cholesky)).sum() + jnp.log(scale).sum())
    return _ScaledFactor(scale, cholesky, logdet)


def _require_factor(factor, name):
    if not bool(
        jnp.isfinite(factor.logdet)
        & jnp.all(jnp.isfinite(factor.cholesky))
        & jnp.all(jnp.diag(factor.cholesky) > 0)
    ):
        raise FloatingPointError(f"{name} is not numerically positive definite")


@jax.jit
def _prior_response(jacobian_transpose, factor):
    return jsp.linalg.solve_triangular(
        factor.cholesky, jacobian_transpose / factor.scale[:, None], lower=True
    )


@jax.jit
def _decode_dense(z, strengths, coordinates):
    """Apply the inverse square root of the prior-whitened GN metric.

    If the prior is A A.T and V = A^-1 J.T, the change of variables is
    A^-T (I + V V.T)^-1/2. The middle action uses eigenvectors in data space.
    """
    source = coordinates.source_factor
    potential = coordinates.potential_factor
    vs, vp = coordinates.source_response, coordinates.potential_response
    vectors = coordinates.eigenvectors
    root_lambda = jnp.sqrt(strengths[0])
    zs, zp = z[: source.scale.size], z[source.scale.size :]
    image = vs.T @ zs / root_lambda + vp.T @ zp
    image = vectors @ (coordinates.coefficients * (vectors.T @ image))
    zs = zs - vs @ image / root_lambda
    zp = zp - vp @ image
    ds = (
        jsp.linalg.solve_triangular(source.cholesky, zs, lower=True, trans="T")
        / source.scale
        / root_lambda
    )
    dp = (
        jsp.linalg.solve_triangular(potential.cholesky, zp, lower=True, trans="T")
        / potential.scale
    )
    return jnp.concatenate((ds, dp))


@jax.jit
def _dense_stationarity(vs, vp, gs, gp, covariance):
    """Woodbury inverse quadratic in a sum-of-squares form."""
    rhs = vs.T @ gs + vp.T @ gp
    step = (
        jsp.linalg.cho_solve((covariance.cholesky, True), rhs / covariance.scale)
        / covariance.scale
    )
    rs = gs - vs @ step
    rp = gp - vp @ step
    return 0.5 * (jnp.vdot(rs, rs) + jnp.vdot(rp, rp) + jnp.vdot(step, step))


class _DenseLBFGSBackend:
    name = "dense"
    determinant_method = "full_data_space"
    configuration = {}
    decode = staticmethod(_decode_dense)

    def __init__(self, problem):
        self.problem = problem
        self.reference = None
        self.value_and_grad = problem._white_kernel(self.decode)

        @jax.jit
        def source_prior():
            return _dense_response_columns(
                lambda source: problem.source_regularizer.matvec_free(
                    source, *problem.renderer.source_bbox, scale=problem.source_scale
                ),
                problem.n_source,
                problem.dtype,
            )

        def row_kernel(forward):
            @jax.jit
            def rows(values, indices):
                _, pullback = jax.vjp(
                    lambda q: forward(q) / problem.noise_active, values
                )
                basis = jax.nn.one_hot(
                    indices, problem.renderer.n_active, dtype=problem.dtype
                )
                return jax.vmap(lambda row: pullback(row)[0])(basis)

            return rows

        self._source_prior = source_prior
        self._current_rows = row_kernel(problem._forward)
        self._reference_rows = (
            self._current_rows
            if problem.reference_interpolation == problem.renderer.source_interpolation
            else row_kernel(problem._reference_forward)
        )

    def _jacobian(self, rows, values):
        # Bound the VJP panel through the native subpixels and PSF convolution.
        n_data = self.problem.renderer.n_active
        return jnp.concatenate(
            [
                rows(
                    values, jnp.arange(start, min(start + 16, n_data), dtype=jnp.int32)
                )
                for start in range(0, n_data, 16)
            ],
            axis=0,
        )

    def prepare_reference(self):
        if self.reference is not None:
            return
        source_factor = _scaled_cholesky(self._source_prior())
        _require_factor(source_factor, "Source prior")
        jacobian = self._jacobian(self._reference_rows, self.problem.reference_values)
        split = self.problem.n_source
        source_response = _prior_response(jacobian[:, :split].T, source_factor)
        covariance = source_response.T @ source_response
        roughness = self.problem.potential_regularizer.to_dense(1.0, 0.0)
        reference = _DenseReference(
            source_factor, source_response, covariance, jacobian[:, split:], roughness
        )
        jax.block_until_ready(reference)
        self.reference = reference

    def prepare_coordinates(self, strengths):
        reference = self.reference
        n_potential = self.problem.n_potential
        potential_factor = _scaled_cholesky(
            strengths[1] * reference.potential_roughness
            + strengths[2] * jnp.eye(n_potential, dtype=self.problem.dtype)
        )
        _require_factor(potential_factor, "Potential prior")
        response = _prior_response(reference.potential_jacobian.T, potential_factor)
        covariance = (
            jnp.eye(self.problem.renderer.n_active, dtype=self.problem.dtype)
            + reference.source_covariance / strengths[0]
            + response.T @ response
        )
        eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
        if not bool(jnp.all(jnp.isfinite(eigenvalues) & (eigenvalues > 0))):
            raise FloatingPointError(
                "Reference data covariance eigendecomposition failed"
            )
        root = jnp.sqrt(eigenvalues)
        return _DenseCoordinates(
            reference.source_factor,
            reference.source_response,
            potential_factor,
            response,
            eigenvectors,
            1 / (root * (1 + root)),
        )

    def endpoint(self, values, gradient, strengths, coordinates):
        source = coordinates.source_factor
        potential = coordinates.potential_factor
        split = self.problem.n_source
        root_lambda = jnp.sqrt(strengths[0])
        jacobian = self._jacobian(self._current_rows, values)
        vs = _prior_response(jacobian[:, :split].T, source) / root_lambda
        vp = _prior_response(jacobian[:, split:].T, potential)
        covariance = _scaled_cholesky(
            jnp.eye(self.problem.renderer.n_active, dtype=self.problem.dtype)
            + vs.T @ vs
            + vp.T @ vp
        )
        _require_factor(covariance, "Endpoint data covariance")
        gs = (
            jsp.linalg.solve_triangular(
                source.cholesky, gradient[:split] / source.scale, lower=True
            )
            / root_lambda
        )
        gp = jsp.linalg.solve_triangular(
            potential.cholesky, gradient[split:] / potential.scale, lower=True
        )
        stationarity = float(_dense_stationarity(vs, vp, gs, gp, covariance))
        prior_logdet = split * jnp.log(strengths[0]) + source.logdet + potential.logdet
        return _Endpoint(
            prior_logdet,
            prior_logdet + covariance.logdet,
            covariance.logdet,
            stationarity,
        )


class LBFGSPotentialCorrection(_PotentialCorrectionObservation):
    """Fit a fixed macro with a cubic Hermite source and local potential.

    Supply a lens-light-subtracted image, its noise, and signed initial
    intensities in row-major source-node order. Each fit starts at these
    intensities and zero potential. The optional whitening reference uses the
    same fitted grid, with its own source values and interpolation mode.

    The three prior strengths are supplied to each fit. L-BFGS-B works in
    coordinates whitened at the reference. The reported image, undamped GN
    stationarity statistic and local evidence use the final Hermite model.
    Both backends use the renderer's Float32 or Float64 dtype for JAX
    computations. SciPy maintains its L-BFGS-B state in Float64.
    """

    def __init__(
        self,
        renderer,
        image_data,
        noise_map,
        initial_source,
        *,
        source_regularization="second-order",
        source_scale=None,
        initialization_provenance=None,
        whitening_reference=None,
        solve_rtol=None,
        solve_atol=0.0,
    ):
        super().__init__(
            renderer,
            image_data,
            noise_map,
            source_regularization=source_regularization,
            source_scale=source_scale,
            solve_rtol=solve_rtol,
            solve_atol=solve_atol,
        )
        if renderer.source_interpolation != "cubic_hermite":
            raise ValueError("L-BFGS-B requires a cubic_hermite renderer")
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            self.initial_values = self._zero_potential_start(
                initial_source, "initial_source"
            )
            if whitening_reference is None:
                self.reference_values = self.initial_values
                self.reference_interpolation = renderer.source_interpolation
            else:
                if not isinstance(
                    whitening_reference, PotentialCorrectionWhiteningReference
                ):
                    raise TypeError(
                        "whitening_reference must be a PotentialCorrectionWhiteningReference"
                    )
                self.reference_values = self._zero_potential_start(
                    whitening_reference.source_values,
                    "whitening_reference.source_values",
                )
                self.reference_interpolation = validate_source_interpolation(
                    whitening_reference.source_interpolation
                )
        self.initialization_provenance = {
            **dict(initialization_provenance or {}),
            "source_initialization": "supplied_source_node_intensities",
            "potential_initialization": "zero",
            "whitening_reference": (
                "initial_source" if whitening_reference is None else "supplied"
            ),
            "whitening_source_interpolation": self.reference_interpolation,
            "whitening_source_boundary": source_boundary_for_interpolation(
                self.reference_interpolation
            ),
            "gradient_policy": "current_autodiff_cubic_hermite",
            "reset_each_fit": True,
        }
        self._objective_kernel = jax.jit(self._objective)
        self._value_and_grad = jax.jit(jax.value_and_grad(self._objective))
        self._dense_backend = _DenseLBFGSBackend(self)
        self._operator_backends = {}

    def _zero_potential_start(self, source, name):
        source = jnp.asarray(source, dtype=self.dtype)
        if source.shape != (self.n_source,) or not bool(jnp.all(jnp.isfinite(source))):
            raise ValueError(
                f"{name} must contain {self.n_source} finite source-node intensities"
            )
        return jnp.concatenate((source, jnp.zeros(self.n_potential, dtype=self.dtype)))

    def _forward_for(self, values, interpolation):
        source, potential = self._joint(values)
        mapping = self._source_mapping(potential, interpolation)
        return self.renderer.source_forward(source, mapping)

    def _forward(self, values):
        return self._forward_for(values, self.renderer.source_interpolation)

    def _reference_forward(self, values):
        return self._forward_for(values, self.reference_interpolation)

    def _penalties(self, source, potential, strengths):
        source_penalty = strengths[0] * jnp.vdot(
            source,
            self.source_regularizer.matvec_free(
                source, *self.renderer.source_bbox, scale=self.source_scale
            ),
        )
        fourth = self.potential_regularizer.forward(potential)
        potential_penalty = strengths[1] * jnp.vdot(fourth, fourth) + strengths[
            2
        ] * jnp.vdot(potential, potential)
        return source_penalty, potential_penalty

    def _objective(self, values, strengths):
        residual = (self._forward(values) - self.data_active) / self.noise_active
        source_penalty, potential_penalty = self._penalties(
            *self._joint(values), strengths
        )
        return 0.5 * (jnp.vdot(residual, residual) + source_penalty + potential_penalty)

    def _white_kernel(self, decode):
        def objective(z, strengths, base, coordinates):
            return self._objective(base + decode(z, strengths, coordinates), strengths)

        return jax.jit(jax.value_and_grad(objective))

    def _minimize(self, numerical, strengths, coordinates, max_iter, timings):
        args = (strengths, self.initial_values, coordinates)

        def evaluate(z):
            started = perf_counter()
            value, gradient = numerical.value_and_grad(
                jnp.asarray(z, dtype=self.dtype), *args
            )
            result = float(value), np.asarray(gradient)
            timings.setdefault(
                "first_objective_evaluation_seconds", perf_counter() - started
            )
            return result

        return minimize(
            evaluate,
            np.zeros(self.n_joint, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            options={**_LBFGSB_OPTIONS, "maxiter": max_iter},
        )

    def fit(
        self,
        strengths: PotentialCorrectionStrengths,
        *,
        backend="dense",
        max_iter=3000,
        gn_stationarity_tol=1e-2,
        block_size=64,
        block_indices=None,
        pcg_max_iter=400,
        pcg_rtol=None,
        pcg_atol=0.0,
    ) -> PotentialCorrectionResult:
        """Fit and qualify the endpoint at one fixed physical strength triple.

        Operator fits use bounded principal blocks for whitening,
        preconditioning, and determinants. PCG applies the full coupled
        undamped GN curvature at the final fields. Its recomputed residual
        must pass the constructor's independent solve tolerances.
        """
        if not isinstance(strengths, PotentialCorrectionStrengths):
            raise TypeError("strengths must be a PotentialCorrectionStrengths")
        strengths = PotentialCorrectionStrengths(
            _positive(strengths.lambda_source, "lambda_source"),
            _positive(strengths.lambda_potential, "lambda_potential"),
            _positive(strengths.lambda_amplitude, "lambda_amplitude"),
        )
        max_iter = _integer(max_iter, "max_iter")
        tolerance = _nonnegative(gn_stationarity_tol, "gn_stationarity_tol")
        if backend not in ("dense", "operator"):
            raise ValueError("backend must be 'dense' or 'operator'")
        with (
            jax.experimental.enable_x64(self.dtype == jnp.float64),
            jax.default_matmul_precision("highest"),
        ):
            numerical = self._dense_backend
            endpoint_controls = {}
            if backend == "operator":
                from .potential_correction_lbfgs_operator import lbfgs_operator_backend

                numerical, endpoint_controls = lbfgs_operator_backend(
                    self, block_size, block_indices, pcg_max_iter, pcg_rtol, pcg_atol
                )
            return self._fit(
                numerical, strengths, max_iter, tolerance, endpoint_controls
            )

    def _fit(self, numerical, strengths, max_iter, tolerance, endpoint_controls):
        weights = jnp.array(
            [
                strengths.lambda_source,
                strengths.lambda_potential,
                strengths.lambda_amplitude,
            ],
            dtype=self.dtype,
        )
        timings = {}
        values = self.initial_values
        native = endpoint = gradient_inf = white_gradient_inf = None
        initial_objective = evidence = None
        reason = None
        try:
            started = perf_counter()
            numerical.prepare_reference()
            timings["fixed_reference_seconds"] = perf_counter() - started
            started = perf_counter()
            initial_objective = float(self._objective_kernel(values, weights))
            timings["initial_objective_seconds"] = perf_counter() - started
            if not np.isfinite(initial_objective):
                raise FloatingPointError("nonfinite_initial_objective")
            started = perf_counter()
            coordinates = jax.block_until_ready(numerical.prepare_coordinates(weights))
            timings["whitening_seconds"] = perf_counter() - started
            started = perf_counter()
            native = self._minimize(numerical, weights, coordinates, max_iter, timings)
            white_gradient_inf = float(np.max(np.abs(native.jac)))
            candidate = values + numerical.decode(
                jnp.asarray(native.x, dtype=self.dtype), weights, coordinates
            )
            if not bool(jnp.all(jnp.isfinite(candidate))):
                raise FloatingPointError("nonfinite_fitted_fields")
            values = candidate
            timings["optimizer_seconds"] = perf_counter() - started
            started = perf_counter()
            objective, gradient = self._value_and_grad(values, weights)
            gradient_inf = float(jnp.max(jnp.abs(gradient)))
            if not (np.isfinite(gradient_inf) and bool(jnp.isfinite(objective))):
                raise FloatingPointError("nonfinite_endpoint")
            endpoint = numerical.endpoint(
                values, gradient, weights, coordinates, **endpoint_controls
            )
            evidence = -objective - 0.5 * (
                endpoint.logdet_ratio
                + self.noise_logdet
                + self.data_active.size * np.log(2 * np.pi)
            )
            if not native.success:
                reason = "native_optimizer_did_not_succeed"
            elif endpoint.failure_reason is not None:
                reason = endpoint.failure_reason
            elif not np.isfinite(white_gradient_inf) or not bool(
                jnp.isfinite(evidence)
                & jnp.isfinite(endpoint.prior_logdet)
                & jnp.isfinite(endpoint.posterior_logdet)
                & jnp.isfinite(endpoint.gn_stationarity)
            ):
                reason = "nonfinite_endpoint"
            elif endpoint.gn_stationarity > tolerance:
                reason = "gn_stationarity_exceeds_tolerance"
            timings["endpoint_seconds"] = perf_counter() - started
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            reason = str(exc) or "numerical_failure"

        started = perf_counter()
        penalty = sum(self._penalties(*self._joint(values), weights))
        products = self._field_products(values, strengths, penalty)
        if reason is None:
            finite_maps = all(
                bool(
                    jnp.all(
                        jnp.isfinite(
                            jnp.where(products["derivative_valid"], products[key], 0)
                        )
                    )
                )
                for key in ("deflection_x", "deflection_y", "convergence")
            )
            if not finite_maps or not bool(
                jnp.all(jnp.isfinite(products["nonlinear_model_image"]))
                & jnp.isfinite(products["objective"])
            ):
                reason = "nonfinite_result_products"
        success = reason is None
        info = PotentialOptimizerInfo(
            native_success=None if native is None else bool(native.success),
            native_status=None if native is None else int(native.status),
            native_message=None if native is None else str(native.message),
            iterations=0 if native is None else int(native.nit),
            function_evaluations=0 if native is None else int(native.nfev),
            gradient_evaluations=0 if native is None else int(native.njev),
            initial_objective=initial_objective,
            gradient_inf=gradient_inf,
            white_gradient_inf=white_gradient_inf,
            gn_stationarity=None if endpoint is None else endpoint.gn_stationarity,
            gn_stationarity_tol=tolerance,
            diagnostics={} if endpoint is None else endpoint.diagnostics,
        )
        jax.block_until_ready(products)
        timings["result_seconds"] = perf_counter() - started
        unavailable = jnp.asarray(jnp.nan, dtype=self.dtype)
        return PotentialCorrectionResult(
            **products,
            log_evidence=evidence if success else None,
            prior_logdet=unavailable if endpoint is None else endpoint.prior_logdet,
            posterior_logdet=(
                unavailable if endpoint is None else endpoint.posterior_logdet
            ),
            backend=numerical.name,
            algorithm="lbfgs",
            status="success" if success else "failed",
            success=success,
            failure_reason=reason,
            termination_reason="converged" if success else reason,
            solver_info=None,
            optimizer_info=info,
            evidence_method="local_gauss_newton",
            determinant_method=numerical.determinant_method,
            evidence_is_exact=False,
            initialization_provenance=dict(self.initialization_provenance),
            configuration={
                **self._configuration(),
                **numerical.configuration,
                **endpoint_controls,
                "max_iter": max_iter,
                "optimizer_state_dtype": "float64",
                "optimizer_options": {**_LBFGSB_OPTIONS, "maxiter": max_iter},
                "gn_stationarity_tol": tolerance,
                "solve_atol": self.solve_atol,
                "timings": timings,
            },
        )
