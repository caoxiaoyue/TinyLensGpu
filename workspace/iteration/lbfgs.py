"""Experimental nonlinear source/potential L-BFGS with full local-GN evidence.

The data-space determinant and low-rank initial whitening are algebraically
equivalent to full joint matrices. No GN/LM update is used by the optimizer.
All numerical work in this experiment uses Float64.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from scipy.optimize import minimize

from TinyLensGpu.ObservationModel.LensImage._potential_correction_common import (
    _dense_response_columns,
)


@jax.jit
def scaled_cholesky(matrix):
    diagonal = jnp.sqrt(jnp.diag(matrix))
    factor = jnp.linalg.cholesky(matrix / diagonal[:, None] / diagonal[None, :])
    logdet = 2 * (jnp.log(jnp.diag(factor)).sum() + jnp.log(diagonal).sum())
    return diagonal, factor, logdet


@jax.jit
def prior_response(jacobian_transpose, diagonal, factor):
    return jsp.linalg.solve_triangular(
        factor, jacobian_transpose / diagonal[:, None], lower=True
    )


def low_rank_whiten(z, source_diagonal, source_factor, source_response,
                    potential_diagonal, potential_factor, potential_response,
                    lambda_source, eigenvectors, coefficients):
    """Apply T=A^-T (I+VV^T)^-1/2, so T^T H_reference T=I."""
    ns = source_diagonal.size
    zs, zp = z[:ns], z[ns:]
    root_lambda = jnp.sqrt(lambda_source)
    image = source_response.T @ zs / root_lambda + potential_response.T @ zp
    image = eigenvectors @ (coefficients * (eigenvectors.T @ image))
    zs = zs - source_response @ image / root_lambda
    zp = zp - potential_response @ image
    ds = jsp.linalg.solve_triangular(
        source_factor, zs, lower=True, trans="T"
    ) / source_diagonal / root_lambda
    dp = jsp.linalg.solve_triangular(
        potential_factor, zp, lower=True, trans="T"
    ) / potential_diagonal
    return jnp.concatenate((ds, dp))


@jax.jit
def stationarity_proxy(source_response, potential_response, white_gradient,
                       covariance_diagonal, covariance_factor):
    """Stable .5*g^T H_GN^-1*g; a local diagnostic, not an error bound."""
    ns = source_response.shape[0]
    gs, gp = white_gradient[:ns], white_gradient[ns:]
    rhs = source_response.T @ gs + potential_response.T @ gp
    step = jsp.linalg.cho_solve(
        (covariance_factor, True), rhs / covariance_diagonal
    ) / covariance_diagonal
    rs = gs - source_response @ step
    rp = gp - potential_response @ step
    return 0.5 * (jnp.vdot(rs, rs) + jnp.vdot(rp, rp) + jnp.vdot(step, step))


class EvidenceEngine:
    """Freeze geometry/prior weights, refresh the nonlinear image at every call."""

    def __init__(self, problem, *, maxiter=3000, proxy_tolerance=1e-3,
                 source_initialization="conditional_source", reference_engine=None):
        self.problem = problem
        self.maxiter = maxiter
        self.proxy_tolerance = proxy_tolerance
        self.source_initialization = source_initialization
        self.ns, self.np = problem.n_source, problem.n_potential
        self.nd = problem.renderer.n_active
        self.q0 = problem.initial_values
        self.reference_problem = problem if reference_engine is None else reference_engine.reference_problem
        self.shared_reference = reference_engine is not None
        if reference_engine is not None:
            other = reference_engine.problem
            pairs = [
                (self.q0, reference_engine.q0),
                (problem.source_scale, other.source_scale),
                (problem.renderer.source_bbox, other.renderer.source_bbox),
                (problem.renderer.query_x, other.renderer.query_x),
                (problem.renderer.query_y, other.renderer.query_y),
                (problem.renderer.potential_grid.node_indices, other.renderer.potential_grid.node_indices),
                (problem.data_active, other.data_active), (problem.noise_active, other.noise_active),
                (problem.macro_mapping.beta_x, other.macro_mapping.beta_x),
                (problem.macro_mapping.beta_y, other.macro_mapping.beta_y),
            ]
            if (self.ns != reference_engine.ns or self.np != reference_engine.np
                    or problem.source_regularizer.regularization_type != other.source_regularizer.regularization_type
                    or any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in pairs)):
                raise ValueError("Shared whitening requires identical geometry, data and physical priors")
        self.eye = jnp.eye(self.nd, dtype=jnp.float64)
        start = time.perf_counter()

        print("[prepare] fixed adaptive source precision", flush=True)
        if reference_engine is None:
            source = jax.jit(lambda: _dense_response_columns(
                lambda s: problem.source_regularizer.matvec_free(
                    s, *problem.renderer.source_bbox, scale=problem.source_scale
                ), self.ns, problem.dtype
            ))()
            self.source_diag_precision = jnp.diag(source)
            self.sd, self.sc, self.source_logdet = scaled_cholesky(source)
            self._finite_factor(self.sc, self.source_logdet, "source prior")
            del source  # Never retain a full joint prior/curvature.
            self.roughness = problem.potential_regularizer.to_dense(1.0, 0.0)
        else:
            for name in ("source_diag_precision", "sd", "sc", "source_logdet", "roughness"):
                setattr(self, name, getattr(reference_engine, name))
        self.roughness_diag = jnp.diag(self.roughness)

        def objective(q, strengths):
            ls, l4, l0 = strengths
            source, potential = q[:self.ns], q[self.ns:]
            residual = (problem.forward(q) - problem.data_active) / problem.noise_active
            source_penalty = ls * jnp.vdot(source,
                problem.source_regularizer.matvec_free(
                    source, *problem.renderer.source_bbox, scale=problem.source_scale
                ))
            fourth = problem.potential_regularizer.forward(potential)
            potential_penalty = l4 * jnp.vdot(fourth, fourth) + l0 * jnp.vdot(potential, potential)
            return 0.5 * (jnp.vdot(residual, residual) + source_penalty + potential_penalty)

        self.objective = jax.jit(objective)
        self.physical_value_grad = jax.jit(jax.value_and_grad(objective))

        def jacobian_rows(q, indices):
            _, pull = jax.vjp(lambda x: problem.forward(x) / problem.noise_active, q)
            return jax.vmap(lambda row: pull(row)[0])(
                jax.nn.one_hot(indices, self.nd, dtype=problem.dtype)
            )

        self._jacobian_rows = jax.jit(jacobian_rows)

        # Defined now, traced only after s0 is ready. Large source factors are
        # arguments below, avoiding their duplication as executable constants.
        def white_dynamic(z, strengths, base, sd, sc, s0, pd, pc, vp, vectors, coefficients):
            delta = low_rank_whiten(z, sd, sc, s0, pd, pc, vp,
                                     strengths[0], vectors, coefficients)
            return objective(base + delta, strengths)

        self.white_value_grad = jax.jit(jax.value_and_grad(white_dynamic))
        self.decode_delta = jax.jit(low_rank_whiten)
        print("[prepare] AD Jacobian at saved M3 source / zero potential", flush=True)
        initial_jacobian = self.jacobian(self.q0)
        self.j0p = initial_jacobian[:, self.ns:]
        self.j0_diagonal = jnp.sum(initial_jacobian**2, axis=0)
        self.s0 = prior_response(initial_jacobian[:, :self.ns].T, self.sd, self.sc)
        self.source_cov0 = jax.jit(lambda s: s.T @ s)(self.s0)
        self.source_cov0.block_until_ready()
        # Source-only initialization always uses this model's actual response.
        # An optional shared metric changes coordinates only, never this solve
        # or the objective/endpoint evidence of the current physical model.
        self.conditional_s0, self.conditional_cov0 = self.s0, self.source_cov0
        if reference_engine is not None:
            for name in ("j0p", "j0_diagonal", "s0", "source_cov0"):
                setattr(self, name, getattr(reference_engine, name))
        del initial_jacobian
        self.noise_normalization = float(problem.noise_logdet) + self.nd * np.log(2 * np.pi)
        self.preparation_seconds = time.perf_counter() - start
        print(f"[prepare] complete in {self.preparation_seconds:.2f}s", flush=True)

    @staticmethod
    def _finite_factor(factor, logdet, name):
        if not bool(jnp.isfinite(logdet) & jnp.all(jnp.isfinite(factor)) &
                    jnp.all(jnp.diag(factor) > 0)):
            raise FloatingPointError(f"{name} is not numerically SPD; no jitter was added")

    def jacobian(self, q):
        # VJP panels bound intermediates from the 36k native subpixels.
        rows = []
        for start in range(0, self.nd, 16):
            indices = jnp.arange(start, min(start + 16, self.nd), dtype=jnp.int32)
            rows.append(self._jacobian_rows(q, indices))
        matrix = jnp.concatenate(rows, axis=0)
        matrix.block_until_ready()
        return matrix

    def bounds(self, lambda_source):
        positive = np.asarray(self.roughness_diag)
        positive = positive[positive > 0]
        informative = np.asarray(self.j0_diagonal[self.ns:])
        informative = informative[informative > 0]
        if not informative.size or not positive.size:
            raise ValueError("Potential grid has no informative response or roughness windows")
        amplitude = float(np.median(informative))
        center4 = np.log10(amplitude / np.median(positive))
        center0 = np.log10(amplitude)
        return np.asarray([
            [np.log10(lambda_source) - 2, np.log10(lambda_source) + 2],
            [center4 - 3, center4 + 6],
            [center0 - 3, center0 + 3],
        ])

    def prepare_coordinates(self, strengths):
        ls, l4, l0 = strengths
        potential = l4 * self.roughness + l0 * jnp.eye(self.np, dtype=jnp.float64)
        pd, pc, logdet = scaled_cholesky(potential)
        self._finite_factor(pc, logdet, "potential prior")
        vp = prior_response(self.j0p.T, pd, pc)
        covariance = self.eye + self.source_cov0 / ls + vp.T @ vp
        eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
        if not bool(jnp.all(jnp.isfinite(eigenvalues)) & jnp.all(eigenvalues > 0)):
            raise FloatingPointError("initial data covariance eigendecomposition failed")
        root = jnp.sqrt(eigenvalues)
        coefficients = 1 / (root * (1 + root))
        coefficients.block_until_ready()
        return pd, pc, vp, eigenvectors, coefficients, logdet

    def decode(self, z, strengths, coordinates, base=None):
        pd, pc, vp, vectors, coefficients, _ = coordinates
        return (self.q0 if base is None else base) + self.decode_delta(
            jnp.asarray(z), self.sd, self.sc, self.s0, pd, pc, vp,
            strengths[0], vectors, coefficients
        )

    def conditional_source(self, lambda_source):
        """Exact source-only MAP at zero potential; a deterministic cold start."""
        response = getattr(self, "conditional_s0", self.s0)
        covariance0 = getattr(self, "conditional_cov0", self.source_cov0)
        covariance = self.eye + covariance0 / lambda_source
        diagonal, factor, logdet = scaled_cholesky(covariance)
        self._finite_factor(factor, logdet, "initial source-only covariance")
        data = self.problem.data_active / self.problem.noise_active
        image = jsp.linalg.cho_solve((factor, True), data / diagonal) / diagonal
        source = jsp.linalg.solve_triangular(
            self.sc, response @ image / lambda_source, lower=True, trans="T"
        ) / self.sd
        return jnp.concatenate((source, jnp.zeros(self.np, dtype=jnp.float64)))

    def fit(self, log10_strengths, *, audit=False, initial_values=None,
            initialization_label=None, prepared_coordinates=None):
        start = time.perf_counter()
        strengths = jnp.asarray(10.0 ** np.asarray(log10_strengths), dtype=jnp.float64)
        if initial_values is None:
            base = self.conditional_source(strengths[0]) if self.source_initialization == "conditional_source" else self.q0
            initialization = self.source_initialization
        else:
            base = jnp.asarray(initial_values, dtype=self.problem.dtype)
            if base.shape != self.q0.shape or not bool(jnp.all(jnp.isfinite(base))):
                raise ValueError("Explicit initial values must be finite and match the joint unknowns")
            initialization = initialization_label or "explicit_joint_values"
        self.last_initial_values = np.asarray(base)
        initial_value = float(self.objective(base, strengths))
        if prepared_coordinates is None:
            coordinates = self.prepare_coordinates(strengths)
        else:
            prepared_strengths, coordinates = prepared_coordinates
            if not np.array_equal(np.asarray(prepared_strengths), np.asarray(strengths)):
                raise ValueError("Prepared coordinates use different regularization strengths")
        coordinate_seconds = time.perf_counter() - start
        pd, pc, vp, vectors, coefficients, logdet_p = coordinates
        args = (strengths, base, self.sd, self.sc, self.s0, pd, pc, vp, vectors, coefficients)
        evaluations = 0
        history = []
        latest_value = None

        def fun(z):
            nonlocal evaluations, latest_value
            value, gradient = self.white_value_grad(jnp.asarray(z), *args)
            evaluations += 1
            latest_value = float(value)
            return latest_value, np.asarray(gradient, dtype=np.float64)

        begin = time.perf_counter()
        result = minimize(fun, np.zeros(self.problem.n_joint), jac=True,
                          method="L-BFGS-B", options={
                              "maxiter": self.maxiter, "maxfun": 120000,
                              "maxcor": 30, "maxls": 80, "ftol": 1e-15, "gtol": 1e-8,
                          }, callback=lambda z: history.append(latest_value))
        q = self.decode(result.x, strengths, coordinates, base=base)
        value, gradient = self.physical_value_grad(q, strengths)
        q.block_until_ready()
        inner_seconds = time.perf_counter() - begin
        endpoint_start = time.perf_counter()
        jacobian = self.jacobian(q)
        vs = prior_response(jacobian[:, :self.ns].T, self.sd, self.sc) / jnp.sqrt(strengths[0])
        vp_final = prior_response(jacobian[:, self.ns:].T, pd, pc)
        covariance = self.eye + vs.T @ vs + vp_final.T @ vp_final
        cd, cc, logdet_ratio = scaled_cholesky(covariance)
        self._finite_factor(cc, logdet_ratio, "endpoint full data covariance")
        gs = jsp.linalg.solve_triangular(self.sc, gradient[:self.ns] / self.sd,
                                         lower=True) / jnp.sqrt(strengths[0])
        gp = jsp.linalg.solve_triangular(pc, gradient[self.ns:] / pd, lower=True)
        proxy = float(stationarity_proxy(vs, vp_final, jnp.concatenate((gs, gp)), cd, cc))
        prior_diagonal = jnp.concatenate((strengths[0] * self.source_diag_precision,
                                           strengths[1] * self.roughness_diag + strengths[2]))
        gamma = float(jnp.max(jnp.abs(gradient) /
                              jnp.sqrt(jnp.sum(jacobian**2, axis=0) + prior_diagonal)))
        log_evidence = -float(value) - 0.5 * (float(logdet_ratio) + self.noise_normalization)
        finite = bool(jnp.all(jnp.isfinite(q)) & jnp.isfinite(value)) and np.isfinite(log_evidence + proxy)
        accepted = bool(result.success and finite and proxy <= self.proxy_tolerance)
        residual = (self.problem.data_active - self.problem.forward(q)) / self.problem.noise_active
        chi2 = float(jnp.vdot(residual, residual))
        row = dict(
            log10_strengths=np.asarray(log10_strengths).tolist(),
            strengths=dict(zip(("lambda_s", "lambda4", "lambda0"), np.asarray(strengths).tolist())),
            native_success=bool(result.success), native_status=int(result.status),
            native_message=str(result.message), iterations=int(result.nit),
            scipy_nfev=int(result.nfev), objective_evaluations=evaluations,
            objective=float(value), chi2=chi2, chi2_per_pixel=chi2 / self.nd,
            initial_objective=initial_value, source_initialization=initialization,
            shared_preconditioner_reference=self.shared_reference,
            preconditioner_reference_boundary=getattr(self.reference_problem.renderer,
                "source_boundary", "hard_zero_outside_bbox"),
            penalty=2 * float(value) - chi2,
            gn_stationarity_proxy=proxy, proxy_tolerance=self.proxy_tolerance,
            gamma=gamma, gradient_inf=float(jnp.max(jnp.abs(gradient))),
            white_gradient_inf=float(np.max(np.abs(result.jac))),
            accepted=accepted, log_evidence=log_evidence if accepted else None,
            rejection_reasons=[reason for condition, reason in (
                (not result.success, "native_optimizer_did_not_succeed"),
                (not finite, "nonfinite_endpoint"),
                (proxy > self.proxy_tolerance, "GN_stationarity_proxy_exceeds_tolerance"),
            ) if condition],
            diagnostic_log_evidence=log_evidence,
            logdet_prior=float(self.ns * jnp.log(strengths[0]) + self.source_logdet + logdet_p),
            logdet_posterior_minus_prior=float(logdet_ratio),
            coordinate_seconds=coordinate_seconds, inner_seconds=inner_seconds,
            endpoint_seconds=time.perf_counter() - endpoint_start,
            objective_history=history,
        )
        if audit:
            rng = np.random.default_rng(9436)
            z = rng.normal(size=self.problem.n_joint)
            z /= np.linalg.norm(z)
            delta = self.decode(z, strengths, coordinates) - self.q0
            jdelta = jax.jvp(self.reference_problem.forward, (self.q0,), (delta,))[1] / self.reference_problem.noise_active
            # Quadratic prior only; subtract objective data action explicitly.
            source, potential = delta[:self.ns], delta[self.ns:]
            quadratic = strengths[0] * jnp.vdot(source,
                self.problem.source_regularizer.matvec_free(source, *self.problem.renderer.source_bbox,
                                                            scale=self.problem.source_scale))
            fourth = self.problem.potential_regularizer.forward(potential)
            quadratic += strengths[1] * jnp.vdot(fourth, fourth) + strengths[2] * jnp.vdot(potential, potential)
            whiten_error = abs(float(jnp.vdot(jdelta, jdelta) + quadratic) - 1)
            augmented = jnp.concatenate((vs, vp_final, self.eye), axis=0)
            qr_root = jnp.linalg.qr(augmented, mode="r")
            qr_logdet = float(2 * jnp.log(jnp.abs(jnp.diag(qr_root))).sum())
            row["algebra_audit"] = dict(
                whitening_quadratic_error=whiten_error,
                qr_minus_cholesky_logdet=qr_logdet - float(logdet_ratio),
                whitening_pass=whiten_error < 1e-5,
                determinant_pass=abs(qr_logdet - float(logdet_ratio)) < 1e-4,
                whitening_reference_boundary=getattr(self.reference_problem.renderer,
                    "source_boundary", "hard_zero_outside_bbox"),
            )
            if not all((row["algebra_audit"]["whitening_pass"], row["algebra_audit"]["determinant_pass"])):
                raise FloatingPointError(f"Failed independent algebra audit: {row['algebra_audit']}")
        row["total_seconds"] = time.perf_counter() - start
        return row, np.asarray(q)
