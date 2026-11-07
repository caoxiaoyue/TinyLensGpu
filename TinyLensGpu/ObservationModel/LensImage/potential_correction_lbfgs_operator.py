"""Bounded GN whitening, coupled stationarity, and block determinants."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from TinyLensGpu.utils.cg_solver import PrincipalBlockPreconditioner, pcg_solve
from TinyLensGpu.utils.curvature_operator import CurvatureOperator

from ._potential_correction_operator_common import (
    _assemble_prior_blocks,
    _block_basis,
    _data_curvature_block,
    _factor_blocks,
    _inverse_lower_factors,
    _padded_partition,
    _partition,
    _pcg_controls,
    _unit_prior_blocks,
)
from .potential_correction_lbfgs import _Endpoint


class _BlockCoordinates(NamedTuple):
    indices: jax.Array
    valid: jax.Array
    inverse_chols: jax.Array
    prior_blocks: jax.Array
    prior_logdet: jax.Array


@jax.jit
def _decode_blocks(white, _strengths, coordinates):
    """Apply the inverse transpose of the reference lower factor blockwise."""
    indices, valid = coordinates.indices, coordinates.valid
    steps = jnp.einsum("bji,bj->bi", coordinates.inverse_chols, white[indices] * valid)
    return jnp.zeros_like(white).at[indices].add(steps * valid)


class _OperatorLBFGSBackend:
    name = "operator"
    determinant_method = "principal_block_cholesky"
    decode = staticmethod(_decode_blocks)

    def __init__(self, problem, blocks, layout, block_size):
        self.problem = problem
        self.reference = None
        indices, valid, digest = _padded_partition(blocks)
        self.configuration = dict(
            whitening="reference_principal_block_cholesky",
            preconditioner="principal_block_cholesky",
            preconditioner_application="inverse_lower_factor_matvec",
            block_layout=layout,
            block_size=block_size,
            block_sizes=tuple(map(len, blocks)),
            block_indices=blocks,
            block_count=len(blocks),
            block_partition_sha256=digest,
            solve_residual="recomputed_full_coupled_undamped_system",
        )
        self.value_and_grad = problem._white_kernel(self.decode)

        def data_block(basis, linearization):
            return _data_curvature_block(
                problem,
                basis,
                lambda increment: problem._jacobian_forward(increment, linearization),
            )

        @jax.jit
        def reference_blocks(values):
            linearization = problem._linearize(values, problem.reference_interpolation)

            def one_block(_, inputs):
                index, mask = inputs
                basis = _block_basis(problem, index, mask)
                source, potential = _unit_prior_blocks(problem, basis, index, mask)
                return None, (source, potential, data_block(basis, linearization))

            return jax.lax.scan(one_block, None, (indices, valid))[1]

        @jax.jit
        def factor_reference(reference, strengths):
            source, potential, data = reference
            prior = _assemble_prior_blocks(
                problem, source, potential, indices, valid, *strengths
            )
            _, prior_logdet, prior_ok = _factor_blocks(prior)
            chols, _, posterior_ok = _factor_blocks(data + prior)
            inverses = _inverse_lower_factors(chols)
            coordinates = _BlockCoordinates(
                indices, valid, inverses, prior, prior_logdet
            )
            return coordinates, prior_ok, posterior_ok & jnp.all(jnp.isfinite(inverses))

        @jax.jit
        def factor_endpoint(values, prior):
            linearization = problem._linearize(
                values, problem.renderer.source_interpolation
            )

            def one_block(_, inputs):
                index, mask = inputs
                basis = _block_basis(problem, index, mask)
                return None, data_block(basis, linearization)

            _, data = jax.lax.scan(one_block, None, (indices, valid))
            chols, logdet, positive = _factor_blocks(data + prior)
            inverses = _inverse_lower_factors(chols)
            preconditioner = PrincipalBlockPreconditioner(
                chols, indices, valid, inverses
            )
            positive = positive & jnp.all(jnp.isfinite(inverses))
            return linearization, preconditioner, logdet, positive

        def curvature(values, data, _spec):
            linearization, strengths = data
            image = problem._jacobian_forward(values, linearization)
            return problem._jacobian_adjoint(
                image / problem.noise_active**2, linearization
            ) + problem._apply_prior(values, *strengths)

        @partial(jax.jit, static_argnames=("max_iter", "rtol", "atol"))
        def solve(
            linearization, preconditioner, strengths, gradient, *, max_iter, rtol, atol
        ):
            operator = CurvatureOperator(
                data=(linearization, strengths),
                kernel=curvature,
                spec=None,
                size=problem.n_joint,
            )
            inverse_gradient, info = pcg_solve(
                operator,
                gradient,
                preconditioner,
                max_iter=max_iter,
                rtol=rtol,
                atol=atol,
            )
            true_residual = gradient - operator.matvec(inverse_gradient)
            residual_norm = jnp.linalg.norm(true_residual)
            rhs_norm = jnp.linalg.norm(gradient)
            threshold = problem.solve_atol + problem.solve_rtol * rhs_norm
            relative_residual = residual_norm / jnp.maximum(
                rhs_norm, jnp.finfo(problem.dtype).tiny
            )
            stationarity = 0.5 * jnp.vdot(gradient, inverse_gradient)
            finite = (
                jnp.all(jnp.isfinite(inverse_gradient))
                & jnp.isfinite(residual_norm)
                & jnp.isfinite(relative_residual)
                & jnp.isfinite(rhs_norm)
                & jnp.isfinite(threshold)
                & jnp.isfinite(stationarity)
                & jnp.isfinite(info.residual_norm)
            )
            return (
                info,
                stationarity,
                residual_norm,
                relative_residual,
                rhs_norm,
                threshold,
                finite,
            )

        self._reference_blocks = reference_blocks
        self._factor_reference = factor_reference
        self._factor_endpoint = factor_endpoint
        self._solve = solve

    def prepare_reference(self):
        if self.reference is None:
            self.reference = jax.block_until_ready(
                self._reference_blocks(self.problem.reference_values)
            )

    def prepare_coordinates(self, strengths):
        coordinates, prior_ok, posterior_ok = self._factor_reference(
            self.reference, strengths
        )
        if not bool(prior_ok):
            raise FloatingPointError(
                "Operator prior blocks are not numerically positive definite"
            )
        if not bool(posterior_ok):
            raise FloatingPointError("Reference GN blocks could not be factored")
        return coordinates

    def endpoint(
        self,
        values,
        gradient,
        strengths,
        coordinates,
        *,
        pcg_max_iter,
        pcg_rtol,
        pcg_atol
    ):
        linearization, preconditioner, posterior_logdet, positive = (
            self._factor_endpoint(values, coordinates.prior_blocks)
        )
        prior_logdet = coordinates.prior_logdet
        logdet_ratio = posterior_logdet - prior_logdet
        if not bool(positive):
            return _Endpoint(
                prior_logdet,
                posterior_logdet,
                logdet_ratio,
                float("nan"),
                failure_reason="endpoint_factorization_failed",
            )

        info, stationarity, norm, relative, rhs_norm, threshold, finite = self._solve(
            linearization,
            preconditioner,
            strengths,
            gradient,
            max_iter=pcg_max_iter,
            rtol=pcg_rtol,
            atol=pcg_atol,
        )
        reason = None
        if bool(info.failed):
            reason = "pcg_breakdown"
        elif not bool(info.converged):
            reason = "pcg_iteration_limit"
        elif not bool(finite) or float(stationarity) < 0:
            reason = "nonfinite_stationarity_solve"
        elif float(norm) > float(threshold):
            reason = "linear_residual_exceeds_tolerance"
        diagnostics = dict(
            stationarity_solver=dict(
                method="pcg",
                iterations=int(info.n_iter),
                converged=reason is None,
                failed=reason is not None,
                reason=reason,
                recursive_converged=bool(info.converged),
                recursive_failed=bool(info.failed),
                recursive_residual_norm=float(info.residual_norm),
                true_residual_norm=float(norm),
                relative_residual=float(relative),
                rhs_norm=float(rhs_norm),
                true_residual_threshold=float(threshold),
            )
        )
        return _Endpoint(
            prior_logdet,
            posterior_logdet,
            logdet_ratio,
            float(stationarity),
            diagnostics,
            reason,
        )


def lbfgs_operator_backend(problem, block_size, block_indices, max_iter, rtol, atol):
    """Reuse the invariant blocks and pass solver controls independently per fit."""
    blocks, layout, block_size = _partition(problem, block_size, block_indices)
    max_iter, rtol, atol = _pcg_controls(problem, max_iter, rtol, atol)
    key = (blocks, layout, block_size)
    if key not in problem._operator_backends:
        problem._operator_backends[key] = _OperatorLBFGSBackend(
            problem, blocks, layout, block_size
        )
    return problem._operator_backends[key], dict(
        pcg_max_iter=max_iter, pcg_rtol=rtol, pcg_atol=atol
    )
