"""Current bounded blocks and exact coupled PCG increments for nonlinear fits.

Only the partition and unit-prior blocks are fixed. The current mapping and
source derivatives are dynamic arguments to every data-block/curvature kernel.
No controller or frozen single-step image action lives in this adapter.
"""

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from TinyLensGpu.utils.cg_solver import PCGInfo, PrincipalBlockPreconditioner, pcg_solve
from TinyLensGpu.utils.curvature_operator import CurvatureOperator
from ._potential_correction_operator_common import (
    _assemble_prior_blocks,
    _block_basis,
    _factor_blocks,
    _padded_partition,
    _partition,
    _pcg_controls,
    _unit_prior_blocks,
)
from .potential_correction_result import PotentialSolverInfo


class _PriorBlocks(NamedTuple):
    strengths: jax.Array
    blocks: jax.Array


class _OperatorSystem(NamedTuple):
    linearization: Any
    prior: _PriorBlocks
    blocks: jax.Array
    metric: jax.Array
    finite: jax.Array


class _OperatorIterationBackend:
    name = "operator"
    determinant_method = "principal_block_cholesky"

    def __init__(self, problem, blocks, layout, block_size):
        self.problem = problem
        indices, valid, digest = _padded_partition(blocks)
        width = indices.shape[1]
        identity = jnp.eye(width, dtype=problem.dtype)[None, :, :]
        self.configuration = dict(
            preconditioner="principal_block_cholesky",
            preconditioner_application="inverse_lower_factor_matvec",
            block_layout=layout,
            block_size=block_size,
            block_sizes=tuple(map(len, blocks)),
            block_indices=blocks,
            block_count=len(blocks),
            block_partition_sha256=digest,
            solve_residual="recomputed_full_coupled_damped_system",
            pcg_refinement_policy="one_separate_residual_correction_within_total_budget",
            pcg_recursive_residual_scope="initial_solve_original_rhs",
        )

        @jax.jit
        def unit_priors():
            def one_block(_, inputs):
                index, mask = inputs

                return None, _unit_prior_blocks(
                    problem, _block_basis(problem, index, mask), index, mask
                )

            return jax.lax.scan(one_block, None, (indices, valid))[1]

        source_blocks, potential_blocks = unit_priors()

        @jax.jit
        def initialize(strengths):
            ls, lp = strengths
            prior = _assemble_prior_blocks(
                problem, source_blocks, potential_blocks, indices, valid, ls, lp
            )
            _, logdet, positive = _factor_blocks(prior)
            return _PriorBlocks(strengths, prior), logdet, positive

        self.initialize = initialize

        @jax.jit
        def prepare(linearization, prior):
            with jax.default_matmul_precision("highest"):

                def one_block(_, inputs):
                    index, mask = inputs
                    panel = (
                        jax.vmap(
                            lambda vector: problem.jacobian_forward(
                                vector, linearization
                            )
                        )(_block_basis(problem, index, mask))
                        / problem.noise_active[None, :]
                    )
                    return None, panel @ panel.T

                _, data = jax.lax.scan(one_block, None, (indices, valid))
                posterior = data + prior.blocks
                diagonal = jnp.diagonal(posterior, axis1=1, axis2=2) * valid
                metric = (
                    jnp.zeros(problem.n_joint, dtype=problem.dtype)
                    .at[indices]
                    .add(diagonal)
                )
                finite = jnp.all(jnp.isfinite(posterior)) & jnp.all(metric > 0)
                return _OperatorSystem(linearization, prior, posterior, metric, finite)

        self.prepare = prepare

        def curvature(values, data, _spec):
            linearization, strengths, metric, damping = data
            return (
                problem.jacobian_adjoint(
                    problem.jacobian_forward(values, linearization)
                    / problem.noise_active**2,
                    linearization,
                )
                + problem.prior_matvec(values, *strengths)
                + damping * metric * values
            )

        @partial(jax.jit, static_argnames=("max_iter", "rtol", "atol"))
        def solve(system, rhs, damping, *, max_iter, rtol, atol):
            with jax.default_matmul_precision("highest"):
                # Padding remains identity and never receives physical damping.
                damped = (
                    system.blocks
                    + (damping * system.metric[indices] * valid)[:, :, None] * identity
                )
                chols, _, positive = _factor_blocks(damped)
                inverses = jax.vmap(
                    lambda chol: jsl.solve_triangular(
                        chol, jnp.eye(width, dtype=problem.dtype), lower=True
                    )
                )(chols)
                positive = positive & jnp.all(jnp.isfinite(inverses))
                operator = CurvatureOperator(
                    data=(
                        system.linearization,
                        system.prior.strengths,
                        system.metric,
                        damping,
                    ),
                    kernel=curvature,
                    spec=None,
                    size=problem.n_joint,
                )
                preconditioner = PrincipalBlockPreconditioner(
                    chols, indices, valid, inverses
                )
                delta, initial_info = pcg_solve(
                    operator,
                    rhs,
                    preconditioner,
                    max_iter=max_iter,
                    rtol=rtol,
                    atol=atol,
                )
                # Recursive PCG convergence alone cannot certify this step.
                residual = rhs - operator.matvec(delta)
                initial_norm = jnp.linalg.norm(residual)
                rhs_norm = jnp.linalg.norm(rhs)
                threshold = problem.solve_atol + problem.solve_rtol * rhs_norm
                initial_finite = jnp.all(jnp.isfinite(delta)) & jnp.isfinite(
                    initial_norm
                )
                remaining = max_iter - initial_info.n_iter
                attempt = (
                    positive
                    & initial_finite
                    & initial_info.converged
                    & ~initial_info.failed
                    & (initial_norm > threshold)
                    & (remaining > 0)
                )

                def correct(_):
                    # Accumulate a small correction separately. Restarting PCG
                    # on the large delta loses more of this stiff-system residual.
                    # The public wrapper makes max_iter static for other
                    # callers. This adapter is already jitted: tracing the
                    # identical body here permits a dynamic remaining budget.
                    correction, correction_info = pcg_solve.__wrapped__(
                        operator,
                        residual,
                        preconditioner,
                        max_iter=remaining,
                        rtol=rtol,
                        atol=atol,
                    )
                    candidate = delta + correction
                    candidate_norm = jnp.linalg.norm(rhs - operator.matvec(candidate))
                    candidate_finite = jnp.all(jnp.isfinite(candidate)) & jnp.isfinite(
                        candidate_norm
                    )
                    improved = candidate_finite & (candidate_norm < initial_norm)
                    return (
                        jnp.where(improved, candidate, delta),
                        jnp.where(improved, candidate_norm, initial_norm),
                        correction_info,
                        candidate_finite,
                        improved,
                        candidate_norm,
                    )

                def unchanged(_):
                    return (
                        delta,
                        initial_norm,
                        PCGInfo(
                            jnp.asarray(0, dtype=initial_info.n_iter.dtype),
                            jnp.asarray(0, dtype=rhs.dtype),
                            jnp.asarray(True),
                            jnp.asarray(False),
                        ),
                        initial_finite,
                        jnp.asarray(False),
                        initial_norm,
                    )

                (
                    delta,
                    norm,
                    correction_info,
                    correction_finite,
                    improved,
                    candidate_norm,
                ) = jax.lax.cond(
                    attempt,
                    correct,
                    unchanged,
                    operand=None,
                )
                info = PCGInfo(
                    initial_info.n_iter + correction_info.n_iter,
                    initial_info.residual_norm,
                    initial_info.converged & correction_info.converged,
                    initial_info.failed | correction_info.failed,
                )
                relative = norm / jnp.maximum(rhs_norm, jnp.finfo(problem.dtype).tiny)
                finite = (
                    initial_finite
                    & correction_finite
                    & jnp.all(jnp.isfinite(delta))
                    & jnp.isfinite(norm)
                    & jnp.isfinite(relative)
                )
                diagnostics = (
                    initial_info,
                    initial_norm,
                    correction_info,
                    attempt,
                    improved,
                    candidate_norm,
                    rhs_norm,
                    threshold,
                )
                return delta, info, norm, relative, positive, finite, diagnostics

        self._solve = solve

        @jax.jit
        def measures(system, point, delta, damping):
            with jax.default_matmul_precision("highest"):
                # Exact undamped coupled H, independent of the block approximation.
                physical = jnp.vdot(
                    delta,
                    curvature(
                        delta,
                        (
                            system.linearization,
                            system.prior.strengths,
                            system.metric,
                            jnp.asarray(0, dtype=problem.dtype),
                        ),
                        None,
                    ),
                )
                scale = jnp.sqrt(system.metric)
                damped_step = jnp.sqrt(damping) * scale * delta
                damped = jnp.vdot(damped_step, damped_step)
                predicted = jnp.vdot(point.negative_gradient, delta) - 0.5 * physical
                fraction = damped / jnp.maximum(
                    physical + damped, jnp.finfo(problem.dtype).tiny
                )
                update = jnp.max(jnp.abs(scale * delta)) / jnp.maximum(
                    1.0, jnp.max(jnp.abs(scale * point.values))
                )
                return predicted, fraction, update

        self.measures = measures

        @jax.jit
        def logdets(system):
            # Final current physical blocks; the step's damped factors are never reused.
            _, logdet, positive = _factor_blocks(system.blocks)
            return logdet, positive

        self.logdets = logdets

    def solve(self, system, rhs, damping):
        delta, info, norm, relative, positive, finite, diagnostics = self._solve(
            system, rhs, damping, max_iter=self.max_iter, rtol=self.rtol, atol=self.atol
        )
        reason = None
        if not bool(positive):
            reason = "preconditioner_factorization_failed"
        elif bool(info.failed):
            reason = "pcg_breakdown"
        elif not bool(info.converged):
            reason = "pcg_iteration_limit"
        elif not bool(finite):
            reason = "nonfinite_increment"
        elif float(norm) > self.problem.solve_atol + self.problem.solve_rtol * float(
            jnp.linalg.norm(rhs)
        ):
            reason = "linear_residual_exceeds_tolerance"
        self.configuration["pcg_recursive_residual_norm"] = float(info.residual_norm)
        (
            initial,
            initial_norm,
            correction,
            attempted,
            improved,
            candidate_norm,
            rhs_norm,
            threshold,
        ) = diagnostics
        phases = [
            dict(
                phase="initial",
                iterations=int(initial.n_iter),
                recursive_converged=bool(initial.converged),
                recursive_failed=bool(initial.failed),
                recursive_residual_norm=float(initial.residual_norm),
                true_residual_norm=float(initial_norm),
            )
        ]
        if bool(attempted):
            phases.append(
                dict(
                    phase="residual_correction",
                    iterations=int(correction.n_iter),
                    recursive_converged=bool(correction.converged),
                    recursive_failed=bool(correction.failed),
                    recursive_residual_norm=float(correction.residual_norm),
                    candidate_true_residual_norm=float(candidate_norm),
                    accepted=bool(improved),
                )
            )
        return delta, PotentialSolverInfo(
            "pcg",
            reason is None,
            reason is not None,
            int(info.n_iter),
            float(norm),
            float(relative),
            reason,
            dict(
                iteration_budget=self.max_iter,
                phases=phases,
                refinement_attempted=bool(attempted),
                refinement_accepted=bool(improved),
                original_rhs_norm=float(rhs_norm),
                true_residual_threshold=float(threshold),
            ),
        )


def iteration_operator_backend(
    problem, *, block_size, block_indices, pcg_max_iter, pcg_rtol, pcg_atol
):
    """Reuse one geometry/partition with dynamic current points and strengths."""
    blocks, layout, block_size = _partition(problem, block_size, block_indices)
    max_iter, pcg_rtol, pcg_atol = _pcg_controls(
        problem, pcg_max_iter, pcg_rtol, pcg_atol
    )
    key = (blocks, layout, block_size)
    if key not in problem._operator_backends:
        problem._operator_backends[key] = _OperatorIterationBackend(
            problem, blocks, layout, block_size
        )
    numerical = problem._operator_backends[key]
    numerical.max_iter, numerical.rtol, numerical.atol = (
        max_iter,
        float(pcg_rtol),
        float(pcg_atol),
    )
    numerical.configuration.update(
        pcg_max_iter=max_iter,
        pcg_rtol=float(pcg_rtol),
        pcg_atol=float(pcg_atol),
        pcg_recursive_residual_norm=None,
    )
    return numerical
