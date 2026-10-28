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
import numpy as np

from TinyLensGpu.utils.cg_solver import PrincipalBlockPreconditioner, pcg_solve
from TinyLensGpu.utils.curvature_operator import CurvatureOperator
from .potential_correction_operator import _partition, _positive_integer
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


def _factor_blocks(matrices):
    """Equilibrated bounded Cholesky factors without a prior/jitter change."""
    scale = jnp.sqrt(jnp.diagonal(matrices, axis1=1, axis2=2))
    normalized = matrices / (scale[:, :, None] * scale[:, None, :])
    chols = jnp.linalg.cholesky(normalized) * scale[:, :, None]
    diagonal = jnp.diagonal(chols, axis1=1, axis2=2)
    logdet = 2 * jnp.sum(jnp.log(diagonal))
    positive = (
        jnp.all(jnp.isfinite(matrices))
        & jnp.all(jnp.isfinite(chols))
        & jnp.all(diagonal > 0)
        & jnp.isfinite(logdet)
    )
    return chols, logdet, positive


class _OperatorIterationBackend:
    name = "operator"
    determinant_method = "principal_block_cholesky"

    def __init__(self, problem, blocks, layout, block_size):
        import hashlib

        self.problem = problem
        width = max(map(len, blocks))
        indices = np.zeros((len(blocks), width), dtype=np.int32)
        valid = np.zeros_like(indices, dtype=bool)
        for row, block in enumerate(blocks):
            indices[row, : len(block)] = block
            valid[row, : len(block)] = True
        indices, valid = jnp.asarray(indices), jnp.asarray(valid)
        identity = jnp.eye(width, dtype=problem.dtype)[None, :, :]
        self.configuration = dict(
            preconditioner="principal_block_cholesky",
            preconditioner_application="inverse_lower_factor_matvec",
            block_layout=layout,
            block_size=block_size,
            block_sizes=tuple(map(len, blocks)),
            block_indices=blocks,
            block_count=len(blocks),
            block_partition_sha256=hashlib.sha256(
                repr(blocks).encode("ascii")
            ).hexdigest(),
            solve_residual="recomputed_full_coupled_damped_system",
        )

        def block_basis(index, mask):
            # This is a bounded panel, never a complete field/joint identity.
            return (
                jax.nn.one_hot(index, problem.n_joint, dtype=problem.dtype)
                * mask[:, None]
            )

        @jax.jit
        def unit_priors():
            def one_block(_, inputs):
                index, mask = inputs

                def unit_prior(values):
                    return jnp.concatenate(
                        (
                            problem.source_regularizer.matvec_free(
                                values[: problem.n_source],
                                *problem.renderer.source_bbox
                            ),
                            problem.potential_regularizer.matvec(
                                values[problem.n_source :], 1, 0
                            ),
                        )
                    )

                principal = jax.vmap(unit_prior)(block_basis(index, mask))[:, index]
                principal = principal * (mask[:, None] & mask[None, :])
                source = index < problem.n_source
                return None, (
                    principal * (source[:, None] & source[None, :]),
                    principal * (~source[:, None] & ~source[None, :]),
                )

            return jax.lax.scan(one_block, None, (indices, valid))[1]

        source_blocks, potential_blocks = unit_priors()

        @jax.jit
        def initialize(strengths):
            ls, lp = strengths
            amplitude = (indices >= problem.n_source) & valid
            prior = (
                ls * source_blocks
                + lp * potential_blocks
                + jnp.asarray(problem.lambda_amplitude, dtype=problem.dtype)
                * amplitude[:, :, None]
                * identity
                + (~valid)[:, :, None] * identity
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
                        )(block_basis(index, mask))
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
                delta, info = pcg_solve(
                    operator,
                    rhs,
                    PrincipalBlockPreconditioner(chols, indices, valid, inverses),
                    max_iter=max_iter,
                    rtol=rtol,
                    atol=atol,
                )
                # Recursive PCG convergence alone cannot certify this step.
                norm = jnp.linalg.norm(operator.matvec(delta) - rhs)
                relative = norm / jnp.maximum(
                    jnp.linalg.norm(rhs), jnp.finfo(problem.dtype).tiny
                )
                finite = (
                    jnp.all(jnp.isfinite(delta))
                    & jnp.isfinite(norm)
                    & jnp.isfinite(relative)
                )
                return delta, info, norm, relative, positive, finite

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
        delta, info, norm, relative, positive, finite = self._solve(
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
        return delta, PotentialSolverInfo(
            "pcg",
            reason is None,
            reason is not None,
            int(info.n_iter),
            float(norm),
            float(relative),
            reason,
        )


def iteration_operator_backend(
    problem, *, block_size, block_indices, pcg_max_iter, pcg_rtol, pcg_atol
):
    """Reuse one geometry/partition with dynamic current points and strengths."""
    blocks, layout, block_size = _partition(problem, block_size, block_indices)
    max_iter = _positive_integer(pcg_max_iter, "pcg_max_iter")
    if pcg_rtol is None:
        pcg_rtol = min(
            problem.solve_rtol * 0.1, 1e-10 if problem.dtype == jnp.float64 else 1e-5
        )
    if not np.isfinite(pcg_rtol) or pcg_rtol <= 0:
        raise ValueError("pcg_rtol must be finite and positive")
    if not np.isfinite(pcg_atol) or pcg_atol < 0:
        raise ValueError("pcg_atol must be finite and nonnegative")
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
