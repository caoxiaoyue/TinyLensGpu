"""Bounded principal blocks and the full matrix-free source–potential solve.

Only preconditioning and determinant evaluation discard inter-block entries.
The fitted vector always solves the complete coupled conditional model.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from TinyLensGpu.utils.cg_solver import (
    PCGInfo,
    PrincipalBlockPreconditioner,
    pcg_solve,
)
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


class _OperatorSolution(NamedTuple):
    values: jax.Array
    log_evidence: jax.Array
    prior_logdet: jax.Array
    posterior_logdet: jax.Array
    residual_norm: jax.Array
    relative_residual: jax.Array
    pcg_info: PCGInfo
    finite_system: jax.Array
    prior_positive: jax.Array
    posterior_positive: jax.Array
    finite_solution: jax.Array


class _SingleStepOperator:
    """Numerical state cached for one fixed problem and partition.

    ``_principal_determinants`` is the determinant-method boundary: it returns
    both terms, factor validity, and the matching preconditioner. Future
    determinant methods can replace that calculation independently of the
    exact curvature action.
    """

    determinant_method = "principal_block_cholesky"

    def __init__(self, problem, blocks, layout, block_size):
        self.problem = problem
        self.blocks = blocks
        self.layout = layout
        self.block_size = block_size
        self.indices, self.valid, self.partition_digest = _padded_partition(blocks)

        @jax.jit
        def prepare():
            def one_block(_, inputs):
                indices, valid = inputs
                # Only this bounded block is batched. scan releases the image
                # panel before advancing; no full identity is ever allocated.
                basis = _block_basis(problem, indices, valid)
                panel = jax.vmap(problem.forward)(basis) / problem.noise_active[None, :]
                source_block, potential_block = _unit_prior_blocks(
                    problem, basis, indices, valid
                )
                with jax.default_matmul_precision("highest"):
                    data_block = panel @ panel.T
                return None, (data_block, source_block, potential_block)

            _, components = jax.lax.scan(one_block, None, (self.indices, self.valid))
            rhs = problem.adjoint(problem.data_active / problem.noise_active**2)
            return components, rhs

        self.components, self.rhs = prepare()

        def curvature(values, strengths, _spec):
            lambda_source, lambda_potential = strengths
            return problem.adjoint(
                problem.forward(values) / problem.noise_active**2
            ) + problem.prior_matvec(values, lambda_source, lambda_potential)

        self.curvature = curvature

        @partial(jax.jit, static_argnames=("max_iter", "rtol", "atol"))
        def solve(lambda_source, lambda_potential, *, max_iter, rtol, atol):
            with jax.default_matmul_precision("highest"):
                operator = CurvatureOperator(
                    data=(lambda_source, lambda_potential),
                    kernel=self.curvature,
                    spec=None,
                    size=problem.n_joint,
                )
                (
                    preconditioner,
                    prior_logdet,
                    posterior_logdet,
                    finite,
                    prior_ok,
                    posterior_ok,
                ) = self._principal_determinants(lambda_source, lambda_potential)
                values, info = pcg_solve(
                    operator,
                    self.rhs,
                    preconditioner,
                    max_iter=max_iter,
                    rtol=rtol,
                    atol=atol,
                )
                # Recursive CG residuals can drift. This is the actual full
                # coupled residual, recomputed independently after the solve.
                residual_norm = jnp.linalg.norm(operator.matvec(values) - self.rhs)
                relative_residual = residual_norm / jnp.maximum(
                    jnp.linalg.norm(self.rhs),
                    jnp.finfo(problem.dtype).tiny,
                )
                residual = problem.data_active - problem.forward(values)
                log_evidence = -0.5 * (
                    jnp.vdot(
                        residual / problem.noise_active, residual / problem.noise_active
                    )
                    + jnp.vdot(
                        values,
                        problem.prior_matvec(values, lambda_source, lambda_potential),
                    )
                    + posterior_logdet
                    - prior_logdet
                    + problem.noise_logdet
                    + problem.data_active.size * jnp.log(2 * jnp.pi)
                )
                return _OperatorSolution(
                    values,
                    log_evidence,
                    prior_logdet,
                    posterior_logdet,
                    residual_norm,
                    relative_residual,
                    info,
                    finite & jnp.all(jnp.isfinite(self.rhs)),
                    prior_ok,
                    posterior_ok,
                    jnp.all(jnp.isfinite(values))
                    & jnp.isfinite(log_evidence)
                    & jnp.isfinite(relative_residual),
                )

        self.solve = solve

    def _principal_determinants(self, lambda_source, lambda_potential):
        data, source, potential = self.components
        width = self.indices.shape[1]
        prior = _assemble_prior_blocks(
            self.problem,
            source,
            potential,
            self.indices,
            self.valid,
            lambda_source,
            lambda_potential,
        )
        posterior = data + prior
        _, prior_logdet, prior_ok = _factor_blocks(prior)
        chols, posterior_logdet, posterior_ok = _factor_blocks(posterior)
        inverse_chols = jax.vmap(
            lambda chol: jsl.solve_triangular(
                chol, jnp.eye(width, dtype=chols.dtype), lower=True
            )
        )(chols)
        return (
            PrincipalBlockPreconditioner(
                chols, self.indices, self.valid, inverse_chols
            ),
            prior_logdet,
            posterior_logdet,
            jnp.all(jnp.isfinite(prior)) & jnp.all(jnp.isfinite(posterior)),
            prior_ok,
            posterior_ok & jnp.all(jnp.isfinite(inverse_chols)),
        )

    def fit(self, lambda_source, lambda_potential, *, max_iter, rtol, atol):
        problem = self.problem
        solved = self.solve(
            jnp.asarray(lambda_source, dtype=problem.dtype),
            jnp.asarray(lambda_potential, dtype=problem.dtype),
            max_iter=max_iter,
            rtol=rtol,
            atol=atol,
        )
        reason = None
        if not bool(solved.finite_system):
            reason = "nonfinite_system"
        elif not bool(solved.prior_positive):
            reason = "prior_factorization_failed"
        elif not bool(solved.posterior_positive):
            reason = "preconditioner_factorization_failed"
        elif bool(solved.pcg_info.failed):
            reason = "pcg_breakdown"
        elif not bool(solved.pcg_info.converged):
            reason = "pcg_iteration_limit"
        elif not bool(solved.finite_solution):
            reason = "nonfinite_solution_or_evidence"
        elif float(solved.relative_residual) > problem.solve_rtol:
            reason = "linear_residual_exceeds_tolerance"
        info = PotentialSolverInfo(
            "pcg",
            reason is None,
            reason is not None,
            int(solved.pcg_info.n_iter),
            float(solved.residual_norm),
            float(solved.relative_residual),
            reason,
        )
        return problem.result_from_solution(
            solved.values,
            lambda_source,
            lambda_potential,
            log_evidence=solved.log_evidence,
            prior_logdet=solved.prior_logdet,
            posterior_logdet=solved.posterior_logdet,
            solver_info=info,
            backend="operator",
            determinant_method=self.determinant_method,
            evidence_is_exact=False,
            failure_reason=reason,
            solver_configuration={
                "preconditioner": "principal_block_cholesky",
                "preconditioner_application": "inverse_lower_factor_matvec",
                "block_layout": self.layout,
                "block_size": self.block_size,
                "block_sizes": tuple(map(len, self.blocks)),
                "block_indices": self.blocks,
                "block_count": len(self.blocks),
                "block_partition_sha256": self.partition_digest,
                "pcg_max_iter": max_iter,
                "pcg_rtol": rtol,
                "pcg_atol": atol,
                "pcg_recursive_residual_norm": float(solved.pcg_info.residual_norm),
                "solve_residual": "recomputed_full_coupled_system",
            },
        )


def fit_operator(
    problem,
    lambda_source,
    lambda_potential,
    *,
    block_size,
    block_indices,
    pcg_max_iter,
    pcg_rtol,
    pcg_atol,
):
    """Validate operator controls and reuse one prepared numerical topology."""
    blocks, layout, block_size = _partition(problem, block_size, block_indices)
    pcg_max_iter, pcg_rtol, pcg_atol = _pcg_controls(
        problem, pcg_max_iter, pcg_rtol, pcg_atol
    )
    key = (blocks, layout, block_size)
    if key not in problem._operator_backends:
        problem._operator_backends[key] = _SingleStepOperator(
            problem,
            blocks,
            layout,
            block_size,
        )
    return problem._operator_backends[key].fit(
        lambda_source,
        lambda_potential,
        max_iter=int(pcg_max_iter),
        rtol=float(pcg_rtol),
        atol=float(pcg_atol),
    )
