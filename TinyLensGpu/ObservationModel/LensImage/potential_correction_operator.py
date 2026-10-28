"""Bounded principal blocks and the full matrix-free source–potential solve.

Only preconditioning and determinant evaluation discard inter-block entries.
The fitted vector always solves the complete coupled conditional model.
"""

from __future__ import annotations

import hashlib
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from TinyLensGpu.utils.cg_solver import (
    PCGInfo,
    PrincipalBlockPreconditioner,
    pcg_solve,
)
from TinyLensGpu.utils.curvature_operator import CurvatureOperator

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


def _positive_integer(value, name):
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or integer != value or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _partition(problem, block_size, block_indices):
    block_size = _positive_integer(block_size, "block_size")
    if block_indices is None:
        blocks = []
        for start, size in (
            (0, problem.n_source),
            (problem.n_source, problem.n_potential),
        ):
            # Even a small field is split: preparation never probes a full
            # source/potential/joint basis in one panel.
            width = min(block_size, max(1, (size + 1) // 2))
            blocks.extend(
                tuple(range(offset, min(offset + width, start + size)))
                for offset in range(start, start + size, width)
            )
        layout = "field_separated_contiguous_v1"
    else:
        blocks = []
        try:
            for block in block_indices:
                values = np.asarray(block)
                if (
                    values.ndim != 1
                    or values.size == 0
                    or not np.issubdtype(values.dtype, np.integer)
                ):
                    raise ValueError
                blocks.append(tuple(int(value) for value in values))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "block_indices must contain nonempty integer index blocks"
            ) from exc
        layout = "explicit_joint_partition_v1"
    flat = [index for block in blocks for index in block]
    if sorted(flat) != list(range(problem.n_joint)):
        raise ValueError(
            "block_indices must cover every joint coefficient exactly once"
        )
    if any(len(block) > block_size for block in blocks):
        raise ValueError("Each block must contain at most block_size coefficients")
    if any(
        len(block) == problem.n_joint
        or sum(index >= problem.n_source for index in block) == problem.n_potential
        for block in blocks
    ):
        raise ValueError(
            "Blocks must not contain the complete joint or potential field"
        )
    return tuple(blocks), layout, block_size


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
        width = max(map(len, blocks))
        indices = np.zeros((len(blocks), width), dtype=np.int32)
        valid = np.zeros_like(indices, dtype=bool)
        for row, block in enumerate(blocks):
            indices[row, : len(block)] = block
            valid[row, : len(block)] = True
        self.indices = jnp.asarray(indices)
        self.valid = jnp.asarray(valid)
        self.partition_digest = hashlib.sha256(repr(blocks).encode("ascii")).hexdigest()

        @jax.jit
        def prepare():
            def one_block(_, inputs):
                indices, valid = inputs
                # Only this bounded block is batched. scan releases the image
                # panel before advancing; no full identity is ever allocated.
                basis = jax.nn.one_hot(indices, problem.n_joint, dtype=problem.dtype)
                basis = basis * valid[:, None]
                panel = jax.vmap(problem.forward)(basis) / problem.noise_active[None, :]

                def unit_prior(values):
                    return jnp.concatenate(
                        (
                            problem.source_regularizer.matvec_free(
                                values[: problem.n_source],
                                *problem.renderer.source_bbox,
                            ),
                            problem.potential_regularizer.matvec(
                                values[problem.n_source :],
                                1,
                                0,
                            ),
                        )
                    )

                principal = jax.vmap(unit_prior)(basis)[:, indices]
                outer_valid = valid[:, None] & valid[None, :]
                principal = principal * outer_valid
                source = indices < problem.n_source
                source_block = principal * (source[:, None] & source[None, :])
                potential_block = principal * (~source[:, None] & ~source[None, :])
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
        identity = jnp.eye(width, dtype=self.problem.dtype)[None, :, :]
        amplitude = (self.indices >= self.problem.n_source) & self.valid
        prior = (
            lambda_source * source
            + lambda_potential * potential
            + self.problem.lambda_amplitude * amplitude[:, :, None] * identity
            + (~self.valid)[:, :, None] * identity
        )
        posterior = data + prior

        def factor(matrices):
            # Equilibration changes neither the represented matrix nor its
            # determinant. No physical prior or hidden jitter is added.
            scale = jnp.sqrt(jnp.diagonal(matrices, axis1=1, axis2=2))
            normalized = matrices / (scale[:, :, None] * scale[:, None, :])
            chols = jnp.linalg.cholesky(normalized) * scale[:, :, None]
            diagonal = jnp.diagonal(chols, axis1=1, axis2=2)
            return (
                chols,
                2 * jnp.sum(jnp.log(diagonal)),
                jnp.all(jnp.isfinite(chols)) & jnp.all(diagonal > 0),
            )

        _, prior_logdet, prior_ok = factor(prior)
        chols, posterior_logdet, posterior_ok = factor(posterior)
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
    pcg_max_iter = _positive_integer(pcg_max_iter, "pcg_max_iter")
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
