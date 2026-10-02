"""Projected FISTA solver for matrix-free non-negative quadratic problems.

Solves ``min_x 0.5 * x.T @ A @ x - b.T @ x`` subject to ``x >= 0`` using
matrix-vector products supplied through the same typed curvature-operator seam
used by the PCG solver.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from TinyLensGpu.utils.curvature_operator import CurvatureOperator
from TinyLensGpu.utils.cg_solver import MaskedPreconditioner, pcg_solve


class FISTAState(NamedTuple):
    """Carry state for fixed-iteration FISTA."""
    x: Array
    y: Array
    t: Array


class FISTAInfo(NamedTuple):
    """Convergence diagnostics returned by :func:`fista_nnls_solve`.

    ``n_iter`` is the fixed iteration budget executed by the current scan-based
    solver, not the first iteration at which the convergence metric passed
    tolerance.  This keeps the loop shape stable for JIT-compiled likelihoods.

    The implicit-gradient wrapper additionally fills ``kkt_violation`` (scaled
    stationarity/dual violation) and ``min_active_residual`` (unscaled, infinity
    for an empty active set). These two fields are NaN for the raw solver.
    """
    n_iter: Array
    convergence_metric: Array
    converged: Array
    failed: Array
    step_size: Array
    # Populated by the implicit-gradient wrapper; unavailable for the raw solve.
    kkt_violation: Array
    min_active_residual: Array


@partial(
    jax.jit,
    static_argnames=(
        "max_iter",
        "rtol",
        "atol",
        "power_iter",
        "step_safety",
        "step_size",
    ),
)
def fista_nnls_solve(
    operator: CurvatureOperator,
    b: Array,
    x0: Array | None = None,
    max_iter: int = 300,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    power_iter: int = 10,
    step_safety: float = 1.2,
    step_size: float | None = None,
) -> tuple[Array, FISTAInfo]:
    """Solve a matrix-free NNLS quadratic with projected FISTA.

    Parameters
    ----------
    operator : CurvatureOperator
        Matrix-free curvature shared with the PCG and PNPG solvers.
    b : Array, shape ``(N,)``
        Right-hand side vector.
    x0 : Array, optional
        Initial point. If omitted, starts from the zero vector.
    max_iter, rtol, atol : optional
        Fixed FISTA iteration count and projected-gradient convergence
        tolerance. The current implementation always runs this many iterations
        to keep JIT behavior predictable; ``info.n_iter`` reports this fixed
        budget rather than an early-stop iteration.
    power_iter : int, optional
        Fixed power-iteration count used to estimate ``lambda_max(A)`` when
        ``step_size`` is not provided.
    step_safety : float, optional
        Multiplicative safety factor applied to the power-iteration spectral
        estimate before taking its reciprocal.
    step_size : float, optional
        Positive explicit step size. When omitted, a matrix-free estimate is
        used.
    """
    b = jnp.asarray(b)
    n = b.shape[0]
    dtype = b.dtype
    eps = jnp.finfo(dtype).eps

    def _A_vec(s: Array) -> Array:
        return operator.matvec(s)

    def _estimate_lipschitz() -> Array:
        idx = jnp.arange(n, dtype=dtype)
        # Deterministic non-constant start avoids missing high-frequency modes
        # of finite-difference regularization operators.
        v0 = jnp.sin((idx + 1.0) * 1.61803398875) + 0.1 * ((idx % 2.0) * 2.0 - 1.0)
        v0 = v0 / jnp.maximum(jnp.linalg.norm(v0), eps)

        def body(v, _):
            Av = _A_vec(v)
            norm_Av = jnp.linalg.norm(Av)
            v_next = Av / jnp.maximum(norm_Av, eps)
            return v_next, None

        v, _ = lax.scan(body, v0, xs=None, length=power_iter)
        Av = _A_vec(v)
        rayleigh = jnp.dot(v, Av) / jnp.maximum(jnp.dot(v, v), eps)
        return rayleigh

    if step_size is None:
        lipschitz = _estimate_lipschitz()
        valid_lipschitz = jnp.isfinite(lipschitz) & (lipschitz > 0.0)
        safe_lipschitz = jnp.where(valid_lipschitz, lipschitz, 1.0)
        step = 1.0 / (jnp.asarray(step_safety, dtype=dtype) * safe_lipschitz)
    else:
        step = jnp.asarray(step_size, dtype=dtype)
        valid_lipschitz = jnp.asarray(True, dtype=bool)

    valid_step = jnp.isfinite(step) & (step > 0.0)
    safe_step = jnp.where(valid_step, step, jnp.asarray(1.0, dtype=dtype))

    x_init = (
        jnp.zeros(n, dtype=dtype)
        if x0 is None
        else jnp.maximum(jnp.asarray(x0, dtype=dtype), 0.0)
    )
    init_state = FISTAState(
        x=x_init,
        y=x_init,
        t=jnp.asarray(1.0, dtype=dtype),
    )

    def body(state: FISTAState, _) -> tuple[FISTAState, None]:
        grad = _A_vec(state.y) - b
        x_next = jnp.maximum(state.y - safe_step * grad, 0.0)
        t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * state.t * state.t))
        momentum = (state.t - 1.0) / t_next
        y_next = x_next + momentum * (x_next - state.x)
        return FISTAState(x=x_next, y=y_next, t=t_next), None

    # Deliberately use a fixed-length scan instead of early stopping. Nested
    # sampling calls repeatedly trace/evaluate this path, so predictable loop
    # shape is preferred over variable iteration count in this first version.
    final_state, _ = lax.scan(body, init_state, xs=None, length=max_iter)

    grad_final = _A_vec(final_state.x) - b
    projected_step = final_state.x - jnp.maximum(final_state.x - grad_final, 0.0)
    metric = jnp.linalg.norm(projected_step)
    tol = rtol * jnp.linalg.norm(b) + atol
    failed = (~valid_lipschitz) | (~valid_step) | (~jnp.isfinite(metric))
    converged = (metric < tol) & (~failed)

    info = FISTAInfo(
        n_iter=jnp.asarray(max_iter, dtype=jnp.int32),
        convergence_metric=metric,
        converged=converged,
        failed=failed,
        step_size=step,
        kkt_violation=jnp.asarray(jnp.nan, dtype=dtype),
        min_active_residual=jnp.asarray(jnp.nan, dtype=dtype),
    )
    return final_state.x, info


def _masked_curvature_matvec(vector, data, spec):
    """Stable kernel: the operator and mask are dynamic PyTree children."""
    operator, free = data
    return free * operator.matvec(free * vector) + (1.0 - free) * vector


def _checked_cotangent(leaf, valid):
    # Operator data include integer indices. Their cotangents have dtype float0,
    # which must not be subjected to floating-point arithmetic or NaN filling.
    if leaf.dtype == jax.dtypes.float0:
        return leaf
    return jnp.where(valid, leaf, jnp.full_like(leaf, jnp.nan))


def fista_nnls_solve_with_implicit_grad(
    operator: CurvatureOperator,
    b: Array,
    x0: Array | None,
    preconditioner,
    *,
    active_tol: float = 1e-8,
    kkt_rel_tol: float = 1e-4,
    kkt_abs_tol: float = 1e-8,
    comp_rel_tol: float = 1e-4,
    comp_abs_tol: float = 1e-8,
    backward_pcg_max_iter: int = 800,
    backward_pcg_rtol: float = 1e-6,
    backward_pcg_atol: float = 1e-12,
    **kwargs,
) -> tuple[Array, FISTAInfo]:
    """FISTA with a first-order, fixed-active-set implicit reverse derivative.

    Forward iterates match :func:`fista_nnls_solve`, even on failure. Gradients
    of the solution with respect to operator data and b are NaN if the forward
    solve, KKT checks, strict complementarity or adjoint solve fail. Diagnostic
    fields are auxiliary outputs, not differentiable objectives. No derivative
    is taken through the initial guess or numerical preconditioner.

    Each KKT residual is checked against ``kkt_abs_tol + kkt_rel_tol *
    (1 + abs(b_i) + abs((Ax)_i))``. ``kkt_violation`` reports the largest
    componentwise normalized residual. On active variables,
    the dual residual must exceed ``comp_abs_tol + comp_rel_tol *
    max(abs(b_i), abs((Ax)_i))`` plus the largest free-variable stationarity
    error. This uses the local amplitude scale instead of allowing a bright,
    unrelated lens-light coefficient to set every variable's threshold.
    Near active-set changes the classical derivative need not exist.
    """
    # A zero numerical cotangent alone does not prevent transposition of an
    # external warm-start while_loop. Detach before entering the custom VJP.
    if x0 is not None:
        x0 = lax.stop_gradient(x0)

    def forward(op, rhs, initial, precond):
        x, info = fista_nnls_solve(op, rhs, x0=initial, **kwargs)
        Ax = op.matvec(x)
        residual = Ax - rhs
        free = x > active_tol
        stationarity_error = jnp.max(jnp.where(free, jnp.abs(residual), 0.0))
        kkt_error = jnp.where(free, jnp.abs(residual), jnp.maximum(-residual, 0.0))
        # A global norm lets a bright, unrelated coefficient hide an incorrect
        # free set in a faint component, producing a finite but wrong VJP.
        scale = 1.0 + jnp.abs(rhs) + jnp.abs(Ax)
        violation = jnp.max(kkt_error / scale)
        min_active = jnp.min(jnp.where(free, jnp.inf, residual))
        margin = (
            comp_abs_tol
            + comp_rel_tol * jnp.maximum(jnp.abs(rhs), jnp.abs(Ax))
            + stationarity_error
        )
        strict_ok = jnp.all(free | (residual > margin))
        kkt_ok = jnp.all(kkt_error <= kkt_abs_tol + kkt_rel_tol * scale)
        valid = (
            info.converged & (~info.failed) & kkt_ok & strict_ok
            & jnp.all(jnp.isfinite(x))
        )
        info = info._replace(kkt_violation=violation, min_active_residual=min_active)
        return (x, info), (op, x, precond, free.astype(x.dtype), valid)

    @jax.custom_vjp
    def solve(op, rhs, initial, precond):
        return forward(op, rhs, initial, precond)[0]

    def backward(saved, cotangents):
        op, x, precond, free, forward_valid = saved
        g_x, _ = cotangents
        masked_op = CurvatureOperator(
            data=(op, free), kernel=_masked_curvature_matvec,
            spec=None, size=op.size,
        )
        adjoint, info = pcg_solve(
            masked_op, free * g_x, MaskedPreconditioner(precond, free),
            max_iter=backward_pcg_max_iter,
            rtol=backward_pcg_rtol, atol=backward_pcg_atol,
        )
        valid = (
            forward_valid & info.converged & (~info.failed)
            & jnp.all(jnp.isfinite(adjoint))
        )
        adjoint = free * adjoint
        _, pullback = jax.vjp(lambda operator: operator.matvec(x), op)
        g_op = pullback(-adjoint)[0]
        g_op = jax.tree_util.tree_map(lambda v: _checked_cotangent(v, valid), g_op)
        g_b = _checked_cotangent(adjoint, valid)
        return g_op, g_b, None, None

    solve.defvjp(forward, backward)
    return solve(operator, b, x0, preconditioner)


__all__ = [
    "fista_nnls_solve", "fista_nnls_solve_with_implicit_grad", "FISTAInfo", "FISTAState",
]
