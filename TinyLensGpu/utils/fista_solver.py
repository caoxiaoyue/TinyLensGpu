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
    """
    n_iter: Array
    convergence_metric: Array
    converged: Array
    failed: Array
    step_size: Array


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
    )
    return final_state.x, info


__all__ = ["fista_nnls_solve", "FISTAInfo", "FISTAState"]
