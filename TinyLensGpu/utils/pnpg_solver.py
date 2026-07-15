"""Diagonally preconditioned projected Nesterov solver for matrix-free NNLS."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from TinyLensGpu.utils.cg_solver import preconditioner_diagonal
from TinyLensGpu.utils.curvature_operator import CurvatureOperator


class PNPGInfo(NamedTuple):
    """Diagnostics returned by :func:`pnpg_nnls_solve`."""

    n_iter: Array
    convergence_metric: Array
    converged: Array
    failed: Array
    step_size: Array


class _PNPGState(NamedTuple):
    x: Array
    y: Array
    A_x: Array
    A_y: Array
    momentum_scale: Array
    step: Array
    failed: Array


class _LineSearchState(NamedTuple):
    x: Array
    A_x: Array
    step: Array
    accepted: Array
    n_trials: Array


@partial(
    jax.jit,
    static_argnames=(
        "max_iter",
        "rtol",
        "power_iter",
        "step_safety",
        "max_backtracking",
    ),
)
def pnpg_nnls_solve(
    operator: CurvatureOperator,
    b: Array,
    preconditioner,
    x0: Array | None = None,
    max_iter: int = 1000,
    rtol: float = 2e-2,
    power_iter: int = 20,
    step_safety: float = 1.1,
    max_backtracking: int = 30,
) -> tuple[Array, PNPGInfo]:
    """Solve a non-negative quadratic in an equilibrated variable metric.

    With ``D = diag(P)`` from the operator preconditioner, the change of
    variables ``x = D**-1/2 y`` preserves non-negativity while removing the
    source/MGE amplitude-scale disparity that makes scalar-step FISTA stall.
    A projected PCG estimate may be supplied as ``x0``; it is rescaled along
    its feasible ray to guarantee an objective no worse than the zero vector.
    """
    b = jnp.asarray(b)
    dtype = b.dtype
    eps = jnp.finfo(dtype).eps

    def A_vec(value: Array) -> Array:
        return operator.matvec(value)

    diagonal = preconditioner_diagonal(preconditioner, b.shape[0])
    diagonal_fallback = jnp.asarray(10.0 * eps, dtype) * jnp.maximum(
        jnp.max(jnp.abs(diagonal)), 1.0
    )
    safe_diagonal = jnp.where(
        jnp.isfinite(diagonal) & (diagonal > 0.0),
        diagonal,
        diagonal_fallback,
    )
    inverse_sqrt_diagonal = lax.rsqrt(safe_diagonal)
    scaled_b = inverse_sqrt_diagonal * b

    def scaled_A(value: Array) -> Array:
        return inverse_sqrt_diagonal * A_vec(inverse_sqrt_diagonal * value)

    index = jnp.arange(b.shape[0], dtype=dtype)
    power_vector = jnp.sin((index + 1.0) * 1.61803398875)
    power_vector = power_vector / jnp.maximum(jnp.linalg.norm(power_vector), eps)

    def power_body(value: Array, _) -> tuple[Array, None]:
        A_value = scaled_A(value)
        return A_value / jnp.maximum(jnp.linalg.norm(A_value), eps), None

    power_vector, _ = lax.scan(power_body, power_vector, xs=None, length=power_iter)
    rayleigh = jnp.dot(power_vector, scaled_A(power_vector))
    valid_lipschitz = jnp.isfinite(rayleigh) & (rayleigh > 0.0)
    safe_lipschitz = jnp.where(valid_lipschitz, rayleigh, 1.0)
    step = 1.0 / (jnp.asarray(step_safety, dtype) * safe_lipschitz)

    projected_warm = (
        jnp.zeros_like(b)
        if x0 is None
        else jnp.maximum(jnp.asarray(x0, dtype=dtype), 0.0)
    )
    warm_A = A_vec(projected_warm)
    warm_curvature = jnp.dot(projected_warm, warm_A)
    warm_drive = jnp.dot(b, projected_warm)
    warm_scale = jnp.clip(warm_drive / jnp.maximum(warm_curvature, eps), 0.0, 1.0)
    warm_scale = jnp.where(
        jnp.isfinite(warm_scale) & (warm_curvature > 0.0), warm_scale, 0.0
    )
    scaled_initial = warm_scale * projected_warm / inverse_sqrt_diagonal
    initial_A = scaled_A(scaled_initial)
    initial = _PNPGState(
        x=scaled_initial,
        y=scaled_initial,
        A_x=initial_A,
        A_y=initial_A,
        momentum_scale=jnp.asarray(1.0, dtype),
        step=step,
        failed=jnp.asarray(False),
    )

    def body(state: _PNPGState, _) -> tuple[_PNPGState, None]:
        gradient = state.A_y - scaled_b

        def line_search_condition(line_state: _LineSearchState) -> Array:
            return (~line_state.accepted) & (line_state.n_trials < max_backtracking)

        def line_search_body(line_state: _LineSearchState) -> _LineSearchState:
            trial_x = jnp.maximum(state.y - line_state.step * gradient, 0.0)
            trial_A_x = scaled_A(trial_x)
            displacement = trial_x - state.y
            directional_curvature = jnp.dot(displacement, trial_A_x - state.A_y)
            majorizer_curvature = jnp.dot(displacement, displacement) / line_state.step
            roundoff_tolerance = jnp.asarray(100.0 * eps, dtype) * jnp.maximum(
                jnp.abs(majorizer_curvature), 1.0
            )
            finite = (
                jnp.all(jnp.isfinite(trial_x))
                & jnp.all(jnp.isfinite(trial_A_x))
                & jnp.isfinite(directional_curvature)
            )
            accepted = finite & (
                directional_curvature <= majorizer_curvature + roundoff_tolerance
            )
            next_step = jnp.where(accepted, line_state.step, 0.5 * line_state.step)
            return _LineSearchState(
                trial_x,
                trial_A_x,
                next_step,
                accepted,
                line_state.n_trials + 1,
            )

        line_initial = _LineSearchState(
            x=state.x,
            A_x=state.A_x,
            step=state.step,
            accepted=jnp.asarray(False),
            n_trials=jnp.asarray(0, dtype=jnp.int32),
        )
        line_final = lax.while_loop(
            line_search_condition, line_search_body, line_initial
        )
        x_next = line_final.x
        A_x_next = line_final.A_x
        momentum_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * state.momentum_scale**2))
        momentum = (state.momentum_scale - 1.0) / momentum_next
        accelerated_y = x_next + momentum * (x_next - state.x)
        accelerated_A_y = A_x_next + momentum * (A_x_next - state.A_x)

        # Gradient restart suppresses oscillation without another operator call.
        restart = jnp.dot(state.y - x_next, x_next - state.x) > 0.0
        y_next = jnp.where(restart, x_next, accelerated_y)
        A_y_next = jnp.where(restart, A_x_next, accelerated_A_y)
        momentum_next = jnp.where(restart, 1.0, momentum_next)
        next_state = _PNPGState(
            x=x_next,
            y=y_next,
            A_x=A_x_next,
            A_y=A_y_next,
            momentum_scale=momentum_next,
            step=line_final.step,
            failed=state.failed | (~line_final.accepted),
        )
        return next_state, None

    final, _ = lax.scan(body, initial, xs=None, length=max_iter)
    solution = jnp.maximum(inverse_sqrt_diagonal * final.x, 0.0)
    A_solution = final.A_x / inverse_sqrt_diagonal
    gradient = A_solution - b
    active_tolerance = jnp.asarray(10.0 * eps, dtype) * jnp.maximum(
        jnp.max(jnp.abs(solution)), 1.0
    )
    projected_gradient = jnp.where(
        solution > active_tolerance,
        gradient,
        jnp.minimum(gradient, 0.0),
    )
    component_scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(A_solution), jnp.abs(b)))
    metric = jnp.max(jnp.abs(projected_gradient) / component_scale)
    finite = (
        jnp.all(jnp.isfinite(solution))
        & jnp.all(jnp.isfinite(A_solution))
        & jnp.isfinite(metric)
    )
    failed = (~finite) | (~valid_lipschitz) | final.failed
    converged = (~failed) & (metric <= jnp.asarray(rtol, dtype))
    info = PNPGInfo(
        n_iter=jnp.asarray(max_iter, dtype=jnp.int32),
        convergence_metric=metric,
        converged=converged,
        failed=failed,
        step_size=final.step,
    )
    return solution, info


__all__ = ["pnpg_nnls_solve", "PNPGInfo"]
