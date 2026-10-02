"""Implicit solver derivatives, independent of the evidence envelope theorem."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.utils.cg_solver import pcg_solve, pcg_solve_with_implicit_grad
from TinyLensGpu.utils.curvature_operator import CurvatureOperator
from TinyLensGpu.utils.fista_solver import (
    fista_nnls_solve,
    fista_nnls_solve_with_implicit_grad,
)


def _dense_matvec(x, matrix, spec):
    return matrix @ x


def _operator(matrix):
    return CurvatureOperator(matrix, _dense_matvec, None, matrix.shape[0])


@pytest.mark.unit
def test_pcg_solution_vjp_matches_dense_with_dynamic_preconditioner():
    """Check A and b independently, including jit/vmap and nonzero cotangents."""
    matrix = jnp.array([[3.0, 0.4], [0.4, 1.5]])
    rhs = jnp.array([0.8, -0.3])
    cotangent = jnp.array([0.7, -1.2])

    def loss(theta, solve):
        a = matrix + theta[0] * jnp.eye(2)
        b = rhs + theta[1] * jnp.array([1.0, 0.2])
        if solve:
            x, _ = pcg_solve_with_implicit_grad(
                _operator(a), b, jnp.linalg.cholesky(a), rtol=1e-6
            )
        else:
            x = jnp.linalg.solve(a, b)
        return cotangent @ x

    points = jnp.array([[0.1, 0.2], [0.3, -0.1]])
    actual = jax.jit(jax.vmap(jax.grad(lambda t: loss(t, True))))(points)
    expected = jax.vmap(jax.grad(lambda t: loss(t, False)))(points)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["forward", "backward", "degenerate", "kkt"])
def test_fista_invalid_gradients_preserve_raw_forward(failure):
    matrix = jnp.array([[2.0, 0.2], [0.2, 1.0]])
    # At theta=0 this is either strictly interior or a degenerate active point.
    rhs = jnp.zeros(2) if failure == "degenerate" else jnp.array([1.0, 0.5])
    budget = 0 if failure in ("forward", "kkt") else 300
    initial = jnp.array([0.47, 0.4]) if failure == "kkt" else None
    rtol = 1.0 if failure == "kkt" else 1e-5

    def solve(theta):
        return fista_nnls_solve_with_implicit_grad(
            _operator(matrix), rhs + theta, initial, jnp.eye(2), rtol=rtol,
            max_iter=budget, backward_pcg_max_iter=0 if failure == "backward" else 100,
        )

    x, info = solve(0.0)
    raw_x, raw_info = fista_nnls_solve(
        _operator(matrix), rhs, x0=initial, max_iter=budget, rtol=rtol,
    )
    np.testing.assert_array_equal(x, raw_x)
    assert info.converged == raw_info.converged
    assert info.failed == raw_info.failed
    if failure in ("degenerate", "kkt"):
        assert info.converged
    assert jnp.all(jnp.isnan(jax.jit(jax.jacrev(lambda t: solve(t)[0]))(0.0)))


@pytest.mark.unit
def test_fista_complementarity_uses_local_scales():
    """An unrelated bright coefficient must not invalidate a stable active pixel."""
    matrix = jnp.diag(jnp.array([1.0, 2.0]))

    def loss(theta):
        rhs = jnp.array([-0.1, 2e5 + theta])
        x, _ = fista_nnls_solve_with_implicit_grad(
            _operator(matrix), rhs, None, jnp.linalg.cholesky(matrix), max_iter=300,
        )
        return jnp.sum(x)

    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.0), 0.5, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("exact_initial", [False, True])
def test_fista_kkt_rejects_wrong_free_set_with_bright_component(exact_initial):
    """A bright independent component cannot hide a small pixel's KKT error."""
    matrix = jnp.array([[4.0, 1.5, 0.0], [1.5, 1.0, 0.0], [0.0, 0.0, 3e5]])
    op = _operator(matrix)
    precond = jnp.linalg.cholesky(matrix)
    solution = jnp.array([0.0, 0.0, 1.0 / 3.0])

    def initial(rhs):
        if exact_initial:
            return solution
        warm, _ = pcg_solve(op, rhs, precond)
        return jnp.maximum(warm, 0.0)

    def solve(theta):
        rhs = jnp.array([theta, -5.0, 1e5])
        return fista_nnls_solve_with_implicit_grad(
            op, rhs, initial(rhs), precond, max_iter=1000,
        )

    rhs = jnp.array([-1.0, -5.0, 1e5])
    x, info = solve(-1.0)
    raw_x, raw_info = fista_nnls_solve(op, rhs, x0=initial(rhs), max_iter=1000)
    np.testing.assert_array_equal(x, raw_x)
    for actual, expected in zip(info[:5], raw_info[:5]):
        np.testing.assert_array_equal(actual, expected)
    assert info.converged
    assert not info.failed
    gradient = jax.jit(jax.grad(lambda theta: solve(theta)[0][0]))(-1.0)
    # The exact NNQP solution has both first pixels strictly active for theta < 0.
    if exact_initial:
        np.testing.assert_allclose(x, solution, atol=1e-7)
        np.testing.assert_allclose(gradient, 0.0, atol=1e-7)
    else:
        assert x[0] > 0.1  # Raw global convergence accepts an incorrect free set.
        assert jnp.isnan(gradient)
        assert info.kkt_violation > 1e-4


@pytest.mark.unit
def test_fista_detaches_external_pcg_warm_start():
    matrix = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    rhs = jnp.array([0.7, -0.3])

    def loss(theta):
        op = _operator(matrix + theta * jnp.eye(2))
        warm, _ = pcg_solve(op, rhs, jnp.eye(2))
        x, _ = fista_nnls_solve_with_implicit_grad(
            op, rhs, warm, jnp.eye(2), max_iter=300,
        )
        return x.sum()

    # Only the first coefficient is free: x_0 = 0.7 / (2 + theta).
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.0), -0.175, atol=2e-6)


@pytest.mark.slow
def test_fista_active_set_stress_matches_known_solution_and_dense_derivative():
    # Double precision separates convergence of a moderately ill-conditioned
    # problem from the single-precision residual floor. The active set is
    # constructed from an exact feasible solution and strictly positive dual.
    with jax.experimental.enable_x64():
        matrix = jnp.diag(jnp.array([1.0, 2.0, 4.0, 8.0, 16.0]) ** 2) + 0.015
        solution = jnp.array([0.6, 0.0, 0.2, 0.1, 0.05])
        rhs = matrix @ solution - jnp.array([0.0, 0.5, 0.0, 0.0, 0.0])
        free = jnp.array([0, 2, 3, 4])
        c = jnp.array([1.0, -1.0, 0.5, -0.5, 2.0])

        def loss(theta, iterative):
            a = matrix + theta * jnp.eye(5)
            if iterative:
                x, _ = fista_nnls_solve_with_implicit_grad(
                    _operator(a), rhs, None, jnp.linalg.cholesky(a),
                    max_iter=8000, rtol=1e-8, backward_pcg_rtol=1e-10,
                )
            else:
                x = jnp.zeros(5).at[free].set(
                    jnp.linalg.solve(a[free][:, free], rhs[free])
                )
            return c @ x

        x, info = fista_nnls_solve_with_implicit_grad(
            _operator(matrix), rhs, None, jnp.linalg.cholesky(matrix),
            max_iter=8000, rtol=1e-8,
        )
        assert info.converged
        assert x[1] == 0.0
        np.testing.assert_allclose(x, solution, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            jax.grad(lambda t: loss(t, True))(0.0),
            jax.grad(lambda t: loss(t, False))(0.0), rtol=1e-6, atol=1e-8,
        )


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["forward", "backward"])
@pytest.mark.parametrize("variable", ["matrix", "rhs"])
def test_pcg_failed_solve_preserves_forward_but_rejects_gradient(failure, variable):
    """NaN primals alone cannot invalidate the independent RHS derivative."""
    def solve(theta):
        a = jnp.array([[2.0]]) + (theta if variable == "matrix" else 0.0)
        b = jnp.array([1.0]) + (theta if variable == "rhs" else 0.0)
        return pcg_solve_with_implicit_grad(
            _operator(a), b, jnp.eye(1),
            max_iter=0 if failure == "forward" else 10,
            backward_max_iter=0 if failure == "backward" else 10,
        )

    x, info = solve(0.1)
    assert bool(info.converged) == (failure != "forward")
    assert jnp.all(jnp.isfinite(x))
    gradient = jax.jit(jax.grad(lambda t: solve(t)[0].sum()))(0.1)
    assert jnp.isnan(gradient)


@pytest.mark.unit
@pytest.mark.parametrize("constrained", [False, True])
def test_fista_solution_vjp_matches_dense_free_set(constrained):
    matrix = jnp.array([[4.0, 0.4, 0.2], [0.4, 2.5, 0.3], [0.2, 0.3, 1.5]])
    solution = jnp.array([0.5, 0.0 if constrained else 0.4, 0.8])
    dual = jnp.array([0.0, 0.3 if constrained else 0.0, 0.0])
    rhs = matrix @ solution - dual
    free = jnp.array([0, 2] if constrained else [0, 1, 2])
    cotangent = jnp.array([0.7, -0.3, 1.1])

    def loss(theta, iterative):
        a = matrix + theta[0] * jnp.eye(3)
        b = rhs + theta[1] * jnp.array([0.2, -0.3, 0.4])
        if iterative:
            x, _ = fista_nnls_solve_with_implicit_grad(
                _operator(a), b, None, jnp.linalg.cholesky(a),
                max_iter=300, rtol=1e-6,
            )
        else:
            x = jnp.zeros(3).at[free].set(jnp.linalg.solve(a[free][:, free], b[free]))
        return cotangent @ x

    x, info = fista_nnls_solve_with_implicit_grad(
        _operator(matrix), rhs, None, jnp.linalg.cholesky(matrix),
        max_iter=300, rtol=1e-6,
    )
    assert info.converged
    np.testing.assert_allclose(x, solution, rtol=2e-5, atol=2e-6)
    points = jnp.array([[0.0, 0.0], [0.01, 0.01]])
    actual = jax.jit(jax.vmap(jax.grad(lambda t: loss(t, True))))(points)
    expected = jax.vmap(jax.grad(lambda t: loss(t, False)))(points)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
