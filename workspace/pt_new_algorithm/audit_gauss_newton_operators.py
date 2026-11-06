"""Small deterministic cross-point audits of the optimized AD operators.

Run with JAX_PLATFORMS=cpu in tinylens_gpu. These are mathematical checks,
not a performance benchmark, and do not modify any core-library tests.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from problems import make_problem
from run_gauss_newton import GaussNewtonPotentialCorrection


def main():
    with jax.experimental.enable_x64(), jax.default_matmul_precision("highest"):
        rng = np.random.default_rng(17)
        checks = 0
        for case in ("coupled", "nodal"):
            problem, strengths, _ = make_problem(case)
            for whitening in ("solve", "explicit"):
                fit = GaussNewtonPotentialCorrection(
                    problem, strengths, 1e-6, whitening=whitening
                )
                metric = jnp.asarray(fit.matrix(problem.initial_values))
                chol = jnp.linalg.cholesky(metric)
                precision = problem.dense_prior(*strengths)
                prior = jnp.linalg.cholesky(precision)
                factor = (
                    chol
                    if whitening == "solve"
                    else jsl.solve_triangular(
                        chol.T, jnp.eye(problem.n_joint), lower=False
                    )
                )
                for point_index in range(3):
                    z = jnp.asarray(
                        rng.normal(size=problem.n_joint) * (0.01 * point_index)
                    )
                    physical = fit.decode(z, factor)
                    residual = fit.residual(z, factor, prior)
                    physical_value, physical_gradient = fit.gradient(physical)
                    value, gradient = fit.value_grad(z, factor)
                    expected = (
                        jsl.solve_triangular(chol, physical_gradient, lower=True)
                        if whitening == "solve"
                        else factor.T @ physical_gradient
                    )
                    np.testing.assert_allclose(
                        value, physical_value, rtol=1e-10, atol=1e-10
                    )
                    np.testing.assert_allclose(
                        gradient, expected, rtol=1e-9, atol=1e-10
                    )
                    np.testing.assert_allclose(
                        0.5 * jnp.vdot(residual, residual),
                        physical_value,
                        rtol=1e-10,
                        atol=1e-10,
                    )
                    linear = fit.linearization_factory(z, factor, prior)
                    vector = jnp.asarray(rng.normal(size=problem.n_joint))
                    weights = jnp.asarray(rng.normal(size=len(residual)))
                    cached_jv = fit.cached_push(linear, vector)
                    cached_jtu = fit.cached_pull(linear, weights)
                    np.testing.assert_allclose(
                        cached_jv,
                        fit.push(z, factor, prior, vector),
                        rtol=1e-9,
                        atol=1e-10,
                    )
                    np.testing.assert_allclose(
                        cached_jtu,
                        fit.pull(z, factor, prior, weights),
                        rtol=1e-9,
                        atol=1e-10,
                    )
                    np.testing.assert_allclose(
                        jnp.vdot(cached_jv, weights),
                        jnp.vdot(vector, cached_jtu),
                        rtol=1e-9,
                        atol=1e-10,
                    )
                    np.testing.assert_allclose(
                        fit.cached_pull(linear, residual),
                        gradient,
                        rtol=1e-8,
                        atol=1e-9,
                    )
                    compiled_diagonal = fit.compiled_diagonal(
                        physical, jnp.diag(precision)
                    )
                    panel_diagonal = fit.matrix(physical, diagonal=True)
                    np.testing.assert_allclose(
                        compiled_diagonal, panel_diagonal, rtol=1e-9, atol=1e-9
                    )
                    checks += 1
                print(
                    case,
                    whitening,
                    "3 changing points: objective, gradient, adjoint, cached/direct, exact diagonal PASS",
                    flush=True,
                )
        print(checks, "cross-point configurations passed", flush=True)


if __name__ == "__main__":
    main()
