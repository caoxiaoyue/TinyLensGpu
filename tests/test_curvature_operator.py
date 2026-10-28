"""Behavioral tests for the matrix-free curvature operator seam."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.utils.curvature_operator import CurvatureOperator


@pytest.mark.unit
def test_curvature_operator_reuses_jit_for_new_dynamic_data():
    """Changing numerical data must not retrace a stable operator topology."""
    traces = []

    def dense_kernel(coefficients, matrix, scale):
        traces.append(None)
        return scale * (matrix @ coefficients)

    apply = jax.jit(lambda operator, value: operator.matvec(value))
    value = jnp.asarray([1.0, -2.0], dtype=jnp.float32)
    first = CurvatureOperator(
        data=jnp.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float32),
        kernel=dense_kernel,
        spec=1.0,
        size=2,
    )
    second = CurvatureOperator(
        data=jnp.asarray([[4.0, 1.0], [1.0, 5.0]], dtype=jnp.float32),
        kernel=dense_kernel,
        spec=1.0,
        size=2,
    )

    first_result = apply(first, value)
    second_result = apply(second, value)

    np.testing.assert_allclose(first_result, jnp.asarray([2.0, -6.0]))
    np.testing.assert_allclose(second_result, jnp.asarray([2.0, -9.0]))
    assert traces == [None]


@pytest.mark.unit
def test_curvature_operator_rejects_wrong_coefficient_shape():
    """The operator should fail before a kernel sees incompatible coefficients."""
    operator = CurvatureOperator(
        data=jnp.eye(2),
        kernel=lambda coefficients, matrix, _spec: matrix @ coefficients,
        spec=None,
        size=2,
    )

    with pytest.raises(ValueError, match=r"expected coefficients shape \(2,\)"):
        operator.matvec(jnp.ones(3))


@pytest.mark.unit
@pytest.mark.parametrize("size", [0, -1])
def test_curvature_operator_rejects_non_positive_size(size):
    """An operator dimension must describe a non-empty vector space."""
    with pytest.raises(ValueError, match="size must be positive"):
        CurvatureOperator(
            data=jnp.eye(1),
            kernel=lambda coefficients, matrix, _spec: matrix @ coefficients,
            spec=None,
            size=size,
        )


@pytest.mark.unit
def test_curvature_operator_rejects_wrong_kernel_output_shape():
    """A malformed adapter must not leak an incompatible result to a solver."""
    operator = CurvatureOperator(
        data=jnp.eye(2),
        kernel=lambda coefficients, matrix, _spec: jnp.append(
            matrix @ coefficients,
            0.0,
        ),
        spec=None,
        size=2,
    )

    with pytest.raises(ValueError, match=r"kernel returned shape \(3,\)"):
        operator.matvec(jnp.ones(2))


@pytest.mark.unit
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_principal_inverse_factor_application_preserves_padded_spd_solve(dtype):
    from TinyLensGpu.utils.cg_solver import (
        PrincipalBlockPreconditioner,
        apply_preconditioner,
        preconditioner_diagonal,
    )

    with jax.experimental.enable_x64(dtype == "float64"):
        factors = np.array(
            [
                [[0.03, 0, 0], [0.2, 2, 0], [0.1, 0.3, 3]],
                [[0.5, 0, 0], [-0.3, 2, 0], [0, 0, 1]],
            ],
            dtype=dtype,
        )
        indices = np.array([[2, 0, 4], [3, 1, 0]])
        valid = np.array([[True, True, True], [True, True, False]])
        rhs = np.array([1.2, -0.7, 0.03, 0.2, -0.9], dtype=dtype)
        inverse = np.linalg.inv(factors.astype(np.float64)).astype(dtype)
        legacy = PrincipalBlockPreconditioner(
            jnp.asarray(factors), jnp.asarray(indices), jnp.asarray(valid)
        )
        cached = PrincipalBlockPreconditioner(
            jnp.asarray(factors),
            jnp.asarray(indices),
            jnp.asarray(valid),
            inverse_chols=jnp.asarray(inverse),
        )
        expected = np.zeros(5)
        expected_diagonal = np.zeros(5)
        for index in range(2):
            selected = indices[index, valid[index]]
            precision = (
                factors[index].astype(np.float64) @ factors[index].astype(np.float64).T
            )
            expected[selected] = np.linalg.solve(
                precision[: len(selected), : len(selected)], rhs[selected]
            )
            expected_diagonal[selected] = np.diag(precision)[: len(selected)]
        previous = jax.config.jax_default_matmul_precision
        for preconditioner in (legacy, cached):
            actual = jax.jit(apply_preconditioner)(preconditioner, jnp.asarray(rhs))
            np.testing.assert_allclose(
                actual, expected, rtol=2e-5 if dtype == "float32" else 2e-12
            )
            np.testing.assert_allclose(
                preconditioner_diagonal(preconditioner, 5),
                expected_diagonal,
                rtol=2e-6 if dtype == "float32" else 2e-13,
            )
            assert actual.dtype == np.dtype(dtype)
        assert jax.config.jax_default_matmul_precision == previous
