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
            matrix @ coefficients, 0.0,
        ),
        spec=None,
        size=2,
    )

    with pytest.raises(ValueError, match=r"kernel returned shape \(3,\)"):
        operator.matvec(jnp.ones(2))
