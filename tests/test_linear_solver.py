import jax.numpy as jnp
import pytest

from TinyLensGpu.utils.linear_solver import fnnls_jax


@pytest.mark.unit
def test_fnnls_jax_zero_columns_stay_zero() -> None:
    """Columns with zero norm should not receive a spurious positive amplitude."""
    design = jnp.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

    coeffs, residual = fnnls_jax(design, data)

    assert coeffs.shape == (2,)
    assert jnp.isclose(coeffs[0], 1.0, atol=1e-5)
    assert jnp.isclose(coeffs[1], 0.0, atol=1e-8)
    assert residual < 1e-4


@pytest.mark.unit
def test_fnnls_jax_preserves_small_signal_scale() -> None:
    """Scaling the full least-squares problem should not change the NNLS solution."""
    base_design = jnp.array([[1.0]], dtype=jnp.float32)
    base_data = jnp.array([1.0], dtype=jnp.float32)
    tiny_scale = jnp.array(1e-7, dtype=jnp.float32)

    base_coeffs, base_residual = fnnls_jax(base_design, base_data)
    tiny_coeffs, tiny_residual = fnnls_jax(base_design * tiny_scale, base_data * tiny_scale)

    assert jnp.isclose(base_coeffs[0], 1.0, atol=1e-5)
    assert jnp.isclose(tiny_coeffs[0], 1.0, atol=1e-4)
    assert jnp.isclose(tiny_coeffs[0], base_coeffs[0], atol=1e-4)
    assert base_residual < 1e-5
    assert tiny_residual < 1e-10
