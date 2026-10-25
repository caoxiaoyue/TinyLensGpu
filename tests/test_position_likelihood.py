"""Derivatives of the shared image-position constraint."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.ObservationModel.LensImage._position_likelihood import (
    compute_position_penalty_jax,
)


@pytest.mark.unit
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(
    "theta,threshold", [(0.0, 0.3), (0.1, 0.3), (0.6, 0.3), (0.6, 0.0)]
)
def test_position_penalty_gradient_handles_zero_pair_distances(x64, theta, threshold):
    """Self-pairs and coincident positions must not poison an inactive penalty."""
    with jax.experimental.enable_x64(x64):
        positions = jnp.array([0.0, 0.0, 1.0])

        class RayModel:
            def __init__(self, scale):
                self.scale = scale

            def deflection(self, x, y):
                return self.scale * x, self.scale * y

        def penalty(scale):
            return compute_position_penalty_jax(
                RayModel(scale),
                positions,
                jnp.zeros_like(positions),
                jnp.asarray(threshold),
                jnp.asarray(-10.0),
            )

        value, gradient = jax.jit(jax.value_and_grad(penalty))(jnp.asarray(theta))
        if threshold > 0 and theta > threshold:
            expected_value = -10.0 * (1.0 - np.exp(-(theta - threshold) / threshold))
            expected_gradient = (
                -10.0 / threshold * np.exp(-(theta - threshold) / threshold)
            )
        else:
            expected_value = expected_gradient = 0.0
        np.testing.assert_allclose(value, expected_value, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.unit
def test_position_penalty_preserves_nonfinite_ray_failure():
    class InvalidRayModel:
        def deflection(self, x, y):
            return jnp.full_like(x, jnp.nan), y

    value = compute_position_penalty_jax(
        InvalidRayModel(),
        jnp.array([0.0, 1.0]),
        jnp.array([0.0, 0.0]),
        jnp.array(0.3),
        jnp.array(-10.0),
    )
    assert jnp.isnan(value)
