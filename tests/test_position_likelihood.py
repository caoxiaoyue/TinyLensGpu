"""Tests for shared image-position likelihood helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.ObservationModel.LensImage._position_likelihood import (
    compute_position_penalty_jax,
    resolve_position_likelihood_attrs,
)


class _IdentityDeflection:
    def deflection(self, x, y):
        return x, y


@pytest.mark.unit
def test_gaussian_position_penalty_uses_source_plane_separation():
    penalty = compute_position_penalty_jax(
        _IdentityDeflection(),
        jnp.asarray([0.0, 0.003]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.001),
    )

    np.testing.assert_allclose(float(penalty), -2.25, rtol=1e-6)


@pytest.mark.unit
def test_gaussian_position_penalty_sums_all_images_about_centroid():
    penalty = compute_position_penalty_jax(
        _IdentityDeflection(),
        jnp.asarray([-0.001, 0.0, 0.002]),
        jnp.asarray([0.0, 0.0, 0.0]),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.001),
    )

    # Centroid is +1/3 milliarcsec; all three residuals contribute.
    np.testing.assert_allclose(float(penalty), -2.3333333, rtol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("sigma", [0.0, -0.1, np.nan, np.inf])
def test_position_likelihood_rejects_nonpositive_gaussian_sigma(sigma):
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        resolve_position_likelihood_attrs(
            {"positions": [[0.0, 0.0], [1.0, 0.0]], "sigma_arcsec": sigma}
        )


@pytest.mark.unit
def test_legacy_threshold_position_configuration_remains_available():
    attrs = resolve_position_likelihood_attrs(
        {
            "positions": [[0.0, 0.0], [1.0, 0.0]],
            "threshold_arcsec": 0.1,
            "min_log_like": -10.0,
        }
    )

    assert float(attrs[4]) == 0.0
    assert attrs[5]
