import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.Inference import ParamU


@pytest.mark.unit
class TestPseudoJaffeSubhalo:
    def test_tidal_radius_is_a_derived_caskade_parameter(self):
        from TinyLensGpu.PhysicalModel import PseudoJaffeSubhalo

        b_sub = ParamU("b_sub", 0.04)
        theta_e_host = ParamU("theta_E_host", 1.44)
        b_sub.to_dynamic()
        theta_e_host.to_dynamic()

        model = PseudoJaffeSubhalo(
            b_sub=b_sub,
            theta_E_host=theta_e_host,
            center_x=0.0,
            center_y=0.0,
        )

        assert model.rt.pointer
        assert model.rt not in model.dynamic_params
        assert set(model.dynamic_params) == {b_sub, theta_e_host}
        np.testing.assert_allclose(model.rt.value, np.sqrt(0.04 * 1.44))

        new_values = {
            b_sub: 0.09,
            theta_e_host: 1.21,
        }
        model.set_values([new_values[param] for param in model.dynamic_params])
        np.testing.assert_allclose(model.rt.value, np.sqrt(0.09 * 1.21))

    def test_deflection_matches_vegetti_singular_pseudo_jaffe(self):
        from TinyLensGpu.PhysicalModel import PseudoJaffeSubhalo

        b_sub = 0.04
        theta_e_host = 1.44
        center_x = 0.1
        center_y = -0.2
        x = jnp.array([0.3, -0.4, 1.2])
        y = jnp.array([0.5, 0.1, -0.8])
        model = PseudoJaffeSubhalo(
            b_sub=b_sub,
            theta_E_host=theta_e_host,
            center_x=center_x,
            center_y=center_y,
        )

        alpha_x, alpha_y = model.deriv(x=x, y=y)

        dx = np.asarray(x) - center_x
        dy = np.asarray(y) - center_y
        radius = np.hypot(dx, dy)
        rt = np.sqrt(b_sub * theta_e_host)
        alpha_r = b_sub * (radius + rt - np.sqrt(radius**2 + rt**2)) / radius
        np.testing.assert_allclose(alpha_x, alpha_r * dx / radius, rtol=1e-6)
        np.testing.assert_allclose(alpha_y, alpha_r * dy / radius, rtol=1e-6)

    def test_zero_strength_switches_the_subhalo_off_without_nans(self):
        from TinyLensGpu.PhysicalModel import PseudoJaffeSubhalo

        model = PseudoJaffeSubhalo(
            b_sub=0.0,
            theta_E_host=1.44,
            center_x=0.1,
            center_y=-0.2,
        )
        alpha_x, alpha_y = model.deriv(
            x=jnp.array([0.1, 0.3]),
            y=jnp.array([-0.2, 0.5]),
        )

        np.testing.assert_array_equal(alpha_x, jnp.zeros(2))
        np.testing.assert_array_equal(alpha_y, jnp.zeros(2))

    def test_singular_limit_is_stable_at_tiny_nonzero_radius(self):
        from TinyLensGpu.PhysicalModel import PseudoJaffeSubhalo

        b_sub = 0.04
        theta_e_host = 1.44
        radii = jnp.array([1.0e-8, 1.0e-7, 1.0e-6])
        model = PseudoJaffeSubhalo(
            b_sub=b_sub,
            theta_E_host=theta_e_host,
            center_x=0.0,
            center_y=0.0,
        )

        alpha_x, alpha_y = model.deriv(x=radii, y=jnp.zeros_like(radii))

        rt = np.sqrt(b_sub * theta_e_host)
        expected = 2.0 * b_sub * rt / (
            np.asarray(radii) + rt + np.hypot(np.asarray(radii), rt)
        )
        np.testing.assert_allclose(alpha_x, expected, rtol=1e-6)
        np.testing.assert_array_equal(alpha_y, jnp.zeros_like(radii))

    def test_zero_strength_deflection_has_finite_parameter_gradients(self):
        from TinyLensGpu.PhysicalModel import PseudoJaffeSubhalo

        b_sub = ParamU("b_sub", 0.0)
        theta_e_host = ParamU("theta_E_host", 1.44)
        b_sub.to_dynamic()
        theta_e_host.to_dynamic()
        model = PseudoJaffeSubhalo(
            b_sub=b_sub,
            theta_E_host=theta_e_host,
            center_x=0.0,
            center_y=0.0,
        )
        initial = jnp.asarray([
            0.0 if param is b_sub else 1.44 for param in model.dynamic_params
        ])

        gradient = jax.grad(
            lambda values: model.deriv(
                x=jnp.array([0.5]),
                y=jnp.array([0.0]),
                params=values,
            )[0][0]
        )(initial)

        assert jnp.all(jnp.isfinite(gradient))

    def test_public_imports_resolve_to_the_same_class(self):
        from TinyLensGpu.PhysicalModel import PseudoJaffeSubhalo as Shallow
        from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import (
            PseudoJaffeSubhalo as Deep,
        )

        assert Shallow is Deep
