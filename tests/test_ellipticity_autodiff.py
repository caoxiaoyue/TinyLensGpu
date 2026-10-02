"""Gradients of physical profiles at axis-aligned and circular ellipticities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Gradient comparisons against finite differences require enough numerical
# precision to avoid catastrophic cancellation in the difference quotient.
jax.config.update("jax_enable_x64", True)

from TinyLensGpu.PhysicalModel import EPL, SIE
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light.gaussian import GaussianEllipse
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import TNFWEllipsePotential
from TinyLensGpu.utils.geometry import ellipticity2phi_q


@pytest.mark.unit
def test_ellipticity_angle_gradient_on_axis():
    angle = lambda e1: ellipticity2phi_q(e1, 0.2)[0]
    np.testing.assert_allclose(jax.grad(angle)(0.0), -2.5, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("e1,e2", [(0.0, 0.0), (0.0, 0.2), (0.2, 0.0)])
def test_gaussian_ellipticity_gradient_matches_finite_difference(e1, e2):
    model = GaussianEllipse()

    def brightness(params):
        return model.light.__wrapped__(
            model, jnp.array(0.7), jnp.array(0.3), flux=1.0, sigma=0.8,
            e1=params[0], e2=params[1], center_x=0.0, center_y=0.0,
        )

    params = jnp.array([e1, e2])
    step = 1e-3
    gradients = jax.jit(jax.grad(brightness))(params)
    finite_difference = jnp.array([
        (brightness(params + step * direction) - brightness(params - step * direction))
        / (2 * step)
        for direction in (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    ])
    np.testing.assert_allclose(gradients, finite_difference, rtol=2e-3, atol=2e-5)


@pytest.mark.unit
def test_sie_circular_ellipticity_jacobian_has_physical_slope():
    model = SIE()
    x, y = 0.7, 0.3
    radius = np.hypot(x, y)
    u, v = x / radius, y / radius

    def deflection(params):
        alpha_x, alpha_y = model.deriv.__wrapped__(
            model, jnp.array([x]), jnp.array([y]), theta_E=1.0,
            e1=params[0], e2=params[1], center_x=0.0, center_y=0.0,
        )
        return jnp.array([alpha_x[0], alpha_y[0]])

    expected = np.array([
        [-u + 2 * u**3 / 3, -2 * v**3 / 3],
        [v - 2 * v**3 / 3, -2 * u**3 / 3],
    ])
    params = jnp.zeros(2)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(deflection))(params), expected,
                               rtol=2e-3, atol=2e-5)
    np.testing.assert_allclose(jax.jit(jax.jacrev(deflection))(params), expected,
                               rtol=2e-3, atol=2e-5)


@pytest.mark.unit
@pytest.mark.parametrize("gamma", [1.5, 2.0, 2.5])
def test_epl_circular_ellipticity_jacobian_has_physical_slope(gamma):
    model = EPL()
    x, y = 0.7, 0.3
    radius = np.hypot(x, y)
    u, v = x / radius, y / radius
    t = gamma - 1.0
    p = 2.0 - t
    amplitude = radius ** (1.0 - t)
    radial = -p**2 / (4.0 - t) * amplitude
    tangential = 2.0 * p / (4.0 - t) * amplitude
    cos2 = u**2 - v**2
    sin2 = 2.0 * u * v
    expected = np.array([
        [radial * cos2 * u - tangential * sin2 * v,
         radial * sin2 * u + tangential * cos2 * v],
        [radial * cos2 * v + tangential * sin2 * u,
         radial * sin2 * v - tangential * cos2 * u],
    ])

    def deflection(params):
        alpha_x, alpha_y = model.deriv.__wrapped__(
            model, jnp.array([x]), jnp.array([y]), theta_E=1.0,
            gamma=gamma, e1=params[0], e2=params[1],
            center_x=0.0, center_y=0.0,
        )
        return jnp.array([alpha_x[0], alpha_y[0]])

    np.testing.assert_allclose(jax.jit(jax.jacrev(deflection))(jnp.zeros(2)),
                               expected, rtol=2e-3, atol=2e-5)


@pytest.mark.unit
@pytest.mark.parametrize("x,y", [(0.7, 0.3), (1.5, 0.0), (0.9, 1.2)])
def test_tnfw_circular_ellipticity_gradient_matches_finite_difference(x, y):
    model = TNFWEllipsePotential()

    def deflection(params):
        alpha_x, alpha_y = model.deriv.__wrapped__(
            model, jnp.array([x]), jnp.array([y]), Rs=1.5,
            alpha_Rs=2.0, r_trunc=5.0, e1=params[0], e2=params[1],
            center_x=0.0, center_y=0.0,
        )
        return alpha_x[0] + alpha_y[0]

    params = jnp.zeros(2)
    step = 1e-3
    finite_difference = jnp.array([
        (deflection(params + step * direction) - deflection(params - step * direction))
        / (2 * step)
        for direction in (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    ])
    np.testing.assert_allclose(jax.jit(jax.grad(deflection))(params),
                               finite_difference, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("gamma", [1.5, 2.5])
def test_epl_near_center_is_continuous_across_circular_switch(dtype, gamma):
    model = EPL()
    radii = jnp.array([1e-6, 1e-8, 1e-10], dtype=dtype)
    threshold = np.sqrt(np.finfo(dtype).eps)
    # q=(1-e)/(1+e): these straddle 1-q=sqrt(eps).
    e_switch = threshold / (2.0 - threshold)

    @jax.jit
    def deflection(e1):
        return model.deriv.__wrapped__(
            model, radii, jnp.zeros_like(radii), theta_E=1.0,
            gamma=gamma, e1=e1, e2=jnp.asarray(0.0, dtype=dtype),
            center_x=0.0, center_y=0.0,
        )[0]

    circular = deflection(jnp.asarray(0.0, dtype=dtype))
    expected = np.maximum(np.asarray(radii), 1e-9) ** (2.0 - gamma)
    np.testing.assert_allclose(circular, expected, rtol=2e-6)
    below = deflection(jnp.asarray(0.99 * e_switch, dtype=dtype))
    above = deflection(jnp.asarray(1.01 * e_switch, dtype=dtype))
    np.testing.assert_allclose(below, above, rtol=2 * threshold)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_tnfw_projection_gradient_at_scale_radius(dtype):
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass.tnfw import _F

    radius = jnp.asarray(1.0, dtype=dtype)
    np.testing.assert_allclose(jax.jit(_F)(radius), 1.0, rtol=1e-6)
    for differentiate in (jax.jacfwd, jax.jacrev):
        gradient = jax.jit(differentiate(_F))(radius)
        np.testing.assert_allclose(gradient, -2.0 / 3.0, rtol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_tnfw_projection_matches_integral_across_series_boundaries(dtype):
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass.tnfw import _F

    radii = jnp.asarray(
        np.sqrt(1.0 + np.array([-0.0101, -0.0099, 0.0, 0.0099, 0.0101])),
        dtype=dtype,
    )
    # Independent quadrature of F(x)=integral_0^1 1/(1+(x^2-1)*t^2) dt.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    nodes, weights = (nodes + 1.0) / 2.0, weights / 2.0
    x = np.asarray(radii, dtype=np.float64)[:, None]
    denominator = 1.0 + (x**2 - 1.0) * nodes**2
    expected = np.sum(weights / denominator, axis=1)
    expected_gradient = np.sum(
        weights * (-2.0 * x * nodes**2) / denominator**2, axis=1
    )
    np.testing.assert_allclose(jax.jit(_F)(radii), expected, rtol=2e-6)
    gradient = jax.jit(jax.vmap(jax.grad(_F)))(radii)
    np.testing.assert_allclose(gradient, expected_gradient,
                               rtol=3e-5 if dtype == jnp.float32 else 1e-9)
