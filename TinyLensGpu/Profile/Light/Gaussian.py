import jax.numpy as jnp
from TinyLensGpu.Profile import util
from TinyLensGpu.Profile.ProfileBase import LightProfile
import functools
from jax import jit


class Gaussian(LightProfile):
    _name = "Gaussian"
    _params = ['flux', 'sigma', 'center_x', 'center_y']

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, flux, sigma, center_x=0, center_y=0):
        c = flux / (2 * jnp.pi * sigma**2)
        factor = (x - center_x) ** 2 / sigma**2 + (y - center_y) ** 2 / sigma**2
        return c * jnp.exp(-factor / 2.)


class GaussianEllipse(LightProfile):
    _name = "GaussianEllipse"
    _params = ['flux', 'sigma', 'e1', 'e2', 'center_x', 'center_y']

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, flux, sigma, e1, e2, center_x=0, center_y=0):
        c = flux / (2 * jnp.pi * sigma**2)
        x_, y_ = util.ellipse2circle_transform(x, y, e1, e2, center_x, center_y)
        factor = x_**2 / sigma**2 + y_**2 / sigma**2
        return c * jnp.exp(-factor / 2.)
