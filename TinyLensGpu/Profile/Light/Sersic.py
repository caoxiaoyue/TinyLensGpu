import functools
import jax.numpy as jnp
from jax import jit
from TinyLensGpu.Profile.ProfileBase import LightProfile
from TinyLensGpu.Profile import util

class Sersic(LightProfile):
    _name = "SERSIC"
    _params = ["R_sersic", "n_sersic", "center_x", "center_y", "Ie"]

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, R_sersic, n_sersic, center_x, center_y, Ie=None):
        x_ = x - center_x
        y_ = y - center_y
        R = jnp.sqrt(x_ ** 2 + y_ ** 2)
        bn = 1.9992 * n_sersic - 0.3271
        return Ie * jnp.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))


class SersicEllipse(Sersic):
    _name = "SERSIC_ELLIPSE"
    _params = ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y", "Ie"]

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie=None):
        x_, y_ = util.ellipse2circle_transform(x, y, e1, e2, center_x, center_y)
        R = jnp.sqrt(x_ ** 2 + y_ ** 2)
        bn = 1.9992 * n_sersic - 0.3271
        return Ie * jnp.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))