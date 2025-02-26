import functools
from jax import jit
from TinyLensGpu.Profile.ProfileBase import MassProfile

class Shear(MassProfile):
    _name = "SHEAR"
    _params = ["gamma1", "gamma2"]

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, gamma1, gamma2):
        return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y
