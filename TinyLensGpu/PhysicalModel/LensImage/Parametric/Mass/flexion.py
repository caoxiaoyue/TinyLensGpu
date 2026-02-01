"""
Flexion mass profile.
"""
from typing import Optional, Tuple
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.Inference.param_u import ParamU


class Flexion(ck.Module):
    """
    Class for flexion.
    """

    def __init__(self, g1: Optional[float] = None, g2: Optional[float] = None, 
                 g3: Optional[float] = None, g4: Optional[float] = None, 
                 ra_0: Optional[float] = None, dec_0: Optional[float] = None) -> None:
        super().__init__()
        self.g1 = g1 if isinstance(g1, ParamU) else ParamU("g1", g1)
        self.g2 = g2 if isinstance(g2, ParamU) else ParamU("g2", g2)
        self.g3 = g3 if isinstance(g3, ParamU) else ParamU("g3", g3)
        self.g4 = g4 if isinstance(g4, ParamU) else ParamU("g4", g4)
        self.ra_0 = ra_0 if isinstance(ra_0, ParamU) else ParamU("ra_0", ra_0)
        self.dec_0 = dec_0 if isinstance(dec_0, ParamU) else ParamU("dec_0", dec_0)

    @ck.forward
    def deriv(self, x: Array, y: Array, g1: Optional[Array] = None, 
              g2: Optional[Array] = None, g3: Optional[Array] = None, 
              g4: Optional[Array] = None, ra_0: Optional[Array] = None, 
              dec_0: Optional[Array] = None) -> Tuple[Array, Array]:
        
        g1 = jnp.asarray(g1)
        g2 = jnp.asarray(g2)
        g3 = jnp.asarray(g3)
        g4 = jnp.asarray(g4)
        ra_0 = jnp.asarray(ra_0)
        dec_0 = jnp.asarray(dec_0)

        x_ = x - ra_0
        y_ = y - dec_0
        
        f_x = 0.5 * g1 * x_**2 + g2 * x_ * y_ + 0.5 * g3 * y_**2
        f_y = 0.5 * g2 * x_**2 + g3 * x_ * y_ + 0.5 * g4 * y_**2
        
        return f_x, f_y
