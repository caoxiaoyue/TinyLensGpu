"""
FlexionFG mass profile.
"""
from typing import Optional, Tuple
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.Inference.param_u import ParamU
from .flexion import Flexion


class Flexionfg(ck.Module):
    """
    Flexion consist of basis F flexion and G flexion (F1,F2,G1,G2).
    """

    def __init__(self, F1: Optional[float] = None, F2: Optional[float] = None, 
                 G1: Optional[float] = None, G2: Optional[float] = None, 
                 ra_0: Optional[float] = None, dec_0: Optional[float] = None) -> None:
        super().__init__()
        # self.flexion_cart = Flexion()
        
        self.F1 = F1 if isinstance(F1, ParamU) else ParamU("F1", F1)
        self.F2 = F2 if isinstance(F2, ParamU) else ParamU("F2", F2)
        self.G1 = G1 if isinstance(G1, ParamU) else ParamU("G1", G1)
        self.G2 = G2 if isinstance(G2, ParamU) else ParamU("G2", G2)
        self.ra_0 = ra_0 if isinstance(ra_0, ParamU) else ParamU("ra_0", ra_0)
        self.dec_0 = dec_0 if isinstance(dec_0, ParamU) else ParamU("dec_0", dec_0)

    @staticmethod
    def transform_fg(F1, F2, G1, G2):
        """Basis transform from (F1,F2,G1,G2) to (g1,g2,g3,g4)."""
        g1 = (3 * F1 + G1) * 0.5
        g2 = (3 * F2 + G2) * 0.5
        g3 = (F1 - G1) * 0.5
        g4 = (F2 - G2) * 0.5
        return g1, g2, g3, g4

    @ck.forward
    def deriv(self, x: Array, y: Array, F1: Optional[Array] = None, 
              F2: Optional[Array] = None, G1: Optional[Array] = None, 
              G2: Optional[Array] = None, ra_0: Optional[Array] = None, 
              dec_0: Optional[Array] = None) -> Tuple[Array, Array]:
        
        F1 = jnp.asarray(F1)
        F2 = jnp.asarray(F2)
        G1 = jnp.asarray(G1)
        G2 = jnp.asarray(G2)
        # ra_0 and dec_0 are passed directly
        
        g1, g2, g3, g4 = self.transform_fg(F1, F2, G1, G2)
        
        flexion_cart = Flexion()
        return flexion_cart.deriv.__wrapped__(flexion_cart, x, y, g1=g1, g2=g2, g3=g3, g4=g4, ra_0=ra_0, dec_0=dec_0)
