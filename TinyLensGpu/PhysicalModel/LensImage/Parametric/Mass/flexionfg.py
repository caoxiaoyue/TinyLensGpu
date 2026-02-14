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
    Represent the `Flexionfg` component in the TinyLensGpu pipeline.
    
    Parameters
    ----------
    F1 : Any
        Configuration argument consumed during construction of this component.
    F2 : Any
        Configuration argument consumed during construction of this component.
    G1 : Any
        Configuration argument consumed during construction of this component.
    G2 : Any
        Configuration argument consumed during construction of this component.
    ra_0 : Any
        Configuration argument consumed during construction of this component.
    dec_0 : Any
        Configuration argument consumed during construction of this component.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    def __init__(self, F1: Optional[float] = None, F2: Optional[float] = None, 
                 G1: Optional[float] = None, G2: Optional[float] = None, 
                 ra_0: Optional[float] = None, dec_0: Optional[float] = None) -> None:
        """
        Initialize a `Flexionfg` instance with validated configuration.
        
        Parameters
        ----------
        F1 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        F2 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        G1 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        G2 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        ra_0 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        dec_0 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
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
        """
        Compute transform fg.
        
        Parameters
        ----------
        F1 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        F2 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        G1 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        G2 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
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
        
        """
        Compute deriv.
        
        Parameters
        ----------
        x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        F1 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        F2 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        G1 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        G2 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        ra_0 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        dec_0 : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        F1 = jnp.asarray(F1)
        F2 = jnp.asarray(F2)
        G1 = jnp.asarray(G1)
        G2 = jnp.asarray(G2)
        # ra_0 and dec_0 are passed directly
        
        g1, g2, g3, g4 = self.transform_fg(F1, F2, G1, G2)
        
        flexion_cart = Flexion()
        return flexion_cart.deriv.__wrapped__(flexion_cart, x, y, g1=g1, g2=g2, g3=g3, g4=g4, ra_0=ra_0, dec_0=dec_0)
