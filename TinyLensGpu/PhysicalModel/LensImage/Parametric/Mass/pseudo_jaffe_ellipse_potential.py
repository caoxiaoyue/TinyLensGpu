"""
Pseudo Jaffe Ellipse Potential mass profile.
"""
from typing import Optional, Tuple
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.utils.geometry import ellipticity2phi_q, q2e, transform_e1e2_square_average
from .pseudo_jaffe import PseudoJaffe


class PseudoJaffeEllipsePotential(ck.Module):
    """
    Pseudo Jaffe profile with ellipticity in the potential.
    """

    def __init__(self, sigma0: Optional[float] = None, Ra: Optional[float] = None, 
                 Rs: Optional[float] = None, e1: Optional[float] = None, 
                 e2: Optional[float] = None, center_x: Optional[float] = None, 
                 center_y: Optional[float] = None) -> None:
        super().__init__()
        # self.spherical = PseudoJaffe()
        
        self.sigma0 = sigma0 if isinstance(sigma0, ParamU) else ParamU("sigma0", sigma0)
        self.Ra = Ra if isinstance(Ra, ParamU) else ParamU("Ra", Ra)
        self.Rs = Rs if isinstance(Rs, ParamU) else ParamU("Rs", Rs)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

    @ck.forward
    def deriv(self, x: Array, y: Array, sigma0: Optional[Array] = None, 
              Ra: Optional[Array] = None, Rs: Optional[Array] = None, 
              e1: Optional[Array] = None, e2: Optional[Array] = None, 
              center_x: Optional[Array] = None, center_y: Optional[Array] = None) -> Tuple[Array, Array]:
        
        sigma0 = jnp.asarray(sigma0)
        Ra = jnp.asarray(Ra)
        Rs = jnp.asarray(Rs)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        
        phi_G, q = ellipticity2phi_q(e1, e2)
        
        x_, y_ = transform_e1e2_square_average(x, y, e1, e2, center_x, center_y)
        
        e = q2e(q)
        cos_phi = jnp.cos(phi_G)
        sin_phi = jnp.sin(phi_G)
        
        # Call spherical deriv. Pass 0 for center because coordinates are already shifted and rotated.
        spherical = PseudoJaffe()
        f_x_prim, f_y_prim = spherical.deriv.__wrapped__(spherical, x_, y_, sigma0=sigma0, Ra=Ra, Rs=Rs, 
                                                  center_x=0., center_y=0.)
        
        f_x_prim *= jnp.sqrt(1 - e)
        f_y_prim *= jnp.sqrt(1 + e)
        
        f_x = cos_phi * f_x_prim - sin_phi * f_y_prim
        f_y = sin_phi * f_x_prim + cos_phi * f_y_prim
        
        return f_x, f_y
