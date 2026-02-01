"""
EPL BoxyDisky profiles.
"""
from typing import Optional, Tuple
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.utils.geometry import ellipticity2phi_q
from .epl import EPL
from .multipole import Multipole, EllipticalMultipole


class EPL_BOXYDISKY(ck.Module):
    """
    EPL mass profile combined with a circular multipole with m=4.
    """

    def __init__(self, theta_E: Optional[float] = None, gamma: Optional[float] = None,
                 e1: Optional[float] = None, e2: Optional[float] = None,
                 center_x: Optional[float] = None, center_y: Optional[float] = None,
                 a4_a: Optional[float] = None) -> None:
        super().__init__()
        object.__setattr__(self, "epl", EPL())
        object.__setattr__(self, "multipole", Multipole())

        self.theta_E = theta_E if isinstance(theta_E, ParamU) else ParamU("theta_E", theta_E)
        self.gamma = gamma if isinstance(gamma, ParamU) else ParamU("gamma", gamma)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)
        self.a4_a = a4_a if isinstance(a4_a, ParamU) else ParamU("a4_a", a4_a)

    @ck.forward
    def deriv(self, x: Array, y: Array, theta_E: Optional[Array] = None, 
              gamma: Optional[Array] = None, e1: Optional[Array] = None, 
              e2: Optional[Array] = None, center_x: Optional[Array] = None, 
              center_y: Optional[Array] = None, a4_a: Optional[Array] = None) -> Tuple[Array, Array]:
        
        theta_E = jnp.asarray(theta_E)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        a4_a = jnp.asarray(a4_a)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        
        phi, q = ellipticity2phi_q(e1, e2)
        rescale_am = theta_E / jnp.sqrt(q)

        f_x_epl, f_y_epl = self.epl.deriv.__wrapped__(self.epl, x, y, theta_E=theta_E, gamma=gamma, e1=e1, e2=e2, 
                                                     center_x=center_x, center_y=center_y)

        f_x_m4, f_y_m4 = self.multipole.deriv.__wrapped__(self.multipole, x, y, m=4, a_m=a4_a * rescale_am, 
                                                         phi_m=phi, 
                                                         center_x=center_x, center_y=center_y)
        
        f_x = f_x_epl + f_x_m4
        f_y = f_y_epl + f_y_m4
        
        return f_x, f_y


class EPL_BOXYDISKY_ELL(ck.Module):
    """
    EPL mass profile combined with an elliptical multipole with m=4.
    """

    def __init__(self, theta_E: Optional[float] = None, gamma: Optional[float] = None,
                 e1: Optional[float] = None, e2: Optional[float] = None,
                 center_x: Optional[float] = None, center_y: Optional[float] = None,
                 a4_a: Optional[float] = None) -> None:
        super().__init__()
        object.__setattr__(self, "epl", EPL())
        object.__setattr__(self, "multipole", EllipticalMultipole())

        self.theta_E = theta_E if isinstance(theta_E, ParamU) else ParamU("theta_E", theta_E)
        self.gamma = gamma if isinstance(gamma, ParamU) else ParamU("gamma", gamma)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)
        self.a4_a = a4_a if isinstance(a4_a, ParamU) else ParamU("a4_a", a4_a)

    @ck.forward
    def deriv(self, x: Array, y: Array, theta_E: Optional[Array] = None, 
              gamma: Optional[Array] = None, e1: Optional[Array] = None, 
              e2: Optional[Array] = None, center_x: Optional[Array] = None, 
              center_y: Optional[Array] = None, a4_a: Optional[Array] = None) -> Tuple[Array, Array]:
        
        theta_E = jnp.asarray(theta_E)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        a4_a = jnp.asarray(a4_a)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        
        phi, q = ellipticity2phi_q(e1, e2)

        f_x_epl, f_y_epl = self.epl.deriv.__wrapped__(self.epl, x, y, theta_E=theta_E, gamma=gamma, e1=e1, e2=e2, 
                                                     center_x=center_x, center_y=center_y)

        f_x_m4, f_y_m4 = self.multipole.deriv.__wrapped__(self.multipole, x, y, m=4, a_m=a4_a * theta_E, 
                                                         phi_m=phi, q=q,
                                                         center_x=center_x, center_y=center_y, r_E=theta_E)
        
        f_x = f_x_epl + f_x_m4
        f_y = f_y_epl + f_y_m4
        
        return f_x, f_y
