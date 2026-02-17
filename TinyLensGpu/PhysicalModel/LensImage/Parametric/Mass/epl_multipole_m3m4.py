"""
EPL + Multipole M3 M4 profiles.
"""
from typing import Optional, Tuple
import caskade as ck
import jax.numpy as jnp
from jax import Array
from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.utils.geometry import ellipticity2phi_q
from .epl import EPL
from .multipole import Multipole, EllipticalMultipole


class EPL_MULTIPOLE_M3M4(ck.Module):
    """
    EPL lens plus circular multipole perturbations (m=3 and m=4).

    This model combines a base elliptical power-law (EPL) mass profile with
    two angular perturbation terms represented by circular multipoles.

    Parameters
    ----------
    theta_E, gamma : float, optional
        EPL Einstein radius and logarithmic slope.
    e1, e2 : float, optional
        EPL ellipticity components.
    center_x, center_y : float, optional
        Lens center coordinates.
    a3_a, a4_a : float, optional
        Relative amplitudes of the m=3 and m=4 multipoles.
    delta_phi_m3, delta_phi_m4 : float, optional
        Phase offsets (radians) of m=3 and m=4 multipoles relative to EPL angle.
    """

    def __init__(self, theta_E: Optional[float] = None, gamma: Optional[float] = None,
                 e1: Optional[float] = None, e2: Optional[float] = None,
                 center_x: Optional[float] = None, center_y: Optional[float] = None,
                 a3_a: Optional[float] = None, delta_phi_m3: Optional[float] = None,
                 a4_a: Optional[float] = None, delta_phi_m4: Optional[float] = None) -> None:
        """
        Initialize EPL + multipole (m=3,m=4) composite model.

        Parameters
        ----------
        theta_E, gamma, e1, e2, center_x, center_y, a3_a, delta_phi_m3, a4_a, delta_phi_m4 : float, optional
            Model parameters converted to :class:`ParamU` when provided as scalars.
        """
        super().__init__()
        object.__setattr__(self, "epl", EPL())
        object.__setattr__(self, "multipole", Multipole())

        self.theta_E = theta_E if isinstance(theta_E, ParamU) else ParamU("theta_E", theta_E)
        self.gamma = gamma if isinstance(gamma, ParamU) else ParamU("gamma", gamma)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)
        self.a3_a = a3_a if isinstance(a3_a, ParamU) else ParamU("a3_a", a3_a)
        self.delta_phi_m3 = delta_phi_m3 if isinstance(delta_phi_m3, ParamU) else ParamU("delta_phi_m3", delta_phi_m3)
        self.a4_a = a4_a if isinstance(a4_a, ParamU) else ParamU("a4_a", a4_a)
        self.delta_phi_m4 = delta_phi_m4 if isinstance(delta_phi_m4, ParamU) else ParamU("delta_phi_m4", delta_phi_m4)

    @ck.forward
    def deriv(self, x: Array, y: Array, theta_E: Optional[Array] = None, 
              gamma: Optional[Array] = None, e1: Optional[Array] = None, 
              e2: Optional[Array] = None, center_x: Optional[Array] = None, 
              center_y: Optional[Array] = None, a3_a: Optional[Array] = None, 
              delta_phi_m3: Optional[Array] = None, a4_a: Optional[Array] = None, 
              delta_phi_m4: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Compute total deflection from EPL + m3 + m4 components.

        Parameters
        ----------
        x, y : Array
            Image-plane coordinates.
        theta_E, gamma, e1, e2, center_x, center_y, a3_a, delta_phi_m3, a4_a, delta_phi_m4 : Array, optional
            Runtime parameter values injected by caskade.

        Returns
        -------
        tuple[Array, Array]
            Deflection components ``(alpha_x, alpha_y)``.
        """
        theta_E = jnp.asarray(theta_E)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        a3_a = jnp.asarray(a3_a)
        delta_phi_m3 = jnp.asarray(delta_phi_m3)
        a4_a = jnp.asarray(a4_a)
        delta_phi_m4 = jnp.asarray(delta_phi_m4)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        
        phi, q = ellipticity2phi_q(e1, e2)
        rescale_am = theta_E / jnp.sqrt(q)

        f_x_epl, f_y_epl = self.epl.deriv.__wrapped__(self.epl, x, y, theta_E=theta_E, gamma=gamma, e1=e1, e2=e2, 
                                                     center_x=center_x, center_y=center_y)
        
        f_x_m3, f_y_m3 = self.multipole.deriv.__wrapped__(self.multipole, x, y, m=3, a_m=a3_a * rescale_am, 
                                                         phi_m=phi + delta_phi_m3, 
                                                         center_x=center_x, center_y=center_y)
        
        f_x_m4, f_y_m4 = self.multipole.deriv.__wrapped__(self.multipole, x, y, m=4, a_m=a4_a * rescale_am, 
                                                         phi_m=phi + delta_phi_m4, 
                                                         center_x=center_x, center_y=center_y)
        
        f_x = f_x_epl + f_x_m3 + f_x_m4
        f_y = f_y_epl + f_y_m3 + f_y_m4
        
        return f_x, f_y


class EPL_MULTIPOLE_M3M4_ELL(ck.Module):
    """
    EPL lens plus elliptical multipole perturbations (m=3 and m=4).

    Compared with :class:`EPL_MULTIPOLE_M3M4`, this variant uses multipole
    operators that depend explicitly on lens ellipticity ``q`` and Einstein radius.

    Parameters
    ----------
    theta_E, gamma : float, optional
        EPL Einstein radius and logarithmic slope.
    e1, e2 : float, optional
        EPL ellipticity components.
    center_x, center_y : float, optional
        Lens center coordinates.
    a3_a, a4_a : float, optional
        Relative amplitudes of the m=3 and m=4 multipoles.
    delta_phi_m3, delta_phi_m4 : float, optional
        Phase offsets (radians) of m=3 and m=4 multipoles.
    """

    def __init__(self, theta_E: Optional[float] = None, gamma: Optional[float] = None,
                 e1: Optional[float] = None, e2: Optional[float] = None,
                 center_x: Optional[float] = None, center_y: Optional[float] = None,
                 a3_a: Optional[float] = None, delta_phi_m3: Optional[float] = None,
                 a4_a: Optional[float] = None, delta_phi_m4: Optional[float] = None) -> None:
        """
        Initialize EPL + elliptical multipole (m=3,m=4) model.

        Parameters
        ----------
        theta_E, gamma, e1, e2, center_x, center_y, a3_a, delta_phi_m3, a4_a, delta_phi_m4 : float, optional
            Model parameters converted to :class:`ParamU` when provided as scalars.
        """
        super().__init__()
        object.__setattr__(self, "epl", EPL())
        object.__setattr__(self, "multipole", EllipticalMultipole())

        self.theta_E = theta_E if isinstance(theta_E, ParamU) else ParamU("theta_E", theta_E)
        self.gamma = gamma if isinstance(gamma, ParamU) else ParamU("gamma", gamma)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)
        self.a3_a = a3_a if isinstance(a3_a, ParamU) else ParamU("a3_a", a3_a)
        self.delta_phi_m3 = delta_phi_m3 if isinstance(delta_phi_m3, ParamU) else ParamU("delta_phi_m3", delta_phi_m3)
        self.a4_a = a4_a if isinstance(a4_a, ParamU) else ParamU("a4_a", a4_a)
        self.delta_phi_m4 = delta_phi_m4 if isinstance(delta_phi_m4, ParamU) else ParamU("delta_phi_m4", delta_phi_m4)

    @ck.forward
    def deriv(self, x: Array, y: Array, theta_E: Optional[Array] = None, 
              gamma: Optional[Array] = None, e1: Optional[Array] = None, 
              e2: Optional[Array] = None, center_x: Optional[Array] = None, 
              center_y: Optional[Array] = None, a3_a: Optional[Array] = None, 
              delta_phi_m3: Optional[Array] = None, a4_a: Optional[Array] = None, 
              delta_phi_m4: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Compute total deflection from EPL + elliptical m3 + elliptical m4 terms.

        Parameters
        ----------
        x, y : Array
            Image-plane coordinates.
        theta_E, gamma, e1, e2, center_x, center_y, a3_a, delta_phi_m3, a4_a, delta_phi_m4 : Array, optional
            Runtime parameter values injected by caskade.

        Returns
        -------
        tuple[Array, Array]
            Deflection components ``(alpha_x, alpha_y)``.
        """
        theta_E = jnp.asarray(theta_E)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        a3_a = jnp.asarray(a3_a)
        delta_phi_m3 = jnp.asarray(delta_phi_m3)
        a4_a = jnp.asarray(a4_a)
        delta_phi_m4 = jnp.asarray(delta_phi_m4)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        
        phi, q = ellipticity2phi_q(e1, e2)

        f_x_epl, f_y_epl = self.epl.deriv.__wrapped__(self.epl, x, y, theta_E=theta_E, gamma=gamma, e1=e1, e2=e2, 
                                                     center_x=center_x, center_y=center_y)
        
        f_x_m3, f_y_m3 = self.multipole.deriv.__wrapped__(self.multipole, x, y, m=3, a_m=a3_a * theta_E, 
                                                         phi_m=phi + delta_phi_m3, q=q,
                                                         center_x=center_x, center_y=center_y, r_E=theta_E)
        
        f_x_m4, f_y_m4 = self.multipole.deriv.__wrapped__(self.multipole, x, y, m=4, a_m=a4_a * theta_E, 
                                                         phi_m=phi + delta_phi_m4, q=q,
                                                         center_x=center_x, center_y=center_y, r_E=theta_E)
        
        f_x = f_x_epl + f_x_m3 + f_x_m4
        f_y = f_y_epl + f_y_m3 + f_y_m4
        
        return f_x, f_y
