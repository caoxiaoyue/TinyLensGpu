"""
Sersic light profile with Chebyshev polynomial wavelength evolution.

This module extends the standard SersicEllipse profile to support
wavelength-dependent parameters via Chebyshev polynomials, following
the GALFITM method.
"""

from typing import Optional
import caskade as ck
import jax.numpy as jnp
from jax import Array

from TinyLensGpu.utils.geometry import ellipse2circle_transform
from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light.sersic import SersicEllipse


class SersicEllipseChebyshev(ck.Module):
    """
    Sersic light profile with Chebyshev polynomial wavelength evolution.
    
    This class implements a Sersic profile where the structural parameters
    (R_sersic, n_sersic) evolve with wavelength according to Chebyshev
    polynomials, while position and ellipticity remain constant across bands.
    
    The intensity (Ie) is treated separately per band (typically linear-solved).
    
    Parameters
    ----------
    R_sersic_coeffs : list of ParamU
        Chebyshev coefficients for R_sersic evolution: [c0, c1, c2]
        R_sersic(λ) = c0*T0(z) + c1*T1(z) + c2*T2(z)
    n_sersic_coeffs : list of ParamU
        Chebyshev coefficients for n_sersic evolution: [c0, c1, c2]
    e1 : float or ParamU
        Ellipticity component 1 (constant across bands)
    e2 : float or ParamU
        Ellipticity component 2 (constant across bands)
    center_x : float or ParamU
        Center x-coordinate (constant across bands)
    center_y : float or ParamU
        Center y-coordinate (constant across bands)
    Ie : float or ParamU, optional
        Intensity at effective radius (band-specific, typically linear-solved)
    wavelength : float, optional
        Wavelength at which this instance is evaluated
    """

    def __init__(
        self,
        R_sersic_coeffs: Optional[list] = None,
        n_sersic_coeffs: Optional[list] = None,
        e1: Optional[float] = None,
        e2: Optional[float] = None,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        Ie: Optional[float] = None,
        wavelength: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Chebyshev coefficients for wavelength evolution (shared across bands)
        if R_sersic_coeffs is None:
            R_sersic_coeffs = [ParamU("R_sersic_c0", 1.0)]
        if n_sersic_coeffs is None:
            n_sersic_coeffs = [ParamU("n_sersic_c0", 1.0)]
            
        self.R_sersic_coeffs = R_sersic_coeffs
        self.n_sersic_coeffs = n_sersic_coeffs

        # Position and ellipticity (constant across bands)
        self.e1 = e1 if isinstance(e1, ParamU) else ParamU("e1", e1)
        self.e2 = e2 if isinstance(e2, ParamU) else ParamU("e2", e2)
        self.center_x = center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        self.center_y = center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)

        # Intensity (band-specific, typically linear-solved)
        self.Ie = Ie if isinstance(Ie, ParamU) else ParamU("Ie", Ie)
        
        # Wavelength for this instance (used in forward evaluation)
        self.wavelength = wavelength
        
        # Store z-value for parameter linking (normalized wavelength in [-1, +1])
        self._z = None

    def set_wavelength_parameters(
        self,
        z: float,
        R_sersic_func: Optional[callable] = None,
        n_sersic_func: Optional[callable] = None,
    ) -> None:
        """
        Set wavelength-dependent parameters using Chebyshev polynomial.
        
        This method links R_sersic and n_sersic to their Chebyshev coefficients
        using caskade's parameter linking mechanism.
        
        Parameters
        ----------
        z : float
            Normalized wavelength in [-1, +1]
        R_sersic_func : callable, optional
            Pre-computed R_sersic parameter linked to coefficients
        n_sersic_func : callable, optional
            Pre-computed n_sersic parameter linked to coefficients
        """
        self._z = z
        
        # If pre-computed functions provided, use them directly
        if R_sersic_func is not None:
            self.R_sersic = R_sersic_func
        else:
            # Create functional link to Chebyshev coefficients
            coeffs = self.R_sersic_coeffs
            
            def R_func(p):
                c0 = p.R_sersic_c0.value if hasattr(p, 'R_sersic_c0') else coeffs[0].value
                if len(coeffs) > 1:
                    c1 = p.R_sersic_c1.value if hasattr(p, 'R_sersic_c1') else coeffs[1].value
                    c2 = p.R_sersic_c2.value if hasattr(p, 'R_sersic_c2') else coeffs[2].value
                    return c0 + c1 * z + c2 * (2 * z**2 - 1)
                return c0
                
            self.R_sersic = R_func
            self.R_sersic.link(self.R_sersic_coeffs)
        
        if n_sersic_func is not None:
            self.n_sersic = n_sersic_func
        else:
            coeffs = self.n_sersic_coeffs
            
            def n_func(p):
                c0 = p.n_sersic_c0.value if hasattr(p, 'n_sersic_c0') else coeffs[0].value
                if len(coeffs) > 1:
                    c1 = p.n_sersic_c1.value if hasattr(p, 'n_sersic_c1') else coeffs[1].value
                    c2 = p.n_sersic_c2.value if hasattr(p, 'n_sersic_c2') else coeffs[2].value
                    return c0 + c1 * z + c2 * (2 * z**2 - 1)
                return c0
                
            self.n_sersic = n_func
            self.n_sersic.link(self.n_sersic_coeffs)

    @ck.forward
    def light(
        self,
        x: Array,
        y: Array,
        R_sersic: Optional[Array] = None,
        n_sersic: Optional[Array] = None,
        e1: Optional[Array] = None,
        e2: Optional[Array] = None,
        center_x: Optional[Array] = None,
        center_y: Optional[Array] = None,
        Ie: Optional[Array] = None,
    ) -> Array:
        """
        Evaluate elliptical Sersic surface brightness.
        
        Parameters
        ----------
        x, y : array_like
            Coordinates where to evaluate surface brightness
        R_sersic, n_sersic, e1, e2, center_x, center_y, Ie : array_like, optional
            Parameter values (passed by caskade from linked parameters)
            
        Returns
        -------
        surface_brightness : array_like
            Surface brightness values
        """
        # Convert to JAX arrays
        R_sersic = jnp.asarray(R_sersic)
        n_sersic = jnp.asarray(n_sersic)
        e1 = jnp.asarray(e1)
        e2 = jnp.asarray(e2)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)
        Ie = jnp.asarray(Ie)

        # Transform ellipse to circle
        x_, y_ = ellipse2circle_transform(x, y, e1, e2, center_x, center_y)
        R = jnp.sqrt(x_**2 + y_**2)

        # Calculate bn coefficient using Ciotti & Bertin (1999) approximation
        inv_n = 1.0 / n_sersic
        bn = (
            2.0 * n_sersic
            - 1.0 / 3.0
            + 4.0 / 405.0 * inv_n
            + 46.0 / 25515.0 * inv_n**2
            + 131.0 / 1148175.0 * inv_n**3
            - 2194697.0 / 30690717750.0 * inv_n**4
        )

        # Sersic profile
        light_profile = Ie * jnp.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))
        
        # Guard against unphysical parameters (R_sersic <= 0 or n_sersic <= 0)
        is_valid = (R_sersic > 0.0) & (n_sersic > 0.0)
        return jnp.where(is_valid, light_profile, jnp.nan)