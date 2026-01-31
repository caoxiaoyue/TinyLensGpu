import numpy as np
import pytest
import jax.numpy as jnp
from jax import jit
import numpy.testing as npt

from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass.epl import EPL
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass.sie import SIE

try:
    from lenstronomy.LensModel.Profiles.epl import EPL as LenstronomyEPL
    LENSTRONOMY_AVAILABLE = True
except ImportError:
    LENSTRONOMY_AVAILABLE = False

@pytest.mark.skipif(not LENSTRONOMY_AVAILABLE, reason="Lenstronomy not available")
def test_epl_consistency_with_lenstronomy():
    """
    Test consistency between TinyLensGpu EPL and lenstronomy EPL.
    """
    # Parameters
    theta_E = 1.5
    gamma = 2.0  # Isothermal case for initial check
    e1 = 0.1
    e2 = -0.05
    center_x = 0.2
    center_y = -0.1
    
    # TinyLensGpu model
    epl_model = EPL(theta_E=theta_E, gamma=gamma, e1=e1, e2=e2, center_x=center_x, center_y=center_y)
    
    # Lenstronomy model
    lenstronomy_epl = LenstronomyEPL()
    
    # Grid
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    
    # Calculate deflection (TinyLensGpu)
    # Use JIT to test compilation
    @jit
    def get_deriv(x, y):
        return epl_model.deriv(x, y)
    
    alpha_x_tl, alpha_y_tl = get_deriv(jnp.array(xx), jnp.array(yy))
    
    # Calculate deflection (Lenstronomy)
    alpha_x_ls, alpha_y_ls = lenstronomy_epl.derivatives(xx, yy, theta_E=theta_E, gamma=gamma, e1=e1, e2=e2, center_x=center_x, center_y=center_y)
    
    # Compare
    npt.assert_allclose(alpha_x_tl, alpha_x_ls, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(alpha_y_tl, alpha_y_ls, rtol=1e-5, atol=1e-5)

@pytest.mark.skipif(not LENSTRONOMY_AVAILABLE, reason="Lenstronomy not available")
def test_epl_non_isothermal():
    """
    Test consistency for non-isothermal case (gamma != 2).
    """
    theta_E = 1.2
    gamma = 2.2 # Steep
    e1 = -0.2
    e2 = 0.1
    center_x = 0.0
    center_y = 0.0
    
    epl_model = EPL(theta_E=theta_E, gamma=gamma, e1=e1, e2=e2, center_x=center_x, center_y=center_y)
    lenstronomy_epl = LenstronomyEPL()
    
    x = np.array([0.5, 1.0, -0.5, -1.0])
    y = np.array([0.0, 0.5, -1.0, 0.5])
    
    alpha_x_tl, alpha_y_tl = epl_model.deriv(jnp.array(x), jnp.array(y))
    alpha_x_ls, alpha_y_ls = lenstronomy_epl.derivatives(x, y, theta_E=theta_E, gamma=gamma, e1=e1, e2=e2, center_x=center_x, center_y=center_y)
    
    npt.assert_allclose(alpha_x_tl, alpha_x_ls, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(alpha_y_tl, alpha_y_ls, rtol=1e-5, atol=1e-5)

def test_epl_consistency_with_sie():
    """
    Test that EPL with gamma=2 matches SIE.
    """
    # Parameters
    theta_E = 1.5
    gamma = 2.0  # Isothermal
    e1 = 0.1
    e2 = -0.05
    center_x = 0.2
    center_y = -0.1

    # TinyLensGpu models
    epl_model = EPL(theta_E=theta_E, gamma=gamma, e1=e1, e2=e2, center_x=center_x, center_y=center_y)
    sie_model = SIE(theta_E=theta_E, e1=e1, e2=e2, center_x=center_x, center_y=center_y)

    # Grid
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    xx, yy = np.meshgrid(x, y)
    xx = jnp.array(xx.flatten())
    yy = jnp.array(yy.flatten())

    # Calculate deflection
    alpha_x_epl, alpha_y_epl = epl_model.deriv(xx, yy)
    alpha_x_sie, alpha_y_sie = sie_model.deriv(xx, yy)

    # Compare
    # Note: different implementations (iterative vs analytic) may have slight numerical differences
    npt.assert_allclose(alpha_x_epl, alpha_x_sie, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(alpha_y_epl, alpha_y_sie, rtol=1e-5, atol=1e-5)
