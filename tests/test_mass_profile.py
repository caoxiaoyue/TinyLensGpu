
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import caskade as ck
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import (
    Multipole, EllipticalMultipole,
    EPL_MULTIPOLE_M3M4, EPL_MULTIPOLE_M3M4_ELL,
    EPL_MULTIPOLE_M1M3M4, EPL_MULTIPOLE_M1M3M4_ELL,
    EPL_BOXYDISKY, EPL_BOXYDISKY_ELL,
    Dipole, Flexion, Flexionfg,
    PseudoJaffe, PseudoJaffeEllipsePotential,
    EPL, SIE, TNFWEllipsePotential, TNFWSpherical
)

try:
    import lenstronomy.LensModel.Profiles as lens_profiles
    from lenstronomy.LensModel.Profiles.multipole import Multipole as L_Multipole
    from lenstronomy.LensModel.Profiles.multipole import EllipticalMultipole as L_EllipticalMultipole
    from lenstronomy.LensModel.Profiles.epl_multipole_m3m4 import EPL_MULTIPOLE_M3M4 as L_EPL_MULTIPOLE_M3M4
    from lenstronomy.LensModel.Profiles.epl_multipole_m3m4 import EPL_MULTIPOLE_M3M4_ELL as L_EPL_MULTIPOLE_M3M4_ELL
    from lenstronomy.LensModel.Profiles.epl_multipole_m1m3m4 import EPL_MULTIPOLE_M1M3M4 as L_EPL_MULTIPOLE_M1M3M4
    from lenstronomy.LensModel.Profiles.epl_multipole_m1m3m4 import EPL_MULTIPOLE_M1M3M4_ELL as L_EPL_MULTIPOLE_M1M3M4_ELL
    from lenstronomy.LensModel.Profiles.epl_boxydisky import EPL_BOXYDISKY as L_EPL_BOXYDISKY
    from lenstronomy.LensModel.Profiles.epl_boxydisky import EPL_BOXYDISKY_ELL as L_EPL_BOXYDISKY_ELL
    from lenstronomy.LensModel.Profiles.dipole import Dipole as L_Dipole
    from lenstronomy.LensModel.Profiles.flexion import Flexion as L_Flexion
    from lenstronomy.LensModel.Profiles.flexionfg import Flexionfg as L_Flexionfg
    from lenstronomy.LensModel.Profiles.pseudo_jaffe import PseudoJaffe as L_PseudoJaffe
    from lenstronomy.LensModel.Profiles.pseudo_jaffe_ellipse_potential import PseudoJaffeEllipsePotential as L_PseudoJaffeEllipsePotential
    from lenstronomy.LensModel.Profiles.epl import EPL as L_EPL
    from lenstronomy.LensModel.Profiles.tnfw import TNFW as L_TNFWSpherical
    from lenstronomy.LensModel.Profiles.tnfw_ellipse_potential import TNFWELLIPSEPotential as L_TNFWEllipsePotential
    HAS_LENSTRONOMY = True
except ImportError:
    HAS_LENSTRONOMY = False

@pytest.mark.skipif(not HAS_LENSTRONOMY, reason="lenstronomy not installed")
class TestLenstronomyMigration:
    
    def setup_method(self):
        self.x_np = np.array([0.5, 1.0, -0.5, 0.0])
        self.y_np = np.array([0.0, 0.5, -1.0, 0.5])
        self.x = jnp.array(self.x_np)
        self.y = jnp.array(self.y_np)
    
    def assert_close(self, val1, val2, rtol=1e-4, atol=1e-6):
        np.testing.assert_allclose(val1, val2, rtol=rtol, atol=atol)

    def test_multipole(self):
        # m=1
        kwargs = {'m': 1, 'a_m': 0.1, 'phi_m': 0.5, 'center_x': 0.1, 'center_y': -0.1, 'r_E': 1.0}
        
        # TinyLensGpu
        model = Multipole(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_Multipole()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

        # m=3
        kwargs['m'] = 3
        model = Multipole(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_epl(self):
        kwargs = {
            'theta_E': 1.5, 'gamma': 2.0, 'e1': 0.1, 'e2': -0.05,
            'center_x': 0.2, 'center_y': -0.1
        }
        
        # TinyLensGpu
        model = EPL(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_EPL()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

        # Non-isothermal
        kwargs['gamma'] = 2.2
        model = EPL(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_epl_consistency_with_sie(self):
        kwargs = {
            'theta_E': 1.5, 'e1': 0.1, 'e2': -0.05,
            'center_x': 0.2, 'center_y': -0.1
        }
        
        epl_model = EPL(gamma=2.0, **kwargs)
        sie_model = SIE(**kwargs)
        
        f_x_epl, f_y_epl = epl_model.deriv(self.x, self.y)
        f_x_sie, f_y_sie = sie_model.deriv(self.x, self.y)
        
        self.assert_close(f_x_epl, f_x_sie)
        self.assert_close(f_y_epl, f_y_sie)

    def test_tnfw_spherical(self):
        kwargs = {
            'Rs': 1.5, 'alpha_Rs': 2.0, 'r_trunc': 5.0,
            'center_x': 0.2, 'center_y': -0.1
        }
        
        # TinyLensGpu
        model = TNFWSpherical(**kwargs)
        model.Rs.to_static()
        model.alpha_Rs.to_static()
        model.r_trunc.to_static()
        model.center_x.to_static()
        model.center_y.to_static()
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_TNFWSpherical()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_tnfw_ellipse_potential(self):
        kwargs = {
            'Rs': 1.5, 'alpha_Rs': 2.0, 'r_trunc': 5.0,
            'e1': 0.1, 'e2': -0.05, 'center_x': 0.2, 'center_y': -0.1
        }
        
        # TinyLensGpu
        model = TNFWEllipsePotential(**kwargs)
        model.Rs.to_static()
        model.alpha_Rs.to_static()
        model.r_trunc.to_static()
        model.e1.to_static()
        model.e2.to_static()
        model.center_x.to_static()
        model.center_y.to_static()
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_TNFWEllipsePotential()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_elliptical_multipole(self):
        # m=3
        kwargs = {'m': 3, 'a_m': 0.1, 'phi_m': 0.5, 'q': 0.8, 'center_x': 0.1, 'center_y': -0.1, 'r_E': 1.0}
        
        # TinyLensGpu
        model = EllipticalMultipole(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_EllipticalMultipole()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

        # m=4
        kwargs['m'] = 4
        model = EllipticalMultipole(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_epl_multipole_m3m4(self):
        kwargs = {
            'theta_E': 1.5, 'gamma': 2.0, 'e1': 0.1, 'e2': -0.05,
            'center_x': 0.1, 'center_y': 0.2,
            'a3_a': 0.02, 'delta_phi_m3': 0.1,
            'a4_a': -0.01, 'delta_phi_m4': 0.2
        }
        
        # TinyLensGpu
        model = EPL_MULTIPOLE_M3M4(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_EPL_MULTIPOLE_M3M4()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_epl_multipole_m3m4_ell(self):
        kwargs = {
            'theta_E': 1.5, 'gamma': 2.0, 'e1': 0.1, 'e2': -0.05,
            'center_x': 0.1, 'center_y': 0.2,
            'a3_a': 0.02, 'delta_phi_m3': 0.1,
            'a4_a': -0.01, 'delta_phi_m4': 0.2
        }
        
        # TinyLensGpu
        model = EPL_MULTIPOLE_M3M4_ELL(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_EPL_MULTIPOLE_M3M4_ELL()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_epl_multipole_m1m3m4(self):
        kwargs = {
            'theta_E': 1.5, 'gamma': 2.0, 'e1': 0.1, 'e2': -0.05,
            'center_x': 0.1, 'center_y': 0.2,
            'a1_a': 0.01, 'delta_phi_m1': 0.05,
            'a3_a': 0.02, 'delta_phi_m3': 0.1,
            'a4_a': -0.01, 'delta_phi_m4': 0.2
        }
        
        # TinyLensGpu
        model = EPL_MULTIPOLE_M1M3M4(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_EPL_MULTIPOLE_M1M3M4()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_epl_multipole_m1m3m4_ell(self):
        kwargs = {
            'theta_E': 1.5, 'gamma': 2.0, 'e1': 0.1, 'e2': -0.05,
            'center_x': 0.1, 'center_y': 0.2,
            'a1_a': 0.01, 'delta_phi_m1': 0.05,
            'a3_a': 0.02, 'delta_phi_m3': 0.1,
            'a4_a': -0.01, 'delta_phi_m4': 0.2
        }
        
        # TinyLensGpu
        model = EPL_MULTIPOLE_M1M3M4_ELL(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_EPL_MULTIPOLE_M1M3M4_ELL()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_epl_boxydisky(self):
        kwargs = {
            'theta_E': 1.5, 'gamma': 2.0, 'e1': 0.1, 'e2': -0.05,
            'center_x': 0.1, 'center_y': 0.2,
            'a4_a': 0.02
        }
        
        # TinyLensGpu
        model = EPL_BOXYDISKY(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_EPL_BOXYDISKY()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_epl_boxydisky_ell(self):
        kwargs = {
            'theta_E': 1.5, 'gamma': 2.0, 'e1': 0.1, 'e2': -0.05,
            'center_x': 0.1, 'center_y': 0.2,
            'a4_a': 0.02
        }
        
        # TinyLensGpu
        model = EPL_BOXYDISKY_ELL(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_EPL_BOXYDISKY_ELL()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_dipole(self):
        kwargs = {
            'com_x': 0.1, 'com_y': 0.2, 'phi_dipole': 0.5, 'coupling': 0.1
        }
        
        # TinyLensGpu
        model = Dipole(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_Dipole()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_flexion(self):
        kwargs = {
            'g1': 0.01, 'g2': 0.02, 'g3': 0.03, 'g4': 0.04,
            'ra_0': 0.1, 'dec_0': 0.2
        }
        
        # TinyLensGpu
        model = Flexion(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_Flexion()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_flexionfg(self):
        kwargs = {
            'F1': 0.01, 'F2': 0.02, 'G1': 0.03, 'G2': 0.04,
            'ra_0': 0.1, 'dec_0': 0.2
        }
        
        # TinyLensGpu
        model = Flexionfg(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_Flexionfg()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_pseudo_jaffe(self):
        kwargs = {
            'sigma0': 1.0, 'Ra': 0.5, 'Rs': 1.5, 'center_x': 0.1, 'center_y': 0.2
        }
        
        # TinyLensGpu
        model = PseudoJaffe(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_PseudoJaffe()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)

    def test_pseudo_jaffe_ellipse_potential(self):
        kwargs = {
            'sigma0': 1.0, 'Ra': 0.5, 'Rs': 1.5, 'e1': 0.1, 'e2': -0.1, 
            'center_x': 0.1, 'center_y': 0.2
        }
        
        # TinyLensGpu
        model = PseudoJaffeEllipsePotential(**kwargs)
        f_x, f_y = model.deriv(self.x, self.y)
        
        # Lenstronomy
        l_model = L_PseudoJaffeEllipsePotential()
        l_fx, l_fy = l_model.derivatives(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(f_x, l_fx)
        self.assert_close(f_y, l_fy)
