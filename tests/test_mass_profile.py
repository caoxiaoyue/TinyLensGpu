
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
    EPL, SIS, SIE, TNFWEllipsePotential, TNFWSpherical
)


@pytest.mark.unit
class TestSIS:
    def setup_method(self):
        self.x = jnp.array([-1.7, -0.4, 0.3, 1.2, 2.1])
        self.y = jnp.array([0.2, 1.3, -0.8, 0.7, -1.1])
        self.theta_E = 1.4
        self.center_x = 0.15
        self.center_y = -0.25

    def _sis_deflection(self, theta_E, center_x, center_y):
        model = SIS()
        return model.deriv.__wrapped__(
            model,
            self.x,
            self.y,
            theta_E=theta_E,
            center_x=center_x,
            center_y=center_y,
        )

    def _sie_deflection(self, theta_E, e1, e2, center_x, center_y):
        model = SIE()
        return model.deriv.__wrapped__(
            model,
            self.x,
            self.y,
            theta_E=theta_E,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
        )

    def test_matches_independent_analytic_deflection(self):
        alpha_x, alpha_y = self._sis_deflection(
            self.theta_E, self.center_x, self.center_y
        )
        dx = self.x - self.center_x
        dy = self.y - self.center_y
        radius = jnp.sqrt(dx**2 + dy**2)

        np.testing.assert_allclose(alpha_x, self.theta_E * dx / radius, rtol=1e-6)
        np.testing.assert_allclose(alpha_y, self.theta_E * dy / radius, rtol=1e-6)
        np.testing.assert_allclose(
            jnp.sqrt(alpha_x**2 + alpha_y**2), self.theta_E, rtol=1e-6
        )

    def test_center_uses_nonzero_singular_deflection_convention(self):
        model = SIS()
        alpha_x, alpha_y = model.deriv.__wrapped__(
            model,
            jnp.array([self.center_x]),
            jnp.array([self.center_y]),
            theta_E=self.theta_E,
            center_x=self.center_x,
            center_y=self.center_y,
        )

        np.testing.assert_allclose(alpha_x, self.theta_E, rtol=1e-6)
        np.testing.assert_allclose(alpha_y, 0.0, atol=1e-7)

    @pytest.mark.parametrize(
        "direction",
        [(1.0, 0.0), (0.0, 1.0), (2**-0.5, 2**-0.5), (-1.0, 0.0)],
    )
    def test_sie_converges_to_sis_from_multiple_ellipticity_directions(
        self, direction
    ):
        sis_x, sis_y = self._sis_deflection(
            self.theta_E, self.center_x, self.center_y
        )
        errors = []
        for ellipticity in (1e-2, 1e-3):
            sie_x, sie_y = self._sie_deflection(
                self.theta_E,
                ellipticity * direction[0],
                ellipticity * direction[1],
                self.center_x,
                self.center_y,
            )
            errors.append(jnp.max(jnp.hypot(sie_x - sis_x, sie_y - sis_y)))

        assert errors[1] < 0.2 * errors[0]

    @pytest.mark.parametrize("e1,e2", [(0.0, 0.0), (1e-6, 0.0), (0.0, -1e-6)])
    def test_near_circular_sie_matches_sis_under_jit(self, e1, e2):
        sis_fn = jax.jit(self._sis_deflection)
        sie_fn = jax.jit(
            lambda theta_E, center_x, center_y: self._sie_deflection(
                theta_E, e1, e2, center_x, center_y
            )
        )

        sis = sis_fn(self.theta_E, self.center_x, self.center_y)
        sie = sie_fn(self.theta_E, self.center_x, self.center_y)
        np.testing.assert_allclose(sie[0], sis[0], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(sie[1], sis[1], rtol=1e-6, atol=1e-7)
        assert jnp.all(jnp.isfinite(sie[0]))
        assert jnp.all(jnp.isfinite(sie[1]))

    def test_near_circular_sie_and_sis_common_parameter_gradients_match(self):
        def flatten(output):
            return jnp.concatenate(output)

        sis_jacobian = jax.jacrev(
            lambda params: flatten(self._sis_deflection(*params))
        )(jnp.array([self.theta_E, self.center_x, self.center_y]))
        sie_jacobian = jax.jacrev(
            lambda params: flatten(
                self._sie_deflection(params[0], 1e-6, -1e-6, params[1], params[2])
            )
        )(jnp.array([self.theta_E, self.center_x, self.center_y]))

        np.testing.assert_allclose(
            sie_jacobian, sis_jacobian, rtol=1e-6, atol=1e-7
        )
        assert jnp.all(jnp.isfinite(sie_jacobian))

    def test_public_import(self):
        from TinyLensGpu.PhysicalModel import SIS as ShallowSIS

        assert ShallowSIS is SIS

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
