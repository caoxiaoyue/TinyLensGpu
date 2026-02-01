
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import (
    HernquistEllipse, MoffatEllipse, PseudoJaffeEllipse, Ellipsoid
)

try:
    from lenstronomy.LightModel.Profiles.hernquist import HernquistEllipse as L_HernquistEllipse
    from lenstronomy.LightModel.Profiles.moffat import Moffat as L_Moffat
    from lenstronomy.LightModel.Profiles.pseudo_jaffe import PseudoJaffeEllipse as L_PseudoJaffeEllipse
    from lenstronomy.LightModel.Profiles.ellipsoid import Ellipsoid as L_Ellipsoid
    HAS_LENSTRONOMY = True
except ImportError:
    HAS_LENSTRONOMY = False

@pytest.mark.skipif(not HAS_LENSTRONOMY, reason="lenstronomy not installed")
class TestLightMigration:
    
    def setup_method(self):
        # Test coordinates
        self.x_np = np.array([0.5, 1.0, -0.5, 0.0, 1.5, -2.0])
        self.y_np = np.array([0.0, 0.5, -1.0, 0.5, -1.0, 2.0])
        self.x = jnp.array(self.x_np)
        self.y = jnp.array(self.y_np)
    
    def assert_close(self, val1, val2, rtol=1e-5, atol=1e-8):
        np.testing.assert_allclose(val1, val2, rtol=rtol, atol=atol)

    def test_hernquist_ellipse(self):
        kwargs = {
            'amp': 1.5, 'Rs': 0.8, 'e1': 0.1, 'e2': -0.05,
            'center_x': 0.1, 'center_y': -0.2
        }
        
        # TinyLensGpu
        model = HernquistEllipse(**kwargs)
        sb = model.light(self.x, self.y)
        
        # Lenstronomy
        l_model = L_HernquistEllipse()
        l_sb = l_model.function(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(sb, l_sb)

    def test_moffat_ellipse(self):
        # Lenstronomy's Moffat is spherical, so we test with e1=0, e2=0
        kwargs = {
            'amp': 1.2, 'alpha': 0.5, 'beta': 2.5,
            'center_x': 0.05, 'center_y': 0.1
        }
        
        # TinyLensGpu
        model = MoffatEllipse(e1=0.0, e2=0.0, **kwargs)
        sb = model.light(self.x, self.y)
        
        # Lenstronomy
        l_model = L_Moffat()
        l_sb = l_model.function(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(sb, l_sb)
        
        # Test elliptical version (internal consistency check, 
        # since lenstronomy doesn't have MoffatEllipse in light profile)
        model_ell = MoffatEllipse(e1=0.1, e2=-0.1, **kwargs)
        sb_ell = model_ell.light(self.x, self.y)
        assert sb_ell.shape == self.x.shape

    def test_pseudo_jaffe_ellipse(self):
        kwargs = {
            'amp': 2.0, 'Ra': 0.2, 'Rs': 1.2, 'e1': -0.1, 'e2': 0.05,
            'center_x': -0.1, 'center_y': 0.15
        }
        
        # TinyLensGpu
        model = PseudoJaffeEllipse(**kwargs)
        sb = model.light(self.x, self.y)
        
        # Lenstronomy
        l_model = L_PseudoJaffeEllipse()
        l_sb = l_model.function(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(sb, l_sb)

    def test_ellipsoid(self):
        kwargs = {
            'amp': 10.0, 'radius': 0.7, 'e1': 0.05, 'e2': -0.1,
            'center_x': 0.0, 'center_y': 0.0
        }
        
        # TinyLensGpu
        model = Ellipsoid(**kwargs)
        sb = model.light(self.x, self.y)
        
        # Lenstronomy
        l_model = L_Ellipsoid()
        l_sb = l_model.function(self.x_np, self.y_np, **kwargs)
        
        self.assert_close(sb, l_sb)
