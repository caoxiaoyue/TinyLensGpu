
import unittest
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass.tnfw import TNFWEllipsePotential as TNFW_JAX, TNFWSpherical as TNFW_Spherical_JAX
try:
    from lenstronomy.LensModel.Profiles.tnfw_ellipse_potential import TNFWELLIPSEPotential as TNFW_Lenstronomy
    from lenstronomy.LensModel.Profiles.tnfw import TNFW as TNFW_Spherical_Lenstronomy
except ImportError:
    print("Warning: lenstronomy not found. Skipping comparison tests.")
    TNFW_Lenstronomy = None
    TNFW_Spherical_Lenstronomy = None

class TestTNFWEllipsePotential(unittest.TestCase):
    def setUp(self):
        if TNFW_Lenstronomy:
            self.lenstronomy_tnfw = TNFW_Lenstronomy()

    def test_derivatives_comparison(self):
        if not TNFW_Lenstronomy:
            return

        # Parameters
        Rs = 1.5
        alpha_Rs = 2.0
        r_trunc = 5.0
        e1 = 0.1
        e2 = -0.05
        center_x = 0.2
        center_y = -0.1
        
        # Coordinates
        x = jnp.array([0.5, 1.0, 2.0, 0.0, -1.0])
        y = jnp.array([0.5, -0.5, 1.0, 0.1, 0.0])

        # TinyLensGpu (JAX) calculation
        # Initialize model with parameters
        jax_tnfw = TNFW_JAX(
            Rs=Rs, alpha_Rs=alpha_Rs, r_trunc=r_trunc,
            e1=e1, e2=e2, center_x=center_x, center_y=center_y
        )
        
        # Set parameters to static to avoid caskade trying to infer them from args or state
        jax_tnfw.Rs.to_static()
        jax_tnfw.alpha_Rs.to_static()
        jax_tnfw.r_trunc.to_static()
        jax_tnfw.e1.to_static()
        jax_tnfw.e2.to_static()
        jax_tnfw.center_x.to_static()
        jax_tnfw.center_y.to_static()
        
        alpha_x_jax, alpha_y_jax = jax_tnfw.deriv(x, y)

        # lenstronomy calculation
        # lenstronomy expects numpy arrays or scalars
        alpha_x_ls, alpha_y_ls = self.lenstronomy_tnfw.derivatives(
            np.array(x), np.array(y), Rs=Rs, alpha_Rs=alpha_Rs, r_trunc=r_trunc,
            e1=e1, e2=e2, center_x=center_x, center_y=center_y
        )

        # Convert JAX arrays to numpy for comparison
        alpha_x_jax = np.array(alpha_x_jax)
        alpha_y_jax = np.array(alpha_y_jax)

        # Assert closeness
        np.testing.assert_allclose(alpha_x_jax, alpha_x_ls, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(alpha_y_jax, alpha_y_ls, rtol=1e-5, atol=1e-8)

    def test_jit_compilation(self):
        # Test if it can be jitted
        import jax
        
        Rs = 1.5
        alpha_Rs = 2.0
        r_trunc = 5.0
        e1 = 0.1
        e2 = -0.05
        center_x = 0.2
        center_y = -0.1
        
        x = jnp.array([0.5])
        y = jnp.array([0.5])

        # Create model
        tnfw = TNFW_JAX(
            Rs=Rs, alpha_Rs=alpha_Rs, r_trunc=r_trunc,
            e1=e1, e2=e2, center_x=center_x, center_y=center_y
        )
        tnfw.Rs.to_static()
        tnfw.alpha_Rs.to_static()
        tnfw.r_trunc.to_static()
        tnfw.e1.to_static()
        tnfw.e2.to_static()
        tnfw.center_x.to_static()
        tnfw.center_y.to_static()
        
        jit_deriv = jax.jit(tnfw.deriv)
        
        # First run (compilation)
        res1 = jit_deriv(x, y)
        
        # Second run (execution)
        res2 = jit_deriv(x, y)
        
        assert len(res1) == 2


class TestTNFWSpherical(unittest.TestCase):
    def setUp(self):
        if TNFW_Spherical_Lenstronomy:
            self.lenstronomy_tnfw = TNFW_Spherical_Lenstronomy()

    def test_derivatives_comparison(self):
        if not TNFW_Spherical_Lenstronomy:
            return

        # Parameters
        Rs = 1.5
        alpha_Rs = 2.0
        r_trunc = 5.0
        center_x = 0.2
        center_y = -0.1
        
        # Coordinates
        x = jnp.array([0.5, 1.0, 2.0, 0.0, -1.0])
        y = jnp.array([0.5, -0.5, 1.0, 0.1, 0.0])

        # TinyLensGpu (JAX) calculation
        # Initialize model with parameters
        jax_tnfw = TNFW_Spherical_JAX(
            Rs=Rs, alpha_Rs=alpha_Rs, r_trunc=r_trunc,
            center_x=center_x, center_y=center_y
        )
        
        # Set parameters to static
        jax_tnfw.Rs.to_static()
        jax_tnfw.alpha_Rs.to_static()
        jax_tnfw.r_trunc.to_static()
        jax_tnfw.center_x.to_static()
        jax_tnfw.center_y.to_static()
        
        alpha_x_jax, alpha_y_jax = jax_tnfw.deriv(x, y)

        # lenstronomy calculation
        alpha_x_ls, alpha_y_ls = self.lenstronomy_tnfw.derivatives(
            np.array(x), np.array(y), Rs=Rs, alpha_Rs=alpha_Rs, r_trunc=r_trunc,
            center_x=center_x, center_y=center_y
        )

        # Convert JAX arrays to numpy for comparison
        alpha_x_jax = np.array(alpha_x_jax)
        alpha_y_jax = np.array(alpha_y_jax)

        # Assert closeness
        np.testing.assert_allclose(alpha_x_jax, alpha_x_ls, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(alpha_y_jax, alpha_y_ls, rtol=1e-5, atol=1e-8)

    def test_jit_compilation(self):
        # Test if it can be jitted
        import jax
        
        Rs = 1.5
        alpha_Rs = 2.0
        r_trunc = 5.0
        center_x = 0.2
        center_y = -0.1
        
        x = jnp.array([0.5])
        y = jnp.array([0.5])

        # Create model
        tnfw = TNFW_Spherical_JAX(
            Rs=Rs, alpha_Rs=alpha_Rs, r_trunc=r_trunc,
            center_x=center_x, center_y=center_y
        )
        tnfw.Rs.to_static()
        tnfw.alpha_Rs.to_static()
        tnfw.r_trunc.to_static()
        tnfw.center_x.to_static()
        tnfw.center_y.to_static()
        
        jit_deriv = jax.jit(tnfw.deriv)
        
        # First run (compilation)
        res1 = jit_deriv(x, y)
        
        # Second run (execution)
        res2 = jit_deriv(x, y)
        
        assert len(res1) == 2

if __name__ == '__main__':
    unittest.main()
