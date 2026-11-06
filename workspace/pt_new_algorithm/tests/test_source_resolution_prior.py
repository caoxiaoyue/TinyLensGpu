"""Physical-scale checks for the workspace source-resolution experiment."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.utils.inversion.regularization import FiniteDifferenceRegularization


PATH = Path(__file__).resolve().parents[1]/"source_resolution_prior.py"
spec = importlib.util.spec_from_file_location("resolution_prior_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PhysicalCurvaturePrior = module.PhysicalCurvaturePrior


def test_reference_49_preserves_original_quadratic_and_action():
    with jax.experimental.enable_x64(True):
        bbox = jnp.asarray([-.4, .4, -.4, .4])
        s = jnp.asarray(np.random.default_rng(49).normal(size=49*49))
        old = FiniteDifferenceRegularization(49, "second-order").matvec_free(s, *bbox)
        new = PhysicalCurvaturePrior(49).matvec_free(s, *bbox)
        np.testing.assert_allclose(new, old, rtol=2e-14, atol=2e-7)


@pytest.mark.parametrize("n", [5, 8])
def test_scaled_prior_dense_action_ad_symmetry_and_positive_definiteness(n):
    with jax.experimental.enable_x64(True):
        bbox = jnp.asarray([-.4, .4, -.4, .4])
        p = PhysicalCurvaturePrior(n, reference_n=5)
        s = jnp.asarray(np.random.default_rng(n).normal(size=n*n))
        dense = p.matrix(*bbox)
        action = p.matvec_free(s, *bbox)
        np.testing.assert_allclose(action, dense@s, rtol=1e-12, atol=1e-10)
        gradient = jax.grad(lambda x: .5*jnp.vdot(x, p.matvec_free(x, *bbox)))(s)
        np.testing.assert_allclose(gradient, action, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(dense, dense.T, atol=1e-12)
        assert np.linalg.eigvalsh(np.asarray(dense)).min() > 0
        assert float(sum(p.components(s, bbox).values())) == pytest.approx(float(s@action), rel=1e-12)


def test_physical_curvature_and_boundary_integrals_match_across_49_and_100():
    with jax.experimental.enable_x64(True):
        bbox = jnp.asarray([-.4, .4, -.4, .4])
        energies = []
        for n in (49, 100):
            x, y = jnp.meshgrid(jnp.linspace(bbox[0], bbox[1], n), jnp.linspace(bbox[2], bbox[3], n))
            # Constant, affine and curved fields separately exercise boundary
            # amplitude, boundary derivative and interior curvature scaling.
            fields = [jnp.ones_like(x), x+y, x*x+y*y]
            energies.append([PhysicalCurvaturePrior(n).components(s.ravel(), bbox) for s in fields])
        for field, component in [(0, "boundary_amplitude"), (1, "boundary_gradient"), (2, "curvature")]:
            ratio = float(energies[1][field][component]/energies[0][field][component])
            assert abs(ratio-1) < .015  # Endpoint quadrature, not a change of continuum coefficient.
        # Plain h^-4 alone would change total curvature strength by ~4.25x.
        assert PhysicalCurvaturePrior(100).spacing_ratio**2 == pytest.approx((48/99)**2)
