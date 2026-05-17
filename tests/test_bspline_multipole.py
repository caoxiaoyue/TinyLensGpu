"""Tests for the B-spline multipole lens-light basis model."""

import importlib.util
from typing import Any, Tuple, cast

import caskade as ck
import jax.numpy as jnp
import numpy as np
import pytest

# pyright: reportMissingImports=false

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.ObservationModel.LensImage import ImageProbModel
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light.bspline_multipole import (
    BsplineMultipoleBasis,
    bspline_basis_k,
    bspline_bkpts,
    build_bspline_multipole_set,
)

HAS_SCIPY = importlib.util.find_spec("scipy") is not None


class TestBsplineMath:
    """Test public B-spline knot and basis helpers."""

    def test_bspline_bkpts_padding(self):
        """Clamped knots should repeat the radial endpoints by degree."""
        full_knots = bspline_bkpts([0.0, 0.5, 1.0], degree=3)
        expected = jnp.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])

        np.testing.assert_allclose(full_knots, expected)

    def test_bspline_bkpts_length(self):
        """Full knot-vector length should include both boundary pads."""
        for n_breakpoints in [3, 5, 9]:
            for degree in [1, 2, 3]:
                rbkpt = jnp.linspace(0.1, 2.0, n_breakpoints)
                full_knots = bspline_bkpts(rbkpt, degree=degree)

                assert len(full_knots) == n_breakpoints + 2 * degree

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_bspline_basis_k_vs_scipy(self):
        """Each JAX basis column should match SciPy's design matrix."""
        from scipy.interpolate import BSpline

        rbkpt = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        full_knots = bspline_bkpts(rbkpt, degree=3)
        r_values = jnp.linspace(0.0, 1.0, 21)
        n_bases = len(full_knots) - 4

        scipy_design = BSpline.design_matrix(
            np.asarray(r_values), np.asarray(full_knots), k=3
        ).toarray()

        for k in range(n_bases):
            basis = bspline_basis_k(r_values, full_knots, k)
            np.testing.assert_allclose(basis, scipy_design[:, k], rtol=1e-6, atol=1e-8)

    def test_bspline_partition_of_unity(self):
        """Cubic B-spline basis functions should sum to one on the knot span."""
        rbkpt = jnp.array([0.1, 0.2, 0.4, 0.8, 1.6])
        full_knots = bspline_bkpts(rbkpt, degree=3)
        r_values = jnp.linspace(rbkpt[0], rbkpt[-1], 50)
        n_bases = len(full_knots) - 4

        basis_sum = sum(bspline_basis_k(r_values, full_knots, k) for k in range(n_bases))

        np.testing.assert_allclose(basis_sum, jnp.ones_like(r_values), rtol=1e-6, atol=1e-6)

    def test_bspline_basis_zero_outside(self):
        """Basis functions should vanish far outside their full knot range."""
        full_knots = bspline_bkpts([0.1, 0.3, 0.7, 1.0], degree=3)
        r_values = jnp.array([-10.0, -1.0, 2.0, 10.0])
        n_bases = len(full_knots) - 4

        for k in range(n_bases):
            basis = bspline_basis_k(r_values, full_knots, k)
            np.testing.assert_allclose(basis, jnp.zeros_like(r_values), atol=1e-12)

    def test_bspline_basis_nonnegative(self):
        """B-spline basis values should be non-negative everywhere."""
        full_knots = bspline_bkpts([0.1, 0.3, 0.7, 1.0], degree=3)
        r_values = jnp.linspace(0.1, 1.0, 100)
        n_bases = len(full_knots) - 4

        for k in range(n_bases):
            basis = bspline_basis_k(r_values, full_knots, k)
            assert jnp.all(basis >= -1e-12)


class TestBsplineMultipoleComponent:
    """Test the BsplineMultipoleBasis light component public behavior."""

    def setup_method(self):
        """Create shared parameters, a compact radial grid, and image grids."""
        self.center_x = ParamU("center_x", 0.0)
        self.center_y = ParamU("center_y", 0.0)
        self.e1 = ParamU("e1", 0.0)
        self.e2 = ParamU("e2", 0.0)
        self.rbkpt = jnp.array([0.01, 0.05, 0.2, 0.8, 2.0, 5.0])
        self.component = BsplineMultipoleBasis(
            3, 0, self.rbkpt, center_x=self.center_x, center_y=self.center_y, e1=self.e1, e2=self.e2
        )

        x = jnp.linspace(-2.0, 2.0, 50)
        y = jnp.linspace(-2.0, 2.0, 50)
        self.X, self.Y = jnp.meshgrid(x, y)

    def test_component_is_ck_module(self):
        """BsplineMultipoleBasis should participate in caskade module trees."""
        assert isinstance(self.component, ck.Module)

    def test_component_has_light_method(self):
        """The component should expose a callable caskade-forward light method."""
        assert callable(self.component.light)
        assert hasattr(self.component.light, "__wrapped__")

    def test_component_params_count(self):
        """The component should expose amp plus four shared shape parameters."""
        params = [self.component.amp, self.component.center_x, self.component.center_y, self.component.e1, self.component.e2]

        assert len(params) == 5
        assert all(isinstance(param, ParamU) for param in params)

    def test_component_frozen_attrs(self):
        """Static B-spline metadata should be readable attributes."""
        assert self.component._k == 3
        assert self.component._m == 0
        np.testing.assert_allclose(self.component._rbkpt, self.rbkpt)
        assert self.component._degree == 3

    def test_light_output_shape(self):
        """Evaluating light on a two-dimensional grid should preserve shape."""
        light = self.component.light(self.X, self.Y)

        assert light.shape == self.X.shape

    def test_light_amp_scaling(self):
        """The light profile should scale linearly with amp."""
        light_amp1 = self.component.light(self.X, self.Y)
        self.component.amp.to_static(2.0)
        light_amp2 = self.component.light(self.X, self.Y)

        np.testing.assert_allclose(light_amp2, 2.0 * light_amp1, rtol=1e-6, atol=1e-8)

    def test_light_no_nan(self):
        """Reasonable inputs should produce finite surface brightness values."""
        light = self.component.light(self.X, self.Y)

        assert not jnp.any(jnp.isnan(light))
        assert not jnp.any(jnp.isinf(light))

    def test_light_monopole_angular(self):
        """A circular monopole should be axisymmetric at fixed radius."""
        component = BsplineMultipoleBasis(3, 0, self.rbkpt, center_x=0.0, center_y=0.0, e1=0.0, e2=0.0)
        radius = 0.5
        x = jnp.array([radius, 0.0, -radius, 0.0])
        y = jnp.array([0.0, radius, 0.0, -radius])

        light = component.light(x, y)

        np.testing.assert_allclose(light, jnp.full_like(light, light[0]), rtol=1e-6, atol=1e-8)

    def test_light_quadrupole_cos(self):
        """The m=2 cosine mode should follow the expected angular symmetries."""
        component = BsplineMultipoleBasis(3, 2, self.rbkpt, center_x=0.0, center_y=0.0, e1=0.0, e2=0.0)
        radius = 0.5

        light_x_y = component.light(jnp.array([radius]), jnp.array([0.0]))[0]
        light_x_neg_y = component.light(jnp.array([radius]), jnp.array([-0.0]))[0]
        light_y_x = component.light(jnp.array([0.0]), jnp.array([radius]))[0]

        np.testing.assert_allclose(light_x_y, light_x_neg_y, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(light_y_x, -light_x_y, rtol=1e-6, atol=1e-8)

    def test_light_quadrupole_sin(self):
        """The m=-2 sine mode should vanish on the positive x-axis."""
        e1 = ParamU("e1", 0.0)
        e2 = ParamU("e2", 0.0)
        e1.to_static(0.0)
        e2.to_static(0.0)
        component = BsplineMultipoleBasis(3, -2, self.rbkpt, center_x=0.0, center_y=0.0, e1=e1, e2=e2)
        x = jnp.array([0.3, 0.5, 0.9])
        y = jnp.zeros_like(x)

        light = component.light(x, y)

        np.testing.assert_allclose(light, jnp.zeros_like(light), atol=1e-8)

    def test_shared_params_identity(self):
        """Changing a shared ParamU should alter both dependent components."""
        comp_a = BsplineMultipoleBasis(3, 0, self.rbkpt, center_x=0.0, center_y=0.0, e1=self.e1, e2=0.0)
        comp_b = BsplineMultipoleBasis(4, 0, self.rbkpt, center_x=0.0, center_y=0.0, e1=self.e1, e2=0.0)
        point_x = jnp.array([0.5])
        point_y = jnp.array([0.25])

        before_a = comp_a.light(point_x, point_y)
        before_b = comp_b.light(point_x, point_y)
        self.e1.to_static(0.3)
        after_a = comp_a.light(point_x, point_y)
        after_b = comp_b.light(point_x, point_y)

        assert comp_a.e1 is comp_b.e1
        assert not jnp.allclose(before_a, after_a)
        assert not jnp.allclose(before_b, after_b)

    def test_ellipticity_effect(self):
        """Non-zero e1 should break equality between x-axis and y-axis samples."""
        component = BsplineMultipoleBasis(3, 0, self.rbkpt, center_x=0.0, center_y=0.0, e1=0.3, e2=0.0)
        radius = 0.5

        light_x_axis = component.light(jnp.array([radius]), jnp.array([0.0]))[0]
        light_y_axis = component.light(jnp.array([0.0]), jnp.array([radius]))[0]

        assert not jnp.isclose(light_x_axis, light_y_axis)


class TestBsplineMultipoleFactory:
    """Test construction of complete B-spline multipole component sets."""

    def test_factory_default_count(self):
        """Default radial and angular settings should create 51 components (17 bases * 3 multipoles)."""
        components = build_bspline_multipole_set()

        # n_bases = n_radial + degree - 1 = 15 + 3 - 1 = 17; 17 * 3 = 51
        assert len(components) == 51

    def test_factory_custom_ntheta(self):
        """The component count should scale with the requested angular modes."""
        monopoles = build_bspline_multipole_set(ntheta=[0])
        five_modes = build_bspline_multipole_set(ntheta=[0, -2, 2, 4, -4])

        # n_bases = n_radial + degree - 1 = 15 + 3 - 1 = 17
        assert len(monopoles) == 17
        assert len(five_modes) == 85

    def test_factory_log_spacing(self):
        """Factory radial breakpoints should be logarithmically spaced."""
        components = build_bspline_multipole_set(r_min=0.01, r_max=10.0, n_radial=8)
        rbkpt = components[0]._rbkpt
        ratios = rbkpt[1:] / rbkpt[:-1]

        np.testing.assert_allclose(ratios, jnp.full_like(ratios, ratios[0]), rtol=1e-6)

    def test_factory_rmin_rmax(self):
        """Factory breakpoints should honor the requested radial bounds."""
        components = build_bspline_multipole_set(r_min=0.02, r_max=3.5, n_radial=9)
        rbkpt = components[0]._rbkpt

        np.testing.assert_allclose(rbkpt[0], 0.02, rtol=1e-6)
        np.testing.assert_allclose(rbkpt[-1], 3.5, rtol=1e-6)

    def test_factory_shared_params(self):
        """All generated components should share the same center parameters."""
        components = build_bspline_multipole_set()
        center_x = components[0].center_x

        assert all(component.center_x is center_x for component in components)

    def test_factory_all_components_light(self):
        """Every generated component should be evaluable on a small grid."""
        components = build_bspline_multipole_set(n_radial=15, r_min=0.01, r_max=2.0)
        x = jnp.linspace(-1.0, 1.0, 12)
        y = jnp.linspace(-1.0, 1.0, 12)
        X, Y = jnp.meshgrid(x, y)

        for component in components:
            light = component.light(X, Y)
            assert light.shape == X.shape
            assert jnp.all(jnp.isfinite(light))

    def test_factory_amp_default(self):
        """Each component should start with a unit amplitude parameter."""
        components = build_bspline_multipole_set()

        for component in components:
            np.testing.assert_allclose(component.amp.value, 1.0)


class TestBsplineMultipoleIntegration:
    """Test integration with PhysicalModel and ImageProbModel."""

    def _build_components(self):
        """Build the default basis with dynamic shared shape parameters."""
        center_x = ParamU("center_x", 0.0)
        center_y = ParamU("center_y", 0.0)
        e1 = ParamU("e1", 0.0)
        e2 = ParamU("e2", 0.0)
        for param in [center_x, center_y, e1, e2]:
            param.to_dynamic()

        return build_bspline_multipole_set(
            r_min=0.01,
            r_max=2.0,
            n_radial=15,
            ntheta=[0, -2, 2],
            center_x=center_x,
            center_y=center_y,
            e1=e1,
            e2=e2,
        )

    def _build_prob_model(self, solver_type="nnls"):
        """Build a compact image probability model using B-spline lens light."""
        components = self._build_components()
        phys_model = PhysicalModel(lens_light=components)
        npix = 20
        image_data = np.ones((npix, npix), dtype=float)
        noise_map = np.ones((npix, npix), dtype=float) * 0.1
        psf_kernel = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        mask = np.zeros((npix, npix), dtype=bool)

        prob_model = ImageProbModel(
            image_data=image_data,
            noise_map=noise_map,
            psf_kernel=psf_kernel,
            dpix=0.1,
            nsub=1,
            phys_model=phys_model,
            use_linear=True,
            solver_type=solver_type,
            mask=mask,
        )
        return prob_model, phys_model, components

    def test_build_physical_model(self):
        """PhysicalModel should accept a default B-spline lens-light basis."""
        components = self._build_components()
        phys_model = PhysicalModel(lens_light=components)

        # n_bases = n_radial + degree - 1 = 15 + 3 - 1 = 17; 17 * 3 = 51
        assert len(phys_model.lens_light) == 51

    def test_physical_model_registration(self):
        """Each lens-light component should be registered under a unique name."""
        components = self._build_components()
        phys_model = PhysicalModel(lens_light=components)
        registered_names = [f"lens_light_{i}" for i in range(len(components))]

        assert all(hasattr(phys_model, name) for name in registered_names)
        assert len(set(registered_names)) == len(components)

    def test_image_prob_model_creation(self):
        """ImageProbModel should initialize with B-spline lens light in linear mode."""
        prob_model, _, _ = self._build_prob_model()

        assert isinstance(prob_model, ImageProbModel)
        assert prob_model.use_linear is True

    def test_dynamic_params_count(self):
        """Only shared center and ellipticity parameters should be dynamic."""
        prob_model, _, _ = self._build_prob_model()
        dynamic_params = prob_model.get_dynamic_params()

        assert len(dynamic_params) == 4
        assert {param.name for param in dynamic_params} == {"center_x", "center_y", "e1", "e2"}

    def test_forward_model_no_crash(self):
        """Linear forward modeling should return a finite model image."""
        prob_model, _, _ = self._build_prob_model(solver_type="normal")
        model_image, intensities = cast(Tuple[Any, Any], prob_model.forward_model(use_linear=True))

        assert model_image is not None
        assert intensities is not None
        assert model_image.shape == prob_model.image_data.shape
        assert jnp.all(jnp.isfinite(model_image))

    def test_linear_solving_positive_amps(self):
        """NNLS linear solving should return non-negative B-spline amplitudes."""
        prob_model, _, _ = self._build_prob_model(solver_type="nnls")
        solved_params = prob_model.get_linear_solved_params([0.0, 0.0, 0.0, 0.0])

        amp_values = [v for name, v in solved_params.items() if name.endswith("_amp")]
        assert len(amp_values) > 0
        for amp in amp_values:
            assert jnp.all(jnp.asarray(amp) >= -1e-10)

    def test_chi2_reasonable(self):
        """The image chi-square from the forward model should be finite."""
        prob_model, _, _ = self._build_prob_model(solver_type="normal")
        model_image, _ = cast(Tuple[Any, Any], prob_model.forward_model(use_linear=True))
        chi2 = jnp.sum(((model_image - prob_model.image_data) / prob_model.noise_map) ** 2)

        assert jnp.isfinite(chi2)
        assert chi2 >= 0.0
