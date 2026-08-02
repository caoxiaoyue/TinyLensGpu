"""
Unit tests for ImageTemplateLight.

The model interpolates a fixed N x N template galaxy image (bilinear)
onto arbitrary (x, y) coordinates in arcseconds, given a pixel size.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import ImageTemplateLight


@pytest.mark.unit
class TestPixelCenterHit:
    """Query points exactly at pixel centers return that pixel's value."""

    def test_pixel_center_exact_values(self):
        img = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # peak = 4.0
        model = ImageTemplateLight(image=img, pixel_size=0.1)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()

        # Pixel (0,0) center at ((-(2-1)/2)*0.1, (-(2-1)/2)*0.1) = (-0.05, -0.05)
        # Pixel (1,1) center at (+0.05, +0.05)
        x = jnp.array([-0.05, 0.05])
        y = jnp.array([-0.05, 0.05])
        b = model.light(x, y)

        # Normalized (peak = 1): [[0.25, 0.5], [0.75, 1.0]]
        assert jnp.allclose(b, jnp.array([0.25, 1.0])), f"got {b}"


@pytest.mark.unit
class TestBilinearInterpolation:
    """Off-center query points match hand-computed bilinear interpolation."""

    def _norm(self, img):
        return img / jnp.max(img)

    def test_center_midpoint(self):
        """Midpoint between the four pixel centers averages the four values."""
        img = self._norm(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        model = ImageTemplateLight(image=img, pixel_size=0.1)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()

        # Query at arcsec origin (0, 0) = index (0.5, 0.5).
        b = float(model.light(jnp.array(0.0), jnp.array(0.0)))
        expected = 0.25 * (float(img[0, 0]) + float(img[0, 1])
                           + float(img[1, 0]) + float(img[1, 1]))
        assert jnp.isclose(b, expected, rtol=1e-5), f"got {b}, expected {expected}"

    def test_asymmetric_point(self):
        """An asymmetric point follows the bilinear formula exactly."""
        img = self._norm(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        model = ImageTemplateLight(image=img, pixel_size=0.1)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()

        # Index coordinates (u=0.25, v=0.75) -> arcsec (-0.025, +0.025).
        x = jnp.array(-0.025)
        y = jnp.array(0.025)
        b = float(model.light(x, y))

        u, v = 0.25, 0.75
        expected = (
            (1 - u) * (1 - v) * float(img[0, 0])
            + u * (1 - v) * float(img[0, 1])
            + (1 - u) * v * float(img[1, 0])
            + u * v * float(img[1, 1])
        )
        assert jnp.isclose(b, expected, rtol=1e-5), f"got {b}, expected {expected}"


@pytest.mark.unit
class TestBoundary:
    """Query points outside the pixel-center grid return zero."""

    def test_outside_returns_zero(self):
        img = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        model = ImageTemplateLight(image=img, pixel_size=0.1)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()

        # Pixel-center grid spans [-0.05, 0.05] in both axes.
        x = jnp.array([0.06, -0.06, 0.0, 0.0])
        y = jnp.array([0.0, 0.0, 0.06, -0.06])
        b = model.light(x, y)
        assert jnp.all(b == 0.0), f"expected all zeros, got {b}"

    def test_mixed_inside_outside(self):
        img = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        model = ImageTemplateLight(image=img, pixel_size=0.1)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()

        x = jnp.array([-0.05, 0.05, 0.2])
        y = jnp.array([-0.05, 0.05, -0.2])
        b = model.light(x, y)
        # Normalized img[0,0] = 0.25, img[1,1] = 1.0, far corner -> 0.
        assert jnp.allclose(b, jnp.array([0.25, 1.0, 0.0])), f"got {b}"


@pytest.mark.unit
class TestScaleAndCenter:
    """Global scale multiplies brightness; centers shift the template."""

    def test_scale_scales_brightness(self):
        img = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        model = ImageTemplateLight(image=img, pixel_size=0.1, scale=2.0)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()

        b = model.light(jnp.array(-0.05), jnp.array(-0.05))
        # Normalized img[0,0] = 0.25, scaled by 2 -> 0.5.
        assert jnp.isclose(float(b), 0.5, rtol=1e-5), f"got {b}"

    def test_center_shifts_template(self):
        img = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        model = ImageTemplateLight(image=img, pixel_size=0.1, center_x=0.1)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()

        # With center_x = 0.1 the template shifts +0.1 in x; the query
        # (0.05, 0.05) now lands on pixel (1, 0) center -> img[1, 0] = 0.75.
        b = model.light(jnp.array(0.05), jnp.array(0.05))
        assert jnp.isclose(float(b), 0.75, rtol=1e-5), f"got {b}"

    def test_center_y_shifts_template(self):
        img = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        model = ImageTemplateLight(image=img, pixel_size=0.1, center_y=0.1)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()

        # With center_y = 0.1 the template shifts +0.1 in y; the query
        # (0.05, 0.05) now lands on pixel (0, 1) center -> img[0, 1] = 0.5.
        b = model.light(jnp.array(0.05), jnp.array(0.05))
        assert jnp.isclose(float(b), 0.5, rtol=1e-5), f"got {b}"

    def test_all_zero_template_stays_zero(self):
        img = jnp.zeros((3, 3))
        model = ImageTemplateLight(image=img, pixel_size=0.1)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()

        b = model.light(jnp.array(0.0), jnp.array(0.0))
        assert float(b) == 0.0

    def test_shape_preserved(self):
        """Brightness matches the query coordinate shape."""
        img = jnp.ones((4, 4))
        model = ImageTemplateLight(image=img, pixel_size=0.1)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()

        x = jnp.linspace(-0.15, 0.15, 7)
        y = jnp.linspace(-0.15, 0.15, 9)
        X, Y = jnp.meshgrid(x, y)
        b = model.light(X, Y)
        assert b.shape == X.shape


@pytest.mark.unit
class TestConstructorValidation:
    """Constructor rejects invalid templates and pixel scales."""

    def test_non_square_rejected(self):
        with pytest.raises(ValueError, match="square"):
            ImageTemplateLight(image=jnp.ones((3, 4)), pixel_size=0.1)

    def test_1d_rejected(self):
        with pytest.raises(ValueError, match="2D"):
            ImageTemplateLight(image=jnp.ones(4), pixel_size=0.1)

    def test_too_small_rejected(self):
        with pytest.raises(ValueError, match="2x2"):
            ImageTemplateLight(image=jnp.ones((1, 1)), pixel_size=0.1)

    def test_nonpositive_pixel_size_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ImageTemplateLight(image=jnp.ones((2, 2)), pixel_size=0.0)


@pytest.mark.unit
class TestPhysicalModelAssembly:
    """The model composes as both a lens-light and a source-light component."""

    def _make_model(self):
        img = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        model = ImageTemplateLight(image=img, pixel_size=0.1)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()
        return model

    def test_as_lens_light(self):
        from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel

        model = self._make_model()
        pm = PhysicalModel(lens_light=[model])
        x = jnp.array([-0.05, 0.0, 0.05])
        y = jnp.array([-0.05, 0.0, 0.05])
        b = pm.lens_surface_brightness(x, y)
        assert b.shape == x.shape
        assert not jnp.any(jnp.isnan(b))
        assert jnp.allclose(b, model.light(x, y)), "assembly must match direct call"

    def test_as_source_light(self):
        from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel

        model = self._make_model()
        pm = PhysicalModel(source_light=[model])
        beta_x = jnp.array([-0.05, 0.0, 0.05])
        beta_y = jnp.array([-0.05, 0.0, 0.05])
        b = pm.source_surface_brightness(beta_x, beta_y)
        assert b.shape == beta_x.shape
        assert not jnp.any(jnp.isnan(b))
        assert jnp.allclose(b, model.light(beta_x, beta_y)), \
            "assembly must match direct call"


@pytest.mark.unit
class TestConsistencyWithMappingMatrix:
    """The gather-based path agrees with the dense mapping-matrix path."""

    def test_matches_dense_mapping_matrix(self):
        from TinyLensGpu.utils.lensing.mapping import (
            build_lens_mapping_matrix,
            build_source_grid,
        )

        img = jnp.array([[1.0, 2.0, 0.5], [3.0, 4.0, 1.5], [0.2, 0.8, 2.0]])
        model = ImageTemplateLight(image=img, pixel_size=0.2)
        model.scale.to_static()
        model.center_x.to_static()
        model.center_y.to_static()

        n = 3
        half = 0.5 * (n - 1) * 0.2
        x_axis, y_axis, _, _ = build_source_grid(n, -half, half, -half, half)

        x = jnp.linspace(-half, half, 5)
        y = jnp.linspace(-half, half, 7)
        X, Y = jnp.meshgrid(x, y)

        matrix = build_lens_mapping_matrix(X, Y, x_axis, y_axis)
        expected = (matrix @ jnp.ravel(model.image)).reshape(X.shape)

        b = model.light(X, Y)
        assert jnp.allclose(b, expected, rtol=1e-5), "gather path must match dense matrix"
