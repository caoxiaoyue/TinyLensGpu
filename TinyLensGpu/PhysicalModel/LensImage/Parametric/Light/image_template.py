"""
Image template light profile.

This module implements a light model that interpolates a fixed N x N
template galaxy image (bilinear interpolation) onto arbitrary (x, y)
coordinates in arcseconds. The template is normalized to a peak of 1 at
construction; the ``scale`` parameter carries the physical peak surface
brightness.
"""

from typing import Optional

import caskade as ck
import jax.numpy as jnp
from jax import Array

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.utils.lensing.mapping import lens_mapping_operator_bilinear_from


class ImageTemplateLight(ck.Module):
    """
    Bilinear-interpolated image template light profile.

    A fixed ``N x N`` template image (in arcsec coordinates defined by
    ``pixel_size``) is evaluated at arbitrary coordinates via bilinear
    interpolation. Query points outside the template pixel-center grid
    return zero brightness.

    Parameters
    ----------
    image : array_like
        2D square template of shape (N, N). Normalized to peak = 1 at
        construction (an all-zero template stays all-zero).
    pixel_size : float
        Pixel scale of the template in arcseconds per pixel.
    scale : float, optional
        Peak surface brightness (cps/arcsec^2); multiplies the
        normalized interpolated template (can be linear parameter).
    center_x : float, optional
        Template center x-coordinate in arcseconds.
    center_y : float, optional
        Template center y-coordinate in arcseconds.
    """

    def __init__(
        self,
        image,
        pixel_size: float,
        scale: Optional[float] = 1.0,
        center_x: Optional[float] = 0.0,
        center_y: Optional[float] = 0.0,
    ) -> None:
        super().__init__()

        img = jnp.asarray(image, dtype=jnp.float32)
        if img.ndim != 2:
            raise ValueError(f"image must be 2D, got shape {img.shape}")
        n_rows, n_cols = img.shape
        if n_rows != n_cols:
            raise ValueError(f"image must be square, got shape {img.shape}")
        if n_rows < 2:
            raise ValueError(f"image must be at least 2x2, got shape {img.shape}")
        pixel_size = float(pixel_size)
        if pixel_size <= 0.0:
            raise ValueError(f"pixel_size must be positive, got {pixel_size}")

        # Normalize to peak = 1 (an all-zero template stays unchanged).
        peak = jnp.max(img)
        img = jnp.where(peak > 0.0, img / peak, img)

        object.__setattr__(self, "image", img)
        object.__setattr__(self, "pixel_size", pixel_size)
        object.__setattr__(self, "n", n_rows)

        self.scale = scale if isinstance(scale, ParamU) else ParamU("scale", scale)
        self.center_x = (
            center_x if isinstance(center_x, ParamU) else ParamU("center_x", center_x)
        )
        self.center_y = (
            center_y if isinstance(center_y, ParamU) else ParamU("center_y", center_y)
        )

    @ck.forward
    def light(
        self,
        x: Array,
        y: Array,
        scale: Optional[Array] = None,
        center_x: Optional[Array] = None,
        center_y: Optional[Array] = None,
    ) -> Array:
        """
        Evaluate the interpolated template surface brightness.

        Parameters
        ----------
        x : array_like
            x-coordinates (arcseconds) where to evaluate brightness.
        y : array_like
            y-coordinates (arcseconds) where to evaluate brightness.
        scale : float, optional
            Peak surface brightness (defaults to self.scale.value).
        center_x : float, optional
            Template center x (defaults to self.center_x.value).
        center_y : float, optional
            Template center y (defaults to self.center_y.value).

        Returns
        -------
        surface_brightness : array_like
            Brightness at the requested coordinates, same shape as x.
        """
        scale = jnp.asarray(scale)
        center_x = jnp.asarray(center_x)
        center_y = jnp.asarray(center_y)

        # Map arcsec coordinates to pixel-center-index coordinates.
        # Pixel (i, j) center sits at ((j-(N-1)/2)*s, (i-(N-1)/2)*s), so:
        #   u = (x - center_x)/s + (N-1)/2,  v = (y - center_y)/s + (N-1)/2
        # lens_mapping_operator_bilinear_from maps the centered (pixel-unit)
        # coordinates onto [0, N-1] internally via x_min/x_max = +/-(N-1)/2.
        half = 0.5 * (self.n - 1)
        u = (x - center_x) / self.pixel_size
        v = (y - center_y) / self.pixel_size

        data_mesh = jnp.stack([jnp.ravel(u), jnp.ravel(v)], axis=1)
        weights, indices, valid = lens_mapping_operator_bilinear_from(
            data_mesh, -half, half, -half, half, self.n,
        )
        img_flat = jnp.ravel(self.image)
        brightness = jnp.sum(weights * img_flat[indices], axis=1)
        brightness = jnp.where(valid, brightness, 0.0) * scale
        return brightness.reshape(jnp.shape(x))
