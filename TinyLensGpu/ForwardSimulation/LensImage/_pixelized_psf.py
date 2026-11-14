"""Zero-extended PSF response between pixelized render and fitted pixels."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.signal import fftconvolve
from scipy.fft import next_fast_len


PIXELIZED_PSF_CONVENTION = "render_footprint_zero_extended_same"


class PSFData(NamedTuple):
    """Numerical leaves for the response on a fixed, padded FFT canvas."""

    render_slots: Array
    fit_slots: Array
    spectrum: Array


def _apply_psf(values, input_slots, output_slots, spectrum, shape):
    canvas = jnp.zeros((*values.shape[:-1], shape[0] * shape[1]), dtype=values.dtype)
    canvas = canvas.at[..., input_slots].set(
        values, indices_are_sorted=True, unique_indices=True,
    )
    fft = jnp.fft.rfft2(canvas.reshape(*values.shape[:-1], *shape))
    blurred = jnp.fft.irfft2(fft * spectrum, s=shape).reshape(canvas.shape)
    return blurred[..., output_slots]


@partial(jax.jit, static_argnames=("shape",))
def psf_forward(values, data: PSFData, shape):
    """Project native render brightness to the fitted pixel vector."""
    return _apply_psf(values, data.render_slots, data.fit_slots, data.spectrum, shape)


@partial(jax.jit, static_argnames=("shape",))
def psf_adjoint(values, data: PSFData, shape):
    """Transpose the exact same scatter, convolution and crop."""
    return _apply_psf(values, data.fit_slots, data.render_slots, jnp.conj(data.spectrum), shape)


def _render_region(fit: np.ndarray, support: np.ndarray) -> np.ndarray:
    # For y[d] = sum_k K[k] x[d + anchor - k], take the preimage of
    # each fitted pixel under every nonzero kernel tap. Keep fit rays too,
    # since the adaptive reference is projected on exactly those rays.
    render = fit.copy()
    anchor = (np.asarray(support.shape) - 1) // 2
    h, w = fit.shape
    for dy, dx in anchor - np.argwhere(support):
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        if y0 < y1 and x0 < x1:
            render[y0 + dy:y1 + dy, x0 + dx:x1 + dx] |= fit[y0:y1, x0:x1]
    return render


def _subpixel_indices(native: np.ndarray, nsub: int) -> np.ndarray:
    return np.flatnonzero(np.repeat(np.repeat(native, nsub, axis=0), nsub, axis=1))


def _subpixel_parents(subpixels, native, width, nsub):
    if nsub == 1:
        return None
    y, x = np.divmod(subpixels, width * nsub)
    return jnp.asarray(np.searchsorted(native, (y // nsub) * width + x // nsub), dtype=jnp.int32)


class PixelizedPSFResponse:
    """Prepare geometry once; apply a finite PSF and its transpose in JAX.

    Fitted and rendered native indices are distinct, both in row-major order.
    Coordinates used by source interpolation come from the caller's subgrid.
    For a traced kernel override, use its full shape as a conservative footprint
    so the array topology does not depend on the values being differentiated.
    """

    convention = PIXELIZED_PSF_CONVENTION

    def __init__(self, mask, kernel, nsub: int):
        self._mask = np.asarray(mask, dtype=bool)
        self.image_shape = self._mask.shape
        self.nsub = nsub
        self.kernel = jnp.asarray(kernel)
        if self.kernel.ndim != 2 or min(self.kernel.shape) < 1 or jnp.iscomplexobj(self.kernel):
            raise ValueError("PSF kernel must be a nonempty real 2D array")
        if isinstance(self.kernel, jax.core.Tracer):
            support = np.ones(self.kernel.shape, dtype=bool)
        else:
            psf = np.asarray(self.kernel)
            if not np.isfinite(psf).all():
                raise ValueError("PSF kernel must be finite")
            support = psf != 0

        fit = ~self._mask
        render = _render_region(fit, support)
        render_flat = np.flatnonzero(render)
        fit_flat = np.flatnonzero(fit)
        render_sub = _subpixel_indices(render, nsub)
        fit_sub = _subpixel_indices(fit, nsub)
        self.n_fit = fit_flat.size
        self.n_render = render_flat.size
        self.render_indices = jnp.asarray(render_flat, dtype=jnp.int32)
        self.render_subpixels = jnp.asarray(render_sub, dtype=jnp.int32)
        self.render_parents = _subpixel_parents(render_sub, render_flat, fit.shape[1], nsub)
        self._fit_parents = _subpixel_parents(fit_sub, fit_flat, fit.shape[1], nsub)
        self._fit_ray_indices = jnp.asarray(np.searchsorted(render_sub, fit_sub), dtype=jnp.int32)

        render_yx = np.array(np.unravel_index(render_flat, self.image_shape))
        fit_yx = np.array(np.unravel_index(fit_flat, self.image_shape))
        lower = render_yx.min(axis=1) if render_flat.size else np.zeros(2, dtype=int)
        extent = render_yx.max(axis=1) - lower + 1 if render_flat.size else (1, 1)
        self.fft_shape = tuple(
            next_fast_len(int(n) + k - 1, real=True) for n, k in zip(extent, self.kernel.shape)
        )
        # Embed at unshifted input coordinates, and gather with the same crop
        # as fftconvolve(..., mode="same"), including even/asymmetric kernels.
        anchor = (np.asarray(self.kernel.shape) - 1) // 2
        render_slots = np.ravel_multi_index(render_yx - lower[:, None], self.fft_shape)
        fit_slots = np.ravel_multi_index(fit_yx + (anchor - lower)[:, None], self.fft_shape)
        self.data = PSFData(
            jnp.asarray(render_slots, dtype=jnp.int32),
            jnp.asarray(fit_slots, dtype=jnp.int32),
            jnp.fft.rfft2(self.kernel, s=self.fft_shape),
        )

    def with_kernel(self, kernel):
        """Prepare geometry for a dense simulator's per-call PSF override."""
        return type(self)(self._mask, kernel, self.nsub)

    def ray_coordinates(self, config):
        """Gather the caller's existing subpixel coordinates in render order."""
        return (config.xgrid_sub.ravel()[self.render_subpixels],
                config.ygrid_sub.ravel()[self.render_subpixels])

    def _validate_stencil(self, weights, indices):
        if (weights.ndim != 2 or weights.shape != indices.shape
                or weights.shape[0] != self.n_render * self.nsub**2):
            raise ValueError("Source stencil must cover the render subpixels in render order")

    def fit_stencil(self, weights, indices):
        """Select only fitted subpixel rays from a render-ordered source stencil.

        The returned rows follow the fitted subgrid's row-major order. PSF-only
        neighbors contribute to imaging but do not enter reference projection.
        """
        self._validate_stencil(weights, indices)
        return weights[self._fit_ray_indices], indices[self._fit_ray_indices]

    def expand_fit_values(self, values):
        """Repeat a native fitted vector on its fitted subpixel rays, without averaging."""
        values = jnp.asarray(values)
        if values.shape != (self.n_fit,):
            raise ValueError(f"Fitted values must have shape ({self.n_fit},)")
        return values if self._fit_parents is None else values[self._fit_parents]

    def integrate_fit_rays(self, values):
        """Average fitted subpixel rows to native fitted rows, retaining trailing axes."""
        values = jnp.asarray(values)
        if values.ndim < 1 or values.shape[0] != self.n_fit * self.nsub**2:
            raise ValueError("Values must cover the fitted subpixel rays in fitted order")
        return average_subpixels(values, self._fit_parents, self.nsub, self.n_fit)

    def convolved_columns(self, weights, indices, n_columns: int):
        """Build selected source columns as a ``(n_columns, n_fit)`` array.

        Accumulate directly onto native render pixels, integrate subpixels,
        then apply the PSF. Indices at or above ``n_columns`` are omitted, so
        bounded source windows use the same path without a full design matrix.
        The column count is static under JIT.
        """
        self._validate_stencil(weights, indices)
        columns = _native_source_columns(
            weights, indices, self.render_parents, self.nsub, self.n_render, n_columns,
        )
        return self.forward(columns)

    def forward(self, brightness):
        return psf_forward(brightness, self.data, self.fft_shape)

    def adjoint(self, fitted_values):
        return psf_adjoint(fitted_values, self.data, self.fft_shape)

    def precision_diagonal(self, fit_ivar):
        """Diagonal of the PSF-only weighted Gram, indexed by render pixels."""
        squared = self.data._replace(spectrum=jnp.fft.rfft2(self.kernel**2, s=self.fft_shape))
        return jnp.maximum(psf_adjoint(fit_ivar, squared, self.fft_shape), 0.0)

    def full_image(self, brightness):
        """Convolve the modeled render region onto the full cutout for output."""
        image = jnp.zeros(self.image_shape[0] * self.image_shape[1], dtype=brightness.dtype)
        image = image.at[self.render_indices].set(brightness)
        return fftconvolve(image.reshape(self.image_shape), self.kernel, mode="same")


def average_subpixels(values, parents, nsub, n_native):
    """Integrate subpixel rows onto their native-pixel domain."""
    if nsub == 1:
        return values
    return jax.ops.segment_sum(values, parents, num_segments=n_native) / nsub**2


def average_subpixels_adjoint(values, parents, nsub):
    """Transpose native-pixel averaging to the render subgrid."""
    return values if nsub == 1 else values[parents] / nsub**2


@partial(jax.jit, static_argnames=("nsub", "n_render", "n_columns"))
def _native_source_columns(weights, indices, parents, nsub, n_render, n_columns):
    if parents is None:
        parents = jnp.arange(n_render, dtype=jnp.int32)
    columns = jnp.zeros((n_columns, n_render), dtype=weights.dtype)
    return columns.at[indices, parents[:, None]].add(weights / nsub**2, mode="drop")


@partial(jax.jit, static_argnames=("nsub", "n_render"))
def render_source(source, weights, indices, parents, nsub, n_render):
    return average_subpixels(jnp.sum(weights * source[indices], axis=1), parents, nsub, n_render)


@partial(jax.jit, static_argnames=("nsub", "n_source"))
def render_source_adjoint(brightness, weights, indices, parents, nsub, n_source):
    subpixels = average_subpixels_adjoint(brightness, parents, nsub)
    contributions = (weights * subpixels[:, None]).ravel()
    return jnp.zeros(n_source, dtype=contributions.dtype).at[indices.ravel()].add(contributions)
