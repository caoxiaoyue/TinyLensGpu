"""Supplied-field source rendering through a fixed macro plus local potential.

This numerical boundary is shared by source–potential reconstruction backends.
It preserves the established pixelized source interpolation and imaging kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from TinyLensGpu.ForwardSimulation.LensImage._pixelized_psf import (
    PixelizedPSFResponse,
    average_subpixels,
    average_subpixels_adjoint,
    render_source,
    render_source_adjoint,
)
from TinyLensGpu.utils.lensing.mapping import (
    _hermite_axis_support,
    build_source_grid,
    lens_mapping_operator_from,
    source_boundary_for_interpolation,
    source_spatial_gradient as evaluate_source_spatial_gradient,
    validate_source_interpolation,
)
from TinyLensGpu.utils.lensing.potential import PotentialGrid


class MacroMapping(NamedTuple):
    """Macro-dependent rays, separate from invariant image/potential topology."""

    beta_x: Array
    beta_y: Array


class SourceMapping(NamedTuple):
    """Current corrected rays, sparse source weights, and geometric support."""

    weights: Array
    indices: Array
    valid: Array
    beta_x: Array
    beta_y: Array


@dataclass(frozen=True)
class PotentialRenderResult:
    """Nonlinear rendering and derivative products on their stated domains.

    ``model_image`` is the native PSF convolution of the modeled render region.
    ``image_active`` is its selected observation vector. Derivative arrays are defined on the
    potential unknowns, with NaN at support-only nodes and ``derivative_valid``
    explicitly marking the core. They do not assert coverage throughout U.
    """

    model_image: Array
    image_active: Array
    source_values: Array
    source_x_axis: Array
    source_y_axis: Array
    source_bbox: tuple
    potential_values: Array
    potential_grid: PotentialGrid
    deflection_x: Array
    deflection_y: Array
    convergence: Array
    derivative_valid: Array
    query_deflection_x: Array
    query_deflection_y: Array
    corrected_beta_x: Array
    corrected_beta_y: Array
    dtype: str


class PotentialCorrectionRenderer:
    """Prepare fixed geometry and render a signed source with supplied potential.

    Parameters
    ----------
    macro_model : PhysicalModel
        Fixed mass model. ``deflection`` must return macro ray-traced source
        coordinates, as in the existing PhysicalModel interface.
    sim_config : SimulatorConfig
        Existing mask, PSF, image coordinates and subpixel integration settings.
    source_n : int
        Number of nodes per side of the fixed square source grid.
    source_bbox : tuple
        Fixed ``(xmin, xmax, ymin, ymax)`` square bounds, in arcseconds.
    potential_spacing : float, optional
        Angular spacing, default twice the native image pixel spacing.
    potential_grid : PotentialGrid, optional
        Explicit geometry matching all subpixels that can illuminate the fit
        through the PSF, including those outside the fit mask.
    dtype : {"float64", "float32"}
        Explicit numerical precision. Calls use a scoped precision context;
        process defaults outside the call are unchanged. Caller-created input
        coordinates retain their original sampling precision when promoted.
    source_interpolation : {"bilinear", "cubic_hermite"}
        Fixed source interpolation, default "bilinear". Hermite uses the same
        fitted grid with two spacings of support beyond each side.
    """

    def __init__(
        self,
        macro_model,
        sim_config,
        source_n,
        source_bbox,
        *,
        potential_spacing=None,
        potential_grid=None,
        dtype="float64",
        source_interpolation="bilinear",
    ):
        self._source_interpolation = validate_source_interpolation(
            source_interpolation
        )
        if dtype not in ("float32", "float64", jnp.float32, jnp.float64):
            raise ValueError("dtype must be float32 or float64")
        self.dtype = jnp.dtype(dtype)
        self.macro_model = macro_model
        self.sim_config = sim_config
        if isinstance(source_n, bool) or int(source_n) != source_n or int(source_n) < 2:
            raise ValueError("source_n must be an integer at least 2")
        self.source_n = int(source_n)
        self.n_source = self.source_n**2
        bbox = np.asarray(source_bbox, dtype=np.float64)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError("source_bbox must contain four finite bounds")
        dx, dy = bbox[1] - bbox[0], bbox[3] - bbox[2]
        if dx <= 0 or dy <= 0 or not np.isclose(dx, dy, rtol=1e-7, atol=0.0):
            raise ValueError("source_bbox must define a positive square")
        mask = np.asarray(sim_config.mask, dtype=bool)
        if mask.shape != (sim_config.npix, sim_config.npix):
            raise ValueError("Observation mask must match the native image shape")
        if (
            isinstance(sim_config.nsub, bool)
            or int(sim_config.nsub) != sim_config.nsub
            or int(sim_config.nsub) < 1
        ):
            raise ValueError("nsub must be a positive integer")
        self.nsub = int(sim_config.nsub)
        self.image_shape = mask.shape
        self.n_active = int(np.count_nonzero(~mask))
        if self.n_active == 0:
            raise ValueError("Observation mask must retain at least one pixel")
        if not np.isfinite(sim_config.dpix) or sim_config.dpix <= 0:
            raise ValueError("Image pixel spacing must be finite and positive")
        psf = np.asarray(sim_config.psf_kernel)
        if (
            psf.ndim != 2
            or min(psf.shape) < 1
            or not np.isfinite(psf).all()
            or any(k > n for k, n in zip(psf.shape, self.image_shape))
        ):
            raise ValueError(
                "PSF must be finite, two-dimensional, and fit the native image"
            )
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            self.source_bbox = tuple(jnp.asarray(v, dtype=self.dtype) for v in bbox)
            self.source_x_axis, self.source_y_axis, _, _ = build_source_grid(
                self.source_n,
                *self.source_bbox,
            )
            self.flat_indices = jnp.asarray(
                np.flatnonzero(~mask.ravel()), dtype=jnp.int32
            )
            self._psf_response = PixelizedPSFResponse(
                mask, jnp.asarray(sim_config.psf_kernel, dtype=self.dtype), self.nsub,
            )
            x, y = self._psf_response.ray_coordinates(sim_config)
            self.query_x = jnp.asarray(x, dtype=self.dtype)
            self.query_y = jnp.asarray(y, dtype=self.dtype)
            self.agg_segment_ids = self._psf_response.render_parents
            if potential_grid is None:
                spacing = (
                    2 * sim_config.dpix
                    if potential_spacing is None
                    else potential_spacing
                )
                potential_grid = PotentialGrid.from_queries(
                    self.query_x,
                    self.query_y,
                    spacing=spacing,
                    dtype=self.dtype.name,
                )
            elif not isinstance(potential_grid, PotentialGrid):
                raise TypeError("potential_grid must be a PotentialGrid")
            elif potential_spacing is not None and potential_grid.spacing != float(
                potential_spacing
            ):
                raise ValueError(
                    "potential_spacing conflicts with explicit potential_grid"
                )
            if (
                potential_grid.query_x.dtype != self.dtype
                or not np.array_equal(
                    np.asarray(potential_grid.query_x), np.asarray(self.query_x)
                )
                or not np.array_equal(
                    np.asarray(potential_grid.query_y), np.asarray(self.query_y)
                )
            ):
                raise ValueError(
                    "Explicit potential geometry must match the PSF render subpixel queries "
                    "and dtype. Rebuild from the current renderer's query_x and query_y."
                )
            self.potential_grid = potential_grid
            self.macro_mapping = self.prepare_macro()

    @property
    def psf_response(self) -> PixelizedPSFResponse:
        """Shared fitted/rendered image response and source-stencil operations."""
        return self._psf_response

    @property
    def psf_fft(self) -> Array:
        """Spectrum of the kernel on the padded PSF workspace."""
        return self._psf_response.data.spectrum

    @property
    def source_interpolation(self) -> str:
        """Interpolation selected at construction."""
        return getattr(self, "_source_interpolation", "bilinear")

    @property
    def source_boundary(self) -> str:
        """Boundary convention for the selected interpolant."""
        return source_boundary_for_interpolation(self.source_interpolation)

    def prepare_macro(self, macro_model=None) -> MacroMapping:
        """Rebuild macro-dependent rays without rebuilding fixed geometry.

        The returned mapping can be passed explicitly to numerical calls. It
        does not mutate the fixed mapping captured when this renderer was made.
        """
        model = self.macro_model if macro_model is None else macro_model
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            bx, by = model.deflection(x=self.query_x, y=self.query_y)
            bx, by = jnp.asarray(bx, dtype=self.dtype), jnp.asarray(
                by, dtype=self.dtype
            )
            if bx.shape != self.query_x.shape or by.shape != self.query_y.shape:
                raise ValueError("Macro rays must match image subpixel query shapes")
            if (
                not np.isfinite(np.asarray(bx)).all()
                or not np.isfinite(np.asarray(by)).all()
            ):
                raise ValueError(
                    "Macro model produced nonfinite source-plane coordinates"
                )
            return MacroMapping(bx, by)

    def source_mapping(
        self, potential_values=None, *, macro_mapping=None
    ) -> SourceMapping:
        """Prepare current source interpolation at corrected macro rays."""
        return self._source_mapping_for(
            self.source_interpolation, potential_values, macro_mapping=macro_mapping
        )

    def _source_mapping_for(
        self, interpolation, potential_values=None, *, macro_mapping=None
    ):
        """Use an alternate interpolant on the same frozen rays and source grid."""
        rays = self.macro_mapping if macro_mapping is None else macro_mapping
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            bx, by = rays
            if potential_values is not None:
                ax, ay = self.potential_grid.deflection(potential_values)
                bx, by = bx - ax, by - ay
            weights, indices, valid = lens_mapping_operator_from(
                jnp.stack((bx, by), axis=1),
                *self.source_bbox,
                self.source_n,
                interpolation=interpolation,
            )
            return SourceMapping(weights, indices, valid, bx, by)

    def _frozen_source_gradient_mask(self, mapping):
        """Domain of a caller-supplied gradient evaluated on macro rays."""
        if self.source_interpolation == "cubic_hermite":
            xmin, xmax, ymin, ymax = self.source_bbox
            _, _, inside_x = _hermite_axis_support(
                mapping.beta_x, xmin, xmax, self.source_n
            )
            _, _, inside_y = _hermite_axis_support(
                mapping.beta_y, ymin, ymax, self.source_n
            )
            return inside_x & inside_y
        return mapping.valid

    def _source_values(self, source_values):
        values = jnp.asarray(source_values, dtype=self.dtype)
        if values.shape != (self.n_source,):
            raise ValueError(f"source_values must have shape ({self.n_source},)")
        return values

    def source_spatial_gradient(self, source_values, mapping=None):
        """Spatial derivatives of the selected interpolant at corrected rays.

        Uses the same cells and boundary convention as source brightness.
        The default mapping is the fixed macro mapping (zero potential).
        """
        mapping = self.source_mapping() if mapping is None else mapping
        return self._source_spatial_gradient_for(
            self.source_interpolation, source_values, mapping
        )

    def _source_spatial_gradient_for(self, interpolation, source_values, mapping):
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            return evaluate_source_spatial_gradient(
                self._source_values(source_values),
                jnp.stack((mapping.beta_x, mapping.beta_y), axis=1),
                *self.source_bbox,
                self.source_n,
                interpolation=interpolation,
            )

    def _source_render_brightness(self, source_values, mapping):
        return render_source(
            self._source_values(source_values), mapping.weights, mapping.indices,
            self.agg_segment_ids, self.nsub, self._psf_response.n_render,
        )

    def source_forward(self, source_values, mapping=None) -> Array:
        """Map a signed source to selected data pixels using current mapping."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            mapping = self.source_mapping() if mapping is None else mapping
            return self._psf_response.forward(self._source_render_brightness(source_values, mapping))

    def _psf_adjoint(self, image_values):
        values = jnp.asarray(image_values, dtype=self.dtype)
        if values.shape != (self.n_active,):
            raise ValueError(f"image_values must have shape ({self.n_active},)")
        return self._psf_response.adjoint(values)

    def source_adjoint(self, image_values, mapping=None) -> Array:
        """Transpose of source_forward for the same corrected source mapping."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            mapping = self.source_mapping() if mapping is None else mapping
            return render_source_adjoint(
                self._psf_adjoint(image_values), mapping.weights, mapping.indices,
                self.agg_segment_ids, self.nsub, self.n_source,
            )

    def image_forward(self, subpixel_values, *, full_image=False) -> Array:
        """Integrate and convolve a response on the PSF render subpixels.

        The default returns selected data pixels. ``full_image=True`` returns
        the native PSF convolution of the modeled render region.
        """
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            values = jnp.asarray(subpixel_values, dtype=self.dtype)
            if values.shape != self.query_x.shape:
                raise ValueError("subpixel_values must match image query shape")
            brightness = average_subpixels(
                values, self.agg_segment_ids, self.nsub, self._psf_response.n_render,
            )
            if full_image:
                return self._psf_response.full_image(brightness)
            return self._psf_response.forward(brightness)

    def image_adjoint(self, image_values) -> Array:
        """Transpose of image_forward to the PSF render subpixels."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            return average_subpixels_adjoint(
                self._psf_adjoint(image_values), self.agg_segment_ids, self.nsub,
            )

    def render(
        self, source_values, potential_values, *, macro_mapping=None
    ) -> PotentialRenderResult:
        """Return a nonlinear image and physical diagnostics for supplied fields."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            source = self._source_values(source_values)
            potential = self.potential_grid._values(potential_values)
            if not bool(
                jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(potential))
            ):
                raise ValueError("Supplied source and potential values must be finite")
            mapping = self.source_mapping(potential, macro_mapping=macro_mapping)
            brightness = self._source_render_brightness(source, mapping)
            image = self._psf_response.full_image(brightness)
            derivatives = self.potential_grid.derivatives(potential)
            diagnostic = (
                lambda value: jnp.full_like(potential, jnp.nan)
                .at[self.potential_grid.core_indices]
                .set(value)
            )
            ax, ay = self.potential_grid.deflection(potential)
            return PotentialRenderResult(
                image,
                image.ravel()[self.flat_indices],
                source,
                self.source_x_axis,
                self.source_y_axis,
                self.source_bbox,
                potential,
                self.potential_grid,
                diagnostic(derivatives.deflection_x),
                diagnostic(derivatives.deflection_y),
                diagnostic(derivatives.convergence),
                self.potential_grid.derivative_valid,
                ax,
                ay,
                mapping.beta_x,
                mapping.beta_y,
                self.dtype.name,
            )
