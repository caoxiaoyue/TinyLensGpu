"""Pixelized source forward simulator for lens-image modeling."""

# pyright: reportMissingImports=false

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array

from ...PhysicalModel.LensImage.composite import PhysicalModel
from ...PhysicalModel.LensImage.Pixelized.Light import is_pixelized_source_model
from ...utils.lensing.mapping import (
    build_lens_mapping_matrix,
    build_source_grid,
    infer_source_bbox,
    lens_mapping_operator_from,
)
from ._pixelized_psf import PixelizedPSFResponse, render_source
from .config import SimulatorConfig
from .parametric import bin_image_general
from .results import SimulationResult


EPSILON_REG = 1e-6  # Tiny Tikhonov regularization for lens-light amplitudes
LENS_LIGHT_INTENSITY_NAMES = ("Ie", "flux", "amp", "intensity")


def validate_lens_light_regularization(value: float) -> float:
    """Return a finite positive lens-light Tikhonov strength."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("lens_light_regularization must be finite and positive")
    return float(value)


def validate_unit_lens_light_bases(phys_model: PhysicalModel) -> None:
    """Validate the unit-amplitude basis contract for joint inversion."""
    for index, component in enumerate(phys_model.lens_light):
        matches = [
            name for name in LENS_LIGHT_INTENSITY_NAMES
            if getattr(component, name, None) is not None
        ]
        if len(matches) != 1:
            raise ValueError(
                f"lens_light[{index}] must expose exactly one supported intensity "
                f"parameter ({', '.join(LENS_LIGHT_INTENSITY_NAMES)})"
            )
        name = matches[0]
        parameter = getattr(component, name)
        if not bool(getattr(parameter, "static", False)):
            raise ValueError(
                f"lens_light[{index}].{name} must be static for a "
                "unit-amplitude lens-light basis"
            )
        value = parameter.value if hasattr(parameter, "value") else parameter
        if not np.isclose(float(np.asarray(value)), 1.0, rtol=1e-7, atol=1e-7):
            raise ValueError(
                f"lens_light[{index}].{name} must equal 1.0 for a "
                "unit-amplitude lens-light basis"
            )


class PixelizedLensSimulator:
    """Forward simulator for a single pixelized source and optional lens light.

    Ray-traces image-plane pixels to the source plane, builds the dense
    mapping matrix using the source model's interpolation, and convolves
    with the PSF via FFT. When ``nsub > 1`` the ray-tracing uses a sub-sampled
    grid and aggregates the mapping back to native resolution before PSF
    convolution.

    When ``lens_light`` is present, the simulator builds a joint design
    matrix ``M = [F | L]`` where ``F`` is the source mapping matrix and
    ``L`` is the lens-light basis matrix, enabling joint linear inversion.

    Parameters
    ----------
    phys_model : PhysicalModel
        Model with lens mass and exactly one pixelized source.
        May optionally include lens light components.
    sim_config : SimulatorConfig
        Image grid, PSF, mask, and subsampling (``nsub``) configuration.
    detach_bbox : bool, optional
        When ``True`` (default), ``stop_gradient`` is applied to the
        source-plane bounding-box bounds so that min/max operations do
        not produce unstable gradients.  Beta ray-tracing and source
        interpolation weights remain differentiable.

    Raises
    ------
    ValueError
        If the model has mixed sources, multiple pixelized sources,
        or a parametric source.
    """

    def __init__(
        self,
        phys_model: PhysicalModel,
        sim_config: SimulatorConfig,
        detach_bbox: bool = True,
    ) -> None:
        self.phys_model = phys_model
        self.sim_config = sim_config
        self.nsub = int(sim_config.nsub)
        self.detach_bbox = detach_bbox

        n_pixelized = sum(is_pixelized_source_model(s) for s in phys_model.source_light)
        if n_pixelized != len(phys_model.source_light):
            raise ValueError("PixelizedLensSimulator does not support a parametric source")
        if n_pixelized != 1:
            raise ValueError("PixelizedLensSimulator requires a single pixelized source")

        source = phys_model.source_light[0]
        self.source_model = source
        self.source_n = int(source.n)
        self.n_source_pixels = self.source_n * self.source_n

        # Lens light
        self.n_lens_light = len(phys_model.lens_light)
        self.has_lens_light = self.n_lens_light > 0
        validate_unit_lens_light_bases(phys_model)

        self.image_shape = tuple(sim_config.mask.shape)
        self.mask = jnp.asarray(sim_config.mask, dtype=bool)
        self.active_mask = ~self.mask
        self.flat_indices = jnp.flatnonzero(self.active_mask.ravel())
        self.image_x_active = jnp.asarray(sim_config.xgrid)[self.active_mask]
        self.image_y_active = jnp.asarray(sim_config.ygrid)[self.active_mask]
        self.psf_kernel = jnp.asarray(sim_config.psf_kernel)
        self._psf_response = PixelizedPSFResponse(sim_config.mask, self.psf_kernel, self.nsub)
        self._render_coords = self._psf_response.ray_coordinates(sim_config)

        # Source seed mask for bounding-box inference (dual-masking strategy)
        self.source_seed_mask = jnp.asarray(sim_config.source_seed_mask, dtype=bool)
        self.seed_active_mask = ~self.source_seed_mask
        # Ensure seed active region is a subset of data active region
        if not jnp.all(self.seed_active_mask <= self.active_mask):
            raise ValueError(
                "source_seed_mask active region must be a subset of mask active region. "
                "Ensure every unmasked pixel in source_seed_mask is also unmasked in mask."
            )
        self.seed_flat_indices = jnp.flatnonzero(self.seed_active_mask.ravel())
        self.image_x_seed = jnp.asarray(sim_config.xgrid)[self.seed_active_mask]
        self.image_y_seed = jnp.asarray(sim_config.ygrid)[self.seed_active_mask]

        if self.nsub > 1:
            self.image_shape_sub = sim_config.subgrid_shape
            self.flat_indices_sub = sim_config.flat_indices
            self.image_x_active_sub = jnp.asarray(sim_config.xgrid_sub_1d)
            self.image_y_active_sub = jnp.asarray(sim_config.ygrid_sub_1d)
        else:
            self.image_shape_sub = self.image_shape
            self.flat_indices_sub = self.flat_indices
            self.image_x_active_sub = self.image_x_active
            self.image_y_active_sub = self.image_y_active

    @property
    def psf_response(self) -> PixelizedPSFResponse:
        """Shared fitted/rendered image response and source-stencil operations."""
        return self._psf_response

    def ray_trace_seed(self) -> tuple[Array, Array]:
        """Ray-trace seed mask pixels to the source plane.

        Returns ``(beta_x_seed, beta_y_seed)`` for all active pixels in the
        source seed mask.  These are the same coordinates used internally by
        :meth:`design_matrix` for bounding-box inference.
        """
        beta_x_seed, beta_y_seed = self.phys_model.deflection(
            x=self.image_x_seed, y=self.image_y_seed
        )
        return beta_x_seed, beta_y_seed

    def infer_source_bbox(
        self,
        beta_x: Array,
        beta_y: Array,
        padding: float | None = None,
        outlier_frac: float | None = None,
    ):
        """Infer a square source-plane bounding box.

        Parameters
        ----------
        beta_x, beta_y : Array
            Ray-traced source-plane coordinates.
        padding : float or None, optional
            Fractional padding.  When ``None``, uses
            ``sim_config.source_bbox_padding``.
        outlier_frac : float or None, optional
            Fraction of extreme points trimmed from each tail.  When
            ``None``, uses ``sim_config.source_bbox_outlier_frac``.

        Returns (xmin, xmax, ymin, ymax).
        """
        pad = (
            self.sim_config.source_bbox_padding
            if padding is None
            else padding
        )
        frac = (
            self.sim_config.source_bbox_outlier_frac
            if outlier_frac is None
            else outlier_frac
        )
        return infer_source_bbox(
            beta_x, beta_y, padding=pad, outlier_frac=frac
        )

    def build_mapping_matrix(
        self,
        *,
        source_bbox: tuple | None = None,
        use_subgrid: bool = False,
    ) -> Array:
        """Build Nd x Ns lens mapping matrix with the configured interpolation.

        Parameters
        ----------
        source_bbox : tuple, optional
            (xmin, xmax, ymin, ymax) bounding box.  Auto-inferred from
            ray-traced beta points when ``None``.
        use_subgrid : bool, optional
            Whether to use the sub-sampled image grid.
        """
        if use_subgrid and self.nsub > 1:
            x_active = self.image_x_active_sub
            y_active = self.image_y_active_sub
        else:
            x_active = self.image_x_active
            y_active = self.image_y_active

        beta_x, beta_y = self.phys_model.deflection(x=x_active, y=y_active)
        if source_bbox is None:
            source_bbox = self.infer_source_bbox(beta_x, beta_y)

        xmin, xmax, ymin, ymax = source_bbox
        if self.detach_bbox:
            xmin = jax.lax.stop_gradient(xmin)
            xmax = jax.lax.stop_gradient(xmax)
            ymin = jax.lax.stop_gradient(ymin)
            ymax = jax.lax.stop_gradient(ymax)

        source_x_axis, source_y_axis, _, _ = build_source_grid(
            self.source_n, xmin, xmax, ymin, ymax
        )
        return build_lens_mapping_matrix(
            beta_x, beta_y, source_x_axis, source_y_axis,
            interpolation=getattr(self.source_model, "interpolation", "bilinear"),
        )

    def build_lens_light_matrix(
        self,
        *,
        psf_kernel: Array | None = None,
    ) -> Array:
        """Build Nd x Nl lens-light basis matrix L.

        Evaluates each lens-light component on the image-plane grid,
        convolves with the PSF, and extracts the active (unmasked) pixels.
        The returned columns are normalized basis images with unit amplitude.

        Parameters
        ----------
        psf_kernel : Array, optional
            Override PSF kernel. If None, uses the simulator's PSF.

        Returns
        -------
        Array
            Lens-light basis matrix of shape (Nd, Nl).
        """
        if not self.has_lens_light:
            return jnp.zeros((self.flat_indices.shape[0], 0), dtype=jnp.float32)

        kernel = self.psf_kernel if psf_kernel is None else jnp.asarray(psf_kernel)
        H, W = self.image_shape

        # Evaluate each lens-light component on the sub-grid for accuracy,
        # then bin to native resolution to match the source simulation path.
        if self.nsub > 1:
            xgrid = jnp.asarray(self.sim_config.xgrid_sub)
            ygrid = jnp.asarray(self.sim_config.ygrid_sub)
        else:
            xgrid = jnp.asarray(self.sim_config.xgrid)
            ygrid = jnp.asarray(self.sim_config.ygrid)

        lens_images = []
        for light_model in self.phys_model.lens_light:
            img = light_model.light(x=xgrid, y=ygrid)
            lens_images.append(img)

        lens_stack = jnp.stack(lens_images, axis=-1)

        # Bin to native resolution when using sub-grid
        if self.nsub > 1:
            lens_stack = bin_image_general(lens_stack, self.nsub)  # (H, W, Nl)

        # PSF convolution for each component (vectorized)
        lens_convolved = jax.vmap(
            lambda x: jsp.signal.fftconvolve(x, kernel, mode="same"),
            in_axes=-1,
            out_axes=-1,
        )(lens_stack)

        # Extract active pixels and flatten
        lens_1d = lens_convolved.reshape(H * W, self.n_lens_light)
        L = lens_1d[self.flat_indices, :]  # (Nd, Nl)
        return L

    def source_stencil(self, source_bbox=None, *, _response=None) -> tuple[Array, Array, tuple]:
        """Ray-trace the render subpixels and prepare their interpolation stencil."""
        if _response is None or _response is self._psf_response:
            x, y = self._render_coords
        else:
            x, y = _response.ray_coordinates(self.sim_config)
        beta_x, beta_y = self.phys_model.deflection(x=x, y=y)
        if source_bbox is None:
            # Use seed mask for bounding box inference (dual-masking strategy)
            beta_x_seed, beta_y_seed = self.phys_model.deflection(
                x=self.image_x_seed, y=self.image_y_seed
            )
            source_bbox = self.infer_source_bbox(beta_x_seed, beta_y_seed)

        xmin, xmax, ymin, ymax = source_bbox
        if self.detach_bbox:
            xmin = jax.lax.stop_gradient(xmin)
            xmax = jax.lax.stop_gradient(xmax)
            ymin = jax.lax.stop_gradient(ymin)
            ymax = jax.lax.stop_gradient(ymax)

        source_x_axis, source_y_axis, _, _ = build_source_grid(
            self.source_n, xmin, xmax, ymin, ymax
        )
        weights, indices, _ = lens_mapping_operator_from(
            jnp.stack((beta_x.ravel(), beta_y.ravel()), axis=1),
            source_x_axis[0], source_x_axis[-1],
            source_y_axis[0], source_y_axis[-1], self.source_n,
            interpolation=getattr(self.source_model, "interpolation", "bilinear"),
        )

        return weights, indices, (xmin, xmax, ymin, ymax)

    def design_matrix(
        self,
        *,
        source_bbox: tuple | None = None,
        psf_kernel: Array | None = None,
        _source_stencil: tuple | None = None,
    ) -> tuple[Array, tuple]:
        """Build Nd x (Ns+Nl) PSF-convolved joint design matrix M = [F | L].

        When lens_light is absent, returns the source-only matrix F with
        shape (Nd, Ns).  When lens_light is present, returns the joint
        matrix [F | L] with shape (Nd, Ns+Nl).

        Returns (M, source_bbox) where source_bbox is (xmin, xmax, ymin, ymax).
        """
        response = self._psf_response if psf_kernel is None else self._psf_response.with_kernel(psf_kernel)
        if _source_stencil is None or psf_kernel is not None:
            _source_stencil = self.source_stencil(source_bbox, _response=response)
        weights, indices, source_bbox = _source_stencil
        F = response.convolved_columns(weights, indices, self.n_source_pixels).T

        # --- Joint with lens light if present ---
        if self.has_lens_light:
            L = self.build_lens_light_matrix(psf_kernel=psf_kernel)  # (N_d, Nl)
            M = jnp.concatenate([F, L], axis=1)  # (N_d, Ns+Nl)
            return M, source_bbox

        return F, source_bbox

    def simulate(
        self,
        source_pixels: Array,
        *,
        lens_light_amplitudes: Array | None = None,
        source_bbox: tuple | None = None,
        psf_kernel: Array | None = None,
    ) -> Array:
        """Return full 2D model image for given source and lens-light parameters.

        Parameters
        ----------
        source_pixels : Array
            Source pixel intensities, shape (Ns,).
        lens_light_amplitudes : Array, optional
            Lens light linear amplitudes, shape (Nl,). Required when
            ``has_lens_light`` is True.
        source_bbox : tuple, optional
            (xmin, xmax, ymin, ymax) bounding box.  Auto-inferred when ``None``.
        psf_kernel : Array, optional
            Override PSF kernel.

        Returns
        -------
        Array
            Full 2D model image.
        """
        response = self._psf_response if psf_kernel is None else self._psf_response.with_kernel(psf_kernel)
        weights, indices, _ = self.source_stencil(source_bbox, _response=response)
        brightness = render_source(
            jnp.asarray(source_pixels), weights, indices,
            response.render_parents, self.nsub, response.n_render,
        )
        fitted = response.forward(brightness)

        # Add lens light contribution if present
        if self.has_lens_light and lens_light_amplitudes is not None:
            L = self.build_lens_light_matrix(psf_kernel=psf_kernel)
            fitted = fitted + L @ jnp.asarray(lens_light_amplitudes)

        H, W = self.image_shape
        return jnp.zeros(H * W, dtype=fitted.dtype).at[self.flat_indices].set(fitted).reshape(H, W)

    def forward(
        self,
        *,
        source_pixels: Array | None = None,
        lens_light_amplitudes: Array | None = None,
        return_mapping: bool = False,
        return_image_2d: bool = True,
        psf_kernel: Array | None = None,
    ) -> SimulationResult:
        """Forward model compatible with the parametric simulator API."""
        if not return_image_2d:
            raise ValueError("Pixelized forward() always returns 2D images")

        # Infer bbox from seed mask first so that mapping_matrix and
        # simulate use the same source-plane grid.
        beta_x_seed, beta_y_seed = self.phys_model.deflection(
            x=self.image_x_seed, y=self.image_y_seed
        )
        source_bbox = self.infer_source_bbox(beta_x_seed, beta_y_seed)

        mapping_matrix = None
        if source_pixels is None or return_mapping:
            mapping_matrix_sub = self.build_mapping_matrix(
                source_bbox=source_bbox, use_subgrid=(self.nsub > 1)
            )
            mapping_matrix = self.psf_response.integrate_fit_rays(mapping_matrix_sub)
        if source_pixels is None:
            source_pixels = jnp.zeros(self.n_source_pixels, dtype=mapping_matrix.dtype)

        model_image = self.simulate(
            source_pixels,
            lens_light_amplitudes=lens_light_amplitudes,
            source_bbox=source_bbox,
            psf_kernel=psf_kernel,
        )
        return SimulationResult(model_image=model_image, source_image=None, mapping_matrix=mapping_matrix, linear_params=None)

    def __repr__(self) -> str:
        return (f"PixelizedLensSimulator(image_shape={self.image_shape}, "
                f"source_shape=({self.source_n}, {self.source_n}), "
                f"n_active={int(self.flat_indices.shape[0])}, "
                f"n_lens_light={self.n_lens_light})")


__all__ = ["PixelizedLensSimulator", "EPSILON_REG"]
