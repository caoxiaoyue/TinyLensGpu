"""Pixelized source forward simulator for lens-image modeling."""

# pyright: reportMissingImports=false

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

from ...PhysicalModel.LensImage.composite import PhysicalModel
from ...PhysicalModel.LensImage.Pixelized.Light.pixelized_source import PixelizedSourceModel
from ...utils.lensing.mapping import build_lens_mapping_matrix, build_source_grid
from .config import SimulatorConfig
from .parametric import bin_image_general
from .results import SimulationResult


def _is_pixelized_source_model(source: object) -> bool:
    """Check whether a source component carries the pixelized-source marker."""
    if hasattr(source, "is_pixelized_source"):
        return bool(getattr(source, "is_pixelized_source"))
    return isinstance(source, PixelizedSourceModel)


class PixelizedLensSimulator:
    """Forward simulator for a single pixelized source and no lens light.

    Ray-traces image-plane pixels to the source plane, builds the dense
    bilinear mapping matrix, and convolves with PSF via FFT.  When
    ``nsub > 1`` the ray-tracing uses a sub-sampled grid and aggregates
    the mapping back to native resolution before PSF convolution.

    Parameters
    ----------
    phys_model : PhysicalModel
        Model with lens mass and exactly one pixelized source.
    sim_config : SimulatorConfig
        Image grid, PSF, mask, and subsampling (``nsub``) configuration.

    Raises
    ------
    ValueError
        If the model has mixed sources, multiple sources, or lens light.
    """

    def __init__(self, phys_model: PhysicalModel, sim_config: SimulatorConfig) -> None:
        self.phys_model = phys_model
        self.sim_config = sim_config
        self.nsub = int(sim_config.nsub)

        if len(phys_model.lens_light) != 0:
            raise ValueError("PixelizedLensSimulator does not support lens_light in this version")

        n_pixelized = sum(_is_pixelized_source_model(s) for s in phys_model.source_light)
        if n_pixelized != len(phys_model.source_light):
            raise ValueError("PixelizedLensSimulator does not support a parametric source")
        if n_pixelized != 1:
            raise ValueError("PixelizedLensSimulator requires a single pixelized source")

        source = phys_model.source_light[0]
        self.source_model = source
        self.source_nx = int(source.nx)
        self.source_ny = int(source.ny)
        self.n_source_pixels = self.source_nx * self.source_ny

        self.image_shape = tuple(sim_config.mask.shape)
        self.mask = jnp.asarray(sim_config.mask, dtype=bool)
        self.active_mask = ~self.mask
        self.flat_indices = jnp.flatnonzero(self.active_mask.ravel())
        self.image_x_active = jnp.asarray(sim_config.xgrid)[self.active_mask]
        self.image_y_active = jnp.asarray(sim_config.ygrid)[self.active_mask]
        self.psf_kernel = jnp.asarray(sim_config.psf_kernel)

        if self.nsub > 1:
            self.image_shape_sub = sim_config.subgrid_shape
            self.flat_indices_sub = sim_config.flat_indices
            self.image_x_active_sub = jnp.asarray(sim_config.xgrid_sub_1d)
            self.image_y_active_sub = jnp.asarray(sim_config.ygrid_sub_1d)
            # Precompute sub→native index mapping for _aggregate_mapping_to_native.
            # Each sub-pixel maps to a native pixel; we store the native-pixel index
            # relative to the active-pixel list so segment_sum can work directly on
            # the (Nd_active,) output without allocating a full (npix*npix,) buffer.
            npix = sim_config.npix
            nsub = self.nsub
            fis = self.flat_indices_sub          # sub-grid active flat indices
            i_native = (fis // (npix * nsub)) // nsub
            j_native = (fis % (npix * nsub)) // nsub
            native_flat_all = i_native * npix + j_native  # native flat index (0..npix²-1)
            # Map native flat index → position in the active-pixel list.
            # flat_indices is sorted (from jnp.flatnonzero), so searchsorted
            # gives the exact index of each native pixel in the active list.
            self._agg_segment_ids = jnp.searchsorted(
                self.flat_indices, native_flat_all
            ).astype(jnp.int32)                 # shape (Nd_sub,)
            self._agg_n_active = int(self.flat_indices.shape[0])
        else:
            self.image_shape_sub = self.image_shape
            self.flat_indices_sub = self.flat_indices
            self.image_x_active_sub = self.image_x_active
            self.image_y_active_sub = self.image_y_active

        # Precompute PSF FFT once — PSF is fixed for the lifetime of this object.
        H, W = self.image_shape
        kernel = self.psf_kernel
        psf_pad = jnp.zeros(self.image_shape, dtype=kernel.dtype)
        psf_pad = psf_pad.at[: kernel.shape[0], : kernel.shape[1]].set(kernel)
        psf_shifted = jnp.roll(psf_pad, (-(kernel.shape[0] // 2), -(kernel.shape[1] // 2)), axis=(0, 1))
        self._psf_fft = jnp.fft.rfft2(psf_shifted)  # (H, W//2+1)

    def infer_source_half_size(self, beta_x: Array, beta_y: Array) -> Array:
        """1.05 * max(|beta_x|, |beta_y|) floored at 1e-6."""
        return jnp.maximum(1.05 * jnp.maximum(jnp.max(jnp.abs(beta_x)), jnp.max(jnp.abs(beta_y))), 1.0e-6)

    def build_mapping_matrix(
        self,
        *,
        source_half_size: Array | float | None = None,
        use_subgrid: bool = False,
    ) -> Array:
        """Build Nd x Ns lens mapping matrix via bilinear interpolation."""
        if use_subgrid and self.nsub > 1:
            x_active = self.image_x_active_sub
            y_active = self.image_y_active_sub
        else:
            x_active = self.image_x_active
            y_active = self.image_y_active

        beta_x, beta_y = self.phys_model.deflection(x=x_active, y=y_active)
        if source_half_size is None:
            source_half_size = self.infer_source_half_size(beta_x, beta_y)

        source_x_axis, source_y_axis, _, _ = build_source_grid(self.source_nx, self.source_ny, source_half_size)
        return build_lens_mapping_matrix(beta_x, beta_y, source_x_axis, source_y_axis)

    def _aggregate_mapping_to_native(self, mapping_matrix_sub: Array) -> Array:
        """Average a sub-grid mapping matrix back to native resolution.

        Parameters
        ----------
        mapping_matrix_sub : Array
            Mapping matrix of shape (Nd_sub, Ns) built on the sub-grid.

        Returns
        -------
        Array
            Mapping matrix of shape (Nd, Ns) at native resolution.
        """
        # Use precomputed segment IDs to sum sub-pixels into native active pixels
        # directly, avoiding a full (npix*npix, Ns) scatter buffer.
        summed = jax.ops.segment_sum(
            mapping_matrix_sub,
            self._agg_segment_ids,
            num_segments=self._agg_n_active,
        )
        return summed / (self.nsub ** 2)

    def design_matrix(
        self,
        *,
        source_half_size: Array | float | None = None,
        psf_kernel: Array | None = None,
    ) -> tuple[Array, Array]:
        """Build Nd x Ns PSF-convolved design matrix M.

        Returns (M, inferred_source_half_size).
        """
        beta_x, beta_y = self.phys_model.deflection(
            x=self.image_x_active_sub, y=self.image_y_active_sub
        )
        if source_half_size is None:
            source_half_size = self.infer_source_half_size(beta_x, beta_y)

        source_x_axis, source_y_axis, _, _ = build_source_grid(self.source_nx, self.source_ny, source_half_size)
        mapping_matrix_sub = build_lens_mapping_matrix(beta_x, beta_y, source_x_axis, source_y_axis)

        if self.nsub > 1:
            mapping_matrix = self._aggregate_mapping_to_native(mapping_matrix_sub)
        else:
            mapping_matrix = mapping_matrix_sub

        kernel = self.psf_kernel if psf_kernel is None else jnp.asarray(psf_kernel)

        # Use precomputed PSF FFT when the kernel hasn't been overridden.
        H, W = self.image_shape
        if psf_kernel is None:
            psf_fft = self._psf_fft
        else:
            psf_pad = jnp.zeros(self.image_shape, dtype=kernel.dtype)
            psf_pad = psf_pad.at[: kernel.shape[0], : kernel.shape[1]].set(kernel)
            psf_shifted = jnp.roll(psf_pad, (-(kernel.shape[0] // 2), -(kernel.shape[1] // 2)), axis=(0, 1))
            psf_fft = jnp.fft.rfft2(psf_shifted)

        # Scatter all N_s columns into images at once: (N_s, H, W)
        imgs = jnp.zeros((self.n_source_pixels, H * W), dtype=mapping_matrix.dtype)
        imgs = imgs.at[:, self.flat_indices].set(mapping_matrix.T)
        imgs = imgs.reshape(self.n_source_pixels, H, W)

        # Single batched rfft2, multiply, irfft2
        imgs_fft = jnp.fft.rfft2(imgs)  # (N_s, H, W//2+1)
        conv_imgs = jnp.fft.irfft2(imgs_fft * psf_fft[None], s=(H, W))  # (N_s, H, W)

        conv_imgs_flat = conv_imgs.reshape(self.n_source_pixels, H * W)
        design_matrix = conv_imgs_flat[:, self.flat_indices].T  # (N_d, N_s)
        return design_matrix, jnp.asarray(source_half_size)

    def simulate(self, source_pixels: Array, *, source_half_size: Array | float | None = None, psf_kernel: Array | None = None) -> Array:
        """Return full 2D model image for given source pixels."""
        source_pixels = jnp.asarray(source_pixels)
        mapping_matrix_sub = self.build_mapping_matrix(source_half_size=source_half_size, use_subgrid=True)
        m_ideal_1d_sub = mapping_matrix_sub @ source_pixels

        if self.nsub > 1:
            H_sub, W_sub = self.image_shape_sub
            model_image_sub = jnp.zeros((H_sub * W_sub,), dtype=m_ideal_1d_sub.dtype)
            model_image_sub = model_image_sub.at[self.flat_indices_sub].set(m_ideal_1d_sub)
            model_image_sub = model_image_sub.reshape(H_sub, W_sub)
            model_image = bin_image_general(model_image_sub, self.nsub)
        else:
            H, W = self.image_shape
            model_image = jnp.zeros(H * W, dtype=m_ideal_1d_sub.dtype)
            model_image = model_image.at[self.flat_indices].set(m_ideal_1d_sub)
            model_image = model_image.reshape(H, W)

        kernel = self.psf_kernel if psf_kernel is None else jnp.asarray(psf_kernel)
        convolved_image = jsp.signal.fftconvolve(model_image, kernel, mode="same")

        return jnp.where(self.active_mask, convolved_image, jnp.zeros_like(convolved_image))

    def forward(self, *, source_pixels: Array | None = None, return_mapping: bool = False, return_image_2d: bool = True, psf_kernel: Array | None = None) -> SimulationResult:
        """Forward model compatible with the parametric simulator API."""
        if not return_image_2d:
            raise ValueError("Pixelized forward() always returns 2D images")

        mapping_matrix = None
        if source_pixels is None:
            mapping_matrix_sub = self.build_mapping_matrix(use_subgrid=(self.nsub > 1))
            if self.nsub > 1:
                mapping_matrix = self._aggregate_mapping_to_native(mapping_matrix_sub)
            else:
                mapping_matrix = mapping_matrix_sub
            source_pixels = jnp.zeros(self.n_source_pixels, dtype=mapping_matrix.dtype)

        model_image = self.simulate(source_pixels, psf_kernel=psf_kernel)
        if return_mapping and mapping_matrix is None:
            mapping_matrix_sub = self.build_mapping_matrix(use_subgrid=(self.nsub > 1))
            if self.nsub > 1:
                mapping_matrix = self._aggregate_mapping_to_native(mapping_matrix_sub)
            else:
                mapping_matrix = mapping_matrix_sub

        return SimulationResult(model_image=model_image, source_image=None, mapping_matrix=mapping_matrix, linear_params=None)

    def __repr__(self) -> str:
        return (f"PixelizedLensSimulator(image_shape={self.image_shape}, "
                f"source_shape=({self.source_ny}, {self.source_nx}), "
                f"n_active={int(self.flat_indices.shape[0])})")


__all__ = ["PixelizedLensSimulator"]
