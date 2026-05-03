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
from .results import SimulationResult


def _is_pixelized_source_model(source: object) -> bool:
    """Check whether a source component carries the pixelized-source marker."""
    if hasattr(source, "is_pixelized_source"):
        return bool(getattr(source, "is_pixelized_source"))
    return isinstance(source, PixelizedSourceModel)


class PixelizedLensSimulator:
    """Forward simulator for a single pixelized source and no lens light.

    Ray-traces native image pixels to the source plane, builds the dense
    bilinear mapping matrix, and convolves with PSF via FFT.

    Parameters
    ----------
    phys_model : PhysicalModel
        Model with lens mass and exactly one pixelized source. ``nsub`` must be 1.
    sim_config : SimulatorConfig
        Image grid, PSF, and mask configuration.

    Raises
    ------
    ValueError
        If the model has mixed sources, multiple sources, lens light, or nsub != 1.
    """

    def __init__(self, phys_model: PhysicalModel, sim_config: SimulatorConfig) -> None:
        self.phys_model = phys_model
        self.sim_config = sim_config

        if sim_config.nsub != 1:
            raise ValueError("PixelizedLensSimulator does not support nsub in this version")
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

    def infer_source_half_size(self, beta_x: Array, beta_y: Array) -> Array:
        """1.05 * max(|beta_x|, |beta_y|) floored at 1e-6."""
        return jnp.maximum(1.05 * jnp.maximum(jnp.max(jnp.abs(beta_x)), jnp.max(jnp.abs(beta_y))), 1.0e-6)

    def build_mapping_matrix(self, *, source_half_size: Array | float | None = None) -> Array:
        """Build Nd x Ns lens mapping matrix via bilinear interpolation."""
        beta_x, beta_y = self.phys_model.deflection(x=self.image_x_active, y=self.image_y_active)
        if source_half_size is None:
            source_half_size = self.infer_source_half_size(beta_x, beta_y)

        source_x_axis, source_y_axis, _, _ = build_source_grid(self.source_nx, self.source_ny, source_half_size)
        return build_lens_mapping_matrix(beta_x, beta_y, source_x_axis, source_y_axis)

    def design_matrix(
        self,
        *,
        source_half_size: Array | float | None = None,
        psf_kernel: Array | None = None,
    ) -> tuple[Array, Array]:
        """Build Nd x Ns PSF-convolved design matrix M.

        Returns (M, inferred_source_half_size).
        """
        beta_x, beta_y = self.phys_model.deflection(x=self.image_x_active, y=self.image_y_active)
        if source_half_size is None:
            source_half_size = self.infer_source_half_size(beta_x, beta_y)

        source_x_axis, source_y_axis, _, _ = build_source_grid(self.source_nx, self.source_ny, source_half_size)
        mapping_matrix = build_lens_mapping_matrix(beta_x, beta_y, source_x_axis, source_y_axis)

        kernel = self.psf_kernel if psf_kernel is None else jnp.asarray(psf_kernel)

        # Precompute PSF FFT once for all columns.
        H, W = self.image_shape
        psf_pad = jnp.zeros(self.image_shape, dtype=kernel.dtype)
        psf_pad = psf_pad.at[: kernel.shape[0], : kernel.shape[1]].set(kernel)
        psf_shifted = jnp.roll(psf_pad, (-(kernel.shape[0] // 2), -(kernel.shape[1] // 2)), axis=(0, 1))
        psf_fft = jnp.fft.rfft2(psf_shifted)  # (H, W//2+1)

        # Scatter all N_s columns into images at once: (N_s, H, W)
        imgs = jnp.zeros((self.n_source_pixels, H, W), dtype=mapping_matrix.dtype)
        imgs = imgs.at[:, self.active_mask].set(mapping_matrix.T)

        # Single batched rfft2, multiply, irfft2
        imgs_fft = jnp.fft.rfft2(imgs)  # (N_s, H, W//2+1)
        conv_imgs = jnp.fft.irfft2(imgs_fft * psf_fft[None], s=(H, W))  # (N_s, H, W)

        design_matrix = conv_imgs[:, self.active_mask].T  # (N_d, N_s)
        return design_matrix, jnp.asarray(source_half_size)

    def simulate(self, source_pixels: Array, *, source_half_size: Array | float | None = None, psf_kernel: Array | None = None) -> Array:
        """Return full 2D model image for given source pixels."""
        source_pixels = jnp.asarray(source_pixels)
        mapping_matrix = self.build_mapping_matrix(source_half_size=source_half_size)
        m_ideal_1d = mapping_matrix @ source_pixels

        model_image = jnp.zeros(self.image_shape, dtype=m_ideal_1d.dtype)
        model_image = model_image.at[self.active_mask].set(m_ideal_1d)

        kernel = self.psf_kernel if psf_kernel is None else jnp.asarray(psf_kernel)
        return jsp.signal.fftconvolve(model_image, kernel, mode="same")

    def forward(self, *, source_pixels: Array | None = None, return_mapping: bool = False, return_image_2d: bool = True, psf_kernel: Array | None = None) -> SimulationResult:
        """Forward model compatible with the parametric simulator API."""
        if not return_image_2d:
            raise ValueError("Pixelized forward() always returns 2D images")

        mapping_matrix = None
        if source_pixels is None:
            mapping_matrix = self.build_mapping_matrix()
            source_pixels = jnp.zeros(self.n_source_pixels, dtype=mapping_matrix.dtype)

        model_image = self.simulate(source_pixels, psf_kernel=psf_kernel)
        if return_mapping and mapping_matrix is None:
            mapping_matrix = self.build_mapping_matrix()

        return SimulationResult(model_image=model_image, source_image=None, mapping_matrix=mapping_matrix, linear_params=None)

    def __repr__(self) -> str:
        return (f"PixelizedLensSimulator(image_shape={self.image_shape}, "
                f"source_shape=({self.source_ny}, {self.source_nx}), "
                f"n_active={int(self.flat_indices.shape[0])})")


__all__ = ["PixelizedLensSimulator"]
