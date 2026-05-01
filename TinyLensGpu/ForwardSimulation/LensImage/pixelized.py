"""Pixelized source forward simulator for lens-image modeling."""

# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

from ...PhysicalModel.LensImage.composite import PhysicalModel
from ...PhysicalModel.LensImage.Pixelized.Light.pixelized_source import PixelizedSourceModel
from ...utils.pixelized_source_utils import build_lens_mapping_matrix, build_source_grid
from .config import SimulatorConfig
from .results import SimulationResult


class PixelizedLensSimulator:
    """Forward simulator for a single pixelized source and no lens light.

    The simulator ray-traces native image pixels to the source plane, builds the
    dense bilinear image-to-source mapping matrix, and applies PSF convolution
    directly to 2D basis images with FFT convolution. The first implementation
    intentionally excludes sub-sampling, explicit PSF matrices, lens light, and
    multiple or mixed source components.

    Parameters
    ----------
    phys_model : PhysicalModel
        Physical model containing lens mass and exactly one pixelized source
        configuration object with ``nx`` and ``ny`` attributes.
    sim_config : SimulatorConfig
        Image grid, PSF, and mask configuration. ``nsub`` must be 1.

    Raises
    ------
    ValueError
        If the physical model violates the first-version pixelized restrictions
        or if sub-sampling is requested.
    """

    def __init__(self, phys_model: PhysicalModel, sim_config: SimulatorConfig) -> None:
        """Initialize the pixelized lens simulator.

        Parameters
        ----------
        phys_model : PhysicalModel
            Model with lens mass, one pixelized source, and no lens light.
        sim_config : SimulatorConfig
            Native image-grid configuration used by this simulator.
        """
        self.phys_model = phys_model
        self.sim_config = sim_config

        if self.sim_config.nsub != 1:
            raise ValueError("PixelizedLensSimulator does not support nsub in this version")

        if len(self.phys_model.lens_light) != 0:
            raise ValueError("PixelizedLensSimulator does not support lens_light in this version")

        n_pixelized = sum(self._is_pixelized_source_model(source) for source in self.phys_model.source_light)
        if n_pixelized != len(self.phys_model.source_light):
            raise ValueError("PixelizedLensSimulator does not support a parametric source")
        if n_pixelized != 1:
            raise ValueError("PixelizedLensSimulator requires a single pixelized source")

        source_model = self.phys_model.source_light[0]
        self.source_model = source_model
        self.source_nx = int(source_model.nx)
        self.source_ny = int(source_model.ny)
        self.n_source_pixels = self.source_nx * self.source_ny

        self.image_shape = tuple(self.sim_config.mask.shape)
        self.mask = jnp.asarray(self.sim_config.mask, dtype=bool)
        self.active_mask = ~self.mask
        self.flat_indices = jnp.flatnonzero(self.active_mask.ravel())
        self.image_x_active = jnp.asarray(self.sim_config.xgrid)[self.active_mask]
        self.image_y_active = jnp.asarray(self.sim_config.ygrid)[self.active_mask]
        self.psf_kernel = jnp.asarray(self.sim_config.psf_kernel)

    @staticmethod
    def _is_pixelized_source_model(source_model: Any) -> bool:
        """Return whether a source object is a pixelized source model.

        Parameters
        ----------
        source_model : Any
            Source component object to validate.

        Returns
        -------
        bool
            ``True`` when the source carries the ``is_pixelized_source`` marker
            or is a :class:`PixelizedSourceModel` instance.
        """
        if hasattr(source_model, "is_pixelized_source"):
            return bool(getattr(source_model, "is_pixelized_source"))
        return isinstance(source_model, PixelizedSourceModel)

    def _compute_deflection(self) -> tuple[Array, Array]:
        """Compute source-plane deflection for active image pixels.

        Returns
        -------
        tuple[Array, Array]
            ``(beta_x, beta_y)`` ray-traced source-plane coordinates.
        """
        return self.phys_model.deflection(x=self.image_x_active, y=self.image_y_active)

    def infer_source_half_size(self, beta_x: Array, beta_y: Array) -> Array:
        """Return 1.05 * max(|beta_x|, |beta_y|) with numerical floor 1e-6.

        Parameters
        ----------
        beta_x, beta_y : Array
            Ray-traced source-plane coordinates.

        Returns
        -------
        Array
            Positive scalar source half-size with a numerical floor.
        """
        beta_x_extent = jnp.max(jnp.abs(beta_x))
        beta_y_extent = jnp.max(jnp.abs(beta_y))
        source_half_size = 1.05 * jnp.maximum(beta_x_extent, beta_y_extent)
        return jnp.maximum(source_half_size, 1.0e-6)

    def build_mapping_matrix(self, *, source_half_size: Array | float | None = None) -> Array:
        """Build Nd x Ns lens mapping matrix L via bilinear interpolation.

        If ``source_half_size`` is ``None``, the source grid size is inferred
        from the current ray-traced active image coordinates.

        Parameters
        ----------
        source_half_size : Array or float, optional
            Source-plane half-size used to build the rectangular source grid.

        Returns
        -------
        Array
            Dense mapping matrix with one row per unmasked image pixel and one
            column per source pixel.
        """
        beta_x, beta_y = self._compute_deflection()
        if source_half_size is None:
            source_half_size = self.infer_source_half_size(beta_x, beta_y)

        source_x_axis, source_y_axis, _, _ = build_source_grid(
            self.source_nx,
            self.source_ny,
            source_half_size,
        )
        return build_lens_mapping_matrix(beta_x, beta_y, source_x_axis, source_y_axis)

    def design_matrix(
        self,
        *,
        source_half_size: Array | float | None = None,
        psf_kernel: Array | None = None,
    ) -> tuple[Array, Array]:
        """Build Nd x Ns PSF-convolved design matrix M.

        The mapping matrix columns are scattered into native 2D basis images and
        convolved with ``jax.scipy.signal.fftconvolve``. No explicit PSF blur
        matrix is constructed.

        Parameters
        ----------
        source_half_size : Array or float, optional
            Source half-size. If omitted, infer it from the current deflection.
        psf_kernel : Array, optional
            PSF kernel. Defaults to ``sim_config.psf_kernel``.

        Returns
        -------
        tuple[Array, Array]
            ``(M, inferred_source_half_size)`` where ``M`` has shape
            ``(N_d, N_s)`` and includes PSF convolution.
        """
        beta_x, beta_y = self._compute_deflection()
        if source_half_size is None:
            source_half_size = self.infer_source_half_size(beta_x, beta_y)

        source_x_axis, source_y_axis, _, _ = build_source_grid(
            self.source_nx,
            self.source_ny,
            source_half_size,
        )
        mapping_matrix = build_lens_mapping_matrix(beta_x, beta_y, source_x_axis, source_y_axis)

        kernel = self.psf_kernel if psf_kernel is None else jnp.asarray(psf_kernel)

        def convolve_basis(column: Array) -> Array:
            """Scatter and convolve one source-pixel basis image."""
            image = jnp.zeros(self.image_shape, dtype=column.dtype)
            image = image.at[self.active_mask].set(column)
            convolved = jsp.signal.fftconvolve(image, kernel, mode="same")
            return convolved[self.active_mask]

        design_matrix = jax.vmap(convolve_basis, in_axes=1, out_axes=1)(mapping_matrix)
        return design_matrix, jnp.asarray(source_half_size)

    def simulate(
        self,
        source_pixels: Array,
        *,
        source_half_size: Array | float | None = None,
        psf_kernel: Array | None = None,
    ) -> Array:
        """Return the full 2D model image for pixelized source pixels.

        Steps are ``L @ source_pixels``, scatter to the unmasked native image
        pixels, then FFT-convolve with the requested PSF kernel.

        Parameters
        ----------
        source_pixels : Array
            Flattened source-plane pixel intensities with length ``nx * ny``.
        source_half_size : Array or float, optional
            Source half-size. If omitted, infer it from the current deflection.
        psf_kernel : Array, optional
            PSF kernel. Defaults to ``sim_config.psf_kernel``.

        Returns
        -------
        Array
            Full 2D model image with shape matching ``sim_config.mask``.
        """
        source_pixels = jnp.asarray(source_pixels)
        mapping_matrix = self.build_mapping_matrix(source_half_size=source_half_size)
        m_ideal_1d = mapping_matrix @ source_pixels

        model_image = jnp.zeros(self.image_shape, dtype=m_ideal_1d.dtype)
        model_image = model_image.at[self.active_mask].set(m_ideal_1d)

        kernel = self.psf_kernel if psf_kernel is None else jnp.asarray(psf_kernel)
        return jsp.signal.fftconvolve(model_image, kernel, mode="same")

    def forward(
        self,
        *,
        source_pixels: Array | None = None,
        return_mapping: bool = False,
        return_image_2d: bool = True,
        psf_kernel: Array | None = None,
    ) -> SimulationResult:
        """Main forward method compatible with the parametric simulator API.

        If ``source_pixels`` is omitted, a mapping matrix is built for side
        effects and a zero source is simulated. When ``return_mapping`` is true,
        the mapping matrix is returned through the ``mapping_matrix`` field.

        Parameters
        ----------
        source_pixels : Array, optional
            Flattened source pixel intensities. Defaults to zeros.
        return_mapping : bool, optional
            Whether to include the unconvolved mapping matrix in the result.
        return_image_2d : bool, optional
            Must be ``True`` in this implementation.
        psf_kernel : Array, optional
            PSF kernel. Defaults to ``sim_config.psf_kernel``.

        Returns
        -------
        SimulationResult
            Result containing the 2D model image and no linear parameters.
        """
        if not return_image_2d:
            raise ValueError("Pixelized forward() always returns 2D images")

        mapping_matrix = None
        if source_pixels is None:
            mapping_matrix = self.build_mapping_matrix()
            source_pixels = jnp.zeros(self.n_source_pixels, dtype=mapping_matrix.dtype)

        model_image = self.simulate(source_pixels, psf_kernel=psf_kernel)
        if return_mapping and mapping_matrix is None:
            mapping_matrix = self.build_mapping_matrix()

        return SimulationResult(
            model_image=model_image,
            source_image=None,
            mapping_matrix=mapping_matrix,
            linear_params=None,
        )

    def __repr__(self) -> str:
        """Return a concise simulator summary.

        Returns
        -------
        str
            Human-readable summary of image and source-grid dimensions.
        """
        return (
            "PixelizedLensSimulator("
            f"image_shape={self.image_shape}, "
            f"source_shape=({self.source_ny}, {self.source_nx}), "
            f"n_active={int(self.flat_indices.shape[0])})"
        )


__all__ = ["PixelizedLensSimulator"]
