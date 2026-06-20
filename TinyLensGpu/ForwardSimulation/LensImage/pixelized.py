"""Pixelized source forward simulator for lens-image modeling."""

# pyright: reportMissingImports=false

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

from ...PhysicalModel.LensImage.composite import PhysicalModel
from ...PhysicalModel.LensImage.Pixelized.Light.pixelized_source import PixelizedSourceModel
from ...utils.lensing.mapping import build_lens_mapping_matrix, build_source_grid, infer_source_bbox
from .config import SimulatorConfig
from .parametric import bin_image_general
from .results import SimulationResult


def _is_pixelized_source_model(source: object) -> bool:
    """Check whether a source component carries the pixelized-source marker."""
    if hasattr(source, "is_pixelized_source"):
        return bool(getattr(source, "is_pixelized_source"))
    return isinstance(source, PixelizedSourceModel)


EPSILON_REG = 1e-6  # Tiny Tikhonov regularization for lens-light amplitudes


class PixelizedLensSimulator:
    """Forward simulator for a single pixelized source and optional lens light.

    Ray-traces image-plane pixels to the source plane, builds the dense
    bilinear mapping matrix, and convolves with PSF via FFT.  When
    ``nsub > 1`` the ray-tracing uses a sub-sampled grid and aggregates
    the mapping back to native resolution before PSF convolution.

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
        not produce unstable gradients.  Beta ray-tracing and bilinear
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

        # Lens light
        self.n_lens_light = len(phys_model.lens_light)
        self.has_lens_light = self.n_lens_light > 0

        self.image_shape = tuple(sim_config.mask.shape)
        self.mask = jnp.asarray(sim_config.mask, dtype=bool)
        self.active_mask = ~self.mask
        self.flat_indices = jnp.flatnonzero(self.active_mask.ravel())
        self.image_x_active = jnp.asarray(sim_config.xgrid)[self.active_mask]
        self.image_y_active = jnp.asarray(sim_config.ygrid)[self.active_mask]
        self.psf_kernel = jnp.asarray(sim_config.psf_kernel)

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

        # Precompute PSF FFT once
        H, W = self.image_shape
        kernel = self.psf_kernel
        psf_pad = jnp.zeros(self.image_shape, dtype=kernel.dtype)
        psf_pad = psf_pad.at[: kernel.shape[0], : kernel.shape[1]].set(kernel)
        psf_shifted = jnp.roll(psf_pad, (-(kernel.shape[0] // 2), -(kernel.shape[1] // 2)), axis=(0, 1))
        self._psf_fft = jnp.fft.rfft2(psf_shifted)  # (H, W//2+1)

    def infer_source_bbox(
        self,
        beta_x: Array,
        beta_y: Array,
        padding: float | None = None,
        outlier_frac: float | None = None,
    ):
        """Infer source-plane bounding box.

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
        """Build Nd x Ns lens mapping matrix via bilinear interpolation.

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
            self.source_nx, self.source_ny, xmin, xmax, ymin, ymax
        )
        return build_lens_mapping_matrix(beta_x, beta_y, source_x_axis, source_y_axis)

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
        source_bbox: tuple | None = None,
        psf_kernel: Array | None = None,
    ) -> tuple[Array, tuple]:
        """Build Nd x (Ns+Nl) PSF-convolved joint design matrix M = [F | L].

        When lens_light is absent, returns the source-only matrix F with
        shape (Nd, Ns).  When lens_light is present, returns the joint
        matrix [F | L] with shape (Nd, Ns+Nl).

        Returns (M, source_bbox) where source_bbox is (xmin, xmax, ymin, ymax).
        """
        # --- Source mapping matrix F ---
        beta_x, beta_y = self.phys_model.deflection(
            x=self.image_x_active_sub, y=self.image_y_active_sub
        )
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
            self.source_nx, self.source_ny, xmin, xmax, ymin, ymax
        )
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
        F = conv_imgs_flat[:, self.flat_indices].T  # (N_d, N_s)

        # --- Joint with lens light if present ---
        if self.has_lens_light:
            L = self.build_lens_light_matrix(psf_kernel=psf_kernel)  # (N_d, Nl)
            M = jnp.concatenate([F, L], axis=1)  # (N_d, Ns+Nl)
            return M, (xmin, xmax, ymin, ymax)

        return F, (xmin, xmax, ymin, ymax)

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
        source_pixels = jnp.asarray(source_pixels)

        # Infer bbox using seed mask coordinates (same as design_matrix)
        if source_bbox is None:
            beta_x_seed, beta_y_seed = self.phys_model.deflection(
                x=self.image_x_seed, y=self.image_y_seed
            )
            source_bbox = self.infer_source_bbox(beta_x_seed, beta_y_seed)

        mapping_matrix_sub = self.build_mapping_matrix(
            source_bbox=source_bbox, use_subgrid=True
        )
        m_ideal_1d_sub = mapping_matrix_sub @ source_pixels

        H, W = self.image_shape
        if self.nsub > 1:
            H_sub, W_sub = self.image_shape_sub
            model_image_sub = jnp.zeros((H_sub * W_sub,), dtype=m_ideal_1d_sub.dtype)
            model_image_sub = model_image_sub.at[self.flat_indices_sub].set(m_ideal_1d_sub)
            model_image_sub = model_image_sub.reshape(H_sub, W_sub)
            model_image = bin_image_general(model_image_sub, self.nsub)
        else:
            model_image = jnp.zeros(H * W, dtype=m_ideal_1d_sub.dtype)
            model_image = model_image.at[self.flat_indices].set(m_ideal_1d_sub)
            model_image = model_image.reshape(H, W)

        kernel = self.psf_kernel if psf_kernel is None else jnp.asarray(psf_kernel)
        convolved_image = jsp.signal.fftconvolve(model_image, kernel, mode="same")

        # Add lens light contribution if present
        if self.has_lens_light and lens_light_amplitudes is not None:
            lens_light_amplitudes = jnp.asarray(lens_light_amplitudes)
            L = self.build_lens_light_matrix(psf_kernel=psf_kernel)
            lens_1d = L @ lens_light_amplitudes
            lens_image = jnp.zeros(H * W, dtype=lens_1d.dtype)
            lens_image = lens_image.at[self.flat_indices].set(lens_1d)
            lens_image = lens_image.reshape(H, W)
            convolved_image = convolved_image + lens_image

        return jnp.where(self.active_mask, convolved_image, jnp.zeros_like(convolved_image))

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
        if source_pixels is None:
            mapping_matrix_sub = self.build_mapping_matrix(
                source_bbox=source_bbox, use_subgrid=(self.nsub > 1)
            )
            if self.nsub > 1:
                mapping_matrix = self._aggregate_mapping_to_native(mapping_matrix_sub)
            else:
                mapping_matrix = mapping_matrix_sub
            source_pixels = jnp.zeros(self.n_source_pixels, dtype=mapping_matrix.dtype)

        model_image = self.simulate(
            source_pixels,
            lens_light_amplitudes=lens_light_amplitudes,
            source_bbox=source_bbox,
            psf_kernel=psf_kernel,
        )
        if return_mapping and mapping_matrix is None:
            mapping_matrix_sub = self.build_mapping_matrix(
                source_bbox=source_bbox, use_subgrid=(self.nsub > 1)
            )
            if self.nsub > 1:
                mapping_matrix = self._aggregate_mapping_to_native(mapping_matrix_sub)
            else:
                mapping_matrix = mapping_matrix_sub

        return SimulationResult(model_image=model_image, source_image=None, mapping_matrix=mapping_matrix, linear_params=None)

    def __repr__(self) -> str:
        return (f"PixelizedLensSimulator(image_shape={self.image_shape}, "
                f"source_shape=({self.source_ny}, {self.source_nx}), "
                f"n_active={int(self.flat_indices.shape[0])}, "
                f"n_lens_light={self.n_lens_light})")


__all__ = ["PixelizedLensSimulator", "EPSILON_REG"]
