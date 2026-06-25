"""Operator-based (matrix-free) pixelized source forward simulator.

Provides :class:`PixelizedLensOperator`, a drop-in replacement for
:class:`PixelizedLensSimulator` that avoids building the dense (Nd x Ns)
design matrix.  Uses precomputed lens-operator data and JIT-compiled
primitives for efficient matrix-vector products inside PCG.

Phase-1 limitation: lens-light joint inversion is not yet supported.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.signal as jsp_signal
from jax import Array

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light.pixelized_source import (
    PixelizedSourceModel,
)
from TinyLensGpu.utils.inversion.regularization import (
    DenseRegularizationBuilder,
)
from TinyLensGpu.utils.lensing.mapping import (
    build_source_grid,
    infer_source_bbox,
    lens_mapping_operator_bilinear_rectangular_from,
)


def _is_pixelized_source_model(source: object) -> bool:
    if hasattr(source, "is_pixelized_source"):
        return bool(getattr(source, "is_pixelized_source"))
    return isinstance(source, PixelizedSourceModel)


# ==================================================================
# Operator data — precomputed once per bbox, reused across matvecs
# ==================================================================

class LensOperatorData(NamedTuple):
    """Precomputed data for lens mapping operator matvec / rmatvec.

    All fields are plain JAX arrays so they can be captured by JIT closures.
    """
    weights: Array       # (Nd_sub, 4)  bilinear interpolation weights
    indices: Array       # (Nd_sub, 4)  flat source-pixel indices
    flat_indices: Array  # (Nd_native,) native active pixel flat indices
    n_source: int
    nsub: int
    agg_segment_ids: Array | None  # (Nd_sub,) for sub→native aggregation
    agg_n_active: int
    psf_fft_conj: Array  # conj(FFT(PSF)) for adjoint convolution Bᵀ


# ==================================================================
# JIT-compiled operator primitives  (no ``self`` — pure array args)
# ==================================================================

@partial(jax.jit, static_argnames=("H", "W"))
def _psf_convolve_full_jit(img_flat: Array, psf_fft: Array, H: int, W: int) -> Array:
    """PSF convolution on a full native image via FFT."""
    img_2d = img_flat.reshape(H, W)
    img_fft = jnp.fft.rfft2(img_2d)
    conv_2d = jnp.fft.irfft2(img_fft * psf_fft, s=(H, W))
    return conv_2d.ravel()


@partial(jax.jit, static_argnames=("H", "W", "nsub", "agg_n_active"))
def _apply_L_jit(
    s: Array,
    weights: Array,
    indices: Array,
    flat_indices: Array,
    H: int, W: int,
    nsub: int,
    agg_segment_ids: Array | None,
    agg_n_active: int,
) -> Array:
    """L(s): source → native full image, with optional sub-grid aggregation."""
    # Bilinear interp at (sub-grid) active pixels
    active = jnp.sum(weights * s[indices], axis=1)  # (Nd_sub,)

    # Aggregate sub → native if needed
    if nsub > 1:
        summed = jax.ops.segment_sum(active, agg_segment_ids, num_segments=agg_n_active)
        active = summed / (nsub ** 2)

    # Scatter into native full image
    img = jnp.zeros(H * W, dtype=active.dtype)
    img = img.at[flat_indices].set(active)
    return img


@partial(jax.jit, static_argnames=("H", "W", "n_source", "nsub"))
def _apply_Lt_jit(
    img: Array,
    weights: Array,
    indices: Array,
    flat_indices: Array,
    H: int, W: int,
    n_source: int,
    nsub: int,
    agg_segment_ids: Array | None,
) -> Array:
    """Lᵀ(img): native full image → source plane, with optional sub-grid expansion."""
    # Extract native active pixels
    active_native = img[flat_indices]  # (Nd_native,)

    # Expand to sub-grid if needed
    if nsub > 1:
        active_sub = active_native[agg_segment_ids] / (nsub ** 2)
    else:
        active_sub = active_native

    # Adjoint bilinear: accumulate weighted contributions to source pixels
    contributions = (weights * active_sub[:, None]).ravel()
    result = jnp.zeros(n_source, dtype=contributions.dtype)
    result = result.at[indices.ravel()].add(contributions)
    return result


def _geom_mean(a: Array, b: Array) -> Array:
    """Geometric mean, clipped away from zero."""
    return jnp.sqrt(jnp.maximum(a, 1e-30) * jnp.maximum(b, 1e-30))


def _geom_mean3(a: Array, b: Array, c: Array) -> Array:
    """Geometric mean of three arrays."""
    return jnp.exp(
        (jnp.log(jnp.maximum(a, 1e-30))
         + jnp.log(jnp.maximum(b, 1e-30))
         + jnp.log(jnp.maximum(c, 1e-30))) / 3.0
    )


def _weighted_first_order_matvec_jit(
    s: Array, scale: Array | None, scale_x: Array, scale_y: Array, nx: int, ny: int
) -> Array:
    """Edge-weighted first-order Laplacian ``R @ s``."""
    s_2d = s.reshape(ny, nx)
    scale_2d = scale.reshape(ny, nx) if scale is not None else None

    out_x = jnp.zeros_like(s_2d)
    out_y = jnp.zeros_like(s_2d)
    if nx > 1:
        if scale_2d is None:
            w_x = jnp.ones((ny, nx - 1), dtype=s.dtype)
        else:
            w_x = _geom_mean(scale_2d[:, :-1], scale_2d[:, 1:])
        diff_x = s_2d[:, 1:] - s_2d[:, :-1]
        wdiff_x = w_x * diff_x
        out_x = out_x.at[:, :-1].add(-wdiff_x)
        out_x = out_x.at[:, 1:].add(wdiff_x)
        out_x = out_x.at[:, -1].add(s_2d[:, -1])
    if ny > 1:
        if scale_2d is None:
            w_y = jnp.ones((ny - 1, nx), dtype=s.dtype)
        else:
            w_y = _geom_mean(scale_2d[:-1, :], scale_2d[1:, :])
        diff_y = s_2d[1:, :] - s_2d[:-1, :]
        wdiff_y = w_y * diff_y
        out_y = out_y.at[:-1, :].add(-wdiff_y)
        out_y = out_y.at[1:, :].add(wdiff_y)
        out_y = out_y.at[-1, :].add(s_2d[-1, :])
    return (scale_x * out_x + scale_y * out_y).ravel()


def _weighted_second_order_matvec_jit(
    s: Array, scale: Array | None, scale_x: Array, scale_y: Array, nx: int, ny: int
) -> Array:
    """Edge-weighted second-order curvature ``R @ s``."""
    s_2d = s.reshape(ny, nx)
    scale_2d = scale.reshape(ny, nx) if scale is not None else None

    out_x = jnp.zeros_like(s_2d)
    out_y = jnp.zeros_like(s_2d)
    if nx > 1:
        if scale_2d is None:
            w_near_x = jnp.ones((ny,), dtype=s.dtype)
        else:
            w_near_x = _geom_mean(scale_2d[:, -2], scale_2d[:, -1])
        diff_near_x = s_2d[:, -1] - s_2d[:, -2]
        out_x = out_x.at[:, -2].add(-w_near_x * diff_near_x)
        out_x = out_x.at[:, -1].add(w_near_x * diff_near_x)
    if nx > 2:
        if scale_2d is None:
            w_x2 = jnp.ones((ny, nx - 2), dtype=s.dtype)
        else:
            w_x2 = _geom_mean3(scale_2d[:, :-2], scale_2d[:, 1:-1], scale_2d[:, 2:])
        c_x = s_2d[:, :-2] - 2.0 * s_2d[:, 1:-1] + s_2d[:, 2:]
        wc_x = w_x2 * c_x
        out_x = out_x.at[:, :-2].add(wc_x)
        out_x = out_x.at[:, 1:-1].add(-2.0 * wc_x)
        out_x = out_x.at[:, 2:].add(wc_x)
    if nx > 1:
        out_x = out_x.at[:, -1].add(s_2d[:, -1])

    if ny > 1:
        if scale_2d is None:
            w_near_y = jnp.ones((nx,), dtype=s.dtype)
        else:
            w_near_y = _geom_mean(scale_2d[-2, :], scale_2d[-1, :])
        diff_near_y = s_2d[-1, :] - s_2d[-2, :]
        out_y = out_y.at[-2, :].add(-w_near_y * diff_near_y)
        out_y = out_y.at[-1, :].add(w_near_y * diff_near_y)
    if ny > 2:
        if scale_2d is None:
            w_y2 = jnp.ones((ny - 2, nx), dtype=s.dtype)
        else:
            w_y2 = _geom_mean3(scale_2d[:-2, :], scale_2d[1:-1, :], scale_2d[2:, :])
        c_y = s_2d[:-2, :] - 2.0 * s_2d[1:-1, :] + s_2d[2:, :]
        wc_y = w_y2 * c_y
        out_y = out_y.at[:-2, :].add(wc_y)
        out_y = out_y.at[1:-1, :].add(-2.0 * wc_y)
        out_y = out_y.at[2:, :].add(wc_y)
    if ny > 1:
        out_y = out_y.at[-1, :].add(s_2d[-1, :])

    return (scale_x * out_x + scale_y * out_y).ravel()


@partial(jax.jit, static_argnames=("H", "W", "n_source", "nsub", "agg_n_active",
                                   "nx", "ny", "reg_type"))
def _A_matvec_jit(
    s: Array,
    weights: Array,
    indices: Array,
    flat_indices: Array,
    H: int, W: int,
    n_source: int,
    nsub: int,
    agg_segment_ids: Array | None,
    agg_n_active: int,
    psf_fft: Array,
    psf_fft_conj: Array,
    noise_var: Array,        # σ² at active pixels (Nd_native,)
    reg_data: tuple,          # RegData: (scale, scale_x, scale_y)
    lambda_reg: Array,
    nx: int,                  # static: source grid x-dim
    ny: int,                  # static: source grid y-dim
    reg_type: str,            # static: "zero-order" / "first-order" / "second-order"
) -> Array:
    """JIT-compiled A(s) = Mᵀ C⁻¹ M s + λ R s.

    All data passed as explicit arrays so JAX traces cleanly.
    ``psf_fft_conj`` is ``conj(FFT(PSF))`` for the adjoint convolution Bᵀ.
    ``reg_data`` is a :class:`~TinyLensGpu.utils.inversion.regularization.RegData`
    tuple holding the per-pixel adaptive ``scale`` array and physical spacing
    factors for the edge-weighted finite-difference regularisation.
    ``nx``, ``ny`` and ``reg_type`` are static grid dimensions / type.
    """
    # ---- L(s) ----
    img_lensed = _apply_L_jit(
        s, weights, indices, flat_indices,
        H, W, nsub, agg_segment_ids, agg_n_active,
    )  # (H*W,)

    # ---- B(L(s)) ----
    img_conv = _psf_convolve_full_jit(img_lensed, psf_fft, H, W)  # (H*W,)

    # ---- extract active, C⁻¹ ----
    active = img_conv[flat_indices]          # (Nd,)
    weighted = active / noise_var             # C⁻¹ = 1/σ²

    # ---- inject, Bᵀ (adjoint uses conj(FFT(PSF))) ----
    img_w = jnp.zeros(H * W, dtype=weighted.dtype)
    img_w = img_w.at[flat_indices].set(weighted)
    img_corr = _psf_convolve_full_jit(img_w, psf_fft_conj, H, W)  # (H*W,)

    # ---- Lᵀ ----
    src = _apply_Lt_jit(
        img_corr, weights, indices, flat_indices,
        H, W, n_source, nsub, agg_segment_ids,
    )  # (Ns,)

    # ---- + λ R s  (edge-weighted matrix-free Laplacian) ----
    scale, scale_x, scale_y = reg_data
    if reg_type == "zero-order":
        reg_term = scale * s if scale is not None else s
    elif reg_type == "first-order":
        reg_term = _weighted_first_order_matvec_jit(s, scale, scale_x, scale_y, nx, ny)
    elif reg_type == "second-order":
        reg_term = _weighted_second_order_matvec_jit(s, scale, scale_x, scale_y, nx, ny)
    else:
        # Unreachable: reg_type is a static arg validated in __init__ to be
        # one of {zero, first, second}-order.  Kept for type-narrowing.
        raise ValueError(f"Unsupported reg_type for operator backend: {reg_type!r}")
    return src + lambda_reg * reg_term


# ==================================================================
# Main class
# ==================================================================

class PixelizedLensOperator:
    """Matrix-free forward simulator for a single pixelized source.

    Parameters
    ----------
    phys_model : PhysicalModel
        Model with lens mass and exactly one pixelized source.
    sim_config : SimulatorConfig
        Image grid, PSF, mask, and subsampling configuration.
    detach_bbox : bool, optional
        When True (default), stop_gradient is applied to bbox bounds.
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

        n_pixelized = sum(
            _is_pixelized_source_model(s) for s in phys_model.source_light
        )
        if n_pixelized != len(phys_model.source_light):
            raise ValueError("PixelizedLensOperator does not support a parametric source")
        if n_pixelized != 1:
            raise ValueError("PixelizedLensOperator requires a single pixelized source")

        source = phys_model.source_light[0]
        self.source_model = source
        _reg_type = str(source.regularization_type)
        if _reg_type not in ("zero-order", "first-order", "second-order"):
            raise ValueError(
                f"PixelizedLensOperator only supports finite-difference "
                f"regularization types (zero-order, first-order, second-order), "
                f"got {_reg_type!r}. Use the dense backend for GP types."
            )
        self.reg_type = _reg_type
        self.source_nx = int(source.nx)
        self.source_ny = int(source.ny)
        self.n_source_pixels = self.source_nx * self.source_ny

        self.n_lens_light = len(phys_model.lens_light)
        self.has_lens_light = self.n_lens_light > 0
        if self.has_lens_light:
            raise NotImplementedError(
                "Lens-light joint inversion is not yet supported in the "
                "operator (matrix-free) backend."
            )

        self.image_shape = tuple(sim_config.mask.shape)
        H, W = self.image_shape
        self.n_full_pixels = H * W

        self.mask = jnp.asarray(sim_config.mask, dtype=bool)
        self.active_mask = ~self.mask
        self.flat_indices = jnp.flatnonzero(self.active_mask.ravel())
        self.n_active = int(self.flat_indices.shape[0])

        self.image_x_active = jnp.asarray(sim_config.xgrid)[self.active_mask]
        self.image_y_active = jnp.asarray(sim_config.ygrid)[self.active_mask]
        self.psf_kernel = jnp.asarray(sim_config.psf_kernel)

        # Source seed mask
        self.source_seed_mask = jnp.asarray(sim_config.source_seed_mask, dtype=bool)
        self.seed_active_mask = ~self.source_seed_mask
        if not jnp.all(self.seed_active_mask <= self.active_mask):
            raise ValueError(
                "source_seed_mask active region must be a subset of mask active region."
            )
        self.seed_flat_indices = jnp.flatnonzero(self.seed_active_mask.ravel())
        self.image_x_seed = jnp.asarray(sim_config.xgrid)[self.seed_active_mask]
        self.image_y_seed = jnp.asarray(sim_config.ygrid)[self.seed_active_mask]

        # Sub-grid setup
        if self.nsub > 1:
            self.image_shape_sub = sim_config.subgrid_shape
            self.flat_indices_sub = sim_config.flat_indices
            self.image_x_active_sub = jnp.asarray(sim_config.xgrid_sub_1d)
            self.image_y_active_sub = jnp.asarray(sim_config.ygrid_sub_1d)
            npix = int(sim_config.npix)
            nsub = self.nsub
            fis = self.flat_indices_sub
            i_native = (fis // (npix * nsub)) // nsub
            j_native = (fis % (npix * nsub)) // nsub
            native_flat_all = i_native * npix + j_native
            self._agg_segment_ids = jnp.searchsorted(
                self.flat_indices, native_flat_all
            ).astype(jnp.int32)
            self._agg_n_active = self.n_active
        else:
            self.image_shape_sub = self.image_shape
            self.flat_indices_sub = self.flat_indices
            self.image_x_active_sub = self.image_x_active
            self.image_y_active_sub = self.image_y_active
            self._agg_segment_ids = None
            self._agg_n_active = 0

        # Precompute PSF FFT once
        kernel = self.psf_kernel
        psf_pad = jnp.zeros(self.image_shape, dtype=kernel.dtype)
        psf_pad = psf_pad.at[: kernel.shape[0], : kernel.shape[1]].set(kernel)
        psf_shifted = jnp.roll(
            psf_pad,
            (-(kernel.shape[0] // 2), -(kernel.shape[1] // 2)),
            axis=(0, 1),
        )
        self._psf_fft = jnp.fft.rfft2(psf_shifted)
        self._psf_fft_conj = jnp.conj(self._psf_fft)

        # Pre-bind static args to _A_matvec_jit so that the closure
        # returned by build_A_matvec only captures JAX arrays (PyTree).
        self._A_matvec_jit_prebound = partial(
            _A_matvec_jit,
            H=self.image_shape[0],
            W=self.image_shape[1],
            n_source=self.n_source_pixels,
            nsub=self.nsub,
            agg_n_active=self._agg_n_active if self.nsub > 1 else 0,
            nx=self.source_nx,
            ny=self.source_ny,
            reg_type=self.reg_type,
        )

    # ------------------------------------------------------------------
    # Bbox helpers
    # ------------------------------------------------------------------

    def _get_beta_sub_and_seed(self):
        beta_x_sub, beta_y_sub = self.phys_model.deflection(
            x=self.image_x_active_sub, y=self.image_y_active_sub
        )
        beta_x_seed, beta_y_seed = self.phys_model.deflection(
            x=self.image_x_seed, y=self.image_y_seed
        )
        return beta_x_sub, beta_y_sub, beta_x_seed, beta_y_seed

    def _infer_and_fix_bbox(self, beta_x_seed, beta_y_seed):
        xmin, xmax, ymin, ymax = infer_source_bbox(
            beta_x_seed, beta_y_seed,
            padding=self.sim_config.source_bbox_padding,
            outlier_frac=self.sim_config.source_bbox_outlier_frac,
        )
        if self.detach_bbox:
            xmin = jax.lax.stop_gradient(xmin)
            xmax = jax.lax.stop_gradient(xmax)
            ymin = jax.lax.stop_gradient(ymin)
            ymax = jax.lax.stop_gradient(ymax)
        return xmin, xmax, ymin, ymax

    def ray_trace_seed(self) -> tuple[Array, Array]:
        """Ray-trace seed mask pixels to the source plane.

        Returns ``(beta_x_seed, beta_y_seed)`` for all active pixels in the
        source seed mask.
        """
        beta_x_seed, beta_y_seed = self.phys_model.deflection(
            x=self.image_x_seed, y=self.image_y_seed
        )
        return beta_x_seed, beta_y_seed

    # ------------------------------------------------------------------
    # Precompute lens-operator data (called ONCE per bbox)
    # ------------------------------------------------------------------

    def precompute_operator_data(
        self, xmin, xmax, ymin, ymax, *,
        _betas_sub: tuple | None = None,
    ) -> LensOperatorData:
        """Ray-trace and compute bilinear weights/indices.

        This is the expensive step (deflection angles + bilinear weights).
        Call once per evidence evaluation; reuse for all matvecs.

        If ``_betas_sub`` is provided as ``(beta_x_sub, beta_y_sub)``, the
        deflection call is skipped — useful when the caller has already
        computed them (e.g., from ``_get_bbox``).
        """
        if _betas_sub is not None:
            beta_x_sub, beta_y_sub = _betas_sub
        else:
            beta_x_sub, beta_y_sub, _, _ = self._get_beta_sub_and_seed()
        source_x_axis, source_y_axis, _, _ = build_source_grid(
            self.source_nx, self.source_ny, xmin, xmax, ymin, ymax,
        )
        data_mesh = jnp.stack(
            [jnp.ravel(beta_x_sub), jnp.ravel(beta_y_sub)], axis=1
        )
        weights, indices, _ = lens_mapping_operator_bilinear_rectangular_from(
            data_mesh,
            source_x_axis[0],
            source_x_axis[-1],
            source_y_axis[0],
            source_y_axis[-1],
            self.source_nx,
            self.source_ny,
        )
        return LensOperatorData(
            weights=weights,
            indices=indices,
            flat_indices=self.flat_indices,
            n_source=self.n_source_pixels,
            nsub=self.nsub,
            agg_segment_ids=self._agg_segment_ids,
            agg_n_active=self._agg_n_active,
            psf_fft_conj=self._psf_fft_conj,
        )

    # ------------------------------------------------------------------
    # Forward model (with optional precomputed data)
    # ------------------------------------------------------------------

    def forward_model(
        self, s: Array, xmin, xmax, ymin, ymax,
        op_data: LensOperatorData | None = None,
    ) -> Array:
        """Evaluate M(s) = B(L(s)) at native active pixels.

        If ``op_data`` is provided, it is reused; otherwise computed fresh.
        """
        if op_data is None:
            op_data = self.precompute_operator_data(xmin, xmax, ymin, ymax)
        H, W = self.image_shape
        img_lensed = _apply_L_jit(
            jnp.asarray(s),
            op_data.weights, op_data.indices, op_data.flat_indices,
            H, W, op_data.nsub, op_data.agg_segment_ids, op_data.agg_n_active,
        )
        img_conv = _psf_convolve_full_jit(img_lensed, self._psf_fft, H, W)
        return img_conv[self.flat_indices]

    # ------------------------------------------------------------------
    # Right-hand side: b = Mᵀ C⁻¹ d
    # ------------------------------------------------------------------

    def build_rhs(
        self,
        data_1d: Array,
        noise_1d: Array,
        xmin, xmax, ymin, ymax,
        op_data: LensOperatorData | None = None,
    ) -> Array:
        """Compute b = Lᵀ Bᵀ C⁻¹ d.

        If ``op_data`` is provided, it is reused; otherwise computed fresh.
        """
        if op_data is None:
            op_data = self.precompute_operator_data(xmin, xmax, ymin, ymax)
        H, W = self.image_shape

        weighted = jnp.asarray(data_1d) / (jnp.asarray(noise_1d) ** 2)

        img_full = jnp.zeros(H * W, dtype=weighted.dtype)
        img_full = img_full.at[op_data.flat_indices].set(weighted)
        img_corr = _psf_convolve_full_jit(img_full, self._psf_fft_conj, H, W)

        return _apply_Lt_jit(
            img_corr,
            op_data.weights, op_data.indices, op_data.flat_indices,
            H, W, op_data.n_source, op_data.nsub, op_data.agg_segment_ids,
        )

    # ------------------------------------------------------------------
    # A-operator: A(s) = Mᵀ C⁻¹ M s + λ R s   (Ns → Ns)
    # ------------------------------------------------------------------

    def build_A_matvec(
        self,
        noise_1d: Array,
        xmin, xmax, ymin, ymax,
        lambda_reg: Array,
        reg_data: tuple,
        op_data: LensOperatorData | None = None,
    ) -> tuple[tuple, callable]:
        """Return ``(A_data, _A_jit_prebound)`` for :func:`pcg_solve`.

        ``A_data`` is a tuple of JAX arrays; ``_A_jit_prebound`` is a
        ``functools.partial`` created once at ``__init__`` (static, stable).

        ``reg_data`` is a :class:`~TinyLensGpu.utils.inversion.regularization.RegData`
        tuple holding the per-pixel adaptive ``scale`` array and physical spacing
        factors for the edge-weighted finite-difference regularisation term.
        """
        if op_data is None:
            op_data = self.precompute_operator_data(xmin, xmax, ymin, ymax)

        A_data = (
            op_data.weights,
            op_data.indices,
            op_data.flat_indices,
            op_data.agg_segment_ids,
            self._psf_fft,
            self._psf_fft_conj,
            jnp.asarray(noise_1d) ** 2,   # noise variance
            reg_data,                       # RegData tuple
            jnp.asarray(lambda_reg),
        )
        return A_data, self._A_matvec_jit_prebound

    def call_A_matvec(self, s: Array, A_data: tuple, _A_jit_prebound=None) -> Array:
        """Apply the A-operator to ``s`` using data from :meth:`build_A_matvec`.

        Convenience wrapper for testing and forward-model use.
        """
        if _A_jit_prebound is None:
            _A_jit_prebound = self._A_matvec_jit_prebound
        return _A_jit_prebound(
            s, A_data[0], A_data[1], A_data[2],
            agg_segment_ids=A_data[3], psf_fft=A_data[4],
            psf_fft_conj=A_data[5],
            noise_var=A_data[6], reg_data=A_data[7], lambda_reg=A_data[8],
        )

    # ------------------------------------------------------------------
    # Block-diagonal preconditioner
    # ------------------------------------------------------------------

    def build_block_diag_preconditioner(
        self,
        noise_1d: Array,
        xmin, xmax, ymin, ymax,
        lambda_reg: Array,
        reg_builder: DenseRegularizationBuilder,
        block_size: int = 10,
        scale: Array | None = None,
    ) -> tuple[Array | list[Array], Array | list[Array]]:
        r"""Build a block-diagonal preconditioner P and its Cholesky factors.

        Partitions the source grid into ``block_size × block_size`` blocks
        (default 10×10), constructs the submatrix of
        :math:`P = L^T W_{\rm eff} L + \lambda R` for each block, and
        Cholesky-factors each one independently.

        When the source grid is uniformly divisible by ``block_size``, uses
        ``jax.lax.scan`` to avoid Python-loop unrolling during JIT tracing
        (significant compilation speedup).  Otherwise falls back to the
        legacy Python-loop path.

        Only finite-difference regularization types are supported.

        Parameters
        ----------
        noise_1d : Array
            Per-pixel noise std at active pixels, shape ``(Nd,)``.
        xmin, xmax, ymin, ymax : float
            Source-plane bounding box.
        lambda_reg : Array
            Regularization strength λ (scalar or broadcastable).
        reg_builder : DenseRegularizationBuilder
            Source-plane regularization builder (provides per-block R).
        block_size : int, optional
            Source-grid block size in pixels (default 10).
        scale : Array or None, optional
            Per-pixel regularization scale of shape ``(Ns,)``.  When
            provided, each block's regularization submatrix incorporates
            the corresponding subset of the scale array.

        Returns
        -------
        tuple[Array | list[Array], Array | list[Array]]
            ``(block_chols, block_masks)`` where ``block_chols[i]`` is the
            lower-triangular Cholesky factor of the i-th block's P submatrix,
            and ``block_masks[i]`` contains the global flat source indices
            belonging to that block.  When all blocks have the same size the
            result is stacked into arrays with shapes ``(n_blocks, b, b)`` and
            ``(n_blocks, b)`` so the PCG preconditioner can be vmapped.
        """
        H, W = self.image_shape
        noise_1d_j = jnp.asarray(noise_1d)

        # ---- W_eff = diag(Bᵀ C⁻¹ B) on full image ----
        w_map = jnp.zeros(H * W, dtype=noise_1d_j.dtype)
        w_map = w_map.at[self.flat_indices].set(1.0 / (noise_1d_j ** 2))
        w_map_2d = w_map.reshape(H, W)

        psf_sq = self.psf_kernel ** 2
        psf_sq_flipped = psf_sq[::-1, ::-1]
        w_eff_2d = jsp_signal.fftconvolve(w_map_2d, psf_sq_flipped, mode="same")
        w_eff_full = w_eff_2d.ravel()

        # ---- Native-resolution weights/indices ----
        beta_x, beta_y = self.phys_model.deflection(
            x=self.image_x_active, y=self.image_y_active
        )
        source_x_axis, source_y_axis, _, _ = build_source_grid(
            self.source_nx, self.source_ny, xmin, xmax, ymin, ymax,
        )
        data_mesh = jnp.stack(
            [jnp.ravel(beta_x), jnp.ravel(beta_y)], axis=1
        )
        weights, indices, _ = lens_mapping_operator_bilinear_rectangular_from(
            data_mesh,
            source_x_axis[0], source_x_axis[-1],
            source_y_axis[0], source_y_axis[-1],
            self.source_nx, self.source_ny,
        )
        Ns = self.n_source_pixels
        nx, ny = self.source_nx, self.source_ny
        w_eff_active = w_eff_full[self.flat_indices]

        # ---- Block partitioning ----
        n_bx = (nx + block_size - 1) // block_size
        n_by = (ny + block_size - 1) // block_size

        # Source pixel → block mapping  (column-major: s = x + y * nx)
        sx = jnp.arange(Ns) % nx
        sy = jnp.arange(Ns) // nx
        block_x = sx // block_size
        block_y = sy // block_size
        block_id = block_x + block_y * n_bx  # (Ns,)

        # Which blocks does each image pixel's bilinear stencil touch?
        bid_per_neighbor = block_id[indices]  # (Nd, 4)

        # Dispatch: scan for uniform grids, legacy loop otherwise
        is_uniform = (nx % block_size == 0) and (ny % block_size == 0)
        if is_uniform:
            chols, masks = self._build_block_diag_precond_scan(
                weights, indices, bid_per_neighbor, w_eff_active,
                lambda_reg, reg_builder, block_size,
                xmin, xmax, ymin, ymax, scale,
                n_bx, n_by, nx,
            )
        else:
            chols, masks = self._build_block_diag_precond_legacy(
                weights, indices, bid_per_neighbor, w_eff_active,
                lambda_reg, reg_builder, block_size,
                xmin, xmax, ymin, ymax, scale,
                n_bx, n_by, nx,
            )

        # Stack if uniform
        if isinstance(chols, (list, tuple)) and chols:
            block_sizes = {int(chol.shape[0]) for chol in chols}
            if len(block_sizes) == 1:
                return jnp.stack(chols, axis=0), jnp.stack(masks, axis=0)

        return chols, masks

    def _build_block_diag_precond_legacy(
        self,
        weights, indices, bid_per_neighbor, w_eff_active,
        lambda_reg, reg_builder, block_size,
        xmin, xmax, ymin, ymax, scale,
        n_bx, n_by, nx,
    ):
        """Legacy Python-loop block-diagonal preconditioner (non-uniform grids)."""
        Ns = self.n_source_pixels
        block_chols = []
        block_masks = []

        for by in range(n_by):
            for bx in range(n_bx):
                bid = bx + by * n_bx
                x_s = bx * block_size
                x_e = min(x_s + block_size, nx)
                y_s = by * block_size
                y_e = min(y_s + block_size, self.source_ny)
                block_nx_b = x_e - x_s
                block_ny_b = y_e - y_s
                block_n = block_nx_b * block_ny_b

                if block_n == 0:
                    continue

                bf = jnp.array(
                    [x + y * nx
                     for y in range(y_s, y_e)
                     for x in range(x_s, x_e)],
                    dtype=jnp.int32,
                )

                affected = jnp.any(bid_per_neighbor == bid, axis=1)
                mask_w = affected[:, None].astype(weights.dtype)
                aff_w = weights * mask_w
                aff_we = w_eff_active * affected.astype(w_eff_active.dtype)
                in_block = bid_per_neighbor == bid

                g2l = -jnp.ones(Ns, dtype=jnp.int32)
                for loc_i, g_idx in enumerate(bf):
                    g2l = g2l.at[g_idx].set(loc_i)
                loc_idx = g2l[indices]

                P_block = jnp.zeros((block_n, block_n), dtype=aff_w.dtype)

                wgt_i = aff_w[:, :, None]
                wgt_j = aff_w[:, None, :]
                w_prod = wgt_i * wgt_j * aff_we[:, None, None]

                loc_i = loc_idx[:, :, None]
                loc_j = loc_idx[:, None, :]
                in_i = in_block[:, :, None]
                in_j = in_block[:, None, :]
                valid = in_i & in_j

                loc_i_b = jnp.broadcast_to(loc_i, valid.shape)
                loc_j_b = jnp.broadcast_to(loc_j, valid.shape)
                valid_f = valid.ravel()
                valid_f_float = valid_f.astype(w_prod.dtype)

                loc_i_f = jnp.where(valid_f, loc_i_b.ravel(), 0)
                loc_j_f = jnp.where(valid_f, loc_j_b.ravel(), 0)
                vals_f = w_prod.ravel() * valid_f_float

                P_block = P_block.at[loc_i_f, loc_j_f].add(vals_f)

                R_block = reg_builder.block_diag_R(
                    x_s, x_e, y_s, y_e, xmin, xmax, ymin, ymax,
                    scale=scale,
                )
                P_block = P_block + lambda_reg * R_block
                P_block = 0.5 * (P_block + P_block.T)

                diag_P = jnp.diag(P_block)
                diag_mean = jnp.mean(jnp.abs(diag_P))
                jitter_scale = jnp.maximum(diag_mean, 1e-8)
                jitter = 1e-6 * jitter_scale * jnp.eye(
                    block_n, dtype=P_block.dtype,
                )
                P_block = P_block + jitter

                chol_block = jnp.linalg.cholesky(P_block)
                block_chols.append(chol_block)
                block_masks.append(bf)

        return block_chols, block_masks

    def _build_block_diag_precond_scan(
        self,
        weights, indices, bid_per_neighbor, w_eff_active,
        lambda_reg, reg_builder, block_size,
        xmin, xmax, ymin, ymax, scale,
        n_bx, n_by, nx,
    ):
        """``lax.scan``-based block-diagonal preconditioner (uniform grids).

        Compiles the block body once and executes it dynamically, avoiding
        the ~100× loop unrolling that dominates JIT compilation time.
        """
        bs = block_size
        block_n = bs * bs
        Ns = self.n_source_pixels

        # Precompute scan inputs: (bid, x_s, y_s) for each block
        bx_arr = jnp.arange(n_bx, dtype=jnp.int32)
        by_arr = jnp.arange(n_by, dtype=jnp.int32)
        bxs, bys = jnp.meshgrid(bx_arr, by_arr, indexing="xy")
        bids = (bxs + bys * n_bx).ravel().astype(jnp.int32)
        x_starts = (bxs * bs).ravel().astype(jnp.int32)
        y_starts = (bys * bs).ravel().astype(jnp.int32)
        scan_inputs = jnp.stack([bids, x_starts, y_starts], axis=-1)

        # Pre-compute local index template (column-major): [0, 1, ..., block_n-1]
        local_i = jnp.arange(block_n, dtype=jnp.int32)
        loc_x_template = local_i % bs
        loc_y_template = local_i // bs

        def scan_body(carry, xs):
            bid = xs[0]
            x_s = xs[1]
            y_s = xs[2]

            # Flat source indices for this block (column-major, vectorized)
            bf = (x_s + loc_x_template) + (y_s + loc_y_template) * nx  # (block_n,)

            # Affected pixels mask
            affected = jnp.any(bid_per_neighbor == bid, axis=1)  # (Nd,)
            mask_w = affected[:, None].astype(weights.dtype)
            aff_w = weights * mask_w
            aff_we = w_eff_active * affected.astype(w_eff_active.dtype)
            in_block = bid_per_neighbor == bid  # (Nd, 4)

            # Global → local index mapping (vectorized, no Python loop)
            g2l = -jnp.ones(Ns, dtype=jnp.int32)
            g2l = g2l.at[bf].set(jnp.arange(block_n, dtype=jnp.int32))
            loc_idx = g2l[indices]  # (Nd, 4)

            # ---- Scatter-add Lᵀ W_eff L into block-sized matrix ----
            P_block = jnp.zeros((block_n, block_n), dtype=aff_w.dtype)

            wgt_i = aff_w[:, :, None]              # (Nd, 4, 1)
            wgt_j = aff_w[:, None, :]              # (Nd, 1, 4)
            w_prod = wgt_i * wgt_j * aff_we[:, None, None]  # (Nd, 4, 4)

            loc_i = loc_idx[:, :, None]            # (Nd, 4, 1)
            loc_j = loc_idx[:, None, :]            # (Nd, 1, 4)
            in_i = in_block[:, :, None]            # (Nd, 4, 1)
            in_j = in_block[:, None, :]            # (Nd, 1, 4)
            valid = in_i & in_j                    # (Nd, 4, 4)

            loc_i_b = jnp.broadcast_to(loc_i, valid.shape)
            loc_j_b = jnp.broadcast_to(loc_j, valid.shape)
            valid_f = valid.ravel()
            valid_f_float = valid_f.astype(w_prod.dtype)

            loc_i_f = jnp.where(valid_f, loc_i_b.ravel(), 0)
            loc_j_f = jnp.where(valid_f, loc_j_b.ravel(), 0)
            vals_f = w_prod.ravel() * valid_f_float

            P_block = P_block.at[loc_i_f, loc_j_f].add(vals_f)

            # ---- + λ R_block (vectorized stencil methods) ----
            R_block = reg_builder.block_diag_R_vec(
                x_s, x_s + bs, y_s, y_s + bs,
                xmin, xmax, ymin, ymax,
                scale=scale, block_size=bs,
            )
            P_block = P_block + lambda_reg * R_block
            P_block = 0.5 * (P_block + P_block.T)

            # ---- Cholesky with scale-adaptive diagonal jitter ----
            diag_P = jnp.diag(P_block)
            diag_mean = jnp.mean(jnp.abs(diag_P))
            jitter_scale = jnp.maximum(diag_mean, 1e-8)
            jitter = 1e-6 * jitter_scale * jnp.eye(
                block_n, dtype=P_block.dtype,
            )
            P_block = P_block + jitter

            chol_block = jnp.linalg.cholesky(P_block)
            return carry, (chol_block, bf)

        init_carry = jnp.array(0, dtype=jnp.int32)
        _, (chols, masks) = jax.lax.scan(scan_body, init_carry, scan_inputs)
        # Convert to lists for uniform post-processing
        return list(chols), list(masks)

    # ------------------------------------------------------------------
    # Legacy: dense preconditioner  (for testing / small grids)
    # ------------------------------------------------------------------
    @staticmethod
    def logdet_block_diag(block_chols: Array | list[Array]) -> Array:
        """Compute ``log|P|`` from block-diagonal Cholesky factors."""
        if not isinstance(block_chols, (list, tuple)):
            return 2.0 * jnp.sum(jnp.log(jnp.diagonal(block_chols, axis1=-2, axis2=-1)))

        if len(block_chols) == 0:
            return jnp.array(0.0)

        total = jnp.array(0.0, dtype=block_chols[0].dtype)
        for chol in block_chols:
            total = total + 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        return total

    # ------------------------------------------------------------------
    # design_matrix — for parity testing only
    # ------------------------------------------------------------------

    def design_matrix(self) -> tuple[Array, tuple]:
        """Build F (Nd × Ns) by applying forward_model to each basis vector.

        **Slow** — for verification only.
        """
        _, _, beta_x_seed, beta_y_seed = self._get_beta_sub_and_seed()
        xmin, xmax, ymin, ymax = self._infer_and_fix_bbox(
            beta_x_seed, beta_y_seed
        )
        eye = jnp.eye(self.n_source_pixels, dtype=self.psf_kernel.dtype)
        columns = [self.forward_model(eye[i], xmin, xmax, ymin, ymax)
                   for i in range(self.n_source_pixels)]
        F = jnp.stack(columns, axis=1)
        return F, (xmin, xmax, ymin, ymax)

    def __repr__(self) -> str:
        return (
            f"PixelizedLensOperator(image_shape={self.image_shape}, "
            f"source_shape=({self.source_ny}, {self.source_nx}), "
            f"n_active={self.n_active})"
        )


__all__ = ["PixelizedLensOperator"]
