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


@partial(jax.jit, static_argnames=("H", "W", "n_source", "nsub", "agg_n_active",
                                   "nx", "ny"))
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
    reg_data: tuple,          # RegData: (rx,ry,scale_x,scale_y,is_gp,gp_matrix)
    lambda_reg: Array,
    nx: int,                  # static: source grid x-dim
    ny: int,                  # static: source grid y-dim
) -> Array:
    """JIT-compiled A(s) = Mᵀ C⁻¹ M s + λ R s.

    All data passed as explicit arrays so JAX traces cleanly.
    ``psf_fft_conj`` is ``conj(FFT(PSF))`` for the adjoint convolution Bᵀ.
    ``reg_data`` is a :class:`~TinyLensGpu.utils.inversion.regularization.RegData`
    tuple supporting both finite-difference (matrix-free Kronecker) and GP
    (dense fallback) regularisation.  ``nx``, ``ny`` are static grid dimensions.
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

    # ---- + λ R s  (matrix-free for FD, dense fallback for GP) ----
    rx, ry, scale_x, scale_y, is_gp, gp_matrix = reg_data

    def _fd_reg_matvec(s_vec: Array) -> Array:
        s_2d = s_vec.reshape(ny, nx).T          # (nx, ny)
        result_2d = rx @ s_2d * scale_x + s_2d @ ry.T * scale_y
        return result_2d.T.ravel()               # back to (Ns,)

    def _gp_reg_matvec(s_vec: Array) -> Array:
        raw = gp_matrix @ s_vec  # (Ns,) for real GP, (1,) for FD placeholder
        # Pad to n_source so both cond branches return (Ns,).
        return jnp.pad(raw, (0, n_source - raw.shape[0]))

    reg_term = jax.lax.cond(is_gp, _gp_reg_matvec, _fd_reg_matvec, s)
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
        tuple (the compact regularisation descriptor) rather than a dense matrix.
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
    # Preconditioner: P = Lᵀ W_eff L + λR  (explicit Ns×Ns)
    # ------------------------------------------------------------------

    def build_preconditioner(
        self,
        noise_1d: Array,
        xmin, xmax, ymin, ymax,
        lambda_reg: Array,
        reg_matrix: Array,
    ) -> tuple[Array, Array]:
        r"""Build sparse preconditioner P and its Cholesky factor.

        .. math::
            P = L^T W_{\rm eff} L + \lambda R, \quad
            W_{\rm eff} = \operatorname{diag}(B^T C^{-1} B)

        Returns ``(P, chol_lower)`` where ``P = chol_lower @ chol_lower.T``.
        """
        H, W = self.image_shape
        noise_1d_j = jnp.asarray(noise_1d)
        reg_matrix_j = jnp.asarray(reg_matrix)

        # ---- W_eff = diag(Bᵀ C⁻¹ B) on full image ----
        w_map = jnp.zeros(H * W, dtype=noise_1d_j.dtype)
        w_map = w_map.at[self.flat_indices].set(1.0 / (noise_1d_j ** 2))
        w_map_2d = w_map.reshape(H, W)

        # To strictly compute diag(Bᵀ C⁻¹ B) for an asymmetric PSF, we need the cross-correlation.
        # This is equivalent to convolving w_map_2d with the flipped psf_sq.
        psf_sq = self.psf_kernel ** 2
        psf_sq_flipped = psf_sq[::-1, ::-1]
        w_eff_2d = jsp_signal.fftconvolve(w_map_2d, psf_sq_flipped, mode="same")
        w_eff_full = w_eff_2d.ravel()

        # ---- Native-resolution weights/indices for P construction ----
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
        w_eff_active = w_eff_full[self.flat_indices]

        # Vectorized scatter-add for Lᵀ diag(w_eff) L
        flat_row = indices[:, :, None]
        flat_col = indices[:, None, :]
        flat_idx = (flat_row * Ns + flat_col).ravel()

        w_prod = weights[:, :, None] * weights[:, None, :]
        values = (w_eff_active[:, None, None] * w_prod).ravel()

        P_flat = jnp.zeros(Ns * Ns, dtype=values.dtype)
        P_flat = P_flat.at[flat_idx].add(values)
        P = P_flat.reshape(Ns, Ns)

        P = P + lambda_reg * reg_matrix_j
        P = 0.5 * (P + P.T)

        chol_lower = jnp.linalg.cholesky(P)
        return P, chol_lower

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
