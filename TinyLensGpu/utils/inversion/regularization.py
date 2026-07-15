"""Dense regularization matrix builders for pixelized source inversion.

This module provides traditional finite-difference penalties and dense
Gaussian-process covariance penalties for source-plane linear inversion.  The
finite-difference operators are precomputed on index space and scaled by the
physical source-grid spacing when a matrix is requested.

For matrix-free (operator) backends, the class also exposes
:meth:`matvec_free`, :meth:`logdet_free`, :meth:`to_dense_free`, and
:meth:`make_reg_data` for finite-difference types, which apply
edge-weighted graph-Laplacian stencils directly (and a block-diagonal
logdet approximation) to avoid materialising the full Ns x Ns
regularisation matrix.

.. note::

    GP-style regularization (``exponential``, ``gaussian``, ``matern*``) is
    **not** supported by the operator backend.  GP types inherently require a
    dense ``(Ns, Ns)`` precision matrix, so they gain no memory benefit from
    matrix-free operators.  Use the dense backend
    (:class:`~TinyLensGpu.ObservationModel.LensImage.pixelized_image_model.PixelizedImageProbModel`)
    for GP regularization instead.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

VALID_REGULARIZATION_TYPES: frozenset[str] = frozenset({
    "zero-order", "first-order", "second-order",
    "exponential", "gaussian",
    "matern32", "matern52", "matern72",
})
"""All supported regularization type names."""

GP_REGULARIZATION_TYPES: frozenset[str] = frozenset({
    "exponential", "gaussian",
    "matern32", "matern52", "matern72",
})
"""GP-style regularization types that require a kernel_scale parameter."""


def source_template_scale_map(
    source_pixels: "jax.Array",
    n: int,
    rho: float,
    *,
    ref_percentile: float = 99.5,
    eps: float = 1.0e-10,
) -> "jax.Array | None":
    """Build a Galan-style adaptive precision scale map from a source template.

    The input source template is assumed to be a regularized MAP source
    reconstruction such as stage-m0 ``S0``.  No additional smoothing is
    applied here; the stage-m0 regularization controls the smoothness of the
    template itself.

    Parameters
    ----------
    source_pixels : array_like
        Source template with shape ``(n * n,)`` or ``(n, n)``.
    n : int
        Source-grid dimension (n x n square grid).
    rho : float or Array
        Galan-style adaptive regularization strength. Values within ``1e-10`` of zero
        return ``None`` for the uniform-regularization fast path.
    ref_percentile : float, optional
        Static percentile of non-negative source brightness used as the
        reference-bright value (default 99.5).
    eps : float, optional
        Reference-brightness floor used to keep all-dark templates finite.
    """
    n = int(n)
    try:
        rho_static = float(rho)
    except (TypeError, jax.errors.ConcretizationTypeError):
        rho_static = None

    if rho_static is not None and rho_static < 0.0:
        raise ValueError(f"rho must be >= 0, got {rho}")
    if rho_static is not None and abs(rho_static) < 1.0e-10:
        return None
    ref_percentile = float(ref_percentile)
    if not 0.0 <= ref_percentile <= 100.0:
        raise ValueError(
            f"ref_percentile must be in [0, 100], got {ref_percentile}"
        )

    source = jnp.asarray(source_pixels, dtype=jnp.float32)
    if source.shape == (n, n):
        source = source.reshape(n * n)
    elif source.shape != (n * n,):
        raise ValueError(
            "source_pixels must have shape "
            f"({n * n},) or ({n}, {n}), got {source.shape}."
        )

    source_pos = jnp.maximum(source, 0.0)
    brightness_ref = jnp.percentile(source_pos, ref_percentile)
    u = source_pos / jnp.maximum(brightness_ref, float(eps))
    u = jnp.clip(u, 0.0, 1.0)
    rho_j = jnp.maximum(jnp.asarray(rho, dtype=jnp.float32), 0.0)
    scale = jnp.exp(rho_j * (1.0 - u))
    if rho_static is None:
        scale = jnp.where(
            jnp.abs(rho_j) < jnp.asarray(1.0e-10, dtype=jnp.float32),
            jnp.ones_like(scale),
            scale,
        )
    return jnp.asarray(scale, dtype=jnp.float32)


class RegData(NamedTuple):
    """Compact finite-difference regularisation data for matrix-free matvec / logdet.

    The operator backend uses **edge-weighted** finite-difference
    regularisation with a single physical scaling factor ``scale_factor``
    (since the source grid and bbox are square, ``dx == dy``).
    ``scale`` is the per-pixel adaptive factor from which edge
    weights are derived as geometric means.

    Only finite-difference types are supported; GP types should use the dense
    backend instead.
    """
    scale: "jax.Array | None"  # (Ns,) per-pixel adaptive scale; None = uniform
    scale_factor: "jax.Array"  # scalar physical scaling factor (1/dx^2 or 1/dx^4)


class DenseRegularizationBuilder:
    """Build dense source-grid regularization matrices.

    Parameters
    ----------
    n : int
        Number of source pixels per side (n x n square grid).
    regularization_type : str
        Regularization family. Supported values are ``"zero-order"``,
        ``"first-order"``, ``"second-order"``, ``"exponential"``,
        ``"gaussian"``, ``"matern32"``, ``"matern52"``, and ``"matern72"``.
    jitter : float, optional
        Diagonal jitter added to GP covariance matrices for numerical stability.

    Raises
    ------
    ValueError
        If the grid shape or regularization configuration is invalid.
    """

    _VALID_TYPES = VALID_REGULARIZATION_TYPES

    def __init__(
        self,
        n: int,
        regularization_type: str,
        *,
        jitter: float = 1e-6,
    ) -> None:
        self.n = int(n)
        if self.n < 2:
            raise ValueError("n must be at least 2.")

        self.n_pixels = self.n * self.n
        self.regularization_type = regularization_type.lower()
        self.jitter = float(jitter)

        if self.regularization_type not in self._VALID_TYPES:
            raise ValueError(f"Unsupported regularization_type: {regularization_type!r}.")
        if self.jitter < 0.0:
            raise ValueError("jitter must be non-negative.")

        self._identity = None
        self._dx_operator = self._dy_operator = None
        self._lx_operator = self._ly_operator = None
        self._unit_coordinates = None

        if self.regularization_type in ("zero-order", *GP_REGULARIZATION_TYPES):
            self._identity = jnp.eye(self.n_pixels)
        if self.regularization_type == "first-order":
            self._dx_operator, self._dy_operator = self._build_first_difference_operators()
        elif self.regularization_type == "second-order":
            self._lx_operator, self._ly_operator = self._build_curvature_difference_operators()
        elif self.regularization_type in GP_REGULARIZATION_TYPES:
            self._unit_coordinates = self._build_unit_coordinates()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_scale(
        self, scale: "jax.Array | None", expected_size: int | None = None,
    ) -> "jax.Array | None":
        """Return validated scale as ``(Ns,)`` or ``None`` for uniform scaling."""
        if scale is None:
            return None

        scale_arr = jnp.asarray(scale)
        expected_size = self.n_pixels if expected_size is None else int(expected_size)
        if scale_arr.shape != (expected_size,):
            raise ValueError(
                "scale must have shape "
                f"({expected_size},), got {scale_arr.shape}."
            )

        valid = jnp.all(jnp.isfinite(scale_arr) & (scale_arr > 0.0))
        try:
            is_valid = bool(valid)
        except jax.errors.TracerBoolConversionError:
            is_valid = True
        if not is_valid:
            raise ValueError("scale values must be finite and strictly positive.")
        return scale_arr

    def _apply_diag_scale(self, matrix: "jax.Array", scale: "jax.Array | None") -> "jax.Array":
        """If *scale* is not None, return ``diag(sqrt(scale)) @ matrix @ diag(sqrt(scale))``.

        Otherwise return *matrix* unchanged.  Kept only for GP regularisation;
        finite-difference types now use edge-weighted Laplacians instead.
        """
        scale = self._check_scale(scale, int(matrix.shape[0]))
        if scale is not None:
            sqrt_scale = jnp.sqrt(scale)
            # Explicitly form the rank-1 outer-product scaling matrix first,
            # then apply it with a single element-wise multiply.  This avoids
            # chained broadcast-multiplies that can produce NaN in XLA's GEMM
            # fusion autotuner when the surrounding computation (kron + add)
            # is fused into a single kernel.
            scale_mat = sqrt_scale[:, None] * sqrt_scale[None, :]
            matrix = matrix * scale_mat
        return matrix

    # ------------------------------------------------------------------
    # Edge-weight helpers for finite-difference adaptive regularisation
    # ------------------------------------------------------------------

    def _scale_to_2d(self, scale: "jax.Array | None") -> "jax.Array | None":
        """Return ``scale`` reshaped to ``(n, n)``, or ``None`` if uniform."""
        if scale is None:
            return None
        return scale.reshape(self.n, self.n)

    @staticmethod
    def _geom_mean(a: "jax.Array", b: "jax.Array") -> "jax.Array":
        """Geometric mean, computed in log-space to avoid float32 overflow.

        ``sqrt(a*b)`` overflows when ``a*b`` exceeds the float32 max even if
        both ``a`` and ``b`` are individually representable; the log-space
        form ``exp(0.5*(log a + log b))`` is stable.  Inputs are clipped away
        from zero to keep ``log`` finite.
        """
        return jnp.exp(
            0.5 * (jnp.log(jnp.maximum(a, 1e-30)) + jnp.log(jnp.maximum(b, 1e-30)))
        )

    @staticmethod
    def _geom_mean3(a: "jax.Array", b: "jax.Array", c: "jax.Array") -> "jax.Array":
        """Geometric mean of three arrays."""
        return jnp.exp(
            (jnp.log(jnp.maximum(a, 1e-30))
             + jnp.log(jnp.maximum(b, 1e-30))
             + jnp.log(jnp.maximum(c, 1e-30))) / 3.0
        )

    def _edge_weights_first_order(
        self, scale_2d: "jax.Array | None"
    ) -> tuple["jax.Array", "jax.Array"]:
        """Return ``(w_x, w_y)`` edge weights for first-order regularisation.

        ``w_x`` has shape ``(n, n-1)``; ``w_y`` has shape ``(n-1, n)``.
        When ``scale_2d`` is ``None`` (uniform), all weights are 1.
        """
        if scale_2d is None:
            w_x = jnp.ones((self.n, max(self.n - 1, 1)), dtype=jnp.float32)
            w_y = jnp.ones((max(self.n - 1, 1), self.n), dtype=jnp.float32)
            return w_x, w_y
        w_x = self._geom_mean(scale_2d[:, :-1], scale_2d[:, 1:])
        w_y = self._geom_mean(scale_2d[:-1, :], scale_2d[1:, :])
        return w_x, w_y

    def _edge_weights_second_order(
        self, scale_2d: "jax.Array | None"
    ) -> tuple["jax.Array", "jax.Array", "jax.Array", "jax.Array"]:
        """Return curvature stencil weights for second-order regularisation.

        Returns ``(w_x2, w_y2, w_x2_near, w_y2_near)`` where:
        - ``w_x2`` : ``(n, n-2)`` full-curvature horizontal weights.
        - ``w_y2`` : ``(n-2, n)`` full-curvature vertical weights.
        - ``w_x2_near`` : ``(n,)`` near-boundary horizontal first-gradient weights.
        - ``w_y2_near`` : ``(n,)`` near-boundary vertical first-gradient weights.
        """
        if scale_2d is None:
            w_x2 = jnp.ones((self.n, max(self.n - 2, 1)), dtype=jnp.float32)
            w_y2 = jnp.ones((max(self.n - 2, 1), self.n), dtype=jnp.float32)
            w_x2_near = jnp.ones((self.n,), dtype=jnp.float32)
            w_y2_near = jnp.ones((self.n,), dtype=jnp.float32)
            return w_x2, w_y2, w_x2_near, w_y2_near
        w_x2 = self._geom_mean3(
            scale_2d[:, :-2], scale_2d[:, 1:-1], scale_2d[:, 2:]
        )
        w_y2 = self._geom_mean3(
            scale_2d[:-2, :], scale_2d[1:-1, :], scale_2d[2:, :]
        )
        w_x2_near = self._geom_mean(scale_2d[:, -2], scale_2d[:, -1])
        w_y2_near = self._geom_mean(scale_2d[-2, :], scale_2d[-1, :])
        return w_x2, w_y2, w_x2_near, w_y2_near

    def _weighted_first_order_matvec(
        self,
        s: "jax.Array",
        scale_2d: "jax.Array | None",
        scale_factor: "jax.Array",
    ) -> "jax.Array":
        """Matrix-free ``R @ s`` for edge-weighted first-order regularisation."""
        s_2d = s.reshape(self.n, self.n)
        w_x, w_y = self._edge_weights_first_order(scale_2d)

        out_x = jnp.zeros_like(s_2d)
        out_y = jnp.zeros_like(s_2d)
        if self.n > 1:
            diff_x = s_2d[:, 1:] - s_2d[:, :-1]
            wdiff_x = w_x * diff_x
            out_x = out_x.at[:, :-1].add(-wdiff_x)
            out_x = out_x.at[:, 1:].add(wdiff_x)
            # boundary zero-order fallback on last column
            out_x = out_x.at[:, -1].add(s_2d[:, -1])
        if self.n > 1:
            diff_y = s_2d[1:, :] - s_2d[:-1, :]
            wdiff_y = w_y * diff_y
            out_y = out_y.at[:-1, :].add(-wdiff_y)
            out_y = out_y.at[1:, :].add(wdiff_y)
            # boundary zero-order fallback on last row
            out_y = out_y.at[-1, :].add(s_2d[-1, :])
        return (scale_factor * (out_x + out_y)).ravel()

    def _weighted_second_order_matvec(
        self,
        s: "jax.Array",
        scale_2d: "jax.Array | None",
        scale_factor: "jax.Array",
    ) -> "jax.Array":
        """Matrix-free ``R @ s`` for edge-weighted second-order regularisation."""
        s_2d = s.reshape(self.n, self.n)
        w_x2, w_y2, w_x2_near, w_y2_near = self._edge_weights_second_order(scale_2d)

        out_x = jnp.zeros_like(s_2d)
        out_y = jnp.zeros_like(s_2d)
        if self.n > 1:
            # near-boundary first-gradient fallback (always present for n>=2)
            diff_near_x = s_2d[:, -1] - s_2d[:, -2]
            out_x = out_x.at[:, -2].add(-w_x2_near * diff_near_x)
            out_x = out_x.at[:, -1].add(w_x2_near * diff_near_x)
        if self.n > 2:
            # full curvature rows
            c_x = s_2d[:, :-2] - 2.0 * s_2d[:, 1:-1] + s_2d[:, 2:]
            wc_x = w_x2 * c_x
            out_x = out_x.at[:, :-2].add(wc_x)
            out_x = out_x.at[:, 1:-1].add(-2.0 * wc_x)
            out_x = out_x.at[:, 2:].add(wc_x)
        if self.n > 1:
            diff_near_y = s_2d[-1, :] - s_2d[-2, :]
            out_y = out_y.at[-2, :].add(-w_y2_near * diff_near_y)
            out_y = out_y.at[-1, :].add(w_y2_near * diff_near_y)
        if self.n > 2:
            c_y = s_2d[:-2, :] - 2.0 * s_2d[1:-1, :] + s_2d[2:, :]
            wc_y = w_y2 * c_y
            out_y = out_y.at[:-2, :].add(wc_y)
            out_y = out_y.at[1:-1, :].add(-2.0 * wc_y)
            out_y = out_y.at[2:, :].add(wc_y)
        # outer boundary zero-order fallback
        if self.n > 1:
            out_x = out_x.at[:, -1].add(s_2d[:, -1])
        if self.n > 1:
            out_y = out_y.at[-1, :].add(s_2d[-1, :])
        return (scale_factor * (out_x + out_y)).ravel()

    def _weighted_first_order_dense(
        self, xmin: float, xmax: float, ymin: float, ymax: float,
        scale: "jax.Array | None",
    ) -> "jax.Array":
        """Dense edge-weighted first-order regularisation matrix."""
        scale_2d = self._scale_to_2d(scale)
        w_x, w_y = self._edge_weights_first_order(scale_2d)
        scale_factor = ((self.n - 1) / (xmax - xmin)) ** 2

        wx_full = jnp.ones(self.n_pixels, dtype=w_x.dtype)
        wy_full = jnp.ones(self.n_pixels, dtype=w_y.dtype)
        if self.n > 1:
            interior_x = jnp.arange(self.n_pixels).reshape(self.n, self.n)[:, :-1].ravel()
            wx_full = wx_full.at[interior_x].set(w_x.ravel())
        if self.n > 1:
            interior_y = jnp.arange(self.n_pixels).reshape(self.n, self.n)[:-1, :].ravel()
            wy_full = wy_full.at[interior_y].set(w_y.ravel())

        R = scale_factor * ((wx_full[:, None] * self._dx_operator).T @ self._dx_operator)
        R = R + scale_factor * ((wy_full[:, None] * self._dy_operator).T @ self._dy_operator)
        return 0.5 * (R + R.T)

    def _weighted_second_order_dense(
        self, xmin: float, xmax: float, ymin: float, ymax: float,
        scale: "jax.Array | None",
    ) -> "jax.Array":
        """Dense edge-weighted second-order regularisation matrix."""
        scale_2d = self._scale_to_2d(scale)
        w_x2, w_y2, w_x2_near, w_y2_near = self._edge_weights_second_order(scale_2d)
        scale_factor = ((self.n - 1) / (xmax - xmin)) ** 4

        wlx_full = jnp.ones(self.n_pixels, dtype=w_x2.dtype)
        wly_full = jnp.ones(self.n_pixels, dtype=w_y2.dtype)
        if self.n > 2:
            full_x = jnp.arange(self.n_pixels).reshape(self.n, self.n)[:, :-2].ravel()
            wlx_full = wlx_full.at[full_x].set(w_x2.ravel())
        if self.n > 1:
            near_x = jnp.arange(self.n_pixels).reshape(self.n, self.n)[:, -2].ravel()
            wlx_full = wlx_full.at[near_x].set(w_x2_near)
        if self.n > 2:
            full_y = jnp.arange(self.n_pixels).reshape(self.n, self.n)[:-2, :].ravel()
            wly_full = wly_full.at[full_y].set(w_y2.ravel())
        if self.n > 1:
            near_y = jnp.arange(self.n_pixels).reshape(self.n, self.n)[-2, :].ravel()
            wly_full = wly_full.at[near_y].set(w_y2_near)

        R = scale_factor * ((wlx_full[:, None] * self._lx_operator).T @ self._lx_operator)
        R = R + scale_factor * ((wly_full[:, None] * self._ly_operator).T @ self._ly_operator)
        return 0.5 * (R + R.T)

    def matrix(self, xmin, xmax, ymin, ymax, *, kernel_scale: float | None = None, scale: "jax.Array | None" = None):
        """Return the dense regularization matrix for a square source grid.

        Parameters
        ----------
        xmin, xmax : float
            Source-plane x-axis bounds.
        ymin, ymax : float
            Source-plane y-axis bounds.
        kernel_scale : float, optional
            GP kernel correlation scale. Required for GP regularization and
            ignored for traditional finite-difference penalties.
        scale : Array or None, optional
            Per-pixel adaptive scale factor of shape ``(Ns,)``.  For
            finite-difference types this is converted into edge weights via the
            geometric mean; the returned matrix is the weighted graph
            Laplacian ``G_x^T W_x G_x + G_y^T W_y G_y`` (with physical spacing).
            When ``None`` (default), uniform edge weights of 1 are used.

        Returns
        -------
        tuple[jax.Array, jax.Array | None]
            ``(precision_matrix, logdet_covariance)`` for all regularization types.
            For finite-difference types, ``logdet_covariance`` is ``None`` (callers
            compute ``logdet`` via ``slogdet`` on the returned matrix).
            For GP types, ``logdet_covariance`` is ``log|K|`` extracted from the
            Cholesky factorization — callers compute ``logdet(H) = -logdet_covariance``
            without an extra O(n³) ``slogdet`` call.

        Raises
        ------
        ValueError
            If the bounds produce a non-positive span, or a GP kernel scale is
            missing or non-positive.
        """
        scale = self._check_scale(scale)
        if self.regularization_type == "zero-order":
            mat = self._identity
            if scale is not None:
                mat = mat * scale[:, None]
            return mat, None
        if self.regularization_type == "first-order":
            return self._weighted_first_order_dense(xmin, xmax, ymin, ymax, scale), None
        if self.regularization_type == "second-order":
            return self._weighted_second_order_dense(xmin, xmax, ymin, ymax, scale), None

        if kernel_scale is None:
            raise ValueError("kernel_scale must be provided for GP regularization.")
        precision, logdet_cov = self._gp_matrix(xmin, xmax, ymin, ymax, kernel_scale)
        if scale is not None:
            precision = self._apply_diag_scale(precision, scale)
            # logdet(diag(sqrt(scale)) @ precision @ diag(sqrt(scale)))
            #  = logdet(precision) + sum(log(scale))
            #  = -logdet_cov + sum(log(scale))
            # So logdet_cov_new = logdet_cov - sum(log(scale))
            logdet_cov = logdet_cov - jnp.sum(jnp.log(scale))
        return precision, logdet_cov

    # ------------------------------------------------------------------
    # Matrix-free path  (used by operator / PCG backend)
    # ------------------------------------------------------------------

    def matvec_free(
        self, s: "jax.Array", xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None,
    ) -> "jax.Array":
        r"""Matrix-free :math:`R @ s` for edge-weighted finite-difference regularisation.

        Applies the weighted graph Laplacian

        .. math::
            R = \frac{1}{\Delta x^2} G_x^T W_x G_x
              + \frac{1}{\Delta y^2} G_y^T W_y G_y

        (or the analogous second-order curvature form) in :math:`O(N_s)`
        without materialising the dense :math:`(N_s, N_s)` matrix.

        Edge weights are derived from *scale* as geometric means of adjacent
        pixel scales.

        Raises :exc:`ValueError` for GP types (caller should use the dense
        fallback).
        """
        scale = self._check_scale(scale)
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            raise ValueError("matvec_free is not supported for GP regularization types.")

        scale_factor = self._get_scale(xmin, xmax, ymin, ymax)
        scale_2d = self._scale_to_2d(scale)

        if self.regularization_type == "zero-order":
            if scale is not None:
                return scale * s
            return s
        if self.regularization_type == "first-order":
            return self._weighted_first_order_matvec(s, scale_2d, scale_factor)
        if self.regularization_type == "second-order":
            return self._weighted_second_order_matvec(s, scale_2d, scale_factor)
        raise RuntimeError(f"Unhandled regularization type: {self.regularization_type!r}")

    def _weighted_first_order_block(
        self,
        x_s: int, x_e: int, y_s: int, y_e: int,
        scale_2d: "jax.Array | None",
        scale_factor: "jax.Array",
    ) -> "jax.Array":
        """Principal submatrix of edge-weighted first-order R for one block.

        Constructs the exact principal submatrix ``R[bf, bf]`` directly from
        stencils, without materialising the full ``(Ns, Ns)`` R.  Edges that
        straddle the block boundary (one endpoint inside, one outside)
        contribute their diagonal term to the in-block pixel; the off-diagonal
        entry is outside the submatrix and dropped.
        """
        block_nx = x_e - x_s
        block_ny = y_e - y_s
        block_n = block_nx * block_ny
        weight_dtype = (
            scale_2d.dtype if scale_2d is not None else scale_factor.dtype
        )
        R = jnp.zeros((block_n, block_n), dtype=weight_dtype)

        # --- Horizontal edges: (x, y) -> (x+1, y) for x in [0, n-2] ---
        # Each edge contributes w to both endpoint diagonals and -w to the
        # off-diagonal.  For the principal submatrix, the off-diagonal is
        # kept only when BOTH endpoints are in the block; a straddling edge
        # (one endpoint outside) contributes only w to the in-block diagonal.
        x_h_start = max(0, x_s - 1)
        x_h_end = min(self.n - 2, x_e - 1)
        for x in range(x_h_start, x_h_end + 1):
            left_in = x >= x_s
            right_in = x + 1 < x_e
            if not (left_in or right_in):
                continue
            yy = jnp.arange(y_s, y_e)
            if scale_2d is None:
                w = jnp.ones_like(yy, dtype=jnp.float32) * scale_factor
            else:
                w = self._geom_mean(scale_2d[yy, x], scale_2d[yy, x + 1]) * scale_factor
            if left_in and right_in:
                k1 = (x - x_s) + (yy - y_s) * block_nx
                k2 = k1 + 1
                R = R.at[k1, k1].add(w)
                R = R.at[k1, k2].add(-w)
                R = R.at[k2, k1].add(-w)
                R = R.at[k2, k2].add(w)
            elif left_in:
                k1 = (x - x_s) + (yy - y_s) * block_nx
                R = R.at[k1, k1].add(w)
            else:
                k2 = (x + 1 - x_s) + (yy - y_s) * block_nx
                R = R.at[k2, k2].add(w)

        # Global boundary fallback (last column, x = n-1): diagonal scale_factor
        if x_e == self.n and block_ny > 0:
            yy = jnp.arange(y_s, y_e)
            k = (block_nx - 1) + (yy - y_s) * block_nx
            R = R.at[k, k].add(scale_factor)

        # --- Vertical edges: (x, y) -> (x, y+1) for y in [0, n-2] ---
        y_v_start = max(0, y_s - 1)
        y_v_end = min(self.n - 2, y_e - 1)
        for y in range(y_v_start, y_v_end + 1):
            top_in = y >= y_s
            bot_in = y + 1 < y_e
            if not (top_in or bot_in):
                continue
            xx = jnp.arange(x_s, x_e)
            if scale_2d is None:
                w = jnp.ones_like(xx, dtype=jnp.float32) * scale_factor
            else:
                w = self._geom_mean(scale_2d[y, xx], scale_2d[y + 1, xx]) * scale_factor
            if top_in and bot_in:
                k1 = (xx - x_s) + (y - y_s) * block_nx
                k2 = k1 + block_nx
                R = R.at[k1, k1].add(w)
                R = R.at[k1, k2].add(-w)
                R = R.at[k2, k1].add(-w)
                R = R.at[k2, k2].add(w)
            elif top_in:
                k1 = (xx - x_s) + (y - y_s) * block_nx
                R = R.at[k1, k1].add(w)
            else:
                k2 = (xx - x_s) + (y + 1 - y_s) * block_nx
                R = R.at[k2, k2].add(w)

        # Global boundary fallback (last row, y = n-1): diagonal scale_factor
        if y_e == self.n and block_nx > 0:
            xx = jnp.arange(x_s, x_e)
            k = (xx - x_s) + (block_ny - 1) * block_nx
            R = R.at[k, k].add(scale_factor)

        return 0.5 * (R + R.T)

    def _weighted_first_order_block_vec(
        self,
        x_s: "jax.Array", x_e: "jax.Array",
        y_s: "jax.Array", y_e: "jax.Array",
        scale_2d: "jax.Array | None",
        scale_factor: "jax.Array",
        block_size: int,
    ) -> "jax.Array":
        """Vectorized :meth:`_weighted_first_order_block` for ``lax.scan``.

        Uses fixed-size arrays and boolean masks instead of Python loops,
        so it is safe inside a ``jax.lax.scan`` body where ``x_s, x_e,
        y_s, y_e`` are traced integers.  Assumes all blocks have uniform
        ``block_size × block_size`` dimensions.
        """
        bs = block_size
        block_n = bs * bs
        weight_dtype = (
            scale_2d.dtype if scale_2d is not None else scale_factor.dtype
        )
        R = jnp.zeros((block_n, block_n), dtype=weight_dtype)

        # ---- Horizontal edges ----
        x_h_start = jnp.maximum(0, x_s - 1)
        x_h_end = jnp.minimum(self.n - 2, x_e - 1)
        max_h = bs + 1
        h_off = jnp.arange(max_h, dtype=jnp.int32)
        x_pos = x_h_start + h_off
        h_active = h_off <= (x_h_end - x_h_start)

        left_in = (x_pos >= x_s) & h_active
        right_in = (x_pos + 1 < x_e) & h_active
        both_in = left_in & right_in
        left_only = left_in & ~right_in
        right_only = ~left_in & right_in

        yy_off = jnp.arange(bs, dtype=jnp.int32)
        yy = y_s + yy_off

        if scale_2d is None:
            w = jnp.ones((bs, max_h), dtype=jnp.float32) * scale_factor
        else:
            y_bc = jnp.clip(yy[:, None], 0, self.n - 1)
            x_bc = jnp.clip(x_pos[None, :], 0, self.n - 2)
            x2_bc = jnp.clip(x_pos[None, :] + 1, 0, self.n - 1)
            s1 = scale_2d[y_bc, x_bc]
            s2 = scale_2d[y_bc, x2_bc]
            w = self._geom_mean(s1, s2) * scale_factor

        # Local indices: k1 = (x - x_s) + (y - y_s) * bs
        k1 = (x_pos[None, :] - x_s) + (yy[:, None] - y_s) * bs
        k2 = k1 + 1

        def _scatter_add(R, ki, kj, vals, mask):
            mf = (jnp.ones_like(vals) * mask).ravel().astype(vals.dtype)
            return R.at[ki.ravel(), kj.ravel()].add(vals.ravel() * mf)

        # Case 1: both endpoints in block
        R = _scatter_add(R, k1, k1, w, both_in[None, :])
        R = _scatter_add(R, k1, k2, -w, both_in[None, :])
        R = _scatter_add(R, k2, k1, -w, both_in[None, :])
        R = _scatter_add(R, k2, k2, w, both_in[None, :])
        # Case 2: left only
        R = _scatter_add(R, k1, k1, w, left_only[None, :])
        # Case 3: right only
        R = _scatter_add(R, k2, k2, w, right_only[None, :])

        # Boundary fallback: if x_e == self.n → add scale_factor on last column
        is_last_col = (x_e == self.n).astype(jnp.float32)
        k_bx = (bs - 1) + yy_off * bs
        R = R.at[k_bx, k_bx].add(scale_factor * is_last_col)

        # ---- Vertical edges ----
        y_v_start = jnp.maximum(0, y_s - 1)
        y_v_end = jnp.minimum(self.n - 2, y_e - 1)
        max_v = bs + 1
        v_off = jnp.arange(max_v, dtype=jnp.int32)
        y_pos = y_v_start + v_off
        v_active = v_off <= (y_v_end - y_v_start)

        top_in = (y_pos >= y_s) & v_active
        bot_in = (y_pos + 1 < y_e) & v_active
        both_in_v = top_in & bot_in
        top_only = top_in & ~bot_in
        bot_only = ~top_in & bot_in

        xx_off = jnp.arange(bs, dtype=jnp.int32)
        xx = x_s + xx_off

        if scale_2d is None:
            w_v = jnp.ones((bs, max_v), dtype=jnp.float32) * scale_factor
        else:
            y_bc_v = jnp.clip(y_pos[None, :], 0, self.n - 2)
            x_bc_v = jnp.clip(xx[:, None], 0, self.n - 1)
            y2_bc_v = jnp.clip(y_pos[None, :] + 1, 0, self.n - 1)
            s1_v = scale_2d[y_bc_v, x_bc_v]
            s2_v = scale_2d[y2_bc_v, x_bc_v]
            w_v = self._geom_mean(s1_v, s2_v) * scale_factor

        k1_v = (xx[:, None] - x_s) + (y_pos[None, :] - y_s) * bs
        k2_v = k1_v + bs

        # Case 1: both endpoints in block
        R = _scatter_add(R, k1_v, k1_v, w_v, both_in_v[None, :])
        R = _scatter_add(R, k1_v, k2_v, -w_v, both_in_v[None, :])
        R = _scatter_add(R, k2_v, k1_v, -w_v, both_in_v[None, :])
        R = _scatter_add(R, k2_v, k2_v, w_v, both_in_v[None, :])
        # Case 2: top only
        R = _scatter_add(R, k1_v, k1_v, w_v, top_only[None, :])
        # Case 3: bottom only
        R = _scatter_add(R, k2_v, k2_v, w_v, bot_only[None, :])

        # Boundary fallback: if y_e == self.n → add scale_factor on last row
        is_last_row = (y_e == self.n).astype(jnp.float32)
        k_by = xx_off + (bs - 1) * bs
        R = R.at[k_by, k_by].add(scale_factor * is_last_row)

        return 0.5 * (R + R.T)

    def _weighted_second_order_block(
        self,
        x_s: int, x_e: int, y_s: int, y_e: int,
        scale_2d: "jax.Array | None",
        scale_factor: "jax.Array",
    ) -> "jax.Array":
        """Principal submatrix of edge-weighted second-order R for one block.

        Constructs the exact principal submatrix ``R[bf, bf]`` directly from
        curvature stencils, without materialising the full ``(Ns, Ns)`` R.
        Stencils that straddle the block boundary contribute only the
        sub-entries whose both indices are in-block.
        """
        block_nx = x_e - x_s
        block_ny = y_e - y_s
        block_n = block_nx * block_ny
        weight_dtype = (
            scale_2d.dtype if scale_2d is not None else scale_factor.dtype
        )
        R = jnp.zeros((block_n, block_n), dtype=weight_dtype)

        # --- Horizontal full-curvature: [1, -2, 1] at (x, y), x in [0, n-3] ---
        # Contribution: w * [[1, -2, 1], [-2, 4, -2], [1, -2, 1]] at (x, x+1, x+2)
        # For the principal submatrix, keep only entries where both indices
        # are in-block.  Stencils straddling the block boundary contribute
        # partial submatrices.
        x_fc_start = max(0, x_s - 2)
        x_fc_end = min(self.n - 3, x_e - 1)
        for x in range(x_fc_start, x_fc_end + 1):
            p0_in = x >= x_s
            p1_in = x + 1 >= x_s and x + 1 < x_e
            p2_in = x + 2 >= x_s and x + 2 < x_e
            if not (p0_in or p1_in or p2_in):
                continue
            yy = jnp.arange(y_s, y_e)
            if scale_2d is None:
                w = jnp.ones_like(yy, dtype=jnp.float32) * scale_factor
            else:
                w = self._geom_mean3(
                    scale_2d[yy, x], scale_2d[yy, x + 1], scale_2d[yy, x + 2]
                ) * scale_factor
            # Build local index arrays for the in-block subset
            if p0_in and p1_in and p2_in:
                k0 = (x - x_s) + (yy - y_s) * block_nx
                k1 = k0 + 1
                k2 = k0 + 2
                R = R.at[k0, k0].add(w)
                R = R.at[k0, k1].add(-2.0 * w)
                R = R.at[k0, k2].add(w)
                R = R.at[k1, k0].add(-2.0 * w)
                R = R.at[k1, k1].add(4.0 * w)
                R = R.at[k1, k2].add(-2.0 * w)
                R = R.at[k2, k0].add(w)
                R = R.at[k2, k1].add(-2.0 * w)
                R = R.at[k2, k2].add(w)
            elif p1_in and p2_in:
                # p0 outside, p1 and p2 inside
                k1 = (x + 1 - x_s) + (yy - y_s) * block_nx
                k2 = k1 + 1
                R = R.at[k1, k1].add(4.0 * w)
                R = R.at[k1, k2].add(-2.0 * w)
                R = R.at[k2, k1].add(-2.0 * w)
                R = R.at[k2, k2].add(w)
            elif p0_in and p1_in:
                # p0 and p1 inside, p2 outside
                k0 = (x - x_s) + (yy - y_s) * block_nx
                k1 = k0 + 1
                R = R.at[k0, k0].add(w)
                R = R.at[k0, k1].add(-2.0 * w)
                R = R.at[k1, k0].add(-2.0 * w)
                R = R.at[k1, k1].add(4.0 * w)
            elif p2_in:
                # only p2 inside
                k2 = (x + 2 - x_s) + (yy - y_s) * block_nx
                R = R.at[k2, k2].add(w)
            elif p1_in:
                # only p1 inside (block_nx == 1 case)
                k1 = (x + 1 - x_s) + (yy - y_s) * block_nx
                R = R.at[k1, k1].add(4.0 * w)
            elif p0_in:
                # only p0 inside
                k0 = (x - x_s) + (yy - y_s) * block_nx
                R = R.at[k0, k0].add(w)

        # --- Horizontal near-boundary first-gradient: [-1, 1] at (n-2, y) ---
        # Contribution: w_near * [[1, -1], [-1, 1]] at (n-2, n-1)
        if self.n >= 2:
            near_x = self.n - 2
            p0_in = near_x >= x_s and near_x < x_e
            p1_in = near_x + 1 >= x_s and near_x + 1 < x_e
            if (p0_in or p1_in) and block_ny > 0:
                yy = jnp.arange(y_s, y_e)
                if scale_2d is None:
                    w = jnp.ones_like(yy, dtype=jnp.float32) * scale_factor
                else:
                    w = self._geom_mean(
                        scale_2d[yy, near_x], scale_2d[yy, near_x + 1]
                    ) * scale_factor
                if p0_in and p1_in:
                    k0 = (near_x - x_s) + (yy - y_s) * block_nx
                    k1 = k0 + 1
                    R = R.at[k0, k0].add(w)
                    R = R.at[k0, k1].add(-w)
                    R = R.at[k1, k0].add(-w)
                    R = R.at[k1, k1].add(w)
                elif p0_in:
                    k0 = (near_x - x_s) + (yy - y_s) * block_nx
                    R = R.at[k0, k0].add(w)
                elif p1_in:
                    k1 = (near_x + 1 - x_s) + (yy - y_s) * block_nx
                    R = R.at[k1, k1].add(w)

        # --- Horizontal outer boundary fallback: [1] at (n-1, y) ---
        if x_e == self.n and block_ny > 0:
            yy = jnp.arange(y_s, y_e)
            k = (block_nx - 1) + (yy - y_s) * block_nx
            R = R.at[k, k].add(scale_factor)

        # --- Vertical full-curvature: [1, -2, 1] at (x, y), y in [0, n-3] ---
        y_fc_start = max(0, y_s - 2)
        y_fc_end = min(self.n - 3, y_e - 1)
        for y in range(y_fc_start, y_fc_end + 1):
            p0_in = y >= y_s
            p1_in = y + 1 >= y_s and y + 1 < y_e
            p2_in = y + 2 >= y_s and y + 2 < y_e
            if not (p0_in or p1_in or p2_in):
                continue
            xx = jnp.arange(x_s, x_e)
            if scale_2d is None:
                w = jnp.ones_like(xx, dtype=jnp.float32) * scale_factor
            else:
                w = self._geom_mean3(
                    scale_2d[y, xx], scale_2d[y + 1, xx], scale_2d[y + 2, xx]
                ) * scale_factor
            if p0_in and p1_in and p2_in:
                k0 = (xx - x_s) + (y - y_s) * block_nx
                k1 = k0 + block_nx
                k2 = k1 + block_nx
                R = R.at[k0, k0].add(w)
                R = R.at[k0, k1].add(-2.0 * w)
                R = R.at[k0, k2].add(w)
                R = R.at[k1, k0].add(-2.0 * w)
                R = R.at[k1, k1].add(4.0 * w)
                R = R.at[k1, k2].add(-2.0 * w)
                R = R.at[k2, k0].add(w)
                R = R.at[k2, k1].add(-2.0 * w)
                R = R.at[k2, k2].add(w)
            elif p1_in and p2_in:
                k1 = (xx - x_s) + (y + 1 - y_s) * block_nx
                k2 = k1 + block_nx
                R = R.at[k1, k1].add(4.0 * w)
                R = R.at[k1, k2].add(-2.0 * w)
                R = R.at[k2, k1].add(-2.0 * w)
                R = R.at[k2, k2].add(w)
            elif p0_in and p1_in:
                k0 = (xx - x_s) + (y - y_s) * block_nx
                k1 = k0 + block_nx
                R = R.at[k0, k0].add(w)
                R = R.at[k0, k1].add(-2.0 * w)
                R = R.at[k1, k0].add(-2.0 * w)
                R = R.at[k1, k1].add(4.0 * w)
            elif p2_in:
                k2 = (xx - x_s) + (y + 2 - y_s) * block_nx
                R = R.at[k2, k2].add(w)
            elif p1_in:
                k1 = (xx - x_s) + (y + 1 - y_s) * block_nx
                R = R.at[k1, k1].add(4.0 * w)
            elif p0_in:
                k0 = (xx - x_s) + (y - y_s) * block_nx
                R = R.at[k0, k0].add(w)

        # --- Vertical near-boundary first-gradient: [-1, 1] at (x, n-2) ---
        if self.n >= 2:
            near_y = self.n - 2
            p0_in = near_y >= y_s and near_y < y_e
            p1_in = near_y + 1 >= y_s and near_y + 1 < y_e
            if (p0_in or p1_in) and block_nx > 0:
                xx = jnp.arange(x_s, x_e)
                if scale_2d is None:
                    w = jnp.ones_like(xx, dtype=jnp.float32) * scale_factor
                else:
                    w = self._geom_mean(
                        scale_2d[near_y, xx], scale_2d[near_y + 1, xx]
                    ) * scale_factor
                if p0_in and p1_in:
                    k0 = (xx - x_s) + (near_y - y_s) * block_nx
                    k1 = k0 + block_nx
                    R = R.at[k0, k0].add(w)
                    R = R.at[k0, k1].add(-w)
                    R = R.at[k1, k0].add(-w)
                    R = R.at[k1, k1].add(w)
                elif p0_in:
                    k0 = (xx - x_s) + (near_y - y_s) * block_nx
                    R = R.at[k0, k0].add(w)
                elif p1_in:
                    k1 = (xx - x_s) + (near_y + 1 - y_s) * block_nx
                    R = R.at[k1, k1].add(w)

        # --- Vertical outer boundary fallback: [1] at (x, n-1) ---
        if y_e == self.n and block_nx > 0:
            xx = jnp.arange(x_s, x_e)
            k = (xx - x_s) + (block_ny - 1) * block_nx
            R = R.at[k, k].add(scale_factor)

        return 0.5 * (R + R.T)

    def _weighted_second_order_block_vec(
        self,
        x_s: "jax.Array", x_e: "jax.Array",
        y_s: "jax.Array", y_e: "jax.Array",
        scale_2d: "jax.Array | None",
        scale_factor: "jax.Array",
        block_size: int,
    ) -> "jax.Array":
        """Vectorized :meth:`_weighted_second_order_block` for ``lax.scan``.

        Uses fixed-size arrays and boolean masks instead of Python loops,
        safe inside a ``jax.lax.scan`` body.  Assumes uniform
        ``block_size × block_size`` blocks.
        """
        bs = block_size
        block_n = bs * bs
        weight_dtype = (
            scale_2d.dtype if scale_2d is not None else scale_factor.dtype
        )
        R = jnp.zeros((block_n, block_n), dtype=weight_dtype)
        yy_off = jnp.arange(bs, dtype=jnp.int32)
        xx_off = jnp.arange(bs, dtype=jnp.int32)

        def _scatter_add(R, ki, kj, vals, mask):
            mf = (jnp.ones_like(vals) * mask).ravel().astype(vals.dtype)
            return R.at[ki.ravel(), kj.ravel()].add(vals.ravel() * mf)

        # ---- Horizontal full-curvature [1, -2, 1] ----
        max_h = bs + 2
        h_off = jnp.arange(max_h, dtype=jnp.int32)
        x_pos = jnp.maximum(0, x_s - 2) + h_off
        h_active = h_off <= (jnp.minimum(self.n - 3, x_e - 1) - jnp.maximum(0, x_s - 2))

        p0_in = (x_pos >= x_s) & (x_pos < x_e) & h_active
        p1_in = (x_pos + 1 >= x_s) & (x_pos + 1 < x_e) & h_active
        p2_in = (x_pos + 2 >= x_s) & (x_pos + 2 < x_e) & h_active
        any_in = p0_in | p1_in | p2_in

        p012 = p0_in & p1_in & p2_in
        p12 = (~p0_in) & p1_in & p2_in & any_in
        p01 = p0_in & p1_in & (~p2_in) & any_in
        p2 = (~p0_in) & (~p1_in) & p2_in & any_in
        p1 = (~p0_in) & p1_in & (~p2_in) & any_in
        p0 = p0_in & (~p1_in) & (~p2_in) & any_in

        yy = y_s + yy_off
        y_mask = jnp.ones((bs, 1), dtype=jnp.float32)

        if scale_2d is None:
            w = jnp.ones((bs, max_h), dtype=jnp.float32) * scale_factor
        else:
            y_bc = jnp.clip(yy[:, None], 0, self.n - 1)
            x_bc = jnp.clip(x_pos[None, :], 0, self.n - 3)
            x1_bc = jnp.clip(x_pos[None, :] + 1, 0, self.n - 1)
            x2_bc = jnp.clip(x_pos[None, :] + 2, 0, self.n - 1)
            w = self._geom_mean3(
                scale_2d[y_bc, x_bc],
                scale_2d[y_bc, x1_bc],
                scale_2d[y_bc, x2_bc],
            ) * scale_factor

        k0 = (x_pos[None, :] - x_s) + (yy[:, None] - y_s) * bs
        k1 = k0 + 1
        k2 = k0 + 2

        # p012: full 3x3 kernel
        R = _scatter_add(R, k0, k0, w, p012[None, :] * y_mask)
        R = _scatter_add(R, k0, k1, -2.0 * w, p012[None, :] * y_mask)
        R = _scatter_add(R, k0, k2, w, p012[None, :] * y_mask)
        R = _scatter_add(R, k1, k0, -2.0 * w, p012[None, :] * y_mask)
        R = _scatter_add(R, k1, k1, 4.0 * w, p012[None, :] * y_mask)
        R = _scatter_add(R, k1, k2, -2.0 * w, p012[None, :] * y_mask)
        R = _scatter_add(R, k2, k0, w, p012[None, :] * y_mask)
        R = _scatter_add(R, k2, k1, -2.0 * w, p012[None, :] * y_mask)
        R = _scatter_add(R, k2, k2, w, p012[None, :] * y_mask)
        # p12: right 2×2 sub-kernel
        R = _scatter_add(R, k1, k1, 4.0 * w, p12[None, :] * y_mask)
        R = _scatter_add(R, k1, k2, -2.0 * w, p12[None, :] * y_mask)
        R = _scatter_add(R, k2, k1, -2.0 * w, p12[None, :] * y_mask)
        R = _scatter_add(R, k2, k2, w, p12[None, :] * y_mask)
        # p01: left 2×2 sub-kernel
        R = _scatter_add(R, k0, k0, w, p01[None, :] * y_mask)
        R = _scatter_add(R, k0, k1, -2.0 * w, p01[None, :] * y_mask)
        R = _scatter_add(R, k1, k0, -2.0 * w, p01[None, :] * y_mask)
        R = _scatter_add(R, k1, k1, 4.0 * w, p01[None, :] * y_mask)
        # p2 only
        R = _scatter_add(R, k2, k2, w, p2[None, :] * y_mask)
        # p1 only
        R = _scatter_add(R, k1, k1, 4.0 * w, p1[None, :] * y_mask)
        # p0 only
        R = _scatter_add(R, k0, k0, w, p0[None, :] * y_mask)

        # Horizontal near-boundary [-1, 1] at (n-2, n-1)
        near_x = self.n - 2
        np0_in = (near_x >= x_s) & (near_x < x_e)
        np1_in = (near_x + 1 >= x_s) & (near_x + 1 < x_e)
        near_any = np0_in | np1_in
        if self.n >= 2:
            if scale_2d is None:
                w_near = jnp.ones(bs, dtype=jnp.float32) * scale_factor
            else:
                y_bc_n = jnp.clip(yy, 0, self.n - 1)
                w_near = self._geom_mean(
                    scale_2d[y_bc_n, jnp.clip(near_x, 0, self.n - 1)],
                    scale_2d[y_bc_n, jnp.clip(near_x + 1, 0, self.n - 1)],
                ) * scale_factor
            nk0 = (near_x - x_s) + yy_off * bs
            nk1 = nk0 + 1
            both_n = (np0_in & np1_in).astype(jnp.float32)
            p0_n = (np0_in & ~np1_in).astype(jnp.float32)
            p1_n = (~np0_in & np1_in).astype(jnp.float32)
            R = R.at[nk0, nk0].add(w_near * both_n)
            R = R.at[nk0, nk1].add(-w_near * both_n)
            R = R.at[nk1, nk0].add(-w_near * both_n)
            R = R.at[nk1, nk1].add(w_near * both_n)
            R = R.at[nk0, nk0].add(w_near * p0_n)
            R = R.at[nk1, nk1].add(w_near * p1_n)

        # Horizontal outer boundary fallback [1] at (n-1, y)
        is_last_col = (x_e == self.n).astype(jnp.float32)
        k_lc = (bs - 1) + yy_off * bs
        R = R.at[k_lc, k_lc].add(scale_factor * is_last_col)

        # ---- Vertical full-curvature [1, -2, 1] ----
        max_v = bs + 2
        v_off = jnp.arange(max_v, dtype=jnp.int32)
        y_pos = jnp.maximum(0, y_s - 2) + v_off
        v_active = v_off <= (jnp.minimum(self.n - 3, y_e - 1) - jnp.maximum(0, y_s - 2))

        q0_in = (y_pos >= y_s) & (y_pos < y_e) & v_active
        q1_in = (y_pos + 1 >= y_s) & (y_pos + 1 < y_e) & v_active
        q2_in = (y_pos + 2 >= y_s) & (y_pos + 2 < y_e) & v_active
        q_any = q0_in | q1_in | q2_in

        q012 = q0_in & q1_in & q2_in
        q12 = (~q0_in) & q1_in & q2_in & q_any
        q01 = q0_in & q1_in & (~q2_in) & q_any
        q2 = (~q0_in) & (~q1_in) & q2_in & q_any
        q1 = (~q0_in) & q1_in & (~q2_in) & q_any
        q0 = q0_in & (~q1_in) & (~q2_in) & q_any

        xx = x_s + xx_off
        x_mask = jnp.ones((bs, 1), dtype=jnp.float32)

        if scale_2d is None:
            w_v = jnp.ones((bs, max_v), dtype=jnp.float32) * scale_factor
        else:
            y_bc_v = jnp.clip(y_pos[None, :], 0, self.n - 3)
            y1_bc_v = jnp.clip(y_pos[None, :] + 1, 0, self.n - 1)
            y2_bc_v = jnp.clip(y_pos[None, :] + 2, 0, self.n - 1)
            x_bc_v = jnp.clip(xx[:, None], 0, self.n - 1)
            w_v = self._geom_mean3(
                scale_2d[y_bc_v, x_bc_v],
                scale_2d[y1_bc_v, x_bc_v],
                scale_2d[y2_bc_v, x_bc_v],
            ) * scale_factor

        vk0 = (xx[:, None] - x_s) + (y_pos[None, :] - y_s) * bs
        vk1 = vk0 + bs
        vk2 = vk1 + bs

        # q012: full 3×3 kernel
        R = _scatter_add(R, vk0, vk0, w_v, q012[None, :] * x_mask)
        R = _scatter_add(R, vk0, vk1, -2.0 * w_v, q012[None, :] * x_mask)
        R = _scatter_add(R, vk0, vk2, w_v, q012[None, :] * x_mask)
        R = _scatter_add(R, vk1, vk0, -2.0 * w_v, q012[None, :] * x_mask)
        R = _scatter_add(R, vk1, vk1, 4.0 * w_v, q012[None, :] * x_mask)
        R = _scatter_add(R, vk1, vk2, -2.0 * w_v, q012[None, :] * x_mask)
        R = _scatter_add(R, vk2, vk0, w_v, q012[None, :] * x_mask)
        R = _scatter_add(R, vk2, vk1, -2.0 * w_v, q012[None, :] * x_mask)
        R = _scatter_add(R, vk2, vk2, w_v, q012[None, :] * x_mask)
        # q12
        R = _scatter_add(R, vk1, vk1, 4.0 * w_v, q12[None, :] * x_mask)
        R = _scatter_add(R, vk1, vk2, -2.0 * w_v, q12[None, :] * x_mask)
        R = _scatter_add(R, vk2, vk1, -2.0 * w_v, q12[None, :] * x_mask)
        R = _scatter_add(R, vk2, vk2, w_v, q12[None, :] * x_mask)
        # q01
        R = _scatter_add(R, vk0, vk0, w_v, q01[None, :] * x_mask)
        R = _scatter_add(R, vk0, vk1, -2.0 * w_v, q01[None, :] * x_mask)
        R = _scatter_add(R, vk1, vk0, -2.0 * w_v, q01[None, :] * x_mask)
        R = _scatter_add(R, vk1, vk1, 4.0 * w_v, q01[None, :] * x_mask)
        # q2 only
        R = _scatter_add(R, vk2, vk2, w_v, q2[None, :] * x_mask)
        # q1 only
        R = _scatter_add(R, vk1, vk1, 4.0 * w_v, q1[None, :] * x_mask)
        # q0 only
        R = _scatter_add(R, vk0, vk0, w_v, q0[None, :] * x_mask)

        # Vertical near-boundary [-1, 1] at (n-2, n-1)
        near_y = self.n - 2
        nq0_in = (near_y >= y_s) & (near_y < y_e)
        nq1_in = (near_y + 1 >= y_s) & (near_y + 1 < y_e)
        if self.n >= 2:
            if scale_2d is None:
                w_near_v = jnp.ones(bs, dtype=jnp.float32) * scale_factor
            else:
                x_bc_vn = jnp.clip(xx, 0, self.n - 1)
                w_near_v = self._geom_mean(
                    scale_2d[jnp.clip(near_y, 0, self.n - 1), x_bc_vn],
                    scale_2d[jnp.clip(near_y + 1, 0, self.n - 1), x_bc_vn],
                ) * scale_factor
            vnk0 = (xx - x_s) + (near_y - y_s) * bs
            vnk1 = vnk0 + bs
            both_nv = (nq0_in & nq1_in).astype(jnp.float32)
            q0_nv = (nq0_in & ~nq1_in).astype(jnp.float32)
            q1_nv = (~nq0_in & nq1_in).astype(jnp.float32)
            R = R.at[vnk0, vnk0].add(w_near_v * both_nv)
            R = R.at[vnk0, vnk1].add(-w_near_v * both_nv)
            R = R.at[vnk1, vnk0].add(-w_near_v * both_nv)
            R = R.at[vnk1, vnk1].add(w_near_v * both_nv)
            R = R.at[vnk0, vnk0].add(w_near_v * q0_nv)
            R = R.at[vnk1, vnk1].add(w_near_v * q1_nv)

        # Vertical outer boundary fallback [1] at (n-1, y)
        is_last_row = (y_e == self.n).astype(jnp.float32)
        k_lr = xx_off + (bs - 1) * bs
        R = R.at[k_lr, k_lr].add(scale_factor * is_last_row)

        return 0.5 * (R + R.T)

    def _logdet_block_diag(
        self,
        xmin: float, xmax: float, ymin: float, ymax: float,
        scale: "jax.Array | None",
        block_size: int,
    ) -> "jax.Array":
        r"""Block-diagonal approximation of :math:`\log\det R`.

        When the source grid is uniformly divisible by ``block_size``, uses
        ``jax.lax.scan`` to avoid Python-loop unrolling during JIT tracing.
        Otherwise falls back to the legacy Python-loop path.
        """
        n_blocks = (self.n + block_size - 1) // block_size
        is_uniform = (self.n % block_size == 0) 

        if is_uniform:
            return self._logdet_block_diag_scan(
                xmin, xmax, ymin, ymax, scale, block_size, n_blocks,
            )
        return self._logdet_block_diag_legacy(
            xmin, xmax, ymin, ymax, scale, block_size, n_blocks,
        )

    def _logdet_block_diag_legacy(
        self,
        xmin: float, xmax: float, ymin: float, ymax: float,
        scale: "jax.Array | None",
        block_size: int,
        n_blocks: int,
    ) -> "jax.Array":
        """Legacy Python-loop path for non-uniform source grids."""
        scale_factor = self._get_scale(xmin, xmax, ymin, ymax)
        scale_2d = self._scale_to_2d(scale)
        logdet = jnp.array(0.0, dtype=jnp.float32)

        for by in range(n_blocks):
            for bx in range(n_blocks):
                x_s = bx * block_size
                x_e = min(x_s + block_size, self.n)
                y_s = by * block_size
                y_e = min(y_s + block_size, self.n)

                if self.regularization_type == "zero-order":
                    if scale_2d is not None:
                        block_scale = scale_2d[y_s:y_e, x_s:x_e].ravel()
                    else:
                        block_scale = jnp.ones(
                            (y_e - y_s) * (x_e - x_s), dtype=jnp.float32,
                        )
                    R_block = jnp.diag(block_scale)
                elif self.regularization_type == "first-order":
                    R_block = self._weighted_first_order_block(
                        x_s, x_e, y_s, y_e, scale_2d, scale_factor)
                else:
                    R_block = self._weighted_second_order_block(
                        x_s, x_e, y_s, y_e, scale_2d, scale_factor)

                diag_mean = jnp.mean(jnp.abs(jnp.diag(R_block)))
                jitter_scale = jnp.maximum(diag_mean, 1.0)
                jitter = 1e-6 * jitter_scale * jnp.eye(
                    R_block.shape[0], dtype=R_block.dtype,
                )
                chol = jnp.linalg.cholesky(R_block + jitter)
                logdet = logdet + 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

        return logdet

    def _logdet_block_diag_scan(
        self,
        xmin: float, xmax: float, ymin: float, ymax: float,
        scale: "jax.Array | None",
        block_size: int,
        n_blocks: int,
    ) -> "jax.Array":
        """``lax.scan``-based logdet for uniform source grids."""
        # Precompute block start positions
        block_indices = jnp.arange(n_blocks, dtype=jnp.int32)
        bxs, bys = jnp.meshgrid(block_indices, block_indices, indexing="xy")
        x_starts = (bxs * block_size).ravel().astype(jnp.int32)
        y_starts = (bys * block_size).ravel().astype(jnp.int32)
        scan_inputs = jnp.stack([x_starts, y_starts], axis=-1)

        def scan_body(carry, xs):
            logdet = carry
            x_s = xs[0]
            y_s = xs[1]
            x_e = x_s + block_size
            y_e = y_s + block_size

            R_block = self.block_diag_R_vec(
                x_s, x_e, y_s, y_e,
                xmin, xmax, ymin, ymax,
                scale=scale, block_size=block_size,
            )

            diag_mean = jnp.mean(jnp.abs(jnp.diag(R_block)))
            jitter_scale = jnp.maximum(diag_mean, 1.0)
            jitter = 1e-6 * jitter_scale * jnp.eye(
                block_size * block_size, dtype=R_block.dtype,
            )
            chol = jnp.linalg.cholesky(R_block + jitter)
            block_logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
            return logdet + block_logdet, ()

        init_logdet = jnp.array(0.0, dtype=jnp.float32)
        final_logdet, _ = jax.lax.scan(scan_body, init_logdet, scan_inputs)
        return final_logdet

    def logdet_free(
        self, xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None, block_size: int = 10,
        exact: bool = False,
    ) -> "jax.Array":
        r"""Log-determinant of the finite-difference regularisation matrix.

        Parameters
        ----------
        scale : Array or None, optional
            Per-pixel adaptive scale of shape ``(Ns,)``.  Edge weights are
            derived as geometric means of adjacent pixel scales.
        block_size : int, optional
            Block size for the approximate path (default 10).
        exact : bool, optional
            When ``True``, compute the exact ``slogdet`` of the full
            ``(Ns, Ns)`` R matrix (O(Ns^3), memory O(Ns^2)).  When ``False``
            (default), use a block-diagonal approximation that drops the
            cross-block off-diagonal couplings — faster and lighter, but
            systematically biased high by the Hadamard-Fischer inequality
            (``prod det(R_ii) >= det(R)``).  The operator evidence backend
            uses ``exact=False`` for speed; callers that need exact logdet
            (e.g. for validation) should pass ``exact=True``.

        Raises :exc:`ValueError` for GP types.
        """
        scale = self._check_scale(scale)
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            raise ValueError("logdet_free is not supported for GP regularization types.")
        if self.regularization_type == "zero-order":
            logdet_r = jnp.array(0.0, dtype=jnp.float32)
            if scale is not None:
                logdet_r = logdet_r + jnp.sum(jnp.log(scale))
            return logdet_r
        if exact:
            R_full, _ = self.matrix(xmin, xmax, ymin, ymax, scale=scale)
            _, logdet_r = jnp.linalg.slogdet(R_full)
            return logdet_r
        return self._logdet_block_diag(xmin, xmax, ymin, ymax, scale, block_size)

    def to_dense_free(
        self, xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None,
    ) -> "jax.Array":
        """Materialise the dense ``(N_s, N_s)`` edge-weighted regularisation matrix.

        For finite-difference types this is a thin wrapper around :meth:`matrix`;
        for GP types it raises (use :meth:`matrix` with a ``kernel_scale``).
        """
        scale = self._check_scale(scale)
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            raise ValueError(
                "to_dense_free is not directly supported for GP types. "
                "Use matrix() with a kernel_scale instead."
            )
        mat, _ = self.matrix(xmin, xmax, ymin, ymax, scale=scale)
        return mat

    # ------------------------------------------------------------------
    # Block-diagonal R  (used by block-diagonal preconditioner)
    # ------------------------------------------------------------------

    def _get_scale(self, xmin: float, xmax: float, ymin: float, ymax: float) -> "jax.Array":
        """Return the physical spacing factor for the current FD type.

        - zero-order  : ``1.0``
        - first-order : ``1/dx^2`` (= 1/dy^2 for square grid/bbox)
        - second-order: ``1/dx^4`` (= 1/dy^4 for square grid/bbox)
        """
        inv_dx = (self.n - 1) / (xmax - xmin)

        if self.regularization_type == "zero-order":
            return jnp.array(1.0, dtype=jnp.float32)
        if self.regularization_type == "first-order":
            return jnp.array(inv_dx ** 2, dtype=jnp.float32)
        if self.regularization_type == "second-order":
            return jnp.array(inv_dx ** 4, dtype=jnp.float32)
        raise RuntimeError(f"Unhandled type: {self.regularization_type!r}")

    def block_diag_R(
        self,
        x_start: int, x_end: int,
        y_start: int, y_end: int,
        xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None,
    ) -> "jax.Array":
        r"""Principal submatrix of the edge-weighted R for one block of the grid.

        The block covers source pixels with x-indices ``[x_start, x_end)``
        and y-indices ``[y_start, y_end)``.  The returned matrix is the exact
        principal submatrix ``R[bf, bf]`` of the full edge-weighted R, including
        diagonal contributions from cross-block stencil rows (edges/curvatures
        that straddle the block boundary still contribute to in-block pixel
        diagonals).  Constructed directly from stencils — does NOT materialise
        the full ``(Ns, Ns)`` R matrix.  Only finite-difference types are
        supported.

        Parameters
        ----------
        x_start, x_end : int
            Column (x-axis) index range for the block.
        y_start, y_end : int
            Row (y-axis) index range for the block.
        xmin, xmax, ymin, ymax : float
            Source-plane bounds for FD scaling.
        scale : Array or None, optional
            Full per-pixel scale array of shape ``(Ns,)``.

        Returns
        -------
        Array
            Dense submatrix of shape ``(block_n, block_n)`` where
            ``block_n = (x_end - x_start) * (y_end - y_start)``.
        """
        scale = self._check_scale(scale)
        scale_factor = self._get_scale(xmin, xmax, ymin, ymax)
        scale_2d = self._scale_to_2d(scale)

        if self.regularization_type == "zero-order":
            block_scale = scale_2d[y_start:y_end, x_start:x_end].ravel() if scale_2d is not None else jnp.ones((y_end - y_start) * (x_end - x_start), dtype=jnp.float32)
            return jnp.diag(block_scale)
        if self.regularization_type == "first-order":
            return self._weighted_first_order_block(
                x_start, x_end, y_start, y_end, scale_2d, scale_factor)
        if self.regularization_type == "second-order":
            return self._weighted_second_order_block(
                x_start, x_end, y_start, y_end, scale_2d, scale_factor)
        raise RuntimeError(f"Unhandled type: {self.regularization_type!r}")

    def block_diag_R_vec(
        self,
        x_start: "jax.Array", x_end: "jax.Array",
        y_start: "jax.Array", y_end: "jax.Array",
        xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None, block_size: int = 10,
    ) -> "jax.Array":
        """Vectorized :meth:`block_diag_R` for ``lax.scan`` bodies.

        Dispatches to the vectorized stencil methods.  Assumes all blocks
        have uniform ``block_size × block_size`` dimensions.
        """
        scale = self._check_scale(scale)
        scale_factor = self._get_scale(xmin, xmax, ymin, ymax)
        scale_2d = self._scale_to_2d(scale)

        if self.regularization_type == "zero-order":
            block_scale = (
                jax.lax.dynamic_slice(
                    scale_2d,
                    (y_start, x_start),
                    (block_size, block_size),
                ).ravel()
                if scale_2d is not None
                else jnp.ones(block_size * block_size, dtype=jnp.float32)
            )
            return jnp.diag(block_scale)
        if self.regularization_type == "first-order":
            return self._weighted_first_order_block_vec(
                x_start, x_end, y_start, y_end,
                scale_2d, scale_factor, block_size,
            )
        if self.regularization_type == "second-order":
            return self._weighted_second_order_block_vec(
                x_start, x_end, y_start, y_end,
                scale_2d, scale_factor, block_size,
            )
        raise RuntimeError(f"Unhandled type: {self.regularization_type!r}")

    def diag_R(
        self,
        xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None,
    ) -> "jax.Array":
        r"""Return the diagonal of the edge-weighted R matrix in :math:`O(N_s)`.

        For finite-difference types the diagonal is computed from the weighted
        Laplacian stencil.  For GP types, use ``jnp.diag`` on the full GP
        precision matrix instead.
        """
        scale = self._check_scale(scale)
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            raise ValueError(
                "diag_R is not supported for GP types. "
                "Use jnp.diag on the full GP precision matrix instead."
            )
        if self.regularization_type == "zero-order":
            if scale is not None:
                return scale
            return jnp.ones(self.n_pixels, dtype=jnp.float32)

        scale_factor = self._get_scale(xmin, xmax, ymin, ymax)
        scale_2d = self._scale_to_2d(scale)

        # x/y accumulators are kept separate for clarity but combined with the same
        # scale_factor because the square grid and square bbox guarantee dx == dy.
        weight_dtype = (
            scale_2d.dtype if scale_2d is not None else scale_factor.dtype
        )
        diag_x = jnp.zeros((self.n, self.n), dtype=weight_dtype)
        diag_y = jnp.zeros((self.n, self.n), dtype=weight_dtype)

        if self.regularization_type == "first-order":
            w_x, w_y = self._edge_weights_first_order(scale_2d)
            if self.n > 1:
                # each interior pixel has left and right horizontal edges
                diag_x = diag_x.at[:, 1:-1].add(w_x[:, :-1] + w_x[:, 1:])
                diag_x = diag_x.at[:, 0].add(w_x[:, 0])
                diag_x = diag_x.at[:, -1].add(w_x[:, -1] + 1.0)  # boundary fallback
            if self.n > 1:
                diag_y = diag_y.at[1:-1, :].add(w_y[:-1, :] + w_y[1:, :])
                diag_y = diag_y.at[0, :].add(w_y[0, :])
                diag_y = diag_y.at[-1, :].add(w_y[-1, :] + 1.0)
        elif self.regularization_type == "second-order":
            w_x2, w_y2, w_x2_near, w_y2_near = self._edge_weights_second_order(scale_2d)
            if self.n > 2:
                # Curvature stencil [1, -2, 1]: each curvature centre contributes
                #  1²=1 to the left-wing pixel, (-2)²=4 to the centre pixel,
                #  and 1²=1 to the right-wing pixel.
                diag_x = diag_x.at[:, :-2].add(w_x2)            # left wing:  +1
                diag_x = diag_x.at[:, 1:-1].add(4.0 * w_x2)     # centre:     (-2)²=4
                diag_x = diag_x.at[:, 2:].add(w_x2)             # right wing: +1
            if self.n > 1:
                # near-boundary first-gradient: each of the two pixels gets
                #  (±1)²·w = 1·w from this edge (the edge contributes w to
                #  EACH pixel's diagonal, not 2w).
                diag_x = diag_x.at[:, -2].add(w_x2_near)
                diag_x = diag_x.at[:, -1].add(w_x2_near + 1.0)  # outer boundary
            if self.n > 2:
                diag_y = diag_y.at[:-2, :].add(w_y2)
                diag_y = diag_y.at[1:-1, :].add(4.0 * w_y2)
                diag_y = diag_y.at[2:, :].add(w_y2)
            if self.n > 1:
                diag_y = diag_y.at[-2, :].add(w_y2_near)
                diag_y = diag_y.at[-1, :].add(w_y2_near + 1.0)

        diag_2d = scale_factor * (diag_x + diag_y)

        return diag_2d.ravel()

    def make_reg_data(
        self,
        xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None,
    ) -> RegData:
        """Return a compact :class:`RegData` tuple for the operator backend.

        Carries the per-pixel adaptive ``scale`` array and the physical spacing
        factors; the JIT matvec computes edge weights from ``scale`` on the fly.
        Only finite-difference types are supported.
        """
        scale = self._check_scale(scale)
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            raise ValueError(
                "Operator backend does not support GP regularization types. "
                "Use the dense backend (PixelizedImageProbModel) for GP regularization."
            )
        scl = self._get_scale(xmin, xmax, ymin, ymax)
        return RegData(scale=scale, scale_factor=scl)

    def _build_first_difference_operators(self):
        """Return first-order x/y finite-difference operators on index space.

        Uses vectorized index-based construction instead of per-element
        loops, giving O(1) JAX array operations regardless of grid size.
        Boundary rows use the Suyu et al. zero-order fallback (diagonal 1).
        """
        ix = jnp.arange(self.n)
        iy = jnp.arange(self.n)
        gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
        flat_idx = (gy * self.n + gx).ravel()

        # Interior x-differences: idx -> idx+1
        interior_x = gx < (self.n - 1)
        interior_rows_x = flat_idx[interior_x.ravel()]
        interior_diag_x = interior_rows_x
        interior_off_x = interior_rows_x + 1

        # Boundary x (last column): diagonal 1
        boundary_x = gx == (self.n - 1)
        boundary_rows_x = flat_idx[boundary_x.ravel()]

        dx_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        dx_operator = dx_operator.at[interior_diag_x, interior_diag_x].add(-1.0)
        dx_operator = dx_operator.at[interior_rows_x, interior_off_x].add(1.0)
        dx_operator = dx_operator.at[boundary_rows_x, boundary_rows_x].add(1.0)

        # Interior y-differences: idx -> idx+n
        interior_y = gy < (self.n - 1)
        interior_rows_y = flat_idx[interior_y.ravel()]
        interior_diag_y = interior_rows_y
        interior_off_y = interior_rows_y + self.n

        # Boundary y (last row): diagonal 1
        boundary_y = gy == (self.n - 1)
        boundary_rows_y = flat_idx[boundary_y.ravel()]

        dy_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        dy_operator = dy_operator.at[interior_diag_y, interior_diag_y].add(-1.0)
        dy_operator = dy_operator.at[interior_rows_y, interior_off_y].add(1.0)
        dy_operator = dy_operator.at[boundary_rows_y, boundary_rows_y].add(1.0)

        return dx_operator, dy_operator

    def _build_curvature_difference_operators(self):
        """Return second-order x/y finite-difference operators on index space.

        Uses vectorized index-based construction instead of per-element
        loops, giving O(1) JAX array operations regardless of grid size.
        Near-boundary curvature reduces to first gradient; outer boundary
        uses zero-order fallback (diagonal 1).
        """
        ix = jnp.arange(self.n)
        iy = jnp.arange(self.n)
        gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
        flat_idx = (gy * self.n + gx).ravel()

        # == X curvature operator ==
        # Full curvature (3-point stencil): ix < n-2
        full_x = gx < (self.n - 2)
        full_rows_x = flat_idx[full_x.ravel()]

        # Near-boundary (2-point first gradient): ix == n-2
        near_x = gx == (self.n - 2)
        near_rows_x = flat_idx[near_x.ravel()]

        # Outer boundary (diagonal): ix == n-1
        outer_x = gx == (self.n - 1)
        outer_rows_x = flat_idx[outer_x.ravel()]

        lx_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        lx_operator = lx_operator.at[full_rows_x, full_rows_x].add(1.0)
        lx_operator = lx_operator.at[full_rows_x, full_rows_x + 1].add(-2.0)
        lx_operator = lx_operator.at[full_rows_x, full_rows_x + 2].add(1.0)
        lx_operator = lx_operator.at[near_rows_x, near_rows_x].add(-1.0)
        lx_operator = lx_operator.at[near_rows_x, near_rows_x + 1].add(1.0)
        lx_operator = lx_operator.at[outer_rows_x, outer_rows_x].add(1.0)

        # == Y curvature operator ==
        # Full curvature (3-point stencil): iy < n-2
        full_y = gy < (self.n - 2)
        full_rows_y = flat_idx[full_y.ravel()]

        # Near-boundary (2-point first gradient): iy == n-2
        near_y = gy == (self.n - 2)
        near_rows_y = flat_idx[near_y.ravel()]

        # Outer boundary (diagonal): iy == n-1
        outer_y = gy == (self.n - 1)
        outer_rows_y = flat_idx[outer_y.ravel()]

        ly_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        ly_operator = ly_operator.at[full_rows_y, full_rows_y].add(1.0)
        ly_operator = ly_operator.at[full_rows_y, full_rows_y + self.n].add(-2.0)
        ly_operator = ly_operator.at[full_rows_y, full_rows_y + 2 * self.n].add(1.0)
        ly_operator = ly_operator.at[near_rows_y, near_rows_y].add(-1.0)
        ly_operator = ly_operator.at[near_rows_y, near_rows_y + self.n].add(1.0)
        ly_operator = ly_operator.at[outer_rows_y, outer_rows_y].add(1.0)

        return lx_operator, ly_operator

    def _build_unit_coordinates(self):
        """Return source-grid coordinates for a unit half-size plane."""
        x_axis = jnp.linspace(-1.0, 1.0, self.n)
        y_axis = jnp.linspace(-1.0, 1.0, self.n)
        source_x_mesh, source_y_mesh = jnp.meshgrid(x_axis, y_axis, indexing='xy')
        return jnp.stack(
            [source_x_mesh.reshape(-1), source_y_mesh.reshape(-1)],
            axis=1,
        )

    def _gp_matrix(self, xmin, xmax, ymin, ymax, kernel_scale: float):
        """Return (precision, logdet_covariance) for a GP regularization matrix.

        Builds physical pixel coordinates from the source-plane bounds and
        computes pairwise Euclidean distances in physical space.

        Uses Cholesky decomposition to compute both the precision matrix K^{-1}
        and log|K| in a single O(n^3) factorization, avoiding a separate slogdet call.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            ``(precision, logdet_covariance)`` where precision is ``K^{-1}`` of shape
            ``(n_pixels, n_pixels)`` and ``logdet_covariance`` is the scalar ``log|K|``.
        """
        x_scale = (xmax - xmin) / 2.0
        y_scale = (ymax - ymin) / 2.0
        x_shift = (xmax + xmin) / 2.0
        y_shift = (ymax + ymin) / 2.0
        x_phys = self._unit_coordinates[:, 0] * x_scale + x_shift
        y_phys = self._unit_coordinates[:, 1] * y_scale + y_shift
        delta_x = x_phys[:, None] - x_phys[None, :]
        delta_y = y_phys[:, None] - y_phys[None, :]
        distances = jnp.sqrt(delta_x ** 2 + delta_y ** 2)
        r = distances / kernel_scale

        if self.regularization_type == "exponential":
            covariance = jnp.exp(-r)
        elif self.regularization_type == "gaussian":
            covariance = jnp.exp(-0.5 * r ** 2)
        elif self.regularization_type == "matern32":
            sqrt3_r = jnp.sqrt(3.0) * r
            covariance = (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)
        elif self.regularization_type == "matern52":
            sqrt5_r = jnp.sqrt(5.0) * r
            covariance = (1.0 + sqrt5_r + 5.0 * r ** 2 / 3.0) * jnp.exp(-sqrt5_r)
        elif self.regularization_type == "matern72":
            sqrt7_r = jnp.sqrt(7.0) * r
            covariance = (1.0 + sqrt7_r + 14.0 * r ** 2 / 5.0 + 7.0 * jnp.sqrt(7.0) * r ** 3 / 15.0) * jnp.exp(-sqrt7_r)
        else:
            raise RuntimeError(f"Unhandled GP regularization type: {self.regularization_type!r}")

        stabilized = covariance + self.jitter * self._identity
        # Single Cholesky factorization gives both precision and logdet(K).
        # logdet(K) = 2 * sum(log(diag(L)))  where L = chol(K)
        # K^{-1} via cho_solve avoids a second O(n^3) inv() call.
        chol_k = jnp.linalg.cholesky(stabilized)
        logdet_covariance = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_k)))
        eye = jnp.eye(self.n_pixels, dtype=stabilized.dtype)
        precision = jsl.cho_solve((chol_k, True), eye)
        # Enforce exact symmetry — cho_solve may introduce tiny asymmetry
        # that can break downstream Cholesky factorizations.
        precision = 0.5 * (precision + precision.T)
        return precision, logdet_covariance

__all__ = ["DenseRegularizationBuilder", "RegData", "VALID_REGULARIZATION_TYPES", "GP_REGULARIZATION_TYPES"]
