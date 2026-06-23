"""Dense regularization matrix builders for pixelized source inversion.

This module provides traditional finite-difference penalties and dense
Gaussian-process covariance penalties for source-plane linear inversion.  The
finite-difference operators are precomputed on index space and scaled by the
physical source-grid spacing when a matrix is requested.

For matrix-free (operator) backends, the class also exposes
:meth:`matvec_free`, :meth:`logdet_free`, :meth:`to_dense_free`, and
:meth:`make_reg_data` for finite-difference types, which exploit the
Kronecker-sum structure of separable 2-D difference operators to avoid
materialising the full Ns x Ns regularisation matrix.

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


class RegData(NamedTuple):
    """Compact finite-difference regularisation data for matrix-free matvec / logdet.

    Passed through ``A_data[7]`` in the PCG solver so that
    :func:`_A_matvec_jit` can apply the regularisation term without a dense
    ``(Ns, Ns)`` matrix.

    ``rx`` / ``ry`` hold the 1-D product matrices ``(nx, nx)`` / ``(ny, ny)``
    and ``scale_x`` / ``scale_y`` are the physical pixel-area scaling factors.

    ``scale`` is an optional per-pixel regularisation scale array of shape
    ``(Ns,)`` for adaptive regularisation.  When ``None`` (default), uniform
    regularisation is used.

    Note: ``nx`` / ``ny`` are *not* stored here; they are passed as static
    arguments to :func:`_A_matvec_jit` via the pre-bound partial.

    Only finite-difference types are supported; GP types should use the dense
    backend instead.
    """
    rx: "jax.Array"        # (nx, nx)  1-D x-regularisation product
    ry: "jax.Array"        # (ny, ny)  1-D y-regularisation product
    scale_x: "jax.Array"   # scalar  physical pixel-area scale for x
    scale_y: "jax.Array"   # scalar  physical pixel-area scale for y
    scale: "jax.Array | None" = None  # (Ns,) per-pixel reg scale, or None=uniform


class DenseRegularizationBuilder:
    """Build dense source-grid regularization matrices.

    Parameters
    ----------
    nx : int
        Number of source pixels along the x-axis.
    ny : int
        Number of source pixels along the y-axis.
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
        nx: int,
        ny: int,
        regularization_type: str,
        *,
        jitter: float = 1e-6,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must both be at least 2.")

        self.n_pixels = self.nx * self.ny
        self.regularization_type = regularization_type.lower()
        self.jitter = float(jitter)

        if self.regularization_type not in self._VALID_TYPES:
            raise ValueError(f"Unsupported regularization_type: {regularization_type!r}.")
        if self.jitter < 0.0:
            raise ValueError("jitter must be non-negative.")

        self._identity = jnp.eye(self.n_pixels)
        self._dx_operator, self._dy_operator = self._build_first_difference_operators()
        self._lx_operator, self._ly_operator = self._build_curvature_difference_operators()
        self._unit_coordinates = self._build_unit_coordinates()

        # Precompute H_unit matrices on index space (half_size=1, spacing=2/(n-1)).
        # H(h) = H_unit / dx(h)^2  for first-order,  / dx(h)^4  for second-order.
        # dx(h) = 2h/(nx-1), dy(h) = 2h/(ny-1).
        # Separate x/y contributions so non-square grids (nx != ny) are handled correctly.
        dx_unit = 2.0 / (self.nx - 1)
        dy_unit = 2.0 / (self.ny - 1)
        sdx1 = self._dx_operator / dx_unit
        sdy1 = self._dy_operator / dy_unit
        self._H1_unit_x = sdx1.T @ sdx1   # x contribution at half_size=1
        self._H1_unit_y = sdy1.T @ sdy1   # y contribution at half_size=1
        sdx2 = self._lx_operator / (dx_unit ** 2)
        sdy2 = self._ly_operator / (dy_unit ** 2)
        self._H2_unit_x = sdx2.T @ sdx2
        self._H2_unit_y = sdy2.T @ sdy2

        # ----- 1-D operator products & eigendecompositions for matrix-free path -----
        self._Rx1: "jax.Array | None" = None
        self._Ry1: "jax.Array | None" = None
        self._Rx2: "jax.Array | None" = None
        self._Ry2: "jax.Array | None" = None
        self._Rx_eigvals: "jax.Array | None" = None
        self._Ry_eigvals: "jax.Array | None" = None

        if self.regularization_type not in GP_REGULARIZATION_TYPES:
            if self.regularization_type in ("first-order", "second-order"):
                Dx1d = self._build_1d_first_diff(self.nx)
                Dy1d = self._build_1d_first_diff(self.ny)
                self._Rx1 = Dx1d.T @ Dx1d
                self._Ry1 = Dy1d.T @ Dy1d

            if self.regularization_type == "second-order":
                Lx1d = self._build_1d_curvature_diff(self.nx)
                Ly1d = self._build_1d_curvature_diff(self.ny)
                self._Rx2 = Lx1d.T @ Lx1d
                self._Ry2 = Ly1d.T @ Ly1d

            # Precompute eigenvalues once (used by logdet_free at each evidence eval).
            if self.regularization_type == "first-order":
                self._Rx_eigvals = jnp.linalg.eigvalsh(self._Rx1)  # type: ignore[arg-type]
                self._Ry_eigvals = jnp.linalg.eigvalsh(self._Ry1)  # type: ignore[arg-type]
            elif self.regularization_type == "second-order":
                self._Rx_eigvals = jnp.linalg.eigvalsh(self._Rx2)  # type: ignore[arg-type]
                self._Ry_eigvals = jnp.linalg.eigvalsh(self._Ry2)  # type: ignore[arg-type]
            # zero-order: eigenvalues stay None (logdet=0, matvec=identity)

    # ------------------------------------------------------------------
    # 1-D difference operator builders (used by matrix-free path)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_1d_first_diff(n: int) -> "jax.Array":
        """Build 1-D first-difference operator ``(n, n)``.

        Interior rows use ``[-1, 1]``; the last row is a zero-order
        fallback ``[1]`` on the diagonal (Suyu et al. convention).
        """
        D = jnp.zeros((n, n))
        interior = jnp.arange(n - 1)
        D = D.at[interior, interior].add(-1.0)
        D = D.at[interior, interior + 1].add(1.0)
        D = D.at[n - 1, n - 1].add(1.0)
        return D

    @staticmethod
    def _build_1d_curvature_diff(n: int) -> "jax.Array":
        """Build 1-D curvature (second-difference) operator ``(n, n)``.

        Full-curvature rows use ``[1, -2, 1]``, the penultimate row
        falls back to a first-gradient ``[-1, 1]``, and the outermost
        row is a zero-order diagonal ``[1]``.
        """
        L = jnp.zeros((n, n))
        full = jnp.arange(n - 2)
        L = L.at[full, full].add(1.0)
        L = L.at[full, full + 1].add(-2.0)
        L = L.at[full, full + 2].add(1.0)
        L = L.at[n - 2, n - 2].add(-1.0)
        L = L.at[n - 2, n - 1].add(1.0)
        L = L.at[n - 1, n - 1].add(1.0)
        return L

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
        except Exception:
            # ``scale`` may be a tracer inside a JIT-compiled likelihood.  Its
            # shape is still checked above; dynamic values are produced by
            # bounded model code such as adaptive_reg_floor.
            is_valid = True
        if not is_valid:
            raise ValueError("scale values must be finite and strictly positive.")
        return scale_arr

    def _apply_diag_scale(self, matrix: "jax.Array", scale: "jax.Array | None") -> "jax.Array":
        """If *scale* is not None, return ``diag(sqrt(scale)) @ matrix @ diag(sqrt(scale))``.

        Otherwise return *matrix* unchanged.
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

    def matrix(self, xmin, xmax, ymin, ymax, *, kernel_scale: float | None = None, scale: "jax.Array | None" = None):
        """Return the dense regularization matrix for a rectangular source grid.

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
            Per-pixel regularization scale factor of shape ``(Ns,)``.
            When provided, the returned matrix is
            ``diag(sqrt(scale)) @ R @ diag(sqrt(scale))``.
            When ``None`` (default), uniform regularisation is used.

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
            return self._apply_diag_scale(mat, scale), None
        if self.regularization_type == "first-order":
            mat = self._first_order_matrix(xmin, xmax, ymin, ymax)
            return self._apply_diag_scale(mat, scale), None
        if self.regularization_type == "second-order":
            mat = self._second_order_matrix(xmin, xmax, ymin, ymax)
            return self._apply_diag_scale(mat, scale), None

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
        r"""Matrix-free :math:`R @ s` for finite-difference regularisation.

        Exploits the Kronecker-sum structure

        .. math::
            R = I_{ny} \otimes \frac{R_x}{\Delta x^2}
              + \frac{R_y}{\Delta y^2} \otimes I_{nx}

        to apply :math:`R` in :math:`O(N_s (n_x + n_y))` without
        materialising the dense :math:`(N_s, N_s)` matrix.

        When *scale* is provided, applies
        :math:`\text{diag}(\sqrt{\text{scale}}) \cdot R \cdot
        \text{diag}(\sqrt{\text{scale}})`.

        Raises :exc:`ValueError` for GP types (caller should use the dense
        fallback).
        """
        scale = self._check_scale(scale)
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            raise ValueError("matvec_free is not supported for GP regularization types.")
        if self.regularization_type == "zero-order":
            if scale is not None:
                return scale * s
            return s

        # Pre-scale: s' = sqrt(scale) * s
        if scale is not None:
            sqrt_scale = jnp.sqrt(scale)
            s = sqrt_scale * s

        # Physical pixel scales: dx = (xmax - xmin) / (nx - 1)
        rx, ry, scale_x, scale_y = self._get_rx_ry_scales(xmin, xmax, ymin, ymax)

        # s has flat index ``i + j * nx`` (column-major on (nx, ny)).
        # Reshape to (ny, nx), transpose → (nx, ny) where axis 0 = x.
        s_2d = s.reshape(self.ny, self.nx).T  # (nx, ny)
        result_2d = rx @ s_2d * scale_x + s_2d @ ry.T * scale_y
        result = result_2d.T.ravel()  # back to flat (Ns,)

        # Post-scale: result = sqrt(scale) * (R @ (sqrt(scale) * s))
        if scale is not None:
            result = sqrt_scale * result
        return result

    def logdet_free(
        self, xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None,
    ) -> "jax.Array":
        r"""Eigenvalue-based :math:`\log\det R` for finite-difference regularisation.

        Uses the Kronecker-sum eigenvalue formula

        .. math::
            \lambda_{ij} = \frac{\mu_i}{\Delta x^2} + \frac{\nu_j}{\Delta y^2}

        where :math:`\mu_i` and :math:`\nu_j` are the eigenvalues of the
        1-D product matrices :math:`R_x` and :math:`R_y`.

        When *scale* is provided, returns
        :math:`\log\det(\text{diag}(\sqrt{\text{scale}}) \cdot R \cdot
        \text{diag}(\sqrt{\text{scale}})) = \log\det R + \sum_i \log(\text{scale}_i)`.

        Complexity :math:`O(n_x^3 + n_y^3)` for the initial eigendecomposition
        (done once in ``__init__``) and :math:`O(n_x n_y)` per call.

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

        _, _, scale_x, scale_y = self._get_rx_ry_scales(xmin, xmax, ymin, ymax)

        # λ_{i,j} = μ_i * scale_x + ν_j * scale_y
        eig_grid = (self._Rx_eigvals[:, None] * scale_x
                    + self._Ry_eigvals[None, :] * scale_y)  # (nx, ny)
        # Guard against numerical zeros from the boundary treatment.
        eig_grid = jnp.maximum(eig_grid, 1e-30)
        logdet_r = jnp.sum(jnp.log(eig_grid))

        if scale is not None:
            logdet_r = logdet_r + jnp.sum(jnp.log(scale))
        return logdet_r

    def to_dense_free(
        self, xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None,
    ) -> "jax.Array":
        """Materialise the dense ``(N_s, N_s)`` regularisation matrix.

        For finite-difference types this uses the Kronecker representation;
        for GP types it delegates to :meth:`matrix` (identical result).

        When *scale* is provided, returns
        ``diag(sqrt(scale)) @ R @ diag(sqrt(scale))``.

        Intended for one-shot use in :meth:`~PixelizedLensOperator.build_preconditioner`
        where an explicit ``R`` is still needed.
        """
        scale = self._check_scale(scale)
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            # GP types don't have a meaningful kernel_scale here; callers
            # should go through _regularization_data which supplies one.
            raise ValueError(
                "to_dense_free is not directly supported for GP types. "
                "Use matrix() with a kernel_scale instead."
            )
        if self.regularization_type == "zero-order":
            return self._apply_diag_scale(self._identity, scale)

        rx, ry, scale_x, scale_y = self._get_rx_ry_scales(xmin, xmax, ymin, ymax)

        # R = I_ny ⊗ (rx * scale_x) + (ry * scale_y) ⊗ I_nx
        mat = jnp.kron(jnp.eye(self.ny, dtype=rx.dtype), rx * scale_x) \
            + jnp.kron(ry * scale_y, jnp.eye(self.nx, dtype=ry.dtype))
        return self._apply_diag_scale(mat, scale)

    # ------------------------------------------------------------------
    # Block-diagonal R  (used by block-diagonal preconditioner)
    # ------------------------------------------------------------------

    def _get_rx_ry_scales(
        self, xmin: float, xmax: float, ymin: float, ymax: float,
    ) -> tuple:
        """Return ``(rx, ry, scale_x, scale_y)`` for the current FD type."""
        inv_dx = (self.nx - 1) / (xmax - xmin)
        inv_dy = (self.ny - 1) / (ymax - ymin)

        if self.regularization_type == "zero-order":
            rx = 0.5 * jnp.eye(self.nx, dtype=jnp.float32)
            ry = 0.5 * jnp.eye(self.ny, dtype=jnp.float32)
            scale_x = jnp.array(1.0, dtype=jnp.float32)
            scale_y = jnp.array(1.0, dtype=jnp.float32)
        elif self.regularization_type == "first-order":
            rx = self._Rx1
            ry = self._Ry1
            scale_x = jnp.array(inv_dx ** 2, dtype=rx.dtype)
            scale_y = jnp.array(inv_dy ** 2, dtype=ry.dtype)
        elif self.regularization_type == "second-order":
            rx = self._Rx2
            ry = self._Ry2
            scale_x = jnp.array(inv_dx ** 4, dtype=rx.dtype)
            scale_y = jnp.array(inv_dy ** 4, dtype=ry.dtype)
        else:
            raise RuntimeError(f"Unhandled type: {self.regularization_type!r}")
        return rx, ry, scale_x, scale_y

    def block_diag_R(
        self,
        x_start: int, x_end: int,
        y_start: int, y_end: int,
        xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None,
    ) -> "jax.Array":
        r"""Build the R submatrix for one block of the source grid.

        The block covers source pixels with x-indices ``[x_start, x_end)``
        and y-indices ``[y_start, y_end)``, using the column-major source
        flat-index convention ``s = x + y * nx``.

        The Kronecker-sum structure is exploited:
        :math:`R_{\rm block} = I_{n_y^b} \otimes (R_x^{\rm sub} \cdot s_x)
        + (R_y^{\rm sub} \cdot s_y) \otimes I_{n_x^b}`.

        When *scale* is provided, extracts the corresponding block of the
        full scale array and applies
        ``diag(sqrt(scale_block)) @ R_block @ diag(sqrt(scale_block))``.

        Only finite-difference types are supported.

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
        block_nx = x_end - x_start
        block_ny = y_end - y_start

        rx, ry, scale_x, scale_y = self._get_rx_ry_scales(xmin, xmax, ymin, ymax)

        rx_sub = rx[x_start:x_end, x_start:x_end] * scale_x  # (block_nx, block_nx)
        ry_sub = ry[y_start:y_end, y_start:y_end] * scale_y  # (block_ny, block_ny)

        # R_block = I_{block_ny} ⊗ rx_sub + ry_sub ⊗ I_{block_nx}
        eye_y = jnp.eye(block_ny, dtype=rx_sub.dtype)
        eye_x = jnp.eye(block_nx, dtype=ry_sub.dtype)
        R_block = jnp.kron(eye_y, rx_sub) + jnp.kron(ry_sub, eye_x)

        if scale is not None:
            # Extract scale values for this block (column-major flat indexing)
            sx = jnp.arange(x_start, x_end)
            sy = jnp.arange(y_start, y_end)
            block_flat_idx = (sx[:, None] + sy[None, :] * self.nx).ravel(order='F')
            scale_block = scale[block_flat_idx]
            R_block = self._apply_diag_scale(R_block, scale_block)
        return R_block

    def diag_R(
        self,
        xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None,
    ) -> "jax.Array":
        r"""Return the diagonal of the dense R matrix in :math:`O(N_s)`.

        Uses the Kronecker-sum structure:
        :math:`\operatorname{diag}(R)_k = s_x \cdot \operatorname{diag}(R_x)_{i}
        + s_y \cdot \operatorname{diag}(R_y)_{j}`
        where :math:`k = i + j \cdot n_x`.

        When *scale* is provided, returns the diagonal of
        ``diag(sqrt(scale)) @ R @ diag(sqrt(scale))``, which simplifies to
        ``scale * diag(R)`` (since D is diagonal).

        For GP types, falls back to ``jnp.diag`` of the full matrix (caller
        must supply ``gp_matrix`` via :meth:`matrix` — this method does not
        accept it directly; the caller should pre-build the GP precision and
        extract its diagonal).
        """
        scale = self._check_scale(scale)
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            raise ValueError(
                "diag_R is not supported for GP types. "
                "Use jnp.diag on the full GP precision matrix instead."
            )
        if self.regularization_type == "zero-order":
            diag = jnp.ones(self.n_pixels, dtype=jnp.float32)
            if scale is not None:
                diag = scale * diag
            return diag

        rx, ry, scale_x, scale_y = self._get_rx_ry_scales(xmin, xmax, ymin, ymax)
        diag_rx = jnp.diag(rx) * scale_x  # (nx,)
        diag_ry = jnp.diag(ry) * scale_y  # (ny,)

        # diag(R)[i + j * nx] = diag_rx[i] + diag_ry[j]
        diag_r = diag_rx[:, None] + diag_ry[None, :]  # (nx, ny)
        diag_r = diag_r.T.ravel()  # match column-major flat order

        if scale is not None:
            diag_r = scale * diag_r
        return diag_r

    def make_reg_data(
        self,
        xmin: float, xmax: float, ymin: float, ymax: float,
        *, scale: "jax.Array | None" = None,
    ) -> RegData:
        """Return a compact :class:`RegData` tuple for passing through ``A_data``.

        Only supports finite-difference regularization types.  GP types are
        not supported by the operator backend.

        When *scale* is provided, it is stored in the returned :class:`RegData`
        for use by the matrix-free matvec / logdet primitives.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : float
            Source-plane bounds.
        scale : Array or None, optional
            Per-pixel regularization scale of shape ``(Ns,)``.

        Raises
        ------
        ValueError
            If ``regularization_type`` is a GP type.
        """
        scale = self._check_scale(scale)
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            raise ValueError(
                "Operator backend does not support GP regularization types. "
                "Use the dense backend (PixelizedImageProbModel) for GP regularization."
            )

        rx, ry, scl_x, scl_y = self._get_rx_ry_scales(xmin, xmax, ymin, ymax)
        return RegData(rx=rx, ry=ry, scale_x=scl_x, scale_y=scl_y, scale=scale)

    def _build_first_difference_operators(self):
        """Return first-order x/y finite-difference operators on index space.

        Uses vectorized index-based construction instead of per-element
        loops, giving O(1) JAX array operations regardless of grid size.
        Boundary rows use the Suyu et al. zero-order fallback (diagonal 1).
        """
        ix = jnp.arange(self.nx)
        iy = jnp.arange(self.ny)
        gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
        flat_idx = (gy * self.nx + gx).ravel()

        # Interior x-differences: idx -> idx+1
        interior_x = gx < (self.nx - 1)
        interior_rows_x = flat_idx[interior_x.ravel()]
        interior_diag_x = interior_rows_x
        interior_off_x = interior_rows_x + 1

        # Boundary x (last column): diagonal 1
        boundary_x = gx == (self.nx - 1)
        boundary_rows_x = flat_idx[boundary_x.ravel()]

        dx_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        dx_operator = dx_operator.at[interior_diag_x, interior_diag_x].add(-1.0)
        dx_operator = dx_operator.at[interior_rows_x, interior_off_x].add(1.0)
        dx_operator = dx_operator.at[boundary_rows_x, boundary_rows_x].add(1.0)

        # Interior y-differences: idx -> idx+nx
        interior_y = gy < (self.ny - 1)
        interior_rows_y = flat_idx[interior_y.ravel()]
        interior_diag_y = interior_rows_y
        interior_off_y = interior_rows_y + self.nx

        # Boundary y (last row): diagonal 1
        boundary_y = gy == (self.ny - 1)
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
        ix = jnp.arange(self.nx)
        iy = jnp.arange(self.ny)
        gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
        flat_idx = (gy * self.nx + gx).ravel()

        # == X curvature operator ==
        # Full curvature (3-point stencil): ix < nx-2
        full_x = gx < (self.nx - 2)
        full_rows_x = flat_idx[full_x.ravel()]

        # Near-boundary (2-point first gradient): ix == nx-2
        near_x = gx == (self.nx - 2)
        near_rows_x = flat_idx[near_x.ravel()]

        # Outer boundary (diagonal): ix == nx-1
        outer_x = gx == (self.nx - 1)
        outer_rows_x = flat_idx[outer_x.ravel()]

        lx_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        lx_operator = lx_operator.at[full_rows_x, full_rows_x].add(1.0)
        lx_operator = lx_operator.at[full_rows_x, full_rows_x + 1].add(-2.0)
        lx_operator = lx_operator.at[full_rows_x, full_rows_x + 2].add(1.0)
        lx_operator = lx_operator.at[near_rows_x, near_rows_x].add(-1.0)
        lx_operator = lx_operator.at[near_rows_x, near_rows_x + 1].add(1.0)
        lx_operator = lx_operator.at[outer_rows_x, outer_rows_x].add(1.0)

        # == Y curvature operator ==
        # Full curvature (3-point stencil): iy < ny-2
        full_y = gy < (self.ny - 2)
        full_rows_y = flat_idx[full_y.ravel()]

        # Near-boundary (2-point first gradient): iy == ny-2
        near_y = gy == (self.ny - 2)
        near_rows_y = flat_idx[near_y.ravel()]

        # Outer boundary (diagonal): iy == ny-1
        outer_y = gy == (self.ny - 1)
        outer_rows_y = flat_idx[outer_y.ravel()]

        ly_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        ly_operator = ly_operator.at[full_rows_y, full_rows_y].add(1.0)
        ly_operator = ly_operator.at[full_rows_y, full_rows_y + self.nx].add(-2.0)
        ly_operator = ly_operator.at[full_rows_y, full_rows_y + 2 * self.nx].add(1.0)
        ly_operator = ly_operator.at[near_rows_y, near_rows_y].add(-1.0)
        ly_operator = ly_operator.at[near_rows_y, near_rows_y + self.nx].add(1.0)
        ly_operator = ly_operator.at[outer_rows_y, outer_rows_y].add(1.0)

        return lx_operator, ly_operator

    def _build_unit_coordinates(self):
        """Return source-grid coordinates for a unit half-size plane."""
        x_axis = jnp.linspace(-1.0, 1.0, self.nx)
        y_axis = jnp.linspace(-1.0, 1.0, self.ny)
        source_x_mesh, source_y_mesh = jnp.meshgrid(x_axis, y_axis, indexing='xy')
        return jnp.stack(
            [source_x_mesh.reshape(-1), source_y_mesh.reshape(-1)],
            axis=1,
        )

    def _first_order_matrix(self, xmin, xmax, ymin, ymax):
        """Return first-order gradient regularization with per-axis pixel scaling.

        H = H1_unit_x / dx² + H1_unit_y / dy²
        where dx = (xmax-xmin)/(nx-1), dy = (ymax-ymin)/(ny-1).
        """
        scale_x = 2.0 / (xmax - xmin)
        scale_y = 2.0 / (ymax - ymin)
        return self._H1_unit_x * (scale_x ** 2) + self._H1_unit_y * (scale_y ** 2)

    def _second_order_matrix(self, xmin, xmax, ymin, ymax):
        """Return second-order curvature regularization with per-axis pixel scaling.

        H = H2_unit_x / dx⁴ + H2_unit_y / dy⁴
        where dx = (xmax-xmin)/(nx-1), dy = (ymax-ymin)/(ny-1).
        """
        scale_x = 2.0 / (xmax - xmin)
        scale_y = 2.0 / (ymax - ymin)
        return self._H2_unit_x * (scale_x ** 4) + self._H2_unit_y * (scale_y ** 4)

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
