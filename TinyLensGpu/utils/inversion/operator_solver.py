"""Matrix-free operator-based inversion solvers for pixelized source SLI."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit
from jax.tree_util import register_pytree_node_class


def _safe_noise_inverse(noise_var: Array, eps: float = 1e-12) -> Tuple[Array, Array]:
    """
    Internal helper to safe noise inverse.
    
    Parameters
    ----------
    noise_var : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    eps : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    n_diag = jnp.maximum(noise_var, eps)
    n_inv = 1.0 / n_diag
    return n_diag, n_inv


def _apply_psf_unmasked_to_unmasked(
    x_unmasked: Array,
    psf_fft: Array,
    image_shape: Tuple[int, int],
    psf_shape: Tuple[int, int],
    unmasked_indices: Tuple[Array, Array],
    *,
    adjoint: bool,
) -> Array:
    """
    Apply PSF convolution to a vector of unmasked pixels.

    Handles the mapping from unmasked vector -> full 2D image -> FFT convolution -> unmasked vector.

    Parameters
    ----------
    x_unmasked : Array
        Input vector of values at unmasked pixel locations.
    psf_fft : Array
        PSF in the Fourier domain.
    image_shape : Tuple[int, int]
        Shape of the full image (H, W).
    psf_shape : Tuple[int, int]
        Shape of the PSF (H_psf, W_psf).
    unmasked_indices : Tuple[Array, Array]
        Indices of unmasked pixels.
    adjoint : bool
        If True, applies the adjoint (transpose) operation.

    Returns
    -------
    Array
        Convolved values at unmasked pixel locations.
    """
    h, w = image_shape
    psf_h, psf_w = psf_shape
    y_indices, x_indices = unmasked_indices

    fft_shape = (h + psf_h - 1, w + psf_w - 1)
    start_h = (psf_h - 1) // 2
    start_w = (psf_w - 1) // 2

    if not adjoint:
        # Forward path:
        # 1) scatter unmasked vector to full image,
        # 2) convolve in FFT domain,
        # 3) crop to "same" size and gather unmasked pixels back.
        x_full = jnp.zeros((h, w), dtype=x_unmasked.dtype)
        x_full = x_full.at[y_indices, x_indices].set(x_unmasked)
        x_fft = jnp.fft.rfft2(x_full, s=fft_shape)
        y_full = jnp.fft.irfft2(x_fft * psf_fft, s=fft_shape)
        y_cropped = y_full[start_h : start_h + h, start_w : start_w + w]
        return y_cropped[y_indices, x_indices]

    # Adjoint path:
    # apply P^T by embedding into padded coordinates and correlating with conj(PSF FFT).
    y_cropped = jnp.zeros((h, w), dtype=x_unmasked.dtype)
    y_cropped = y_cropped.at[y_indices, x_indices].set(x_unmasked)
    y_padded = jnp.zeros(fft_shape, dtype=y_cropped.dtype)
    y_padded = y_padded.at[start_h : start_h + h, start_w : start_w + w].set(y_cropped)
    y_fft = jnp.fft.rfft2(y_padded, s=fft_shape)
    x_padded = jnp.fft.irfft2(y_fft * jnp.conj(psf_fft), s=fft_shape)
    x_full = x_padded[:h, :w]
    return x_full[y_indices, x_indices]


def _apply_mapping(source: Array, weights: Array, indices: Array) -> Array:
    """
    Internal helper to apply mapping.
    
    Parameters
    ----------
    source : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    weights : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    indices : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    vals = jnp.take(source, indices, axis=0)
    return jnp.sum(weights * vals, axis=1)


def _apply_mapping_transpose(x_unmasked: Array, weights: Array, indices: Array, n_source: int) -> Array:
    """
    Internal helper to apply mapping transpose.
    
    Parameters
    ----------
    x_unmasked : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    weights : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    indices : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    n_source : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    contrib = weights * x_unmasked[:, None]
    out = jnp.zeros((n_source,), dtype=contrib.dtype)
    out = out.at[indices.reshape(-1)].add(contrib.reshape(-1))
    return out


def _build_forward_and_adjoint(
    *,
    weights: Array,
    indices: Array,
    psf_fft: Array,
    image_shape: Tuple[int, int],
    psf_shape: Tuple[int, int],
    unmasked_indices: Tuple[Array, Array],
    n_source: int,
):
    """
    Internal helper to build forward and adjoint.
    
    Parameters
    ----------
    weights : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    indices : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    psf_fft : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    image_shape : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    psf_shape : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    unmasked_indices : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    n_source : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """

    def forward(x: Array) -> Array:
        """
        Compute forward.
        
        Parameters
        ----------
        x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        unblur = _apply_mapping(x, weights, indices)
        return _apply_psf_unmasked_to_unmasked(
            unblur,
            psf_fft,
            image_shape,
            psf_shape,
            unmasked_indices,
            adjoint=False,
        )

    def adjoint(y: Array) -> Array:
        """
        Compute adjoint.
        
        Parameters
        ----------
        y : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        pre = _apply_psf_unmasked_to_unmasked(
            y,
            psf_fft,
            image_shape,
            psf_shape,
            unmasked_indices,
            adjoint=True,
        )
        return _apply_mapping_transpose(pre, weights, indices, n_source)

    return forward, adjoint


def _apply_sparse_matrix(rows: Array, cols: Array, values: Array, n: int, x: Array) -> Array:
    """
    Internal helper to apply sparse matrix.
    
    Parameters
    ----------
    rows : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    cols : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    values : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    n : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    x : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    y = jnp.zeros((int(n),), dtype=x.dtype)
    contrib = values * x[cols]
    y = y.at[rows].add(contrib)
    return y


def _cg_solve(matvec, b: Array, *, tol: float, maxiter: int) -> Tuple[Array, Array]:
    """
    Internal helper to cg solve.
    
    Parameters
    ----------
    matvec : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    b : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    tol : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    maxiter : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    x = jnp.zeros_like(b)
    r = b
    p = r
    rs_old = jnp.dot(r, r)
    tol2 = jnp.array(float(tol) ** 2, dtype=rs_old.dtype)

    def body(state, _):
        """
        Compute body.
        
        Parameters
        ----------
        state : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        _ : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        x_k, r_k, p_k, rs_k, done = state

        # Standard CG step on SPD system A x = b, where matvec(p_k) computes A p_k.
        ap = matvec(p_k)
        denom = jnp.dot(p_k, ap)
        alpha = rs_k / (denom + 1e-12)

        x_new = x_k + alpha * p_k
        r_new = r_k - alpha * ap
        rs_new = jnp.dot(r_new, r_new)
        beta = rs_new / (rs_k + 1e-12)
        p_new = r_new + beta * p_k
        done_new = done | (rs_new <= tol2)

        def keep(_):
            """
            Compute keep.
            
            Parameters
            ----------
            _ : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            return x_k, r_k, p_k, rs_k, done

        def update(_):
            """
            Compute update.
            
            Parameters
            ----------
            _ : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            return x_new, r_new, p_new, rs_new, done_new

        return jax.lax.cond(done, keep, update, operand=None), None

    (x, _, _, rs_final, _), _ = jax.lax.scan(
        body,
        (x, r, p, rs_old, jnp.array(False)),
        xs=None,
        length=maxiter,
    )
    return x, rs_final


def _lanczos_logdet(matvec, n_dim: int, *, seed: int, probes: int, steps: int) -> Array:
    """
    Internal helper to lanczos logdet.
    
    Parameters
    ----------
    matvec : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    n_dim : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    seed : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    probes : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    steps : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """

    def lanczos_one(z: Array) -> Array:
        """
        Compute lanczos one.
        
        Parameters
        ----------
        z : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        z = z.astype(jnp.float32)
        eps = jnp.array(1e-12, dtype=z.dtype)
        z_norm = jnp.linalg.norm(z)
        q = z / (z_norm + eps)
        q_prev = jnp.zeros_like(q)
        beta_prev = jnp.array(0.0, dtype=q.dtype)

        def body(carry, _):
            """
            Compute body.
            
            Parameters
            ----------
            carry : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            _ : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            q_k, q_prev_k, beta_prev_k = carry
            # Three-term Lanczos recurrence for Krylov basis construction.
            w = jnp.asarray(matvec(q_k), dtype=q_k.dtype) - beta_prev_k * q_prev_k
            alpha = jnp.asarray(jnp.dot(q_k, w), dtype=q_k.dtype)
            w = w - alpha * q_k
            beta = jnp.asarray(jnp.linalg.norm(w), dtype=q_k.dtype)
            q_next = w / (beta + eps)
            return (q_next, q_k, beta), (alpha, beta)

        (_, _, _), (alphas, betas) = jax.lax.scan(body, (q, q_prev, beta_prev), xs=None, length=steps)
        betas = betas.at[-1].set(jnp.array(0.0, dtype=betas.dtype))
        # Build tridiagonal T and approximate z^T log(A) z via eigendecomposition of T.
        t = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
        eigvals, eigvecs = jnp.linalg.eigh(t)
        eigvals = jnp.maximum(eigvals, jnp.array(1e-12, dtype=eigvals.dtype))
        w0 = eigvecs[0, :] ** 2
        return (z_norm * z_norm) * jnp.sum(w0 * jnp.log(eigvals))

    key = jax.random.PRNGKey(int(seed))
    z = jax.random.rademacher(key, (int(probes), int(n_dim)), dtype=jnp.int32).astype(jnp.float32)
    # SLQ uses multiple probe vectors and averages unbiased stochastic estimates.
    values = jax.vmap(lanczos_one)(z)
    return jnp.mean(values)


def _choose_slq_size(probes: int, steps: int) -> Tuple[int, int]:
    """
    Internal helper to choose slq size.
    
    Parameters
    ----------
    probes : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    steps : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    return int(probes), int(steps)


def _estimate_lipschitz_power_iteration(
    grad_fn,
    n_dim: int,
    *,
    n_iter: int = 12,
    seed: int = 0,
) -> Array:
    """
    Internal helper to estimate lipschitz power iteration.
    
    Parameters
    ----------
    grad_fn : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    n_dim : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    n_iter : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    seed : Any
        Input argument used by this routine. Shapes/units follow the surrounding
        simulation or inference convention in the calling context.
    
    Returns
    -------
    value : Any
        Computed output produced by this routine. For array outputs, shape follows
        the input mesh/matrix conventions used by the corresponding pipeline stage.
    
    """
    key = jax.random.PRNGKey(seed)
    v = jax.random.normal(key, (n_dim,), dtype=jnp.float32)
    eps = jnp.array(1e-12, dtype=v.dtype)
    v = v / (jnp.linalg.norm(v) + eps)

    def body(vec, _):
        """
        Compute body.
        
        Parameters
        ----------
        vec : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        _ : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        w = jnp.asarray(grad_fn(vec), dtype=vec.dtype)
        nrm = jnp.asarray(jnp.linalg.norm(w), dtype=vec.dtype)
        return w / (nrm + eps), nrm

    _, norms = jax.lax.scan(body, v, xs=None, length=n_iter)
    # Clamp to a small positive floor to keep downstream step sizes well-defined.
    return jnp.maximum(norms[-1], jnp.array(1e-6, dtype=norms.dtype))


class _OperatorSolverBase:
    """
    Represent the `_OperatorSolverBase` component in the TinyLensGpu pipeline.
    
    Parameters
    ----------
    d : Any
        Configuration argument consumed during construction of this component.
    noise_var : Any
        Configuration argument consumed during construction of this component.
    H : Any
        Configuration argument consumed during construction of this component.
    weights : Any
        Configuration argument consumed during construction of this component.
    indices : Any
        Configuration argument consumed during construction of this component.
    psf_fft : Any
        Configuration argument consumed during construction of this component.
    image_shape : Any
        Configuration argument consumed during construction of this component.
    psf_shape : Any
        Configuration argument consumed during construction of this component.
    unmasked_indices : Any
        Configuration argument consumed during construction of this component.
    lens_basis : Any
        Configuration argument consumed during construction of this component.
    lens_light_ridge : Any
        Configuration argument consumed during construction of this component.
    jitter : Any
        Configuration argument consumed during construction of this component.
    slq_seed : Any
        Configuration argument consumed during construction of this component.
    slq_probes : Any
        Configuration argument consumed during construction of this component.
    slq_steps : Any
        Configuration argument consumed during construction of this component.
    dense_logdet_max_n : Any
        Configuration argument consumed during construction of this component.
    reg_operator_mode : Any
        Configuration argument consumed during construction of this component.
    H_sparse_rows : Any
        Configuration argument consumed during construction of this component.
    H_sparse_cols : Any
        Configuration argument consumed during construction of this component.
    H_sparse_values : Any
        Configuration argument consumed during construction of this component.
    H_sparse_n_source : Any
        Configuration argument consumed during construction of this component.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    def __init__(
        self,
        d: Array,
        noise_var: Array,
        H: Array,
        weights: Array,
        indices: Array,
        psf_fft: Array,
        image_shape: Tuple[int, int],
        psf_shape: Tuple[int, int],
        unmasked_indices: Tuple[Array, Array],
        *,
        lens_basis: Array | None,
        lens_light_ridge: float,
        jitter: float,
        slq_seed: int,
        slq_probes: int,
        slq_steps: int,
        dense_logdet_max_n: int,
        reg_operator_mode: str,
        H_sparse_rows: Array | None,
        H_sparse_cols: Array | None,
        H_sparse_values: Array | None,
        H_sparse_n_source: int | None,
    ) -> None:
        """
        Initialize the OperatorSolverBase.

        Parameters
        ----------
        d : Array
            Data vector (1D, unmasked pixels).
        noise_var : Array
            Noise variance vector (1D, matching d).
        H : Array
            Regularization matrix (or operator representation).
        weights : Array
            Interpolation weights for mapping.
        indices : Array
            Interpolation indices for mapping.
        psf_fft : Array
            PSF in the Fourier domain.
        image_shape : Tuple[int, int]
            Shape of the full 2D image (height, width).
        psf_shape : Tuple[int, int]
            Shape of the PSF kernel.
        unmasked_indices : Tuple[Array, Array]
            Indices (y, x) of unmasked pixels in the 2D grid.
        lens_basis : Array | None
            Basis functions for lens light model (optional).
        lens_light_ridge : float
            Ridge regression coefficient for lens light.
        jitter : float
            Small constant added to diagonal for numerical stability.
        slq_seed : int
            Random seed for Stochastic Lanczos Quadrature (SLQ).
        slq_probes : int
            Number of probe vectors for SLQ.
        slq_steps : int
            Number of Lanczos steps for SLQ.
        dense_logdet_max_n : int
            Threshold for switching from dense to SLQ log-determinant computation.
        reg_operator_mode : str
            Mode for regularization operator ('dense_gp', 'sparse_knn', etc.).
        H_sparse_rows : Array | None
            Row indices for sparse regularization matrix.
        H_sparse_cols : Array | None
            Column indices for sparse regularization matrix.
        H_sparse_values : Array | None
            Values for sparse regularization matrix.
        H_sparse_n_source : int | None
            Number of source pixels (for sparse mode).
        """
        self.d = jnp.asarray(d, dtype=jnp.float32)
        self.noise_var = jnp.asarray(noise_var, dtype=jnp.float32)
        self.H = jnp.asarray(H, dtype=jnp.float32)
        self.weights = jnp.asarray(weights, dtype=jnp.float32)
        self.indices = jnp.asarray(indices, dtype=jnp.int32)
        self.psf_fft = jnp.asarray(psf_fft, dtype=jnp.complex64)
        self.unmasked_indices = (
            jnp.asarray(unmasked_indices[0], dtype=jnp.int32),
            jnp.asarray(unmasked_indices[1], dtype=jnp.int32),
        )

        if lens_basis is None:
            self.lens_basis = jnp.zeros((self.d.shape[0], 0), dtype=jnp.float32)
        else:
            self.lens_basis = jnp.asarray(lens_basis, dtype=jnp.float32)
        if self.lens_basis.ndim != 2:
            raise ValueError("lens_basis must be a 2D matrix with shape (n_data, n_lens).")
        if self.lens_basis.shape[0] != self.d.shape[0]:
            raise ValueError("lens_basis row dimension must match unmasked data length.")
        self.n_lens = int(self.lens_basis.shape[1])
        self.lens_light_ridge = float(max(lens_light_ridge, 0.0))

        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.psf_shape = (int(psf_shape[0]), int(psf_shape[1]))

        self.jitter = float(jitter)
        self.slq_seed = int(slq_seed)
        self.slq_probes = int(slq_probes)
        self.slq_steps = int(slq_steps)
        self.dense_logdet_max_n = int(dense_logdet_max_n)
        self.reg_operator_mode = str(reg_operator_mode).strip().lower()

        if self.reg_operator_mode not in {"dense_gp", "sparse_knn", "sparse_rectangular"}:
            raise ValueError(
                f"Unknown reg_operator_mode: '{reg_operator_mode}'. Must be one of {'dense_gp', 'sparse_knn', 'sparse_rectangular'}."
            )

        if H_sparse_rows is None:
            H_sparse_rows = jnp.zeros((0,), dtype=jnp.int32)
        if H_sparse_cols is None:
            H_sparse_cols = jnp.zeros((0,), dtype=jnp.int32)
        if H_sparse_values is None:
            H_sparse_values = jnp.zeros((0,), dtype=jnp.float32)
        self.H_sparse_rows = jnp.asarray(H_sparse_rows, dtype=jnp.int32)
        self.H_sparse_cols = jnp.asarray(H_sparse_cols, dtype=jnp.int32)
        self.H_sparse_values = jnp.asarray(H_sparse_values, dtype=jnp.float32)
        self.H_sparse_n_source = int(H_sparse_n_source) if H_sparse_n_source is not None else int(self.H.shape[0])

        if self.reg_operator_mode in {"sparse_knn", "sparse_rectangular"}:
            if self.H_sparse_values.shape[0] == 0:
                raise ValueError(
                    "sparse reg_operator_mode requires non-empty sparse regularization entries."
                )
            self.n_source = int(self.H_sparse_n_source)
        else:
            if self.H.ndim != 2 or self.H.shape[0] != self.H.shape[1]:
                raise ValueError("Dense regularization mode requires square dense matrix H.")
            self.n_source = int(self.H.shape[0])
        self.n_dim = int(self.n_source + self.n_lens)

    def _ops(self):
        """
        Internal helper to ops.
        
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        source_forward, source_adjoint = _build_forward_and_adjoint(
            weights=self.weights,
            indices=self.indices,
            psf_fft=self.psf_fft,
            image_shape=self.image_shape,
            psf_shape=self.psf_shape,
            unmasked_indices=self.unmasked_indices,
            n_source=self.n_source,
        )

        n_source = self.n_source
        n_lens = self.n_lens

        if n_lens == 0:
            return source_forward, source_adjoint

        lens_basis = self.lens_basis

        def forward(x: Array) -> Array:
            """
            Compute forward.
            
            Parameters
            ----------
            x : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            x_src = x[:n_source]
            x_lens = x[n_source:]
            return source_forward(x_src) + lens_basis @ x_lens

        def adjoint(y: Array) -> Array:
            """
            Compute adjoint.
            
            Parameters
            ----------
            y : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            src_term = source_adjoint(y)
            lens_term = lens_basis.T @ y
            return jnp.concatenate([src_term, lens_term], axis=0)

        return forward, adjoint

    def _apply_H_source(self, x: Array) -> Array:
        """
        Internal helper to apply H source.
        
        Parameters
        ----------
        x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        if self.reg_operator_mode not in {"sparse_knn", "sparse_rectangular"} or self.H_sparse_values.shape[0] == 0:
            return self.H @ x
        return _apply_sparse_matrix(self.H_sparse_rows, self.H_sparse_cols, self.H_sparse_values, self.H_sparse_n_source, x)

    def _apply_H(self, x: Array) -> Array:
        """
        Internal helper to apply H.
        
        Parameters
        ----------
        x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        if self.n_lens == 0:
            return self._apply_H_source(x)

        x_src = x[: self.n_source]
        x_lens = x[self.n_source :]
        src_term = self._apply_H_source(x_src)
        lens_term = self.lens_light_ridge * x_lens
        return jnp.concatenate([src_term, lens_term], axis=0)

    def _half_log_det_H_source(self) -> Tuple[Array, Array]:
        """
        Internal helper to half log det H source.
        
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        n_source = self.n_source
        if self.reg_operator_mode not in {"sparse_knn", "sparse_rectangular"} or self.H_sparse_values.shape[0] == 0:
            h_stab = self.H + self.jitter * jnp.eye(n_source, dtype=self.H.dtype)
            sign_h, logdet_h = jnp.linalg.slogdet(h_stab)
            return sign_h, 0.5 * logdet_h

        def hvec(v: Array) -> Array:
            """
            Compute hvec.
            
            Parameters
            ----------
            v : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            return self._apply_H_source(v) + self.jitter * v

        if n_source <= self.dense_logdet_max_n:
            eye = jnp.eye(n_source, dtype=self.d.dtype)
            h_dense = jax.vmap(hvec, in_axes=1, out_axes=1)(eye)
            sign_h, logdet_h = jnp.linalg.slogdet(h_dense)
            return sign_h, 0.5 * logdet_h

        probes, steps = _choose_slq_size(self.slq_probes, self.slq_steps)
        logdet_h = _lanczos_logdet(hvec, n_source, seed=self.slq_seed + 113, probes=probes, steps=steps)
        return jnp.array(1.0, dtype=self.d.dtype), 0.5 * logdet_h

    def _half_log_det_H(self) -> Tuple[Array, Array]:
        """
        Internal helper to half log det H.
        
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        sign_src, half_log_det_src = self._half_log_det_H_source()
        if self.n_lens == 0:
            return sign_src, half_log_det_src

        ridge_diag = jnp.asarray(self.lens_light_ridge + self.jitter, dtype=self.d.dtype)
        sign_lens = jnp.where(ridge_diag > 0.0, 1.0, 0.0).astype(self.d.dtype)
        ridge_safe = jnp.maximum(ridge_diag, jnp.asarray(1e-12, dtype=self.d.dtype))
        half_log_det_lens = 0.5 * jnp.asarray(self.n_lens, dtype=self.d.dtype) * jnp.log(ridge_safe)
        return sign_src * sign_lens, half_log_det_src + half_log_det_lens

    @jit
    def model_predict(self, x: Array) -> Array:
        """
        Compute model predict.
        
        Parameters
        ----------
        x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        forward, _ = self._ops()
        return forward(jnp.asarray(x, dtype=jnp.float32))

    @jit
    def objective_value(self, x: Array) -> Array:
        """
        Compute objective value.
        
        Parameters
        ----------
        x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        _, n_inv = _safe_noise_inverse(self.noise_var)
        model = self.model_predict(x)
        resid = self.d - model
        chi2 = jnp.sum((resid * resid) * n_inv)
        reg = jnp.dot(x, self._apply_H(x)) + self.jitter * jnp.dot(x, x)
        return chi2 + reg

    def _get_common_children_aux(self):
        """
        Internal helper to get common children aux.
        
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        children = (
            self.d,
            self.noise_var,
            self.H,
            self.weights,
            self.indices,
            self.psf_fft,
            self.unmasked_indices[0],
            self.unmasked_indices[1],
            self.lens_basis,
            self.H_sparse_rows,
            self.H_sparse_cols,
            self.H_sparse_values,
        )
        aux_data = (
            self.image_shape,
            self.psf_shape,
            self.jitter,
            self.slq_seed,
            self.slq_probes,
            self.slq_steps,
            self.dense_logdet_max_n,
            self.reg_operator_mode,
            self.H_sparse_n_source,
            self.lens_light_ridge,
        )
        return children, aux_data


@register_pytree_node_class
class OperatorInversion(_OperatorSolverBase):
    """
    Represent the `OperatorInversion` component in the TinyLensGpu pipeline.
    
    Parameters
    ----------
    d : Any
        Configuration argument consumed during construction of this component.
    noise_var : Any
        Configuration argument consumed during construction of this component.
    H : Any
        Configuration argument consumed during construction of this component.
    weights : Any
        Configuration argument consumed during construction of this component.
    indices : Any
        Configuration argument consumed during construction of this component.
    psf_fft : Any
        Configuration argument consumed during construction of this component.
    image_shape : Any
        Configuration argument consumed during construction of this component.
    psf_shape : Any
        Configuration argument consumed during construction of this component.
    unmasked_indices : Any
        Configuration argument consumed during construction of this component.
    lens_basis : Any
        Configuration argument consumed during construction of this component.
    lens_light_ridge : Any
        Configuration argument consumed during construction of this component.
    jitter : Any
        Configuration argument consumed during construction of this component.
    cg_tol : Any
        Configuration argument consumed during construction of this component.
    cg_maxiter : Any
        Configuration argument consumed during construction of this component.
    slq_seed : Any
        Configuration argument consumed during construction of this component.
    slq_probes : Any
        Configuration argument consumed during construction of this component.
    slq_steps : Any
        Configuration argument consumed during construction of this component.
    dense_logdet_max_n : Any
        Configuration argument consumed during construction of this component.
    reg_operator_mode : Any
        Configuration argument consumed during construction of this component.
    H_sparse_rows : Any
        Configuration argument consumed during construction of this component.
    H_sparse_cols : Any
        Configuration argument consumed during construction of this component.
    H_sparse_values : Any
        Configuration argument consumed during construction of this component.
    H_sparse_n_source : Any
        Configuration argument consumed during construction of this component.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    def __init__(
        self,
        d: Array,
        noise_var: Array,
        H: Array,
        weights: Array,
        indices: Array,
        psf_fft: Array,
        image_shape: Tuple[int, int],
        psf_shape: Tuple[int, int],
        unmasked_indices: Tuple[Array, Array],
        *,
        lens_basis: Array | None = None,
        lens_light_ridge: float = 1e-8,
        jitter: float = 1e-6,
        cg_tol: float = 1e-6,
        cg_maxiter: int = 300,
        slq_seed: int = 0,
        slq_probes: int = 32,
        slq_steps: int = 60,
        dense_logdet_max_n: int = 256,
        reg_operator_mode: str = "dense_gp",
        H_sparse_rows: Array | None = None,
        H_sparse_cols: Array | None = None,
        H_sparse_values: Array | None = None,
        H_sparse_n_source: int | None = None,
    ) -> None:
        """
        Initialize a `OperatorInversion` instance with validated configuration.
        
        Parameters
        ----------
        d : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        noise_var : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        H : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        weights : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        indices : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        psf_fft : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        image_shape : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        psf_shape : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        unmasked_indices : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        lens_basis : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        lens_light_ridge : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        jitter : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        cg_tol : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        cg_maxiter : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        slq_seed : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        slq_probes : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        slq_steps : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        dense_logdet_max_n : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        reg_operator_mode : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        H_sparse_rows : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        H_sparse_cols : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        H_sparse_values : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        H_sparse_n_source : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        super().__init__(
            d, noise_var, H, weights, indices, psf_fft, image_shape, psf_shape, unmasked_indices,
            lens_basis=lens_basis,
            lens_light_ridge=lens_light_ridge,
            jitter=jitter, slq_seed=slq_seed, slq_probes=slq_probes, slq_steps=slq_steps,
            dense_logdet_max_n=dense_logdet_max_n,
            reg_operator_mode=reg_operator_mode, H_sparse_rows=H_sparse_rows,
            H_sparse_cols=H_sparse_cols, H_sparse_values=H_sparse_values,
            H_sparse_n_source=H_sparse_n_source
        )
        self.cg_tol = float(cg_tol)
        self.cg_maxiter = int(cg_maxiter)

    @jit
    def solve(self) -> Array:
        """
        Solve the linear system for source (and optionally lens) light intensities.

        Uses Conjugate Gradient (CG) to solve:
        (M^T N^{-1} M + H + jitter*I) x = M^T N^{-1} d

        Returns
        -------
        Array
            The solution vector x.
        """
        _, n_inv = _safe_noise_inverse(self.noise_var)
        forward, adjoint = self._ops()

        def mvec(x: Array) -> Array:
            """
            Compute mvec.
            
            Parameters
            ----------
            x : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            return adjoint(forward(x) * n_inv) + self._apply_H(x) + self.jitter * x

        b = adjoint(self.d * n_inv)
        x, _ = _cg_solve(mvec, b, tol=self.cg_tol, maxiter=self.cg_maxiter)
        return x

    @jit
    def log_evidence(self) -> Array:
        """
        Compute log evidence.
        
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        n_data = self.d.shape[0]
        n_dim = self.n_dim
        n_diag, n_inv = _safe_noise_inverse(self.noise_var)

        half_log_det_n = 0.5 * jnp.sum(jnp.log(n_diag))
        log_evidence_const = -0.5 * n_data * jnp.log(2.0 * jnp.pi) - half_log_det_n

        sign_h, half_log_det_h = self._half_log_det_H()

        forward, adjoint = self._ops()

        def mvec(x: Array) -> Array:
            """
            Compute mvec.
            
            Parameters
            ----------
            x : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            return adjoint(forward(x) * n_inv) + self._apply_H(x) + self.jitter * x

        b = adjoint(self.d * n_inv)
        s, _ = _cg_solve(mvec, b, tol=self.cg_tol, maxiter=self.cg_maxiter)

        d_ninv_d = jnp.sum(self.d * self.d * n_inv)
        combined_chi2_reg = d_ninv_d - jnp.dot(s, b)
        n_dim_int = int(n_dim)
        probes, steps = _choose_slq_size(self.slq_probes, self.slq_steps)
        if n_dim_int <= self.dense_logdet_max_n:
            eye = jnp.eye(n_dim_int, dtype=self.H.dtype)
            m_dense = jax.vmap(mvec, in_axes=1, out_axes=1)(eye)
            _, logdet_m = jnp.linalg.slogdet(m_dense)
        else:
            logdet_m = _lanczos_logdet(
                mvec,
                n_dim_int,
                seed=self.slq_seed,
                probes=probes,
                steps=steps,
            )

        is_valid = (
            (sign_h > 0)
            & jnp.isfinite(log_evidence_const)
            & jnp.isfinite(half_log_det_h)
            & jnp.isfinite(logdet_m)
        )

        def valid(_):
            """
            Compute valid.
            
            Parameters
            ----------
            _ : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            val = log_evidence_const
            val += half_log_det_h
            val -= 0.5 * combined_chi2_reg
            val -= 0.5 * logdet_m
            return val

        return jax.lax.cond(
            is_valid,
            valid,
            lambda _: jnp.asarray(-jnp.inf, dtype=log_evidence_const.dtype),
            operand=None,
        )

    def tree_flatten(self):
        """
        Compute tree flatten.
        
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        children, aux = self._get_common_children_aux()
        aux_final = aux + (self.cg_tol, self.cg_maxiter)
        return children, aux_final

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Compute tree unflatten.
        
        Parameters
        ----------
        aux_data : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        children : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        (
            image_shape, psf_shape, jitter, slq_seed, slq_probes, slq_steps,
            dense_logdet_max_n, reg_operator_mode, H_sparse_n_source, lens_light_ridge,
            cg_tol, cg_maxiter
        ) = aux_data
        (
            d, noise_var, H, weights, indices, psf_fft, y_indices, x_indices,
            lens_basis, H_sparse_rows, H_sparse_cols, H_sparse_values
        ) = children
        return cls(
            d=d, noise_var=noise_var, H=H, weights=weights, indices=indices,
            psf_fft=psf_fft, image_shape=image_shape, psf_shape=psf_shape,
            unmasked_indices=(y_indices, x_indices), jitter=jitter,
            lens_basis=lens_basis, lens_light_ridge=lens_light_ridge,
            cg_tol=cg_tol, cg_maxiter=cg_maxiter,
            slq_seed=slq_seed, slq_probes=slq_probes, slq_steps=slq_steps,
            dense_logdet_max_n=dense_logdet_max_n,
            reg_operator_mode=reg_operator_mode, H_sparse_rows=H_sparse_rows,
            H_sparse_cols=H_sparse_cols, H_sparse_values=H_sparse_values,
            H_sparse_n_source=H_sparse_n_source,
        )


@register_pytree_node_class
class OperatorNNLSInversion(_OperatorSolverBase):
    """
    Represent the `OperatorNNLSInversion` component in the TinyLensGpu pipeline.
    
    Parameters
    ----------
    d : Any
        Configuration argument consumed during construction of this component.
    noise_var : Any
        Configuration argument consumed during construction of this component.
    H : Any
        Configuration argument consumed during construction of this component.
    weights : Any
        Configuration argument consumed during construction of this component.
    indices : Any
        Configuration argument consumed during construction of this component.
    psf_fft : Any
        Configuration argument consumed during construction of this component.
    image_shape : Any
        Configuration argument consumed during construction of this component.
    psf_shape : Any
        Configuration argument consumed during construction of this component.
    unmasked_indices : Any
        Configuration argument consumed during construction of this component.
    lens_basis : Any
        Configuration argument consumed during construction of this component.
    lens_light_ridge : Any
        Configuration argument consumed during construction of this component.
    jitter : Any
        Configuration argument consumed during construction of this component.
    maxiter : Any
        Configuration argument consumed during construction of this component.
    tol : Any
        Configuration argument consumed during construction of this component.
    lipschitz_iters : Any
        Configuration argument consumed during construction of this component.
    fista_seed : Any
        Configuration argument consumed during construction of this component.
    slq_seed : Any
        Configuration argument consumed during construction of this component.
    slq_probes : Any
        Configuration argument consumed during construction of this component.
    slq_steps : Any
        Configuration argument consumed during construction of this component.
    dense_logdet_max_n : Any
        Configuration argument consumed during construction of this component.
    reg_operator_mode : Any
        Configuration argument consumed during construction of this component.
    H_sparse_rows : Any
        Configuration argument consumed during construction of this component.
    H_sparse_cols : Any
        Configuration argument consumed during construction of this component.
    H_sparse_values : Any
        Configuration argument consumed during construction of this component.
    H_sparse_n_source : Any
        Configuration argument consumed during construction of this component.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    def __init__(
        self,
        d: Array,
        noise_var: Array,
        H: Array,
        weights: Array,
        indices: Array,
        psf_fft: Array,
        image_shape: Tuple[int, int],
        psf_shape: Tuple[int, int],
        unmasked_indices: Tuple[Array, Array],
        *,
        lens_basis: Array | None = None,
        lens_light_ridge: float = 1e-8,
        jitter: float = 1e-6,
        maxiter: int = 600,
        tol: float = 1e-6,
        lipschitz_iters: int = 12,
        fista_seed: int = 0,
        slq_seed: int = 0,
        slq_probes: int = 32,
        slq_steps: int = 60,
        dense_logdet_max_n: int = 256,
        reg_operator_mode: str = "dense_gp",
        H_sparse_rows: Array | None = None,
        H_sparse_cols: Array | None = None,
        H_sparse_values: Array | None = None,
        H_sparse_n_source: int | None = None,
    ) -> None:
        """
        Initialize a `OperatorNNLSInversion` instance with validated configuration.
        
        Parameters
        ----------
        d : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        noise_var : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        H : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        weights : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        indices : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        psf_fft : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        image_shape : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        psf_shape : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        unmasked_indices : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        lens_basis : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        lens_light_ridge : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        jitter : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        maxiter : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        tol : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        lipschitz_iters : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        fista_seed : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        slq_seed : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        slq_probes : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        slq_steps : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        dense_logdet_max_n : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        reg_operator_mode : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        H_sparse_rows : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        H_sparse_cols : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        H_sparse_values : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        H_sparse_n_source : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        super().__init__(
            d, noise_var, H, weights, indices, psf_fft, image_shape, psf_shape, unmasked_indices,
            lens_basis=lens_basis,
            lens_light_ridge=lens_light_ridge,
            jitter=jitter, slq_seed=slq_seed, slq_probes=slq_probes, slq_steps=slq_steps,
            dense_logdet_max_n=dense_logdet_max_n,
            reg_operator_mode=reg_operator_mode, H_sparse_rows=H_sparse_rows,
            H_sparse_cols=H_sparse_cols, H_sparse_values=H_sparse_values,
            H_sparse_n_source=H_sparse_n_source
        )
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.lipschitz_iters = int(lipschitz_iters)
        self.fista_seed = int(fista_seed)

    @jit
    def _gradient(self, x: Array) -> Array:
        """
        Internal helper to gradient.
        
        Parameters
        ----------
        x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        _, n_inv = _safe_noise_inverse(self.noise_var)
        forward, adjoint = self._ops()
        resid = forward(x) - self.d
        return adjoint(resid * n_inv) + self._apply_H(x) + self.jitter * x

    @jit
    def solve(self) -> Array:
        """
        Compute solve.
        
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        n_dim = self.n_dim
        grad_fn = lambda vec: self._gradient(vec)

        l_est = _estimate_lipschitz_power_iteration(
            grad_fn,
            n_dim,
            n_iter=self.lipschitz_iters,
            seed=self.fista_seed,
        )
        step = 1.0 / (l_est + 1e-12)

        x0 = jnp.zeros((n_dim,), dtype=jnp.float32)
        y0 = x0
        t0 = jnp.array(1.0, dtype=jnp.float32)
        obj0 = self.objective_value(x0)

        def body(state, _):
            """
            Compute body.
            
            Parameters
            ----------
            state : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            _ : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            x_prev, y_prev, t_prev, obj_prev, done = state

            grad_y = grad_fn(y_prev)
            x_next = jnp.maximum(y_prev - step * grad_y, 0.0)
            t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t_prev * t_prev))
            momentum = (t_prev - 1.0) / (t_next + 1e-12)
            y_next = x_next + momentum * (x_next - x_prev)

            obj_next = self.objective_value(x_next)
            rel = jnp.abs(obj_prev - obj_next) / (jnp.abs(obj_prev) + 1e-12)
            done_next = done | (rel <= self.tol)

            def keep(_):
                """
                Compute keep.
                
                Parameters
                ----------
                _ : Any
                    Input argument used by this routine. Shapes/units follow the surrounding
                    simulation or inference convention in the calling context.
                
                Returns
                -------
                value : Any
                    Computed output produced by this routine. For array outputs, shape follows
                    the input mesh/matrix conventions used by the corresponding pipeline stage.
                
                """
                return x_prev, y_prev, t_prev, obj_prev, done

            def update(_):
                """
                Compute update.
                
                Parameters
                ----------
                _ : Any
                    Input argument used by this routine. Shapes/units follow the surrounding
                    simulation or inference convention in the calling context.
                
                Returns
                -------
                value : Any
                    Computed output produced by this routine. For array outputs, shape follows
                    the input mesh/matrix conventions used by the corresponding pipeline stage.
                
                """
                return x_next, y_next, t_next, obj_next, done_next

            return jax.lax.cond(done, keep, update, operand=None), None

        (x_final, _, _, _, _), _ = jax.lax.scan(
            body,
            (x0, y0, t0, obj0, jnp.array(False)),
            xs=None,
            length=self.maxiter,
        )
        return x_final

    @jit
    def log_evidence(self) -> Array:
        """
        Compute log evidence.
        
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        n_data = self.d.shape[0]
        n_dim = self.n_dim
        n_diag, n_inv = _safe_noise_inverse(self.noise_var)

        half_log_det_n = 0.5 * jnp.sum(jnp.log(n_diag))
        log_evidence_const = -0.5 * n_data * jnp.log(2.0 * jnp.pi) - half_log_det_n

        sign_h, half_log_det_h = self._half_log_det_H()

        x = self.solve()
        obj = self.objective_value(x)

        forward, adjoint = self._ops()

        def mvec(v: Array) -> Array:
            """
            Compute mvec.
            
            Parameters
            ----------
            v : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            return adjoint(forward(v) * n_inv) + self._apply_H(v) + self.jitter * v

        n_dim_int = int(n_dim)
        probes, steps = _choose_slq_size(self.slq_probes, self.slq_steps)
        if n_dim_int <= self.dense_logdet_max_n:
            eye = jnp.eye(n_dim_int, dtype=self.H.dtype)
            m_dense = jax.vmap(mvec, in_axes=1, out_axes=1)(eye)
            _, logdet_m = jnp.linalg.slogdet(m_dense)
        else:
            logdet_m = _lanczos_logdet(
                mvec,
                n_dim_int,
                seed=self.slq_seed,
                probes=probes,
                steps=steps,
            )

        is_valid = (sign_h > 0) & jnp.isfinite(log_evidence_const) & jnp.isfinite(half_log_det_h) & jnp.isfinite(logdet_m)

        def valid(_):
            """
            Compute valid.
            
            Parameters
            ----------
            _ : Any
                Input argument used by this routine. Shapes/units follow the surrounding
                simulation or inference convention in the calling context.
            
            Returns
            -------
            value : Any
                Computed output produced by this routine. For array outputs, shape follows
                the input mesh/matrix conventions used by the corresponding pipeline stage.
            
            """
            val = log_evidence_const
            val += half_log_det_h
            val -= 0.5 * obj
            val -= 0.5 * logdet_m
            return val

        return jax.lax.cond(
            is_valid,
            valid,
            lambda _: jnp.asarray(-jnp.inf, dtype=log_evidence_const.dtype),
            operand=None,
        )

    def tree_flatten(self):
        """
        Compute tree flatten.
        
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        children, aux = self._get_common_children_aux()
        aux_final = aux + (self.maxiter, self.tol, self.lipschitz_iters, self.fista_seed)
        return children, aux_final

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Compute tree unflatten.
        
        Parameters
        ----------
        aux_data : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        children : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        (
            image_shape, psf_shape, jitter, slq_seed, slq_probes, slq_steps,
            dense_logdet_max_n, reg_operator_mode, H_sparse_n_source, lens_light_ridge,
            maxiter, tol, lipschitz_iters, fista_seed
        ) = aux_data
        (
            d, noise_var, H, weights, indices, psf_fft, y_indices, x_indices,
            lens_basis, H_sparse_rows, H_sparse_cols, H_sparse_values
        ) = children
        return cls(
            d=d, noise_var=noise_var, H=H, weights=weights, indices=indices,
            psf_fft=psf_fft, image_shape=image_shape, psf_shape=psf_shape,
            unmasked_indices=(y_indices, x_indices), jitter=jitter,
            lens_basis=lens_basis, lens_light_ridge=lens_light_ridge,
            maxiter=maxiter, tol=tol, lipschitz_iters=lipschitz_iters, fista_seed=fista_seed,
            slq_seed=slq_seed, slq_probes=slq_probes, slq_steps=slq_steps,
            dense_logdet_max_n=dense_logdet_max_n,
            reg_operator_mode=reg_operator_mode, H_sparse_rows=H_sparse_rows,
            H_sparse_cols=H_sparse_cols, H_sparse_values=H_sparse_values,
            H_sparse_n_source=H_sparse_n_source,
        )


__all__ = [
    "OperatorInversion",
    "OperatorNNLSInversion",
    "_apply_psf_unmasked_to_unmasked",
    "_apply_mapping",
    "_apply_mapping_transpose",
]
