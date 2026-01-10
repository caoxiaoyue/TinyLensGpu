"""
Linear solvers for intensity parameter estimation.

This module provides linear solvers (NNLS and normal least squares) adapted
for use with caskade models, handling batch processing and regularization.

The FNNLS implementation was copied from the legacy Simulator.Image.fnnls module
to make the implementation independent of legacy code.
"""

from typing import Tuple, Callable, Optional
import jax
import jax.numpy as jnp
from jax import jit, Array


@jax.jit
def fnnls_jax(Z: Array, x: Array, epsilon: Optional[float] = None) -> Tuple[Array, float]:
    """
    JAX implementation of the Fast Non-Negative Least Squares (FNNLS) algorithm.

    Parameters
    ----------
    Z: jnp.ndarray
        m x n matrix.
    x: jnp.ndarray
        m vector.
    epsilon: float or None
        Numerical tolerance. If None, uses jnp.finfo(float).eps.

    Returns
    -------
    d: jnp.ndarray
        n vector, solution to min ||x - Zd|| s.t. d >= 0
    res: float
        Residual norm ||x - Zd||
    """
    m, n = Z.shape
    # Ensure Z and x are float arrays
    Z = Z.astype(jnp.float32)
    x = x.astype(jnp.float32)
    if epsilon is None:
        epsilon = jnp.finfo(jnp.float32).eps

    ZTZ = Z.T @ Z
    ZTx = Z.T @ x

    tolerance = epsilon * n
    max_repetitions = 5

    def body_fun(state):
        d, s, P, w, no_update, _ = state
        current_P = P

        # B2 + B3: Move the element in active set with largest value of w into the passive set
        idx = jnp.argmax(jnp.where(~P, w, -jnp.inf))
        P = P.at[idx].set(True)

        # B4: Set s to the least squares solution along the passive set
        def masked_lstsq(ZTZ, ZTx, P):
            # Masked solve: for inactive variables, set ZTZ to identity and ZTx to zero
            mask = P.astype(ZTZ.dtype)
            eye = jnp.eye(ZTZ.shape[0], dtype=ZTZ.dtype)
            ZTZ_masked = ZTZ * (mask[None, :] * mask[:, None]) + eye * (1.0 - mask[None, :] * mask[:, None])
            ZTx_masked = ZTx * mask
            return jnp.linalg.solve(ZTZ_masked, ZTx_masked)
        s = masked_lstsq(ZTZ, ZTx, P)

        # C1: Loop until all s[P] > tolerance
        def c1_cond_fn(carry):
            s, d, P = carry
            return jnp.any(P) & (jnp.min(jnp.where(P, s, jnp.inf)) <= tolerance)

        def c1_body_fn(carry):
            s, d, P = carry
            # C2: Find largest alpha such that d + alpha(s-d) >= 0 for indices where s <= tolerance
            q = P & (s <= tolerance)
            safe = jnp.where(q, d / (d - s + 1e-12), jnp.inf)
            alpha = jnp.min(safe)
            alpha = jnp.where(jnp.isfinite(alpha), alpha, 0.0)
            # C3: Update d
            d = d + alpha * (s - d)
            # C4: Move elements with d <= tolerance to active set
            P = P & (d > tolerance)
            # C5: Update s
            s = masked_lstsq(ZTZ, ZTx, P)
            # C6: Set s[~P] = 0
            s = s * P.astype(s.dtype)
            return (s, d, P)

        s, d, P = jax.lax.while_loop(c1_cond_fn, c1_body_fn, (s, d, P))

        # B5: d = s
        d = s
        # B6: w = ZTx - ZTZ @ d
        w = ZTx - ZTZ @ d

        # Check if there has been a change to the passive set
        no_update = jnp.where(jnp.all(current_P == P), no_update + 1, 0)
        return (d, s, P, w, no_update, 0)

    def cond_fun(state):
        d, s, P, w, no_update, _ = state
        return (~jnp.all(P)) & (jnp.max(jnp.where(~P, w, -jnp.inf)) > tolerance) & (no_update < max_repetitions)

    # Initializations
    P = jnp.zeros(n, dtype=bool)
    d = jnp.zeros(n, dtype=Z.dtype)
    s = jnp.zeros(n, dtype=Z.dtype)
    w = ZTx - ZTZ @ d
    no_update = 0

    # Main loop
    state = (d, s, P, w, no_update, 0)
    d, s, P, w, no_update, _ = jax.lax.while_loop(cond_fun, body_fun, state)

    res = jnp.linalg.norm(x - Z @ d)
    return d, res




class LinearSolver:
    """
    Linear solver for intensity parameters.

    Supports two solver types:
    - 'nnls': Non-negative least squares (recommended for optical components)
    - 'normal': Standard least squares

    Parameters
    ----------
    solver_type : str
        Solver type, either 'nnls' or 'normal' (default: 'nnls')
    """

    def __init__(self, solver_type: str = 'nnls') -> None:
        if solver_type not in ['nnls', 'normal']:
            raise ValueError("solver_type must be either 'nnls' or 'normal'")
        self.solver_type = solver_type

    def solve(self, A_mat: Array, D_vec: Array) -> Tuple[Array, Optional[float]]:
        """
        Solve linear system AX = D.

        Parameters
        ----------
        A_mat : array_like
            Design matrix, shape [m, n]
        D_vec : array_like
            Data vector, shape [m]

        Returns
        -------
        X_vec : array_like
            Solution vector, shape [n]
        residuals : array_like or None
            Residuals (only for NNLS), None for normal solver
        """
        if self.solver_type == 'nnls':
            return fnnls_jax(A_mat, D_vec)
        else:
            X_vec = solve_linear(A_mat, D_vec)
            return X_vec, None


@jit
def solve_linear(A: Array, b: Array) -> Array:
    """
    Normal least squares solver using pseudoinverse.

    Solves the least squares problem min ||Ax - b||^2 using the
    normal equations and pseudoinverse.

    Parameters
    ----------
    A : array_like
        Design matrix of shape [m, n] where m >= n
    b : array_like
        Data vector of shape [m]

    Returns
    -------
    x : array_like
        Solution vector of shape [n]
    """
    # Compute A^T A
    ATA = A.T @ A  # shape: [n, n]

    # Compute pseudoinverse
    ATA_inv = jnp.linalg.pinv(ATA, rtol=1e-6)  # shape: [n, n]

    # Compute solution: x = (A^T A)^-1 A^T b
    x = ATA_inv @ (A.T @ b)

    return x


def prepare_linear_system(
    img_lens_sub: Array,
    img_arc_sub: Array,
    psf_kernel: Array,
    image_map: Array,
    noise_map: Array,
    nsub: int,
    n_lens_light: int,
    n_src: int,
    bin_func: Callable[[Array, int], Array],
    fftconvolve_func: Callable
) -> Tuple[Array, Array]:
    """
    Prepare linear system for intensity solving.

    This function constructs the design matrix A and data vector D
    for the linear least squares problem.

    Parameters
    ----------
    img_lens_sub : array_like
        Lens light images at subsampled resolution, shape [ny_sub, nx_sub, n_lens]
    img_arc_sub : array_like
        Source light images at subsampled resolution, shape [ny_sub, nx_sub, n_src]
    psf_kernel : array_like
        PSF kernel, shape [ny_psf, nx_psf]
    image_map : array_like
        Observed image, shape [ny, nx]
    noise_map : array_like
        Noise map, shape [ny, nx]
    nsub : int
        Subsampling factor
    n_lens_light : int
        Number of lens light components
    n_src : int
        Number of source light components
    bin_func : callable
        Binning function
    fftconvolve_func : callable
        FFT convolution function

    Returns
    -------
    A_mat : array_like
        Design matrix, shape [m, n_lens+n_src]
    D_vec : array_like
        Data vector, shape [m]
    """
    # Flatten observed image and noise
    img_1d = jnp.ravel(image_map)  # shape: [ny*nx]
    n_1d = jnp.ravel(noise_map)    # shape: [ny*nx]
    snr_1d = img_1d / n_1d

    # Bin and convolve each component
    img_lens = bin_func(img_lens_sub, nsub)  # [ny, nx, n_lens]
    img_arc = bin_func(img_arc_sub, nsub)    # [ny, nx, n_src]

    # Convolve each component with PSF
    img_lens_convolved = jnp.zeros_like(img_lens)
    img_arc_convolved = jnp.zeros_like(img_arc)

    for i in range(n_lens_light):
        img_lens_convolved = img_lens_convolved.at[..., i].set(
            fftconvolve_func(img_lens[..., i], psf_kernel, mode='same')
        )

    for i in range(n_src):
        img_arc_convolved = img_arc_convolved.at[..., i].set(
            fftconvolve_func(img_arc[..., i], psf_kernel, mode='same')
        )

    # Concatenate and reshape
    img = jnp.concatenate([img_arc_convolved, img_lens_convolved], axis=-1)  # [ny, nx, n_total]
    img = jnp.reshape(img, (-1, n_src + n_lens_light))  # [ny*nx, n_total]

    # Prepare data vector
    D_vec = snr_1d  # [ny*nx]

    # Prepare design matrix (weighted by noise)
    A_mat = img / n_1d[:, jnp.newaxis]  # [ny*nx, n_total]

    # Add regularization (see https://arxiv.org/pdf/2403.16253 eq.15)
    n_total = n_lens_light + n_src
    Reg_mat = jnp.eye(n_total) * 0.001  # [n_total, n_total]

    A_mat = jnp.concatenate([A_mat, Reg_mat], axis=0)  # [ny*nx+n_total, n_total]
    D_vec = jnp.concatenate([D_vec, jnp.zeros(n_total)], axis=0)  # [ny*nx+n_total]

    return A_mat, D_vec
