"""
Linear solvers for intensity parameter estimation.

This module provides linear solvers (NNLS and normal least squares) adapted
for use with caskade models, handling batch processing and regularization.
"""

from typing import Tuple, Callable, Optional
import jax
import jax.numpy as jnp
from jax import jit, Array
import jaxnnls


@jax.jit
def fnnls_jax(Z: Array, x: Array) -> Tuple[Array, float]:
    """NNLS solver backed by jaxnnls PDIP.

    Solves :math:`\\min_d \\|Z d - x\\|^2` subject to :math:`d \\ge 0`
    using the primal-dual interior-point method from ``jaxnnls``.
    This wrapper adds three numerical safeguards around the external solver:

    1. Columns are normalized to unit norm to reduce conditioning issues.
    2. Columns that are negligible relative to the largest basis vector in the
       same problem are removed from the NNLS system and forced to zero.
    3. The normal equations are globally rescaled so that the right-hand side
       has order-unity magnitude, which avoids premature convergence when the
       physical signal is very small in absolute units.
    """
    n = Z.shape[1]
    dtype = Z.dtype
    eps = jnp.finfo(dtype).eps

    col_norms = jnp.sqrt(jnp.sum(Z * Z, axis=0))
    max_col_norm = jnp.max(col_norms)

    # Relative threshold: a column is inactive only if its norm is
    # negligible compared to the largest column norm in the same problem.
    active = col_norms > (eps * 10) * max_col_norm
    safe_norms = jnp.where(active, col_norms, 1.0)
    Z_scaled = jnp.where(
        active[jnp.newaxis, :],
        Z / safe_norms[jnp.newaxis, :],
        0.0,
    )

    ZTZ = Z_scaled.T @ Z_scaled
    # Diagonal jitter guards against Cholesky failure in edge cases
    ZTZ = ZTZ + (eps * 10) * jnp.eye(n, dtype=dtype)
    ZTx = Z_scaled.T @ x

    def zero_solution(_: None) -> Tuple[Array, float]:
        """Return the trivial solution when no active or driven columns remain."""
        d = jnp.zeros(n, dtype=dtype)
        return d, jnp.linalg.norm(x)

    def solve_active_system(_: None) -> Tuple[Array, float]:
        """Solve the stabilized NNLS subproblem on the active column set."""
        rhs_max = jnp.max(jnp.abs(ZTx))

        # jaxnnls uses absolute KKT tolerances internally. Rescaling both the
        # matrix and right-hand side preserves the minimizer while preventing
        # very small ``Z^T x`` values from being treated as numerically zero.
        # A target RHS magnitude above unity is intentional here: for tiny
        # signals, jaxnnls may still stop too early when the scaled RHS is only
        # O(1), especially if inactive dummy dimensions remain in the system.
        equation_scale = jnp.minimum(1e3 / jnp.maximum(rhs_max, eps), 1e10)
        d_scaled = jaxnnls.solve_nnls(equation_scale * ZTZ, equation_scale * ZTx)[0]
        d_unscaled = d_scaled / safe_norms
        d = jnp.where(active, d_unscaled, 0.0)
        res = jnp.linalg.norm(x - Z @ d)
        return d.astype(dtype), res.astype(dtype)

    should_solve = jnp.any(active) & jnp.any(jnp.abs(ZTx) > 0)
    return jax.lax.cond(should_solve, solve_active_system, zero_solution, operand=None)


def solve_linear_system(A_mat: Array, D_vec: Array, solver_type: str = 'nnls') -> Tuple[Array, Optional[float]]:
    """Solve linear system AX = D.

    Parameters
    ----------
    A_mat : Array, shape [m, n]
        Design matrix.
    D_vec : Array, shape [m]
        Data vector.
    solver_type : {'nnls', 'normal'}
        'nnls' enforces non-negativity; 'normal' uses standard least squares.

    Returns
    -------
    X_vec : Array, shape [n]
        Solution vector.
    residuals : float or None
        Residuals for NNLS, None for normal solver.
    """
    if solver_type == 'nnls':
        return fnnls_jax(A_mat, D_vec)
    if solver_type == 'normal':
        return solve_linear(A_mat, D_vec), None
    raise ValueError("solver_type must be either 'nnls' or 'normal'")


class LinearSolver:
    """Lightweight solver that remembers solver_type between calls."""

    def __init__(self, solver_type: str = 'nnls') -> None:
        if solver_type not in ['nnls', 'normal']:
            raise ValueError("solver_type must be either 'nnls' or 'normal'")
        self.solver_type = solver_type

    def solve(self, A_mat: Array, D_vec: Array) -> Tuple[Array, Optional[float]]:
        return solve_linear_system(A_mat, D_vec, self.solver_type)


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
    fftconvolve_func: Callable,
    unmask_1d: Optional[Array] = None
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
    unmask_1d : array_like, optional
        1D array of valid integer indices to keep (from flatnonzero(~mask)).

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

    # Vectorized convolution using vmap
    def convolve_func(x):
        """
        Convolve one component image with the PSF kernel.

        Parameters
        ----------
        x : Array
            One 2D image component.

        Returns
        -------
        Array
            Convolved 2D image with ``mode='same'``.
        """
        return fftconvolve_func(x, psf_kernel, mode='same')

    img_lens_convolved = jax.vmap(convolve_func, in_axes=-1, out_axes=-1)(img_lens)
    img_arc_convolved = jax.vmap(convolve_func, in_axes=-1, out_axes=-1)(img_arc)

    # Concatenate and reshape
    img = jnp.concatenate([img_arc_convolved, img_lens_convolved], axis=-1)  # [ny, nx, n_total]
    img = jnp.reshape(img, (-1, n_src + n_lens_light))  # [ny*nx, n_total]

    # Prepare data vector
    D_vec = snr_1d  # [ny*nx]

    # Prepare design matrix (weighted by noise)
    A_mat = img / n_1d[:, jnp.newaxis]  # [ny*nx, n_total]

    # Apply mask if provided
    if unmask_1d is not None:
        A_mat = A_mat[unmask_1d]
        D_vec = D_vec[unmask_1d]

    # Add regularization (see https://arxiv.org/pdf/2403.16253 eq.15)
    n_total = n_lens_light + n_src
    Reg_mat = jnp.eye(n_total) * 0.001  # [n_total, n_total]

    A_mat = jnp.concatenate([A_mat, Reg_mat], axis=0)  # [n_unmasked+n_total, n_total]
    D_vec = jnp.concatenate([D_vec, jnp.zeros(n_total)], axis=0)  # [n_unmasked+n_total]

    return A_mat, D_vec
