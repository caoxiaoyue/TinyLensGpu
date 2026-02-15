"""
Linear Inversion Solver for Source Reconstruction

This module provides an optimized linear inversion framework for gravitational 
lensing source reconstruction. It implements regularized linear inversion with
Bayesian evidence calculation.

The solver is registered as a JAX PyTree for efficient JIT compilation.
"""

import jax
import jax.numpy as jnp
from jax import jit, Array
from jax.tree_util import register_pytree_node_class
from functools import partial

from TinyLensGpu.utils.linear_solver import fnnls_jax


@partial(jit, static_argnames=['jitter', 'eps'])
def _precompute_terms_common(d: Array, F: Array, noise_cov: Array, H: Array, *, jitter: float, eps: float):
    r"""
    Precompute linear algebra terms for inversion and Bayesian evidence.

    Computes terms required for the posterior mean (solution) and marginal likelihood (evidence).
    Handles both diagonal and full noise covariance matrices efficiently.

    Mathematical Context:
    The regularized least squares solution is $s = M^{-1} F^T N^{-1} d$, where $M = F^T N^{-1} F + H$.
    The log-evidence is $\ln P(d) \approx -\frac{1}{2} (d^T N^{-1} d - d_{eff}^T M^{-1} d_{eff}) - \frac{1}{2} \ln|N| + \frac{1}{2} \ln|H| - \frac{1}{2} \ln|M|$.

    Parameters
    ----------
    d : Array
        Data vector (1D, shape [n_data]).
    F : Array
        Lensing mapping matrix (shape [n_data, n_source]).
    noise_cov : Array
        Noise covariance. Can be 1D (diagonal variances) or 2D (full covariance matrix).
    H : Array
        Regularization matrix (shape [n_source, n_source]).
    jitter : float
        Small diagonal constant for numerical stability (added to H and M).
    eps : float
        Minimum value for noise variance to avoid division by zero.

    Returns
    -------
    tuple
        A tuple containing precomputed matrices, vectors, and scalars:
        (N_diag, N_inv_diag, FT_Ninv_F, FT_Ninv_d, d_Ninv_d, M_stab,
         half_log_det_H, half_log_det_M, log_evidence_const, is_valid)

    """
    n_data = d.shape[0]
    n_source = H.shape[0]
    
    is_diagonal = (noise_cov.ndim == 1)
    is_valid = jnp.array(True)

    N_diag = jnp.ones((n_data,), dtype=d.dtype)
    N_inv_diag = jnp.ones((n_data,), dtype=d.dtype)
    
    if is_diagonal:
        # Cheap path: element-wise weighting when noise covariance is diagonal.
        N_diag = jnp.clip(noise_cov, min=eps)
        N_inv_diag = 1.0 / N_diag
        
        half_log_det_N = 0.5 * jnp.sum(jnp.log(N_diag))
        
        weighted_F = F * N_inv_diag[:, None]
        FT_Ninv_F = weighted_F.T @ F
        
        FT_Ninv_d = F.T @ (d * N_inv_diag)
        
        d_Ninv_d = jnp.sum(d**2 * N_inv_diag)
        
    else:
        # General path: solve with full covariance matrix (potentially expensive O(N^3)).
        N = noise_cov + jitter * jnp.eye(n_data)
        
        sign, logdet = jnp.linalg.slogdet(N)
        half_log_det_N = 0.5 * logdet
        is_valid = is_valid & (sign > 0) & jnp.isfinite(logdet)
        
        N_inv_F = jnp.linalg.solve(N, F)
        N_inv_d = jnp.linalg.solve(N, d)
        
        FT_Ninv_F = F.T @ N_inv_F
        FT_Ninv_d = F.T @ N_inv_d
        d_Ninv_d = d.T @ N_inv_d

    log_evidence_const = -0.5 * n_data * jnp.log(2.0 * jnp.pi) - half_log_det_N
    is_valid = is_valid & jnp.isfinite(log_evidence_const)

    H_stab = H + jitter * jnp.eye(n_source)
    sign_H, logdet_H = jnp.linalg.slogdet(H_stab)
    half_log_det_H = 0.5 * logdet_H
    is_valid = is_valid & (sign_H > 0) & jnp.isfinite(logdet_H)

    M = FT_Ninv_F + H
    
    # Stabilize the posterior precision before log-determinant / linear solve operations.
    M_stab = M + jitter * jnp.eye(n_source)
    sign_M, logdet_M = jnp.linalg.slogdet(M_stab)
    half_log_det_M = 0.5 * logdet_M
    is_valid = is_valid & (sign_M > 0) & jnp.isfinite(logdet_M)
    
    return (
        N_diag,
        N_inv_diag,
        FT_Ninv_F,
        FT_Ninv_d,
        d_Ninv_d,
        M_stab,
        half_log_det_H,
        half_log_det_M,
        log_evidence_const,
        is_valid,
    )


@partial(jit, static_argnames=['jitter'])
def _precision_sqrt_factor_from(H: Array, *, jitter: float) -> Array:
    r"""
    Compute a square root factor B such that B^T B approx H.
    
    This is used for NNLS to transform the regularization term $x^T H x$ into a least squares form.
    Specifically, we want $||B x||^2 \approx x^T H x$.
    We use eigendecomposition $H = V D V^T$ so $B = \sqrt{D} V^T$.
    
    Parameters
    ----------
    H : Array
        Symmetric positive-definite regularization matrix.
    jitter : float
        Stabilization constant.
    
    Returns
    -------
    Array
        Matrix B (shape [n_source, n_source]).
    """
    n = H.shape[0]
    H_stab = H + jitter * jnp.eye(n, dtype=H.dtype)
    H_stab = 0.5 * (H_stab + H_stab.T)
    eigvals, eigvecs = jnp.linalg.eigh(H_stab)
    eigvals = jnp.clip(eigvals, min=jitter)
    sqrt_eigvals = jnp.sqrt(eigvals)
    return sqrt_eigvals[:, None] * eigvecs.T


@register_pytree_node_class
class LinearInversion:
    r"""
    Analytic Linear Inversion Solver.

    Solves the unconstrained regularized least squares problem:
    $s^* = \text{argmin}_s \frac{1}{2} (d - F s)^T N^{-1} (d - F s) + \frac{1}{2} s^T H s$

    The solution is closed-form:
    $s^* = (F^T N^{-1} F + H)^{-1} F^T N^{-1} d$

    This class also computes the fully analytic Bayesian log-evidence.

    Parameters
    ----------
    d : Array
        Data vector (1D, shape [n_data]).
    F : Array
        Linear mapping matrix (shape [n_data, n_source]).
    noise_cov : Array
        Noise covariance (1D diagonal or 2D full).
    H : Array
        Regularization matrix (shape [n_source, n_source]).
    _precomputed : tuple, optional
        Internal state for efficient PyTree reconstruction.
    """

    def __init__(self, d, F, noise_cov, H, _precomputed=None):
        """Initialize LinearInversion solver."""
        self.d = jnp.asarray(d, dtype=jnp.float32)
        self.F = jnp.asarray(F, dtype=jnp.float32)
        self.H = jnp.asarray(H, dtype=jnp.float32)
        self.noise_cov = jnp.asarray(noise_cov, dtype=jnp.float32)
        
        self.jitter = 1e-6
        self.eps = 1e-12
        self.n_data = self.d.shape[0]
        self.n_source = self.H.shape[0]

        if _precomputed is None:
            (_N_diag, _N_inv_diag,
             self.FT_Ninv_F, self.FT_Ninv_d, self.d_Ninv_d, self.M_stab,
             self.half_log_det_H, self.half_log_det_M, self.log_evidence_const, self.is_valid) = \
                _precompute_terms_common(self.d, self.F, self.noise_cov, self.H, jitter=self.jitter, eps=self.eps)
        else:
            (_N_diag, _N_inv_diag,
             self.FT_Ninv_F, self.FT_Ninv_d, self.d_Ninv_d, self.M_stab,
             self.half_log_det_H, self.half_log_det_M, self.log_evidence_const, self.is_valid) = _precomputed

    @jit
    def solve(self):
        """
        Compute the MAP source reconstruction.

        Returns
        -------
        Array
            Source vector s.
        """
        s = jnp.linalg.solve(self.M_stab, self.FT_Ninv_d)
        return s

    @jit
    def model_predict(self, s: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the model prediction for a given source.

        Parameters
        ----------
        s : Array
            Source vector.

        Returns
        -------
        Array
            Predicted data vector $d_{model} = F s$.
        """
        return self.F @ s

    def invert(self):
        r"""
        Compute both the solution and its covariance matrix.

        Returns
        -------
        s : Array
            Source reconstruction.
        Sigma : Array
            Posterior covariance matrix $\Sigma = (F^T N^{-1} F + H)^{-1}$.
        """
        s = self.solve()
        Sigma = jnp.linalg.inv(self.M_stab)
        Sigma = 0.5 * (Sigma + Sigma.T)
        return s, Sigma

    @jit
    def log_evidence(self):
        """
        Compute the Bayesian log-evidence.

        Returns
        -------
        Array
            Log-evidence value.
        """
        def _valid(_):
            s = self.solve()
            combined_chi2_reg = self.d_Ninv_d - jnp.dot(s, self.FT_Ninv_d)
            log_ev = self.log_evidence_const
            log_ev += self.half_log_det_H
            log_ev -= 0.5 * combined_chi2_reg
            log_ev -= self.half_log_det_M
            return log_ev

        return jax.lax.cond(self.is_valid, _valid, lambda _: -jnp.inf, operand=None)

    def tree_flatten(self):
        """Flatten for JAX pytree registration."""
        children = (self.d, self.F, self.noise_cov, self.H,
                    self.FT_Ninv_F, self.FT_Ninv_d, self.d_Ninv_d, self.M_stab,
                    self.half_log_det_H, self.half_log_det_M, self.log_evidence_const, self.is_valid)
        aux_data = (self.jitter, self.eps)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten for JAX pytree registration."""
        obj = cls.__new__(cls)
        
        (d, F, noise_cov, H,
         FT_Ninv_F, FT_Ninv_d, d_Ninv_d, M_stab,
         half_log_det_H, half_log_det_M, log_evidence_const, is_valid) = children
        
        jitter, eps = aux_data
        
        obj.d = d
        obj.F = F
        obj.noise_cov = noise_cov
        obj.H = H
        obj.jitter = jitter
        obj.eps = eps
        obj.n_data = d.shape[0]
        obj.n_source = H.shape[0]
        
        obj.FT_Ninv_F = FT_Ninv_F
        obj.FT_Ninv_d = FT_Ninv_d
        obj.d_Ninv_d = d_Ninv_d
        obj.M_stab = M_stab
        obj.half_log_det_H = half_log_det_H
        obj.half_log_det_M = half_log_det_M
        obj.log_evidence_const = log_evidence_const
        obj.is_valid = is_valid
        
        return obj


@register_pytree_node_class
class NNLSInversion:
    r"""
    Non-Negative Least Squares (NNLS) Inversion Solver.

    Solves the constrained problem:
    $s^* = \text{argmin}_{s \ge 0} \frac{1}{2} (d - F s)^T N^{-1} (d - F s) + \frac{1}{2} s^T H s$

    Uses a fast coordinate descent algorithm (FNNLS) adapted for JAX.
    Only supports diagonal noise covariance for efficiency.

    Parameters
    ----------
    d : Array
        Data vector.
    F : Array
        Mapping matrix.
    noise_cov : Array
        Diagonal noise covariance (1D).
    H : Array
        Regularization matrix.
    _precomputed : tuple, optional
        Internal state.
    """
    def __init__(self, d, F, noise_cov, H, _precomputed=None):
        """Initialize NNLSInversion solver."""
        self.d = jnp.asarray(d, dtype=jnp.float32)
        self.F = jnp.asarray(F, dtype=jnp.float32)
        self.H = jnp.asarray(H, dtype=jnp.float32)
        self.noise_cov = jnp.asarray(noise_cov, dtype=jnp.float32)

        self.jitter = 1e-6
        self.eps = 1e-12
        self.n_data = self.d.shape[0]
        self.n_source = self.H.shape[0]

        if self.noise_cov.ndim != 1:
            raise ValueError("NNLSInversion currently supports diagonal noise covariance only.")

        if _precomputed is None:
            (self.N_diag, self.N_inv_diag,
             _FT_Ninv_F, _FT_Ninv_d, _d_Ninv_d, _M_stab,
             self.half_log_det_H, self.half_log_det_M, self.log_evidence_const, self.is_valid) = \
                _precompute_terms_common(self.d, self.F, self.noise_cov, self.H, jitter=self.jitter, eps=self.eps)
        else:
            (self.N_diag, self.N_inv_diag,
             _FT_Ninv_F, _FT_Ninv_d, _d_Ninv_d, _M_stab,
             self.half_log_det_H, self.half_log_det_M, self.log_evidence_const, self.is_valid) = _precomputed

        self.sqrt_w = 1.0 / jnp.sqrt(self.N_diag)
        self.B = _precision_sqrt_factor_from(self.H, jitter=self.jitter)

    @jit
    def solve(self) -> Array:
        """
        Compute the non-negative source reconstruction.

        Returns
        -------
        Array
            Non-negative source vector s.
        """
        Z = self.F * self.sqrt_w[:, None]
        y = self.d * self.sqrt_w
        Z_aug = jnp.concatenate([Z, self.B], axis=0)
        y_aug = jnp.concatenate([y, jnp.zeros(self.n_source, dtype=y.dtype)], axis=0)
        x, _ = fnnls_jax(Z_aug, y_aug)
        return x

    @jit
    def model_predict(self, x: Array) -> Array:
        """Compute model prediction."""
        return self.F @ x

    @jit
    def objective_value(self, x: Array) -> Array:
        """Compute the objective function value (chi-squared + regularization)."""
        model = self.model_predict(x)
        resid = self.d - model
        chi2 = jnp.sum((resid * resid) * self.N_inv_diag)
        reg = jnp.dot(x, self.H @ x) + self.jitter * jnp.dot(x, x)
        return chi2 + reg

    @jit
    def log_evidence(self) -> Array:
        """
        Compute approximate log-evidence for NNLS solution.

        Approximation uses the Laplace approximation at the MAP solution.

        Returns
        -------
        Array
            Log-evidence value.
        """
        def _valid(_):
            x = self.solve()
            obj = self.objective_value(x)
            log_ev = self.log_evidence_const
            log_ev += self.half_log_det_H
            log_ev -= 0.5 * obj
            log_ev -= self.half_log_det_M
            return log_ev

        return jax.lax.cond(self.is_valid, _valid, lambda _: -jnp.inf, operand=None)

    def tree_flatten(self):
        """Flatten for JAX pytree registration."""
        children = (
            self.d,
            self.F,
            self.noise_cov,
            self.H,
            self.N_diag,
            self.N_inv_diag,
            self.sqrt_w,
            self.B,
            self.half_log_det_H,
            self.half_log_det_M,
            self.log_evidence_const,
            self.is_valid,
        )
        aux_data = (self.jitter, self.eps)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten for JAX pytree registration."""
        obj = cls.__new__(cls)

        (
            d,
            F,
            noise_cov,
            H,
            N_diag,
            N_inv_diag,
            sqrt_w,
            B,
            half_log_det_H,
            half_log_det_M,
            log_evidence_const,
            is_valid,
        ) = children

        jitter, eps = aux_data

        obj.d = d
        obj.F = F
        obj.noise_cov = noise_cov
        obj.H = H
        obj.N_diag = N_diag
        obj.N_inv_diag = N_inv_diag
        obj.sqrt_w = sqrt_w
        obj.B = B
        obj.half_log_det_H = half_log_det_H
        obj.half_log_det_M = half_log_det_M
        obj.log_evidence_const = log_evidence_const
        obj.is_valid = is_valid
        obj.jitter = jitter
        obj.eps = eps
        obj.n_data = d.shape[0]
        obj.n_source = H.shape[0]
        return obj


__all__ = [
    'LinearInversion',
    'NNLSInversion',
]
