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
    n_data = d.shape[0]
    n_source = H.shape[0]
    
    is_diagonal = (noise_cov.ndim == 1)
    is_valid = jnp.array(True)

    N_diag = jnp.ones((n_data,), dtype=d.dtype)
    N_inv_diag = jnp.ones((n_data,), dtype=d.dtype)
    
    if is_diagonal:
        N_diag = jnp.clip(noise_cov, min=eps)
        N_inv_diag = 1.0 / N_diag
        
        half_log_det_N = 0.5 * jnp.sum(jnp.log(N_diag))
        
        weighted_F = F * N_inv_diag[:, None]
        FT_Ninv_F = weighted_F.T @ F
        
        FT_Ninv_d = F.T @ (d * N_inv_diag)
        
        d_Ninv_d = jnp.sum(d**2 * N_inv_diag)
        
    else:
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
    n = H.shape[0]
    H_stab = H + jitter * jnp.eye(n, dtype=H.dtype)
    H_stab = 0.5 * (H_stab + H_stab.T)
    eigvals, eigvecs = jnp.linalg.eigh(H_stab)
    eigvals = jnp.clip(eigvals, min=jitter)
    sqrt_eigvals = jnp.sqrt(eigvals)
    return sqrt_eigvals[:, None] * eigvecs.T


@register_pytree_node_class
class LinearInversion:
    """
    Linear inversion framework for gravitational lensing source reconstruction.
    
    Optimized for JAX JIT compilation.
    
    Implements:
    - Regularized linear inversion s = (F^T N^{-1} F + H)^{-1} F^T N^{-1} d
    - Solution covariance Σ = (F^T N^{-1} F + H)^{-1}
    - Bayesian evidence calculation
    
    Key optimization: 
    - Pre-computes all linear algebra terms including Cholesky decompositions.
    - Registered as JAX PyTree to allow efficient passing to JIT-compiled functions.
    - Methods are JIT-compatible without static_argnums.
    
    Parameters
    ----------
    d : array_like
        Data vector (observed image pixels), shape [n_data]
    F : array_like
        Blurred lensing mapping matrix, shape [n_data, n_source]
    noise_cov : array_like
        Noise covariance matrix N, shape [n_data, n_data] or [n_data] for diagonal.
        Note: 2D arrays are treated as full matrices even if diagonal.
    H : array_like
        Regularization matrix (with λ already absorbed), shape [n_source, n_source]
    """

    def __init__(self, d, F, noise_cov, H, _precomputed=None):
        """Initialize and pre-compute common terms for efficiency."""
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
        Solve for the source reconstruction 's' only.
        
        Returns
        -------
        s : jnp.ndarray
            Source reconstruction, shape [n_source]
        """
        s = jnp.linalg.solve(self.M_stab, self.FT_Ninv_d)
        return s

    @jit
    def model_predict(self, s: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the model data (unmasked pixels) given the source.
        
        Parameters
        ----------
        s : jnp.ndarray
            Source intensities, shape [n_source]
            
        Returns
        -------
        model_data : jnp.ndarray
            Model data vector, shape [n_data]
        """
        return self.F @ s

    def invert(self):
        """
        Solve the regularized linear inversion and compute covariance.
        
        Returns
        -------
        s : jnp.ndarray
            Source reconstruction, shape [n_source]
        Sigma : jnp.ndarray
            Solution covariance matrix, shape [n_source, n_source]
        """
        s = self.solve()
        Sigma = jnp.linalg.inv(self.M_stab)
        Sigma = 0.5 * (Sigma + Sigma.T)
        return s, Sigma

    @jit
    def log_evidence(self):
        """
        Compute the Bayesian log evidence.
        
        Returns
        -------
        log_evidence : float
            Log of the Bayesian evidence
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
        """Flatten the object for JAX PyTree registration."""
        children = (self.d, self.F, self.noise_cov, self.H,
                    self.FT_Ninv_F, self.FT_Ninv_d, self.d_Ninv_d, self.M_stab,
                    self.half_log_det_H, self.half_log_det_M, self.log_evidence_const, self.is_valid)
        aux_data = (self.jitter, self.eps)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the object for JAX PyTree registration."""
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
    def __init__(self, d, F, noise_cov, H, _precomputed=None):
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
        Z = self.F * self.sqrt_w[:, None]
        y = self.d * self.sqrt_w
        Z_aug = jnp.concatenate([Z, self.B], axis=0)
        y_aug = jnp.concatenate([y, jnp.zeros(self.n_source, dtype=y.dtype)], axis=0)
        x, _ = fnnls_jax(Z_aug, y_aug)
        return x

    @jit
    def model_predict(self, x: Array) -> Array:
        return self.F @ x

    @jit
    def objective_value(self, x: Array) -> Array:
        model = self.model_predict(x)
        resid = self.d - model
        chi2 = jnp.sum((resid * resid) * self.N_inv_diag)
        reg = jnp.dot(x, self.H @ x) + self.jitter * jnp.dot(x, x)
        return chi2 + reg

    @jit
    def log_evidence(self) -> Array:
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
