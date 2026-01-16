"""
Linear Inversion Solver for Source Reconstruction

This module provides an optimized linear inversion framework for gravitational 
lensing source reconstruction. It implements regularized linear inversion with
Bayesian evidence calculation.

The solver is registered as a JAX PyTree for efficient JIT compilation.
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class
from functools import partial


@partial(jit, static_argnames=['jitter', 'eps'])
def _precompute_terms(d, F, noise_cov, H, jitter=1e-6, eps=1e-12):
    """
    JIT-compiled helper to precompute all linear algebra terms.
    """
    n_data = d.shape[0]
    n_source = H.shape[0]
    
    is_diagonal = (noise_cov.ndim == 1)
    is_valid = jnp.array(True)
    
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
    
    return (FT_Ninv_F, FT_Ninv_d, d_Ninv_d, M_stab, 
            half_log_det_H, half_log_det_M, log_evidence_const, is_valid)


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
            (self.FT_Ninv_F, self.FT_Ninv_d, self.d_Ninv_d, self.M_stab, 
             self.half_log_det_H, self.half_log_det_M, self.log_evidence_const, self.is_valid) = \
                _precompute_terms(self.d, self.F, self.noise_cov, self.H, self.jitter, self.eps)
        else:
            (self.FT_Ninv_F, self.FT_Ninv_d, self.d_Ninv_d, self.M_stab, 
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


__all__ = [
    'LinearInversion'
]
