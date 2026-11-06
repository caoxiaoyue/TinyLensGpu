"""Variance-matched extra Gaussian noise for the fixed coarse observation.

Noise is added after the original block-mean observation operator. The fine
rendering/PSF/potential geometry and all physical priors remain unchanged.
"""

import numpy as np


def add_matched_noise(data, sigma, target_sigma, seed):
    data=np.asarray(data,dtype=np.float64)
    sigma=np.broadcast_to(np.asarray(sigma,dtype=np.float64),data.shape).copy()
    target=np.broadcast_to(np.asarray(target_sigma,dtype=np.float64),data.shape).copy()
    if data.ndim!=1 or not np.isfinite(data).all():
        raise ValueError("data must be a finite observation vector")
    if not np.all(np.isfinite(sigma)&(sigma>0)&np.isfinite(target)&(target>0)):
        raise ValueError("Noise standard deviations must be finite and positive")
    variance=target**2-sigma**2
    if np.any(variance<0):
        raise ValueError("Additional independent noise cannot lower the existing variance")
    extra_sigma=np.sqrt(variance)
    z=np.random.default_rng(seed).normal(size=data.shape)
    extra=extra_sigma*z
    return dict(data=data+extra,sigma=target,base_data=data.copy(),base_sigma=sigma,
                extra_sigma=extra_sigma,extra_standard_normal=z,extra_noise=extra)


def set_projected_likelihood(problem, data, sigma):
    """Set the new coarse likelihood immediately after construction, before JIT.

    Each worker owns a fresh problem. Never call this on a previously traced
    problem: its compiled kernels can capture the old observation arrays.
    Only the coarse observation changes; the fine renderer is left intact.
    """
    import jax.numpy as jnp
    data=np.asarray(data,dtype=np.float64)
    sigma=np.asarray(sigma,dtype=np.float64)
    if data.shape!=problem.data_active.shape or sigma.shape!=data.shape:
        raise ValueError("Coarse data/noise shapes must match the projected likelihood")
    if not np.isfinite(data).all() or not np.all(np.isfinite(sigma)&(sigma>0)):
        raise ValueError("Coarse likelihood arrays must be finite with positive noise")
    problem.data_active=jnp.asarray(data,dtype=problem.dtype)
    problem.noise_active=jnp.asarray(sigma,dtype=problem.dtype)
    problem.noise_logdet=2*jnp.log(problem.noise_active).sum()
    problem.initialization_provenance.update(observation_noise="matched_extra_coarse_gaussian")
