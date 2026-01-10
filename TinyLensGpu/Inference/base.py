from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
import numpy as np
import jax
import jax.numpy as jnp

from .build_prior import make_prior_transformation

class AbstractInference(ABC): 
    def __init__(
        self,
        prob_model: Optional[Any] = None,
        ndim: Optional[int] = None,
        prior_transform: Optional[Callable[[Any], Any]] = None,
    ):
        self.prob_model = prob_model
        self.ndim = ndim
        self.prior_transform = prior_transform


    def _ensure_prior_transform(self) -> None:
        if self.prior_transform is not None and self.ndim is not None:
            return
        if self.prob_model is None:
            raise ValueError("prob_model is not set")

        prior_transform, specs = make_prior_transformation(self.prob_model)
        if self.prior_transform is None:
            self.prior_transform = prior_transform
        if self.ndim is None:
            self.ndim = len(specs)


    def loglike_jax(self, theta):
        """JAX-friendly log-likelihood. Safe to use with jax.jit/jax.vmap."""
        theta = jnp.asarray(theta, dtype=jnp.float32)
        return self.prob_model(theta)


    def likelihood(self, theta):
        """
        Returns the log likelihood of the parameters.
        
        Notes
        -----
        This wrapper returns Python/numpy values for compatibility with
        SciPy and nested samplers that expect numpy/float outputs.
        For JAX vectorization, use `loglike_jax`.
        """
        theta = np.asarray(theta)
        if theta.ndim > 1:
            theta_jax = jnp.asarray(theta, dtype=jnp.float32)
            out = jax.vmap(self.loglike_jax)(theta_jax)
            return np.asarray(out)
        return float(np.asarray(self.loglike_jax(theta)))


    def prior(self, u):
        """Prior transform from unit cube to parameter space."""
        self._ensure_prior_transform()
        return np.asarray(self.prior_transform(u))


    @abstractmethod
    def run(self, nlive=1000, **kwargs):
        """
        Runs the inference
        """
        pass
