from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
import numpy as np
import jax
import jax.numpy as jnp

from .build_prior import make_prior_transformation

class AbstractInference(ABC): 
    """
    Base interface shared by samplers and optimizers.

    The class standardizes three operations used by downstream inference
    backends: likelihood evaluation, prior transformation, and execution through
    ``run``.

    Parameters
    ----------
    prob_model : Any, optional
        Callable probability model taking parameter vectors and returning
        log-likelihood values.
    ndim : int, optional
        Number of free parameters. If omitted, inferred from the prior builder.
    prior_transform : Callable[[Any], Any], optional
        Unit-cube to physical-parameter transform.
    """
    def __init__(
        self,
        prob_model: Optional[Any] = None,
        ndim: Optional[int] = None,
        prior_transform: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Initialize inference wrapper state.

        Parameters
        ----------
        prob_model : Any, optional
            Probability model callable.
        ndim : int, optional
            Number of parameters.
        prior_transform : Callable[[Any], Any], optional
            Prior transform callable.
        """
        self.prob_model = prob_model
        self.ndim = ndim
        self.prior_transform = prior_transform


    def _ensure_prior_transform(self) -> None:
        """
        Lazily initialize prior transform and dimensionality.

        Raises
        ------
        ValueError
            If ``prob_model`` is unavailable and prior metadata cannot be derived.
        """
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
        """
        Evaluate log-likelihood in JAX space.

        Parameters
        ----------
        theta : array_like
            Parameter vector in physical space.

        Returns
        -------
        jnp.ndarray | float
            Log-likelihood value returned by ``prob_model``.
        """
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
        """
        Transform unit-cube variables to physical parameters.

        Parameters
        ----------
        u : array_like
            Unit-hypercube sample in ``[0, 1]^ndim``.

        Returns
        -------
        np.ndarray
            Parameter vector in model space.
        """
        self._ensure_prior_transform()
        return np.asarray(self.prior_transform(u))


    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Execute inference procedure.

        Parameters
        ----------
        *args
            Positional arguments consumed by specific backend implementations.
        **kwargs
            Backend-specific run options.
        """
        pass
