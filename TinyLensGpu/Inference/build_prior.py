"""
Prior specification and transformation following example_v4.py style.

This module provides PriorSpec for transforming unit-cube samples to parameter space,
and utilities for extracting prior specs from caskade modules with ParamU parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, List, Callable, Optional
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erfinv
import caskade as ck


@dataclass(frozen=True)
class PriorSpec:
    """
    Metadata for transforming unit-cube samples to parameter space.
    
    Parameters
    ----------
    name : str
        Parameter name
    prior_type : {'uniform', 'gaussian', 'log_uniform'}
        Type of prior distribution
    settings : tuple of float
        Prior parameters (min, max) for uniform/log_uniform or (mean, std) for gaussian
    limits : tuple of float, optional
        Hard limits to clip parameter values
    
    Examples
    --------
    >>> # Uniform prior
    >>> spec = PriorSpec("slope", "uniform", (0.0, 2.0))
    >>> u = jnp.array([0.5])
    >>> spec.transform(u)  # Returns 1.0
    """
    
    name: str
    prior_type: Literal["uniform", "gaussian", "log_uniform"]
    settings: Tuple[float, float]
    limits: Tuple[float, float] | None = None
    
    def transform(self, u: Array) -> Array:
        """
        Transform unit samples to parameter space.
        
        Parameters
        ----------
        u : Array
            Unit cube samples in [0, 1]
        
        Returns
        -------
        Array
            Transformed parameter values
        """
        u = jnp.clip(u, 1e-9, 1 - 1e-9)
        a, b = self.settings
        
        if self.prior_type == "uniform":
            val = a + u * (b - a)
        elif self.prior_type == "log_uniform":
            val = jnp.exp(jnp.log(a) + u * (jnp.log(b) - jnp.log(a)))
        elif self.prior_type == "gaussian":
            val = a + b * jnp.sqrt(2.0) * erfinv(2.0 * u - 1.0)
        else:
            raise ValueError(f"Unsupported prior type: {self.prior_type}")
        
        return jnp.clip(val, *self.limits) if self.limits else val
    
    def describe(self) -> str:
        """
        Format prior settings for logs and summaries.

        Returns
        -------
        str
            Human-readable prior description including optional hard limits.
        """
        a, b = self.settings
        desc = f"N({a:.2f}, {b:.2f})" if self.prior_type == "gaussian" else f"[{a:.2f}, {b:.2f}]"
        return f"{desc}, limits={self.limits}" if self.limits else desc


def extract_prior_specs(module: ck.Module) -> List[PriorSpec]:
    """
    Extract prior specifications from a caskade module.
    
    This function traverses the module tree and collects all dynamic parameters
    that have prior metadata (ParamU instances).
    
    Parameters
    ----------
    module : ck.Module
        module to extract priors from
    
    Returns
    -------
    list of PriorSpec
        List of prior specifications for all dynamic parameters
    
    Raises
    ------
    ValueError
        If dynamic parameters are missing prior metadata
    
    Examples
    --------
    >>> from TinyLensGpu.PhysicalModel import SersicEllipse
    >>> from TinyLensGpu.Inference import ParamU
    >>> sersic = SersicEllipse(
    ...     R_sersic=ParamU("R_sersic", 1.0, prior_type="uniform", prior_settings=[0.1, 2.0])
    ... )
    >>> sersic.R_sersic.to_dynamic()
    >>> specs = extract_prior_specs(sersic)
    """
    specs = []
    
    # Check if module has get_dynamic_params method (for LensLikelihood)
    if hasattr(module, 'get_dynamic_params'):
        dynamic_params = module.get_dynamic_params()
    else:
        dynamic_params = module.dynamic_params
    
    for param in dynamic_params:
        if not hasattr(param, "prior_type") or not hasattr(param, "prior_settings"):
            raise ValueError(f"Dynamic param '{param.name}' missing prior metadata")
        
        specs.append(
            PriorSpec(
                name=param.name,
                prior_type=param.prior_type,
                settings=tuple(param.prior_settings),
                limits=tuple(param.limits) if param.limits else None,
            )
        )
    
    if not specs:
        raise ValueError("Module has no dynamic parameters")
    return specs


def make_prior_transformation(
    module: ck.Module,
    *,
    dtype: Optional[jnp.dtype] = None,
) -> Tuple[Callable[[Array], Array], List[PriorSpec]]:
    """
    Create prior transformation function for nested sampler.
    
    Parameters
    ----------
    module : ck.Module
        module with dynamic parameters
    
    Returns
    -------
    transform : callable
        Function that transforms unit cube to parameter space
    specs : list of PriorSpec
        Prior specifications for all parameters
    
    Examples
    --------
    >>> prior_transform, specs = make_prior_transformation(likelihood_module)
    >>> # Use with Nautilus sampler
    >>> sampler = Sampler(prior_transform, loglike, n_dim=len(specs))
    """
    specs = extract_prior_specs(module)
    
    def transform(u):
        """
        Transform unit-cube parameters into physical parameter values.

        Parameters
        ----------
        u : Array
            Unit-cube sample whose last dimension equals ``len(specs)``.

        Returns
        -------
        Array
            Transformed parameters with same leading dimensions as ``u``.

        Raises
        ------
        ValueError
            If the last dimension of ``u`` does not match the number of priors.
        """
        u = jnp.asarray(u, dtype=dtype) if dtype is not None else jnp.asarray(u)
        if u.shape[-1] != len(specs):
            raise ValueError(f"Expected {len(specs)} parameters, got {u.shape[-1]}")
        
        return jnp.stack([s.transform(u[..., i]) for i, s in enumerate(specs)], axis=-1)
    
    return transform, specs
