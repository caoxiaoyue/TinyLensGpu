"""Typed JAX PyTree seam for matrix-free curvature operators."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from jax import Array


DataT = TypeVar("DataT")
SpecT = TypeVar("SpecT", bound=Hashable)
CurvatureKernel: TypeAlias = Callable[[Array, DataT, SpecT], Array]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CurvatureOperator(Generic[DataT, SpecT]):
    """Matrix-free curvature with dynamic data and static topology.

    ``data`` is the only dynamic PyTree child. ``kernel``, ``spec``, and
    ``size`` form the static JIT cache key, so changing numerical values with
    stable shapes and dtypes does not retrace solver loops.
    """

    data: DataT
    kernel: CurvatureKernel[DataT, SpecT]
    spec: SpecT
    size: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")

    def matvec(self, coefficients: Array) -> Array:
        """Apply the curvature operator to a one-dimensional coefficient vector."""
        coefficients = jnp.asarray(coefficients)
        expected_shape = (self.size,)
        if coefficients.shape != expected_shape:
            raise ValueError(
                f"expected coefficients shape {expected_shape}, "
                f"got {coefficients.shape}"
            )
        result = jnp.asarray(self.kernel(coefficients, self.data, self.spec))
        if result.shape != expected_shape:
            raise ValueError(
                f"kernel returned shape {result.shape}, expected {expected_shape}"
            )
        return result

    def tree_flatten(self):
        """Expose numerical data as dynamic leaves and topology as static metadata."""
        return (self.data,), (self.kernel, self.spec, self.size)

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        """Reconstruct an operator during JAX PyTree transformations."""
        kernel, spec, size = auxiliary_data
        (data,) = children
        return cls(data=data, kernel=kernel, spec=spec, size=size)


__all__ = ["CurvatureKernel", "CurvatureOperator"]
