"""Differentiation policy for a pixelized inversion's sampling score."""

import jax
import jax.numpy as jnp
from jax import Array


@jax.custom_jvp
def penalize_invalid_evidence(score: Array, valid: Array, penalty: Array) -> Array:
    """Return a finite sampling score and reject gradients of invalid scores."""
    return jnp.where(valid, score, penalty)


@penalize_invalid_evidence.defjvp
def _penalize_invalid_evidence_jvp(primals, tangents):
    score, valid, penalty = primals
    score_tangent, _, _ = tangents
    multiplier = jnp.where(valid, jnp.ones_like(score), jnp.full_like(score, jnp.nan))
    return penalize_invalid_evidence(score, valid, penalty), multiplier * score_tangent
