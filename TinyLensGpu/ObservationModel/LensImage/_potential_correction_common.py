"""Shared numerical preparation for source–potential reconstruction."""

import jax
import jax.numpy as jnp


def _dense_response_columns(action, size, dtype):
    """Materialize a dense response with bounded subpixel/image temporaries.

    Only sixteen basis responses are evaluated simultaneously. Generating each
    basis vector inside the map avoids an additional complete identity array.
    The returned dense design is intentional for the reference backend.
    """
    return jax.lax.map(
        lambda index: action(jax.nn.one_hot(index, size, dtype=dtype)),
        jnp.arange(size, dtype=jnp.int32),
        batch_size=16,
    ).T
