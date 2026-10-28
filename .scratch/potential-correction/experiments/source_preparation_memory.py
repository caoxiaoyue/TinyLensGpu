"""Report live JAX allocation from preparing matrix-free source priors.

Run from the repository root in tinylens_gpu with
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. python <this-file>.
This counts live array bytes, independently of allocator pool reservations.
"""

import json

import jax

from TinyLensGpu.utils.inversion.regularization import FiniteDifferenceRegularization


def prepare_bytes(n, kind):
    before = {id(array) for array in jax.live_arrays()}
    builder = FiniteDifferenceRegularization(n, kind)
    data = builder.make_reg_data(-1.0, 1.0, -1.0, 1.0)
    jax.block_until_ready(data)
    return sum(array.nbytes for array in jax.live_arrays() if id(array) not in before)


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "jax": jax.__version__,
                "device": str(jax.devices()[0]),
                "x64_enabled": jax.config.x64_enabled,
                "measurements": [
                    {
                        "n": n,
                        "prior": kind,
                        "preparation_live_bytes": prepare_bytes(n, kind),
                        "single_dense_float32_matrix_bytes": 4 * n**4,
                    }
                    for n in (16, 32, 64)
                    for kind in ("zero-order", "first-order", "second-order")
                ],
            },
            indent=2,
        )
    )
