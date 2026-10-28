"""Synthetic bounded principal-block application benchmark through public code.

Measures repeated application after inverse-factor preparation. Blocks are
well conditioned; this is not an ill-conditioned accuracy or whole-fit claim.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from TinyLensGpu.utils.cg_solver import (
    PrincipalBlockPreconditioner,
    apply_preconditioner,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=9)
    parser.add_argument("--width", type=int, default=834)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    with jax.experimental.enable_x64():
        rng = np.random.default_rng(907)
        factors = jnp.asarray(
            np.tril(rng.normal(scale=0.002, size=(args.blocks, args.width, args.width)))
            + 2 * np.eye(args.width)[None]
        )
        indices = jnp.arange(args.blocks * args.width).reshape(args.blocks, args.width)
        valid = jnp.ones_like(indices, dtype=bool)
        rhs = jnp.asarray(rng.normal(size=args.blocks * args.width))
        prepare = jax.jit(
            jax.vmap(
                lambda factor: jsl.solve_triangular(
                    factor, jnp.eye(args.width, dtype=factor.dtype), lower=True
                )
            )
        )
        prepare(factors).block_until_ready()
        started = time.perf_counter()
        inverses = prepare(factors).block_until_ready()
        preparation = time.perf_counter() - started
        apply = jax.jit(apply_preconditioner)
        methods = dict(
            triangular=PrincipalBlockPreconditioner(factors, indices, valid),
            inverse_factor_matvec=PrincipalBlockPreconditioner(
                factors, indices, valid, inverses
            ),
        )
        reference = apply(methods["triangular"], rhs).block_until_ready()
        candidate = apply(methods["inverse_factor_matvec"], rhs).block_until_ready()
        rows = []
        for repeat in range(3):
            for method, preconditioner in methods.items():
                started = time.perf_counter()
                for _ in range(args.repeats):
                    apply(preconditioner, rhs).block_until_ready()
                rows.append(
                    dict(
                        round=repeat,
                        method=method,
                        seconds_per_apply=(time.perf_counter() - started)
                        / args.repeats,
                    )
                )
        record = dict(
            scope=__doc__,
            platform=jax.devices()[0].platform,
            device=str(jax.devices()[0]),
            blocks=args.blocks,
            width=args.width,
            dtype="float64",
            inverse_preparation_seconds=preparation,
            measurements=rows,
            max_abs_difference=float(jnp.max(jnp.abs(reference - candidate))),
            relative_l2_difference=float(
                jnp.linalg.norm(reference - candidate) / jnp.linalg.norm(reference)
            ),
        )
    if args.output is not None:
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return record


if __name__ == "__main__":
    main()
