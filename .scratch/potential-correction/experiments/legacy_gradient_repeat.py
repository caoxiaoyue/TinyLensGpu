"""Reproduce float32 finite-difference variability in the existing PCG fixture.

This is a compatibility diagnostic, not potential-correction calibration.
Use --checkout to compare an unchanged baseline with the current checkout.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tight-rtol", type=float, default=1e-8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout.resolve()))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from tests.test_operator_autodiff import _build_model

    rows = []
    for rtol in (1e-5, args.tight_rtol):
        model, theta, steps = _build_model("pcg", joint=True, adaptive=True)
        model.pcg_rtol = rtol
        weights = jnp.linspace(0.5, 1.5, 25)

        def outputs(parameters):
            result = model.evaluate(parameters)
            return jnp.stack(
                [
                    weights @ result.source_pixels,
                    result.log_evidence,
                    result.lens_amplitudes[0],
                ]
            )

        actual = np.asarray(jax.jit(jax.jacrev(outputs))(theta))
        ratios = []
        for _ in range(args.repeats):
            columns = []
            for i, step in enumerate(steps):
                delta = jnp.zeros_like(theta).at[i].set(step)
                columns.append(
                    (outputs(theta + delta) - outputs(theta - delta)) / (2 * step)
                )
            expected = np.asarray(jnp.stack(columns, axis=1))
            ratios.append(
                float(
                    np.max(np.abs(actual - expected) / (0.03 + 0.05 * np.abs(expected)))
                )
            )
        rows.append(
            {
                "pcg_rtol": rtol,
                "failed_repeats": sum(value > 1 for value in ratios),
                "max_tolerance_ratio": max(ratios),
                "min_tolerance_ratio": min(ratios),
                "ratios": ratios,
            }
        )
    record = {
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(args.checkout), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "jax_version": jax.__version__,
        "device": jax.devices()[0].device_kind,
        "model": "existing float32 adaptive joint source-lens-light PCG fixture",
        "source_n": 5,
        "image_shape": [10, 10],
        "repeats": args.repeats,
        "assertion_rtol": 0.05,
        "assertion_atol": 0.03,
        "results": rows,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
