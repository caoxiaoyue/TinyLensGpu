"""Minimal diagnostic: the actual hard source boundary can stall L-BFGS.

The hard variant intentionally fails the convergence assertion.
It uses one whitened deflection coordinate, a nonzero source edge, and the
same interpolation function as the real renderer. No J0946 data are fitted.
"""

from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from TinyLensGpu.utils.lensing.mapping import lens_mapping_operator_bilinear_from


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", choices=("hard", "zero-ring"), default="hard")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    source = jnp.full((3, 3), 0.04, dtype=jnp.float64)
    bounds = (-0.105, -0.005, -0.05, 0.05)
    if args.boundary == "zero-ring":
        # Keep original nodes/brightness and spacing. Extra nodes are fixed
        # zeros, not additional optimization parameters. This is an explicit
        # alternative source boundary model, tested only in this small probe.
        source = jnp.pad(source, 1)
        bounds = (-0.155, 0.045, -0.10, 0.10)
    source_n = source.shape[0]
    source = source.ravel()
    precision = 4000.0
    initial_position = -0.01

    def brightness(position):
        weights, indices, _ = lens_mapping_operator_bilinear_from(
            jnp.array([[position, 0.0]]), *bounds, source_n
        )
        return jnp.sum(weights * source[indices])

    def residual(z):
        # Initial GN curvature in this coordinate is exactly one: the source
        # is constant inside its grid. Thus poor conditioning is not required.
        position = initial_position + z[0] / jnp.sqrt(precision)
        return jnp.array([(brightness(position) - 0.04) / 0.01,
                          jnp.sqrt(precision) * position])

    def objective(z):
        r = residual(z)
        return 0.5 * jnp.vdot(r, r)

    value_gradient = jax.jit(jax.value_and_grad(objective))

    def fun(z):
        value, gradient = value_gradient(jnp.asarray(z))
        return float(value), np.asarray(gradient)

    attempts = []
    for _ in range(3):
        result = minimize(fun, np.zeros(1), jac=True, method="L-BFGS-B",
                          options=dict(maxiter=3000, maxls=80, maxcor=30,
                                       ftol=1e-15, gtol=1e-8))
        _, gradient = fun(result.x)
        curvature = float(jnp.sum(jax.jacfwd(residual)(jnp.asarray(result.x))**2))
        proxy = float(0.5 * gradient @ gradient / curvature)
        attempts.append(dict(native_success=bool(result.success),
            native_message=str(result.message), iterations=int(result.nit),
            source_position=float(initial_position + result.x[0]/np.sqrt(precision)),
            original_source_boundary=-0.005, objective=float(result.fun),
            endpoint_GN_curvature=curvature,
            white_gradient=float(gradient[0]), gn_proxy=proxy,
            passed=bool(result.success and proxy <= 1e-3)))
    edge_differences = {str(h): float(abs(brightness(-0.005-h)-brightness(-0.005+h)))
                        for h in (1e-3, 1e-5, 1e-7)}
    report = dict(purpose="Minimal boundary-stall reproduction, not a scientific lens fit",
                  dtype="float64", boundary=args.boundary,
                  initial_whitened_curvature=1.0,
                  brightness_difference_across_original_edge=edge_differences, attempts=attempts)
    path = Path(__file__).with_name("boundary_probe_" + args.boundary + ".json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("[DEBUG-boundary] " + json.dumps(report, indent=2), flush=True)
    assert all(row["passed"] for row in attempts), (
        "Boundary stall reproduced: L-BFGS stops with a nonzero gradient "
        "at the hard source edge despite unit initial curvature."
    )


if __name__ == "__main__":
    main()
