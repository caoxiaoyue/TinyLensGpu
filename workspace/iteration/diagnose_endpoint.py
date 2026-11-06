"""Read a saved rejected endpoint and test local descent without altering data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pt  # Environment and checkout paths are configured by the entry point.
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from lbfgs import EvidenceEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    row = json.loads((args.run / "evaluations.jsonl").read_text().splitlines()[0])
    q = np.load(args.run / "candidate_0000.npz")["joint_values"]
    problem, _, _ = pt.load_problem()
    engine = EvidenceEngine(problem, source_initialization="saved_M3")
    strengths = jnp.asarray(10.0 ** np.asarray(row["log10_strengths"]))
    coords = engine.prepare_coordinates(strengths)
    pd, pc, vp, vectors, coefficients, _ = coords
    delta = jnp.asarray(q) - engine.q0
    ys = jnp.sqrt(strengths[0]) * (engine.sc.T @ (engine.sd * delta[:engine.ns]))
    yp = pc.T @ (pd * delta[engine.ns:])
    image = engine.s0.T @ ys / jnp.sqrt(strengths[0]) + vp.T @ yp
    # B^-1 = I+V U diag{1/(sqrt(c)+1)} U^T V^T.
    covariance = engine.eye + engine.source_cov0 / strengths[0] + vp.T @ vp
    eigenvalues, _ = jnp.linalg.eigh(covariance)
    image = vectors @ ((1 / (jnp.sqrt(eigenvalues) + 1)) * (vectors.T @ image))
    z = jnp.concatenate((ys + engine.s0 @ image / jnp.sqrt(strengths[0]), yp + vp @ image))
    decode_error = float(jnp.max(jnp.abs(engine.decode(z, strengths, coords) - q)))
    kernel_args = (strengths, engine.q0, engine.sd, engine.sc, engine.s0, pd, pc, vp, vectors, coefficients)

    def fun(z):
        value, gradient = engine.white_value_grad(jnp.asarray(z), *kernel_args)
        return float(value), np.asarray(gradient)

    f0, gradient = fun(z)
    direction = -gradient / np.linalg.norm(gradient)
    probes = []
    for step in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11):
        plus, _ = fun(z + step * direction)
        minus, _ = fun(z - step * direction)
        probes.append(dict(step=step, forward_difference=plus-f0, reverse_difference=minus-f0,
                           predicted_forward=step * float(gradient @ direction)))
    mapping = problem.linearize(jnp.asarray(q)).mapping
    box = np.asarray(problem.renderer.source_bbox)
    uv = np.c_[(np.asarray(mapping.beta_x)-box[0])/(box[1]-box[0]),
               (np.asarray(mapping.beta_y)-box[2])/(box[3]-box[2])]
    edge_distance = np.min(np.abs(np.concatenate((uv, uv-1), axis=1)), axis=1)
    knot_distance = np.min(np.abs(uv*99 - np.round(uv*99)), axis=1)
    records = dict(source_run=str(args.run), decode_error=decode_error, objective=f0,
                   white_gradient_norm=float(np.linalg.norm(gradient)),
                   closest_bbox_distance=float(edge_distance.min()),
                   closest_knot_distance_node_units=float(knot_distance.min()),
                   probes=probes, restarts=[])
    print(json.dumps(records, indent=2), flush=True)
    for restart in range(3):
        start = time.perf_counter()
        result = minimize(fun, np.asarray(z), method="L-BFGS-B", jac=True,
                          options=dict(maxiter=1000, maxcor=30, maxls=80, ftol=0.0, gtol=1e-8))
        z = result.x
        record = dict(restart=restart, objective=float(result.fun), iterations=result.nit,
                      evaluations=result.nfev, success=result.success, message=str(result.message),
                      white_gradient_inf=float(np.max(np.abs(result.jac))),
                      seconds=time.perf_counter()-start)
        records["restarts"].append(record)
        print(record, flush=True)
    pt.write_json(args.run / "endpoint_diagnosis.json", records)


if __name__ == "__main__":
    main()
