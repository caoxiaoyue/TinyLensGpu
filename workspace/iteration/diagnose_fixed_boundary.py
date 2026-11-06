"""Read-only directional probes at the four fixed-lambda J0946 endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pt
import jax
import jax.numpy as jnp
import numpy as np

from lbfgs import EvidenceEngine
from run_fixed_boundary import zero_ring_problem


def ray_coordinates(engine, q):
    mapping = engine.problem.renderer.source_mapping(q[engine.ns:])
    box = np.asarray(engine.problem.renderer.source_bbox)
    n = engine.problem.renderer.source_n
    return np.column_stack(((np.asarray(mapping.beta_x)-box[0])/(box[1]-box[0]),
                            (np.asarray(mapping.beta_y)-box[2])/(box[3]-box[2]))) * (n-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    protocol = json.loads((args.run / "protocol.json").read_text())
    strengths = jnp.asarray([protocol["effective_strengths"][name]
                             for name in ("lambda_s", "lambda4", "lambda0")])
    hard, _, arrays = pt.load_problem(Path(protocol["arguments"]["macro_dir"]),
        protocol["configuration"]["potential_spacing"], float(strengths[2]))
    engines = {"hard": EvidenceEngine(hard)}
    engines["zero_ring"] = EvidenceEngine(zero_ring_problem(hard, arrays, float(strengths[2])),
                                          reference_engine=engines["hard"])
    coordinates = engines["hard"].prepare_coordinates(strengths)
    # q + T z uses exactly the same fixed reference metric as the fits. No
    # inverse-coordinate approximation or optimization restart is needed.
    delta = lambda z: engines["hard"].decode(z, strengths, coordinates)-engines["hard"].q0
    _, pullback = jax.vjp(delta, jnp.zeros(hard.n_joint))
    output = {}
    for boundary, engine in engines.items():
        for initialization in ("source_only", "single_step"):
            directory = args.run / boundary / initialization / "repeat_00"
            with np.load(directory / "reconstruction.npz") as archive:
                q = jnp.asarray(archive["joint_values"])
            value, gradient = engine.physical_value_grad(q, strengths)
            wg = pullback(gradient)[0]
            norm = float(jnp.linalg.norm(wg))
            direction = delta(-wg/norm)
            uv = ray_coordinates(engine, q)
            n = engine.problem.renderer.source_n
            active = np.all((uv >= -1) & (uv <= n), axis=1)
            interior = np.all((uv > 0) & (uv < n-1), axis=1)
            knot_distance = np.abs(uv-np.round(uv))
            outer_distance = np.minimum(np.abs(uv), np.abs(uv-(n-1)))
            rows = []
            for step in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
                plus, minus = q+step*direction, q-step*direction
                up, um = ray_coordinates(engine, plus), ray_coordinates(engine, minus)
                crossed = np.any(np.floor(up) != np.floor(um), axis=1) & active
                outer_crossed = np.any((up >= 0) != (um >= 0), axis=1) | np.any((up <= n-1) != (um <= n-1), axis=1)
                rows.append(dict(step_in_reference_white_coordinates=step,
                    objective_plus_minus_base=float(engine.objective(plus, strengths)-value),
                    objective_minus_minus_base=float(engine.objective(minus, strengths)-value),
                    predicted_forward_change=-norm*step,
                    interior_queries_changing_cells=int(np.count_nonzero(crossed & interior)),
                    queries_crossing_original_outer_edge=int(np.count_nonzero(outer_crossed))))
            output[f"{boundary}/{initialization}"] = dict(
                endpoint=str(directory), objective=float(value), white_gradient_norm=norm,
                minimum_interior_knot_distance_node_units=float(knot_distance[interior].min()),
                minimum_original_outer_edge_distance_node_units=float(outer_distance[active].min()),
                minimum_fixed_zero_ring_distance_node_units=float(np.minimum(np.abs(uv[active]+1),
                    np.abs(uv[active]-n)).min()), probes=rows)
            print(boundary, initialization, "gradient norm", norm, flush=True)
    pt.write_json(args.run / "endpoint_directional_profiles.json", dict(
        fitting_or_restart_performed=False, fixed_reference="hard M3", profiles=output))


if __name__ == "__main__":
    main()
