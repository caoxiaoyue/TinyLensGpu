"""Replay an actual J0946 interior grid-line crossing, without optimization."""

import argparse
import json
from pathlib import Path

import pt
import jax
import jax.numpy as jnp
import numpy as np

from examples.potential_correction.result_products import write_json
from source_boundary import zero_ring_gradient, zero_ring_mapping


HERE = Path(__file__).resolve().parent


def knot_probe(source, beta, bbox, n, mapping_function, gradient_function):
    bbox = np.asarray(bbox)
    spacing = np.array([bbox[1]-bbox[0], bbox[3]-bbox[2]])/(n-1)
    origin = bbox[[0, 2]]
    uv = (np.asarray(beta)-origin)/spacing
    inside = np.all((uv > 2) & (uv < n-3), axis=1)
    distance = np.where(inside[:, None], abs(uv-np.round(uv)), np.inf)
    ray, axis = np.unravel_index(np.argmin(distance), distance.shape)
    center = np.array(beta[ray], dtype=float)
    center[axis] = origin[axis]+np.round(uv[ray, axis])*spacing[axis]
    records = []
    for h in (1e-5, 1e-7, 1e-9):
        pair = np.tile(center, (2, 1))
        pair[:, axis] += np.array([-1., 1.])*h*spacing[axis]
        pair = jnp.asarray(pair)
        weights, indices, _ = mapping_function(pair, tuple(bbox), n)
        brightness = np.asarray(jnp.sum(weights*jnp.asarray(source)[indices], axis=1))
        gx, gy = gradient_function(jnp.asarray(source), pair, tuple(bbox), n)
        gradient = np.column_stack((gx, gy))
        records.append(dict(offset_node_units=h, brightness_difference=float(abs(np.diff(brightness)[0])),
            gradient_jump_norm=float(np.linalg.norm(gradient[1]-gradient[0])),
            gradients=gradient.tolist()))
    jumps = [r["gradient_jump_norm"] for r in records]
    passed = jumps[-1] <= 1e-5*max(1., np.linalg.norm(gradient)) and jumps[-1] < max(1e-10, jumps[0]/1000)
    return dict(ray=int(ray), axis=int(axis), crossing_arcsec=center.tolist(),
        initial_distance_node_units=float(distance[ray, axis]), records=records,
        derivative_continuity_pass=bool(passed), optimization_rerun=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bilinear", "cubic"), default="bilinear")
    parser.add_argument("--require-c1", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    hard, _, _ = pt.load_problem()
    with np.load(HERE/"m3_source_fixed_v1/trials/0038/reconstruction.npz") as archive:
        source, potential = archive["source_values"], archive["potential_values"]
    mapping = hard.renderer.source_mapping(jnp.asarray(potential))
    beta = np.column_stack((mapping.beta_x, mapping.beta_y))
    functions = (zero_ring_mapping, zero_ring_gradient)
    if args.method == "cubic":
        from source_cubic import cubic_mapping, cubic_gradient
        functions = cubic_mapping, cubic_gradient
    result = knot_probe(source, beta, hard.renderer.source_bbox, hard.renderer.source_n, *functions)
    result["method"] = args.method
    print(json.dumps(result, indent=2))
    if args.output:
        write_json(args.output, result)
    if args.require_c1 and not result["derivative_continuity_pass"]:
        raise AssertionError("Interior source derivative jump survives as crossing offsets shrink")


if __name__ == "__main__":
    main()
