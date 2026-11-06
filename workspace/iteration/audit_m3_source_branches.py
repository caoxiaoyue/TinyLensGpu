"""Read-only derivative/branch audit of the selected fixed-source-hyper repeats."""

import argparse
import json
from pathlib import Path

import pt
import jax
import jax.numpy as jnp
import numpy as np

from lbfgs import EvidenceEngine
from run_fixed_boundary import zero_ring_problem
from examples.potential_correction.result_products import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    jax.config.update("jax_enable_x64",True)
    jax.config.update("jax_default_matmul_precision","highest")
    summary=json.loads((args.run/"summary.json").read_text())
    protocol=json.loads((args.run/"protocol.json").read_text())
    if summary["best"] is None:
        raise ValueError("No selected candidate to audit")
    strengths=jnp.asarray([summary["best"]["strengths"][k] for k in ("lambda_s","lambda4","lambda0")])
    hard,_,arrays=pt.load_problem(Path(protocol["arguments"]["macro_dir"]),.1,float(strengths[2]))
    problem=zero_ring_problem(hard,arrays,float(strengths[2]))
    engine=EvidenceEngine(problem)
    previous=None;records=[]
    for index in summary["selected_repeat_indices"]:
        row=summary["rows"][index]
        with np.load(args.run/row["directory"]/"reconstruction.npz") as archive:
            q=np.array(archive["joint_values"])
        point=jnp.asarray(q)
        value,gradient=engine.physical_value_grad(point,strengths)
        jacobian=engine.jacobian(point)
        residual=(problem.forward(point)-problem.data_active)/problem.noise_active
        from_jacobian=jacobian.T@residual+problem.prior_matvec(point,strengths[0],strengths[1])
        mapping=problem.renderer.source_mapping(point[problem.n_source:])
        bbox=np.asarray(problem.renderer.source_bbox);n=problem.renderer.source_n
        uv=np.column_stack(((np.asarray(mapping.beta_x)-bbox[0])/(bbox[1]-bbox[0]),
                            (np.asarray(mapping.beta_y)-bbox[2])/(bbox[3]-bbox[2])))*(n-1)
        inside=np.all((uv>0)&(uv<n-1),axis=1)
        cells=np.floor(uv).astype(int)
        item=dict(index=index,saved_proxy=row["gn_stationarity_proxy"],recomputed_objective=float(value),
            AD_vs_batched_J_gradient_error=float(jnp.linalg.norm(gradient-from_jacobian)/jnp.maximum(1.,jnp.linalg.norm(gradient))),
            minimum_interior_knot_distance_node_units=float(np.abs(uv[inside]-np.round(uv[inside])).min()),
            interior_queries_within_1e_12_node_units=int(np.count_nonzero(
                np.min(np.abs(uv-np.round(uv)),axis=1)[inside]<=1e-12)))
        if previous is not None:
            item.update(max_joint_difference_from_previous=float(np.max(abs(q-previous[0]))),
                max_potential_difference_from_previous=float(np.max(abs(q[problem.n_source:]-previous[0][problem.n_source:]))),
                mapping_cell_changes_from_previous=int(np.count_nonzero(np.any(cells!=previous[1],axis=1))))
        records.append(item);previous=(q,cells)
    write_json(args.run/"postfit_branch_audit.json",dict(optimization_rerun=False,
        original_results_modified=False,records=records))
    print(json.dumps(records,indent=2))


if __name__=="__main__":
    main()
