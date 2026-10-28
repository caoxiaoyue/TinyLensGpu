"""Isolated observable operator-allocation probe; no dense reference is built."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import (
    IterativePotentialCorrection,
    SingleStepPotentialCorrection,
)
from TinyLensGpu.PhysicalModel import PhysicalModel, Shear
from TinyLensGpu.utils.lensing.potential import PotentialGrid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-n", type=int, default=96)
    parser.add_argument("--extra-potential", type=int, default=9216)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument(
        "--algorithm", choices=("single_step", "iterative"), default="single_step"
    )
    args = parser.parse_args(argv)
    with jax.experimental.enable_x64():
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.0, gamma2=0.0)], source_light=[], lens_light=[]
        )
        config = SimulatorConfig(0.1, 3, jnp.ones((1, 1)), nsub=1)
        base = PotentialGrid.from_queries(
            config.xgrid_sub_1d, config.ygrid_sub_1d, spacing=0.2
        )
        # Additional isolated support-only unknowns exercise normalization and
        # potential storage without enlarging the image or derivative domain.
        extra = np.column_stack(
            (
                100 + 3 * np.arange(args.extra_potential),
                np.full(args.extra_potential, 100),
            )
        )
        grid = PotentialGrid.from_queries(
            config.xgrid_sub_1d,
            config.ygrid_sub_1d,
            spacing=0.2,
            node_indices=np.vstack((base.node_indices, extra)),
        )
        renderer = PotentialCorrectionRenderer(
            macro,
            config,
            args.source_n,
            (-0.3, 0.3, -0.3, 0.3),
            potential_grid=grid,
        )
        if args.algorithm == "single_step":
            problem = SingleStepPotentialCorrection(
                renderer,
                np.zeros((3, 3)),
                np.ones((3, 3)),
                lambda bx, by: (jnp.ones_like(bx), -jnp.ones_like(by)),
                lambda_amplitude=0.4,
            )
            fitted = problem.fit(
                1.3, 0.3, backend="operator", block_size=args.block_size
            )
            assert fitted.success, fitted.failure_reason
        else:
            problem = IterativePotentialCorrection(
                renderer,
                np.full((3, 3), 0.1),
                np.ones((3, 3)),
                np.zeros(renderer.n_source),
                lambda_amplitude=0.4,
                source_regularization="zero-order",
            )
            # Two current points exercise refreshing data blocks and real PCG
            # increments, rather than stopping at the initial zero-gradient point.
            fitted = problem.fit(
                1.3,
                0.3,
                backend="operator",
                block_size=args.block_size,
                max_iter=2,
            )
            assert fitted.solver_info.converged, fitted.failure_reason
            assert (
                fitted.termination_reason == "iteration_limit"
            ), fitted.termination_reason
            assert not fitted.success and fitted.log_evidence is None
        fitted.source_values.block_until_ready()
        arrays = jax.live_arrays()
        shapes = [list(array.shape) for array in arrays]
        stats = jax.devices()[0].memory_stats()
        record = {
            "device": str(jax.devices()[0]),
            "algorithm": args.algorithm,
            "termination_reason": fitted.termination_reason,
            "completed_linear_solves": sum(
                "solver_info" in item and item["solver_info"]["converged"]
                for item in fitted.history
            ),
            "n_source": problem.n_source,
            "n_potential": problem.n_potential,
            "n_joint": problem.n_joint,
            "n_data": renderer.n_active,
            "block_size": args.block_size,
            "block_count": fitted.configuration["block_count"],
            "largest_live_array_elements": max(array.size for array in arrays),
            "live_shapes": shapes,
            "peak_bytes_in_use": (
                None if stats is None else stats.get("peak_bytes_in_use")
            ),
            "bytes_in_use": None if stats is None else stats.get("bytes_in_use"),
        }
    print(json.dumps(record))
    return record


if __name__ == "__main__":
    main()
