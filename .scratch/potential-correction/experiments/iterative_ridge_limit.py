"""Compare a coupled weak-prior ridge saddle with a controlled ridge limit.

This is a numerical fixture audit, not an astrophysical precision experiment.
GPU reduction order can change whether a weak-prior run leaves the symmetric
point. The strong-prior runs must retain the source-only ridge limit.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from TinyLensGpu.PhysicalModel import PhysicalModel, Shear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = dict(
        scope="coupled-ridge fixture audit; no physical inference claim", cases=[]
    )
    for amplitude in (2.0, 1e8):
        with jax.experimental.enable_x64(False):
            renderer = PotentialCorrectionRenderer(
                PhysicalModel(
                    lens_mass=[Shear(gamma1=0.0, gamma2=0.0)],
                    source_light=[],
                    lens_light=[],
                ),
                SimulatorConfig(0.1, 4, jnp.ones((1, 1))),
                2,
                (-0.5, 0.5, -0.5, 0.5),
                dtype="float32",
            )
        problem = IterativePotentialCorrection(
            renderer,
            np.full((4, 4), -2.0),
            np.ones((4, 4)),
            np.full(4, 5.0),
            lambda_amplitude=amplitude,
            source_regularization="zero-order",
        )
        case = dict(
            lambda_amplitude=amplitude,
            source_only_ridge_objective=96 / 7,
            source_only_ridge_values=-8 / 7,
            runs=[],
        )
        for index in range(args.repeats):
            result = problem.fit(3.0, 0.4, initial_damping=1.0)
            case["runs"].append(
                dict(
                    run=index,
                    success=result.success,
                    termination=result.termination_reason,
                    objective=float(result.objective),
                    source=np.asarray(result.source_values).tolist(),
                    potential_norm=float(np.linalg.norm(result.potential_values)),
                )
            )
            if amplitude == 1e8:
                if not result.success:
                    raise RuntimeError(
                        f"Controlled ridge limit failed: {result.termination_reason}"
                    )
                np.testing.assert_allclose(result.source_values, -8 / 7, atol=2e-4)
        record["cases"].append(case)
    text = json.dumps(record, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
