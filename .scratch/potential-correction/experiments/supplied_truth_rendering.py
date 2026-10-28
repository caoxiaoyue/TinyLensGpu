"""Measure supplied-truth discretization error without reconstructing fields.

Run from the repository root with PYTHONPATH=. in the tinylens_gpu environment.
The default output is a compact reproducibility record, not a fitted result.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.potential_correction.mock import MockConfig, generate_mock_pair
from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        ".scratch/potential-correction/research/supplied-truth-rendering.json"
    ))
    args = parser.parse_args()
    config = MockConfig()
    pair = generate_mock_pair(config)
    bbox = (-0.6, 0.6, -0.6, 0.6)
    rows = []
    with jax.experimental.enable_x64():
        for source_n in (41, 61):
            for name in ("subhalo", "control"):
                observation = getattr(pair, name)
                renderer = PotentialCorrectionRenderer(
                    observation.macro, observation.sim_config, source_n, bbox,
                    dtype="float64",
                )
                sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
                source = observation.source_truth.brightness(sx, sy).ravel()
                grid = renderer.potential_grid
                potential = observation.subhalo_truth.potential(grid.x, grid.y)
                rendered = renderer.render(source, potential)
                reference = observation.model_image.ravel()[renderer.flat_indices]
                discrepancy = rendered.image_active - reference
                rms = float(jnp.sqrt(jnp.mean(discrepancy**2)))
                rows.append({
                    "source_n": source_n, "case": name,
                    "n_data": renderer.n_active, "n_core": grid.n_core,
                    "n_potential": grid.n_unknown,
                    "rms_intensity": rms,
                    "rms_in_noise_units": rms / config.noise_sigma,
                })
    report = {
        "mock_configuration": asdict(config), "source_bbox": bbox,
        "potential_spacing": 2 * config.dpix, "rows": rows,
        "software": {"jax": jax.__version__, "numpy": np.__version__},
        "device": str(jax.devices()[0]),
        "interpretation": (
            "Supplied analytic truth sampled onto the fixed source/potential grids. "
            "No reconstruction, smoothing, alignment, or evidence evaluation. "
            "Differences from the analytic noiseless mock include source bilinear "
            "interpolation and potential finite-difference/interpolation error. "
            "The matched control helps separate source discretization from the "
            "additional potential discretization. These are not recovery errors."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
