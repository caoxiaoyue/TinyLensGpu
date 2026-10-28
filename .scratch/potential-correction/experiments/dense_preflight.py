"""Exploratory dense single-step fits, before backend or scientific calibration.

Run from the repository root with PYTHONPATH=. in tinylens_gpu. The compact
record is written to /tmp/tinylens-pc-dense-preflight.json. This preliminary
grid does not establish search bounds, an optimum, or acceptable recovery.
"""

from dataclasses import asdict
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from examples.potential_correction.mock import MockConfig, generate_mock_pair
from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection

config = MockConfig(npix=48, dpix=0.05, nsub=2)
observation = generate_mock_pair(config).subhalo
renderer = PotentialCorrectionRenderer(observation.macro, observation.sim_config, 41,
    (-0.5, 0.5, -0.5, 0.5), dtype='float64')
problem = SingleStepPotentialCorrection(renderer, observation.image_data, observation.noise_map,
    observation.source_truth.gradient, lambda_amplitude=1e3)
rows = []
with jax.experimental.enable_x64():
    sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
    source_truth = observation.source_truth.brightness(sx, sy).ravel()
    for ls in (1e-6, 1e-4, 1e-2):
        for lp in (1e4, 1e6, 1e8):
            result = problem.fit(ls, lp)
            row = dict(lambda_source=ls, lambda_potential=lp, success=result.success,
                failure_reason=result.failure_reason, relative_residual=result.solver_info.relative_residual)
            if result.success:
                row.update(log_evidence=float(result.log_evidence), linear_residual_rms=float(jnp.sqrt(jnp.mean(result.linear_normalized_residual[jnp.isfinite(result.linear_normalized_residual)]**2))),
                    nonlinear_residual_rms=float(jnp.sqrt(jnp.mean(result.nonlinear_normalized_residual[jnp.isfinite(result.nonlinear_normalized_residual)]**2))),
                    source_relative_l2=float(jnp.linalg.norm(result.source_values-source_truth)/jnp.linalg.norm(source_truth)),
                    amplitude_prior_penalty=float(1e3*jnp.vdot(result.potential_values,result.potential_values)))
            print(json.dumps(row), flush=True)
            rows.append(row)
report = dict(configuration=asdict(config), source_n=41, source_bbox=[-0.5,0.5,-0.5,0.5],
    potential_spacing=renderer.potential_grid.spacing, n_data=renderer.n_active,
    n_core=renderer.potential_grid.n_core, n_potential=renderer.potential_grid.n_unknown,
    lambda_amplitude=1e3, rows=rows,
    scope='Exploratory fixed-strength dense single-step preflight, not backend calibration, bound selection, or acceptance of scientific recovery.')
Path('/tmp/tinylens-pc-dense-preflight.json').write_text(json.dumps(report, indent=2)+'\n')
