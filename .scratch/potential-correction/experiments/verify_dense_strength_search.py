"""Retain the small DIRECT-versus-independent-covariance scan comparison.

The host Gaussian construction lives with its maintained core regression;
this experiment reuses that construction, not a private library solve.
Run from the repository root in tinylens_gpu with GPU preallocation disabled.
"""

import json
from pathlib import Path
import platform
import runpy
import sys

import jax
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from TinyLensGpu.Inference import optimize_potential_correction


def main():
    regression = runpy.run_path(
        str(ROOT / "tests/test_potential_correction_inference.py")
    )
    problem, _ = regression["_problem"]()
    reference = regression["_independent_negative_log_evidence"](problem)
    bounds = np.log([[0.05, 30.0], [0.001, 30.0]])
    coordinates = np.array(
        [
            (x, y)
            for x in np.linspace(*bounds[0], 31)
            for y in np.linspace(*bounds[1], 31)
        ]
    )
    scan = np.array([reference(point) for point in coordinates])
    scan_index = int(np.argmin(scan))
    result = optimize_potential_correction(
        problem,
        log_bounds=bounds,
        maxfun=180,
        maxiter=100,
        len_tol=1e-3,
        vol_tol=1e-12,
    )
    assert result.has_valid_fit
    error = result.objective - reference(result.log_strengths)
    assert abs(error) <= 2e-10
    assert result.objective <= scan.min() + 1e-3
    summary = {
        "model": "3x3 zero-shear, 2x2 zero-order source, constant gradient (1.2,-0.8)",
        "python": platform.python_version(),
        "jax": jax.__version__,
        "scipy": scipy.__version__,
        "devices": [device.device_kind for device in jax.devices()],
        "configuration": result.configuration,
        "log_bounds": result.log_bounds,
        "grid_scan_size": len(scan),
        "grid_scan_log_strengths": coordinates[scan_index].tolist(),
        "grid_scan_objective": float(scan[scan_index]),
        "direct_log_strengths": result.log_strengths,
        "direct_strengths": {
            "lambda_source": result.strengths.lambda_source,
            "lambda_potential": result.strengths.lambda_potential,
            "lambda_amplitude": result.strengths.lambda_amplitude,
        },
        "direct_objective": result.objective,
        "independent_objective_at_direct": reference(result.log_strengths),
        "objective_error": error,
        "direct_minus_grid": result.objective - float(scan.min()),
        "nfev": result.nfev,
        "optimizer_success": result.optimizer_success,
        "optimizer_status": result.optimizer_status,
        "optimizer_message": result.message,
        "boundary_sides": result.boundary_sides,
        "boundary_distances": result.boundary_distances,
        "evidence_method": result.reconstruction.evidence_method,
        "determinant_method": result.reconstruction.determinant_method,
        "evidence_is_exact": result.reconstruction.evidence_is_exact,
        "scope": "numerical optimizer check; amplitude prior and bounds uncalibrated",
    }
    output = (
        ROOT
        / ".scratch/potential-correction/research/dense-strength-optimization-summary.json"
    )
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
