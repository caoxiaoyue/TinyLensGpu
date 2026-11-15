"""Reproduce the maintained tiny nonlinear scan/DIRECT and rejected layout.

Use tinylens_gpu with XLA_PYTHON_CLIENT_PREALLOCATE=false. This imports the
maintained independent test reference so the retained record uses the same
assertions; it does not provide another production optimizer or fit backend.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np
import scipy

from TinyLensGpu.Inference import optimize_potential_correction
from tests.potential_correction_nested_reference import (
    SEARCH_CONTROLS,
    STRENGTH_BOUNDS,
    central_jacobian,
    fit_options,
    independent_fit,
    mixed_partition,
    numerical_model,
    recovery_metrics,
)
from tests.test_potential_correction_iterative import _coupled_problem
from tests.test_potential_correction_iterative_inference import _run_nested_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / ".scratch/potential-correction/research/iterative-strength-search.json",
    )
    args = parser.parse_args(argv)
    record = _run_nested_comparison()
    problem = _coupled_problem()
    forward, _, _ = numerical_model(problem, 1.0, 1.0)
    initial_jacobian = central_jacobian(forward, np.asarray(problem.initial_values))
    response_scores = np.sum(initial_jacobian[:, 4:] ** 2, axis=0)
    rejected = ((0, 1, *range(4, 21)), (2, 3, *range(21, 37)))
    options = {**fit_options(), "block_indices": rejected}
    search = optimize_potential_correction(
        problem,
        log_bounds=np.log(STRENGTH_BOUNDS),
        backend="operator",
        fit_options=options,
        **SEARCH_CONTROLS
    )
    assert search.has_valid_fit
    fitted = search.reconstruction
    ls, lp = fitted.strengths.lambda_source, fitted.strengths.lambda_potential
    reference = independent_fit(problem, ls, lp, rejected)
    values = np.r_[fitted.source_values, fitted.potential_values]
    recovery = recovery_metrics(problem, values)
    dense_recovery = record["selected"]["dense"]["recovery"]
    ratios = {
        name: recovery[name] / dense_recovery[name]
        for name in (
            "source_rmse",
            "potential_rmse",
            "deflection_rmse",
            "convergence_rmse",
        )
    }
    assert max(ratios.values()) > 1.1
    np.testing.assert_allclose(
        fitted.log_evidence, reference["block_log_evidence"], atol=2e-6, rtol=0
    )
    record["partition_calibration"] = dict(
        selected_rule="all four tiny source nodes plus sixteen potential nodes ranked by initial independent response squared norm; remaining seventeen potential nodes in second block",
        runtime_rule="frozen explicit indices; no runtime dense Jacobian or candidate-dependent repartition",
        selected_indices=mixed_partition(),
        selected_block_sizes=(20, 17),
        initial_potential_response_squared_norms=response_scores.tolist(),
        scope="this fixed 37-unknown numerical model, initialization and bounds only; no scientific-grid transfer",
        rejected_layout=dict(
            indices=rejected,
            block_sizes=(19, 18),
            strengths=dict(
                lambda_source=ls, lambda_potential=lp, lambda_amplitude=20.0
            ),
            log_evidence=float(fitted.log_evidence),
            independent_full_log_evidence=float(reference["full_log_evidence"]),
            independent_block_log_evidence=float(reference["block_log_evidence"]),
            full_evidence_regret=float(
                record["selected"]["dense"]["independent_edge_log_evidence"]
                - reference["full_log_evidence"]
            ),
            dense_selected_minus_rejected_full_evidence=float(
                record["selected"]["dense"]["full_log_evidence"]
                - reference["full_log_evidence"]
            ),
            recovery=recovery,
            recovery_ratios_to_dense_selection=ratios,
            optimizer_success=search.optimizer_success,
            optimizer_status=search.optimizer_status,
            nfev=search.nfev,
            nit=search.nit,
            message=search.message,
            boundary_sides=search.boundary_sides,
            accepted_for_validation=False,
            rejection="potential and deflection recovery errors exceed the unchanged 10% increase guard",
        ),
    )
    files = [
        ROOT / "TinyLensGpu/Inference/potential_correction.py",
        ROOT / "tests/test_potential_correction_iterative_inference.py",
        ROOT / "tests/potential_correction_nested_reference.py",
        ROOT / "examples/potential_correction/mock/iterative/tune.py",
        Path(__file__),
    ]
    record["software"] = dict(
        python=platform.python_version(),
        jax=jax.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
        devices=[str(device) for device in jax.devices()],
        git_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        git_dirty=bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        ),
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    )
    record["model"] = dict(
        image_shape=(4, 4),
        dpix=0.1,
        nsub=1,
        psf_shape=(1, 1),
        n_data=16,
        n_source=4,
        n_potential=33,
        n_joint=37,
        source_n=2,
        source_bbox=(-0.5, 0.5, -0.5, 0.5),
        source_truth=(0.4, 1.1, 0.2, 1.4),
        source_regularization="zero-order",
        potential_truth="0.003*exp(-((x-.07)^2+(y+.03)^2)/.02), raw nodal values; no alignment",
        data="nonlinear nodal source/potential render + .002*sin(pixel_index)",
        noise_sigma=0.07,
        macro="fixed zero shear",
        dtype="float64",
        initialization="same supplied source truth nodes and zero potential every fit",
        evidence="local Gauss-Newton, undamped; full versus bounded principal determinants independently labeled",
        locality="all reference solutions and finite-difference probes remain in the single interior source cell; halved-step derivative check enforced",
        scope="controlled numerical nodal validation, not scientific subhalo recovery or detection",
    )
    record["guards"] = dict(
        independent_scaled_gradient=1e-8,
        independent_derivative_step_error=1e-8,
        field_atol=4e-7,
        field_rtol=3e-6,
        objective_atol=2e-12,
        same_state_evidence_atol=3e-8,
        independent_map_evidence_atol=2e-6,
        same_strength_backend_atol=2e-8,
        same_strength_backend_rtol=2e-7,
        budget_edge_log_evidence_gap=0.005,
        selected_full_evidence_regret=0.005,
        maximum_recovery_error_ratio_to_dense=1.1,
        selected_image_difference_in_noise_units=0.1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": record["selected"],
                "full_evidence_regret": record[
                    "full_evidence_regret_of_operator_selection"
                ],
            },
            indent=2,
        )
    )
    return record


if __name__ == "__main__":
    main()
