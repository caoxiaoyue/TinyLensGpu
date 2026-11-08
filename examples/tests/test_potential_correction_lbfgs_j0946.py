"""The J0946 qualification workflow with regenerated test inputs."""

import pytest


def test_j0946_cli_rejects_a_changed_archive_before_writing(tmp_path, capsys):
    from examples.potential_correction.validate_j0946_lbfgs import main

    macro = tmp_path / "macro"
    macro.mkdir()
    (macro / "m3_handoff.json").write_text('{"complete": false}\n')
    archive = tmp_path / "archive"
    archive.mkdir()
    output = tmp_path / "refit"

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--macro-dir",
                str(macro),
                "--archive-dir",
                str(archive),
                "--output",
                str(output),
            ]
        )
    assert exc.value.code == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err
    assert not output.exists()


@pytest.fixture(scope="module")
def small_dense_refit():
    import numpy as np
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.ObservationModel import (
        LBFGSPotentialCorrection,
        PotentialCorrectionStrengths,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear
    from examples.potential_correction.validate_j0946_lbfgs import (
        _GN_STATIONARITY_TOL,
        _fit_record,
        _result_convergence_fields,
    )

    macro = PhysicalModel(
        lens_mass=[Shear(gamma1=0.0, gamma2=0.0)],
        source_light=[],
        lens_light=[],
    )
    renderer = PotentialCorrectionRenderer(
        macro,
        SimulatorConfig(0.1, 4, np.ones((1, 1))),
        source_n=3,
        source_bbox=(-0.3, 0.3, -0.3, 0.3),
        source_interpolation="cubic_hermite",
    )
    fitter = LBFGSPotentialCorrection(
        renderer, np.zeros((4, 4)), np.ones((4, 4)), np.zeros(9)
    )
    refit = fitter.fit(
        PotentialCorrectionStrengths(0.1, 0.2, 1.0),
        gn_stationarity_tol=_GN_STATIONARITY_TOL,
    )
    assert refit.success, refit.failure_reason
    return _fit_record(refit), _result_convergence_fields(refit)


@pytest.mark.parametrize("offset, passed", [(-8e-5, True), (-2e-4, False)])
def test_j0946_compares_nonpeak_convergence_on_the_full_valid_domain(
    small_dense_refit, offset, passed
):
    import numpy as np
    from examples.potential_correction.validate_j0946_lbfgs import (
        compare_dense_to_trial,
    )

    fit, fields = small_dense_refit
    valid = fields["derivative_valid"]
    convergence = fields["convergence"].copy()
    convergence[np.flatnonzero(valid)[-1]] += offset
    trial_fields = {**fields, "convergence": convergence}
    trial = {"objective": 0.0002, "log_evidence": float(fit["log_evidence"]) + 0.005}
    report = compare_dense_to_trial(fit, fields, trial, trial_fields)
    assert report["passed"] is passed
    assert report["absolute_differences"]["objective"] == pytest.approx(0.0002)
    assert report["absolute_differences"]["log_evidence"] == pytest.approx(0.005)
    assert report["absolute_differences"]["convergence"] == pytest.approx(abs(offset))
    assert report["convergence"]["derivative_valid_count"] == valid.sum()


@pytest.mark.parametrize("field", ["potential_node_indices", "derivative_valid"])
def test_j0946_rejects_an_unmatched_comparison_domain(small_dense_refit, field):
    from examples.potential_correction.validate_j0946_lbfgs import (
        compare_dense_to_trial,
    )

    fit, fields = small_dense_refit
    trial_fields = {**fields, field: fields[field].copy()}
    if field == "derivative_valid":
        trial_fields[field][0] = ~trial_fields[field][0]
    else:
        trial_fields[field][0, 0] += 1
    trial = {
        "objective": float(fit["objective"]),
        "log_evidence": float(fit["log_evidence"]),
    }
    with pytest.raises(ValueError, match=field):
        compare_dense_to_trial(fit, fields, trial, trial_fields)


def test_j0946_operator_requires_a_retained_dense_reference(tmp_path, capsys):
    from examples.potential_correction.validate_j0946_lbfgs import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--backend",
                "operator",
                "--macro-dir",
                str(tmp_path / "macro"),
                "--archive-dir",
                str(tmp_path / "archive"),
                "--output",
                str(tmp_path / "operator"),
            ]
        )
    assert exc.value.code == 2
    assert (
        "operator qualification requires --dense-reference" in capsys.readouterr().err
    )
    assert not (tmp_path / "operator").exists()


@pytest.mark.parametrize(
    "shift, residual, passed",
    [(1, 1e-12, True), (2, 1e-12, False), (1, 2e-8, False)],
)
def test_j0946_operator_scores_peak_distance_and_true_residual(shift, residual, passed):
    import numpy as np
    from examples.potential_correction.validate_j0946_lbfgs import (
        compare_operator_to_dense,
    )

    nodes = np.array([[-7, 11], [-7, 12], [-7, 13], [-7, 14]])
    geometry = {
        "potential_node_indices": nodes,
        "potential_x": nodes[:, 0] * 0.1,
        "potential_y": nodes[:, 1] * 0.1,
        "derivative_valid": np.array([True, True, True, False]),
    }
    dense_fields = {**geometry, "convergence": np.array([0.1, 0.098, 0.098, np.nan])}
    operator_convergence = np.array([0.098, 0.098, 0.098, np.nan])
    operator_convergence[shift] = 0.1
    operator_fields = {**geometry, "convergence": operator_convergence}
    optimizer_info = {
        "native_success": True,
        "gn_stationarity": 4e-5,
        "gn_stationarity_tol": 1e-3,
    }
    shared = {
        "algorithm": "lbfgs",
        "success": True,
        "objective": 4.0,
        "log_evidence": 12.0,
        "evidence_method": "local_gauss_newton",
        "evidence_is_exact": False,
        "optimizer_info": optimizer_info,
        "configuration": {
            "potential_spacing": 0.1,
            "solve_rtol": 1e-8,
            "solve_atol": 0.0,
        },
    }
    dense = {**shared, "backend": "dense", "determinant_method": "full_data_space"}
    operator = {
        **shared,
        "backend": "operator",
        "determinant_method": "principal_block_cholesky",
        "log_evidence": 15.0,
        "optimizer_info": {
            **optimizer_info,
            "diagnostics": {
                "stationarity_solver": {
                    "method": "pcg",
                    "converged": True,
                    "failed": False,
                    "rhs_norm": 1.0,
                    "true_residual_norm": residual,
                    "true_residual_threshold": 1e-8,
                },
            },
        },
    }

    comparison = compare_operator_to_dense(
        operator, operator_fields, dense, dense_fields
    )
    assert comparison["passed"] is passed
    assert comparison["gates"]["true_residual"] is (residual <= 1e-8)
    assert comparison["absolute_differences"][
        "peak_separation_arcsec"
    ] == pytest.approx(shift * 0.1)
    assert comparison["absolute_differences"]["convergence"] == pytest.approx(0.002)
    assert comparison["evidence"]["operator_minus_dense"] == 3.0


@pytest.mark.parametrize(
    "change, error",
    [
        (None, None),
        ("digest", "archive identity"),
        ("max_iter", "configuration differs"),
        ("evidence", "does not pass the archive comparison"),
        ("reconstruction", "reconstruction SHA-256"),
    ],
)
def test_j0946_dense_reference_is_bound_to_the_archive_and_fit(tmp_path, change, error):
    import hashlib
    import json
    import numpy as np
    from examples.potential_correction.validate_j0946_lbfgs import load_dense_reference

    archive = {
        "trial_index": 497,
        "protocol": "regenerated",
        "files": {"macro": {"path": "original", "sha256": "a" * 64}},
        "objective": 4.0,
        "log_evidence": 12.0,
    }
    fields = {
        "potential_node_indices": np.array([[0, 0], [1, 0]]),
        "potential_x": np.array([0.0, 0.1]),
        "potential_y": np.array([0.0, 0.0]),
        "derivative_valid": np.array([True, False]),
        "convergence": np.array([0.1, np.nan]),
        "source_bbox": np.array([-0.1, 0.1, -0.1, 0.1]),
        "source_x_axis": np.array([-0.1, 0.1]),
        "source_y_axis": np.array([-0.1, 0.1]),
    }
    expected_fit = {
        "strengths": {
            "lambda_source": 0.2,
            "lambda_potential": 0.3,
            "lambda_amplitude": 0.4,
        },
        "configuration": {"max_iter": 3000},
        "initialization_provenance": {"source_start_archive": "regenerated"},
    }
    stored_archive = {
        **archive,
        "files": {
            "macro": {
                "path": "relocated",
                "sha256": ("b" if change == "digest" else "a") * 64,
            },
        },
    }
    fit = {
        **expected_fit,
        "configuration": {"max_iter": 42 if change == "max_iter" else 3000},
        "algorithm": "lbfgs",
        "backend": "dense",
        "success": True,
        "objective": 4.0,
        "log_evidence": 13.0 if change == "evidence" else 12.0,
        "evidence_method": "local_gauss_newton",
        "determinant_method": "full_data_space",
        "evidence_is_exact": False,
        "optimizer_info": {
            "native_success": True,
            "gn_stationarity_tol": 1e-3,
            "gn_stationarity": 4e-7,
        },
    }
    path = tmp_path / "reconstruction.npz"
    np.savez_compressed(path, **fields)
    record = {
        "archive": stored_archive,
        "result": fit,
        "comparison": {"passed": True},
        "reconstruction": {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }
    (tmp_path / "result.json").write_text(json.dumps(record))
    if change == "reconstruction":
        path.write_bytes(path.read_bytes() + b"changed")
    if error is None:
        saved, saved_fields, provenance = load_dense_reference(
            tmp_path, archive, fields, expected_fit
        )
        assert saved["log_evidence"] == 12.0
        np.testing.assert_array_equal(
            saved_fields["derivative_valid"], fields["derivative_valid"]
        )
        assert provenance["comparison"]["passed"]
    else:
        with pytest.raises(ValueError, match=error):
            load_dense_reference(tmp_path, archive, fields, expected_fit)
