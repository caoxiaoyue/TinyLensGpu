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
    refit = fitter.fit(PotentialCorrectionStrengths(0.1, 0.2, 1.0))
    assert refit.success, refit.failure_reason
    return refit


@pytest.mark.parametrize("offset, passed", [(-8e-5, True), (-2e-4, False)])
def test_j0946_compares_nonpeak_convergence_on_the_full_valid_domain(
    small_dense_refit, offset, passed
):
    import numpy as np
    from examples.potential_correction.validate_j0946_lbfgs import (
        compare_dense_to_trial,
    )

    fit = small_dense_refit
    grid = fit.potential_grid
    valid = np.asarray(fit.derivative_valid)
    convergence = np.array(fit.convergence)
    convergence[np.flatnonzero(valid)[-1]] += offset
    trial_fields = {
        "potential_node_indices": np.asarray(grid.node_indices),
        "potential_x": np.asarray(grid.x),
        "potential_y": np.asarray(grid.y),
        "derivative_valid": valid,
        "convergence": convergence,
    }
    trial = {"objective": 0.0002, "log_evidence": float(fit.log_evidence) + 0.005}
    report = compare_dense_to_trial(fit, trial, trial_fields)
    assert report["passed"] is passed
    assert report["absolute_differences"]["objective"] == pytest.approx(0.0002)
    assert report["absolute_differences"]["log_evidence"] == pytest.approx(0.005)
    assert report["absolute_differences"]["convergence"] == pytest.approx(abs(offset))
    assert report["convergence"]["derivative_valid_count"] == valid.sum()


@pytest.mark.parametrize("field", ["potential_node_indices", "derivative_valid"])
def test_j0946_rejects_an_unmatched_comparison_domain(small_dense_refit, field):
    import numpy as np
    from examples.potential_correction.validate_j0946_lbfgs import (
        compare_dense_to_trial,
    )

    fit = small_dense_refit
    grid = fit.potential_grid
    trial_fields = {
        "potential_node_indices": np.array(grid.node_indices),
        "potential_x": np.asarray(grid.x),
        "potential_y": np.asarray(grid.y),
        "derivative_valid": np.array(fit.derivative_valid),
        "convergence": np.asarray(fit.convergence),
    }
    if field == "derivative_valid":
        trial_fields[field][0] = ~trial_fields[field][0]
    else:
        trial_fields[field][0, 0] += 1
    trial = {"objective": float(fit.objective), "log_evidence": float(fit.log_evidence)}
    with pytest.raises(ValueError, match=field):
        compare_dense_to_trial(fit, trial, trial_fields)
