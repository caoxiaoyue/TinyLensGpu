"""Iterative scientific assembly and public-result artifacts keep honest histories."""

from dataclasses import asdict
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _observation_problem():
    from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection
    from examples.potential_correction.mock import (
        MockConfig,
        SourceTruth,
        generate_mock_pair,
    )

    config = MockConfig(
        npix=12,
        dpix=0.2,
        nsub=1,
        theta_E=0.7,
        b_sub=0.03,
        subhalo_x=0.65,
        subhalo_y=0.1,
        mask_inner=0.4,
        mask_outer=1.0,
        psf_size=3,
        noise_sigma=0.03,
        source=SourceTruth(sigma_x=0.16, sigma_y=0.12, clump_scale=0.7),
    )
    observation = generate_mock_pair(config).subhalo
    with jax.experimental.enable_x64():
        renderer = PotentialCorrectionRenderer(
            observation.macro,
            observation.sim_config,
            9,
            (-0.4, 0.4, -0.4, 0.4),
            potential_spacing=0.4,
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        problem = IterativePotentialCorrection(
            renderer,
            observation.image_data,
            observation.noise_map,
            observation.source_truth.brightness(sx, sy).ravel(),
            lambda_amplitude=1e6,
        )
    return observation, problem


def test_failed_iterative_export_keeps_diagnostic_history_without_success_products(
    tmp_path,
):
    from examples.potential_correction.shared.result_products import export_result

    observation, problem = _observation_problem()
    result = problem.fit(0.001, 1000.0, max_iter=0)
    assert not result.success and result.termination_reason == "iteration_limit"
    export_result(
        result, observation, tmp_path, configuration={"purpose": "failure contract"}
    )
    metadata = json.loads((tmp_path / "failure.json").read_text())
    assert metadata["success"] is False
    assert metadata["history"] == list(result.history)
    assert metadata["termination_reason"] == "iteration_limit"
    assert (tmp_path / "failure-history.png").is_file()
    assert not (tmp_path / "history.png").exists()
    assert not (tmp_path / "reconstruction.npz").exists()
    assert not (tmp_path / "images.png").exists()


def _configuration():
    from test_potential_correction_scientific import small_configuration

    config = small_configuration()
    config.update(
        backends=["dense", "operator"],
        inversion_nsub=1,
        lambda_amplitude=1e6,
        source_regularization="zero-order",
        partition_source="initial_nodal",
        solve_atol=1e-8,
        iteration_options={"gradient_atol": 1e-4, "max_iter": 80},
    )
    config["mock"]["nsub"] = 1
    return config


@pytest.mark.parametrize("mode", [None, "cubic_hermite"])
def test_iterative_scientific_pair_exports_node_initialization_none_linear_and_history(
    tmp_path, mode,
):
    from examples.potential_correction.mock.iterative.run import (
        run_iterative_scientific,
    )
    from examples.potential_correction.shared.result_products import load_result

    config = _configuration()
    if mode is not None:
        config.update(source_interpolation=mode, partition_source="single_step")
    expected_mode = mode or "bilinear"
    run = run_iterative_scientific(config, output=tmp_path)
    assert run.record["success"]
    assert set(run.results) == {
        "subhalo/dense",
        "subhalo/operator",
        "control/dense",
        "control/operator",
    }
    for key, result in run.results.items():
        metadata = json.loads((tmp_path / key / "result.json").read_text())
        assert metadata["algorithm"] == "iterative"
        assert metadata["evidence_method"] == "local_gauss_newton"
        assert metadata["evidence_is_exact"] is False
        assert metadata["evidence_image"] == "nonlinear_model_image"
        assert (
            metadata["initialization_provenance"]["source_initialization"]
            == "supplied_truth_derived_source_node_intensities"
        )
        policy = {
            "bilinear": "current_analytic_bilinear_branch",
            "cubic_hermite": "current_analytic_cubic_hermite",
        }
        assert (
            metadata["initialization_provenance"]["gradient_policy"]
            == policy[expected_mode]
        )
        assert metadata["initialization_provenance"]["reset_each_fit"] is True
        assert metadata["history"][0]["reason"] == "initial"
        assert metadata["termination_reason"].startswith("converged_")
        assert (tmp_path / key / "history.png").is_file()
        restored = load_result(tmp_path / key)
        assert restored.configuration["source_interpolation"] == expected_mode
        assert restored.configuration == result.configuration
        assert restored.linear_model_image is restored.linear_residual is None
        np.testing.assert_array_equal(restored.source_values, result.source_values)
        np.testing.assert_array_equal(
            restored.potential_values, result.potential_values
        )
        with np.load(
            tmp_path / key / "reconstruction.npz", allow_pickle=False
        ) as saved:
            assert "linear_model_image" not in saved.files
            assert "linear_residual" not in saved.files
            assert np.isnan(saved["convergence"][~saved["derivative_valid"]]).all()
    partition = run.record["block_partition"]
    assert partition["source"] == "initial_nodal"
    assert partition["source_interpolation"] == expected_mode
    if mode is not None:
        assert "source_interpolation" in partition["reuse_rejected"]["mismatched_fields"]


def test_iterative_scientific_nested_search_exports_only_converged_results(tmp_path):
    from examples.potential_correction.mock.iterative.run import (
        run_iterative_scientific,
    )

    config = _configuration()
    config.update(
        mode="tune",
        make_plots=False,
        direct_maxfun=8,
        log_bounds=np.log([[0.0005, 0.002], [100.0, 1000.0]]).tolist(),
    )
    run = run_iterative_scientific(config, output=tmp_path)
    assert run.record["success"]
    for key, result in run.results.items():
        metadata = json.loads((tmp_path / key / "result.json").read_text())
        search = metadata["search"]
        assert search["has_valid_fit"] is True
        assert search["optimizer_success"] is False
        assert search["log_bounds"] == config["log_bounds"]
        assert search["evaluations"]
        assert all(row["algorithm"] == "iterative" for row in search["evaluations"])
        assert all(
            row["evidence_method"] == "local_gauss_newton"
            for row in search["evaluations"]
        )
        assert result.strengths.lambda_amplitude == config["lambda_amplitude"]
        assert result.success
    config["iteration_options"]["max_iter"] = 0
    failed = run_iterative_scientific(config, output=tmp_path)
    assert failed.record["success"] is False
    for key in run.results:
        metadata = json.loads((tmp_path / key / "failure.json").read_text())
        assert metadata["search"]["has_valid_fit"] is False
        assert all(
            row["termination_reason"] == "iteration_limit"
            for row in metadata["search"]["evaluations"]
        )
        assert not (tmp_path / key / "reconstruction.npz").exists()
        assert not (tmp_path / key / "history.png").exists()


def test_iterative_scientific_accepts_explicit_fixed_partition_from_other_mode(tmp_path):
    from examples.potential_correction.mock.iterative.run import (
        run_iterative_scientific,
    )
    from examples.potential_correction.mock.single_step.run import run_scientific

    config = _configuration()
    config.update(make_plots=False, backends=["dense"])
    reference = run_scientific(config, output=tmp_path / "single_step")
    assert reference.record["success"]
    blocks = reference.record["block_partition"]["indices"]
    config.update(
        source_interpolation="cubic_hermite",
        cases=["control"],
        backends=["operator"],
        block_indices=blocks,
    )
    run = run_iterative_scientific(config, output=tmp_path / "iterative")
    assert run.record["success"]
    partition = run.record["block_partition"]
    assert partition["origin"] == "explicit_fixed"
    assert partition["indices"] == blocks
    result = run.results["control/operator"]
    assert result.configuration["source_interpolation"] == "cubic_hermite"
    assert result.configuration["block_indices"] == tuple(map(tuple, blocks))

    config.update(backends=["dense"], block_indices=blocks + blocks[:1])
    with pytest.raises(ValueError, match="cover every joint coefficient exactly once"):
        run_iterative_scientific(config, output=tmp_path / "invalid")


def test_algorithm_comparison_keeps_single_step_linear_evidence_and_nonlinear_objective_distinct():
    from TinyLensGpu.ObservationModel import (
        SingleStepPotentialCorrection,
        IterativePotentialCorrection,
    )
    from examples.potential_correction.shared.result_products import compare_reconstructions

    observation, prepared = _observation_problem()
    renderer = prepared.renderer
    initial = prepared.initial_values[: renderer.n_source]
    single = SingleStepPotentialCorrection(
        renderer,
        observation.image_data,
        observation.noise_map,
        observation.source_truth.gradient,
        lambda_amplitude=1e6,
        source_regularization="zero-order",
    ).fit(0.001, 1000.0)
    iterative = IterativePotentialCorrection(
        renderer,
        observation.image_data,
        observation.noise_map,
        initial,
        lambda_amplitude=1e6,
        source_regularization="zero-order",
    ).fit(0.001, 1000.0, gradient_atol=1e-4)
    assert single.success and iterative.success
    report = compare_reconstructions(single, iterative)
    assert report["same_strengths"]
    assert report["evidence_models"] == [
        "conditional_linear_gaussian",
        "local_gauss_newton",
    ]
    assert report["reference_inference_objective"] == float(single.objective)
    expected = 0.5 * np.sum(
        (
            (np.asarray(single.data_active) - np.asarray(single.nonlinear_image_active))
            / np.asarray(single.noise_active)
        )
        ** 2
    ) + 0.5 * float(single.regularization_penalty)
    np.testing.assert_allclose(
        report["reference_nonlinear_objective"], expected, atol=1e-10, rtol=0
    )
    assert (
        abs(
            report["reference_nonlinear_objective"]
            - report["reference_inference_objective"]
        )
        > 1e-8
    )
    np.testing.assert_allclose(
        report["candidate_nonlinear_objective"], iterative.objective, atol=1e-10, rtol=0
    )
    assert report["raw_fields_aligned"] is False


def test_iterative_runner_compares_saved_dense_single_step_without_refitting_it(
    tmp_path,
):
    from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
    from examples.potential_correction.mock.iterative.run import (
        run_iterative_scientific,
    )
    from examples.potential_correction.shared.result_products import export_result

    observation, prepared = _observation_problem()
    single = SingleStepPotentialCorrection(
        prepared.renderer,
        observation.image_data,
        observation.noise_map,
        observation.source_truth.gradient,
        lambda_amplitude=1e6,
        source_regularization="zero-order",
    ).fit(0.001, 1000.0)
    export_result(
        single,
        observation,
        tmp_path / "single/subhalo/dense",
        configuration={"purpose": "matched reference"},
        make_plots=False,
    )
    config = _configuration()
    config.update(cases=["subhalo"], backends=["dense"], make_plots=False)
    run = run_iterative_scientific(
        config, output=tmp_path / "iterative", single_step_reference=tmp_path / "single"
    )
    report = run.record["comparisons"]["algorithms"]["subhalo"]
    assert report["same_strengths"]
    assert report["evidence_models"] == [
        "conditional_linear_gaussian",
        "local_gauss_newton",
    ]
    assert report["reference_inference_objective"] == float(single.objective)
    assert (
        report["reference_nonlinear_objective"]
        != report["reference_inference_objective"]
    )
    assert (run.output / "comparisons.json").is_file()


def test_case_stopping_controls_remain_explicit_in_public_results(tmp_path):
    from examples.potential_correction.mock.iterative.run import (
        run_iterative_scientific,
    )

    config = _configuration()
    config.update(
        cases=["control"],
        backends=["dense"],
        make_plots=False,
        case_iteration_options={"control": {"gradient_atol": 2e-4}},
    )
    run = run_iterative_scientific(config, output=tmp_path)
    assert run.record["success"]
    assert run.results["control/dense"].configuration["gradient_atol"] == 2e-4
    assert run.record["configuration"]["iteration_options"]["gradient_atol"] == 1e-4
