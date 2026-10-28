"""Iterative scientific assembly and public-result artifacts keep honest histories."""

from dataclasses import asdict
import json

import jax
import jax.numpy as jnp
import numpy as np


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
    from examples.potential_correction.result_products import export_result

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


def test_iterative_scientific_pair_exports_node_initialization_none_linear_and_history(
    tmp_path,
):
    from examples.potential_correction.iterative_scientific import (
        run_iterative_scientific,
    )
    from examples.potential_correction.result_products import load_result

    config = _configuration()
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
        assert (
            metadata["initialization_provenance"]["gradient_policy"]
            == "current_analytic_bilinear_branch"
        )
        assert metadata["initialization_provenance"]["reset_each_fit"] is True
        assert metadata["history"][0]["reason"] == "initial"
        assert metadata["termination_reason"].startswith("converged_")
        assert (tmp_path / key / "history.png").is_file()
        restored = load_result(tmp_path / key)
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
    assert run.record["block_partition"]["source"] == "initial_nodal"


def test_iterative_scientific_nested_search_exports_only_converged_results(tmp_path):
    from examples.potential_correction.iterative_scientific import (
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
