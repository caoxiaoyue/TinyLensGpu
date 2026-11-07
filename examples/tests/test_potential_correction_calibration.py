"""The calibration workflow preserves public fits and honest inference records."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CONSUMER_FIXTURE = Path(__file__).parent / "fixtures/potential_correction_consumers"


def test_source_incidence_votes_with_signed_source_weights():
    from examples.potential_correction.calibrate_single_step import (
        source_incidence_blocks,
    )

    grid = SimpleNamespace(
        n_unknown=5,
        query_x=np.zeros(1),
        query_y=np.zeros(1),
        derivative_indices=np.array([[0, 1, 2, 3, 4]]),
        interpolation_indices=np.zeros((1, 4), dtype=int),
        interpolation_weights=np.array([[1.0, 0.0, 0.0, 0.0]]),
        spacing=1.0,
        node_indices=np.array([[1, 1], [0, 1], [2, 1], [1, 0], [1, 2]]),
    )
    renderer = SimpleNamespace(source_n=4, n_source=16, potential_grid=grid)
    mapping = SimpleNamespace(
        indices=np.array([[0, 1, 2] + [0] * 13]),
        weights=np.array([[0.3, -0.3, 0.4] + [0.0] * 13]),
        valid=np.array([True]),
    )
    blocks = source_incidence_blocks(
        renderer, mapping, [1.0], [0.0], source_tile=2, block_size=5
    )
    # The first potential block attaches to tile 0, whose unsigned vote is
    # 0.60. A signed sum would instead attach it to tile 1 (vote 0.40).
    assert 0 in blocks[0] and 17 in blocks[0]
    np.testing.assert_array_equal(np.sort(np.concatenate(blocks)), np.arange(21))
    with pytest.raises(ValueError, match="query positions"):
        source_incidence_blocks(
            renderer, mapping, [1.0, 1.0], [0.0], source_tile=2, block_size=5
        )
    extra_ray = SimpleNamespace(
        indices=np.repeat(mapping.indices, 2, axis=0),
        weights=np.repeat(mapping.weights, 2, axis=0),
        valid=np.ones(1, dtype=bool),
    )
    with pytest.raises(ValueError, match="query positions"):
        source_incidence_blocks(
            renderer, extra_ray, [1.0], [0.0], source_tile=2, block_size=5
        )


@pytest.fixture
def tiny_partition_renderer():
    axis = np.array([0.0, 1.0])
    return SimpleNamespace(
        n_source=4,
        source_x_axis=axis,
        source_y_axis=axis,
        source_interpolation="cubic_hermite",
        source_boundary="C1_cardinal_bicubic_fixed_zero_exterior_nodes",
        macro_mapping=(np.array([0.3]), np.array([0.4])),
        potential_grid=SimpleNamespace(
            n_unknown=3,
            node_indices=np.array([[0, 0], [0, 1], [1, 0]]),
            core_indices=np.array([0]),
            x=np.array([0.0, 1.0, 0.0]),
            y=np.array([0.0, 0.0, 1.0]),
            query_x=np.array([0.3]),
            query_y=np.array([0.4]),
        ),
    )


@pytest.mark.parametrize(
    "mismatch", [None, "geometry_sha256", "source_interpolation", "vote_rule"]
)
def test_saved_partition_reuse_checks_provenance(
    tmp_path, tiny_partition_renderer, mismatch
):
    from examples.potential_correction._scientific_partitions import prepare_partition

    renderer = tiny_partition_renderer
    fresh = [[0, 1], [2, 3], [4, 5], [6]]
    controls = dict(
        source="initial_nodal",
        method="source_incidence",
        incidence_gradients=(np.array([1.0]), np.array([0.0])),
    )
    config = {"source_tile": 1, "block_size": 2}
    _, saved = prepare_partition(
        renderer, config, lambda: [[0, 4], [1, 5], [2, 6], [3]], **controls,
    )
    if mismatch is not None:
        saved[mismatch] = "unmatched"
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"block_partition": saved}))
    blocks, provenance = prepare_partition(
        renderer,
        {**config, "partition_file": str(path)},
        lambda: fresh,
        **controls,
    )
    if mismatch is None:
        assert blocks == saved["indices"]
        assert provenance["origin"] == "automatic_reuse"
    else:
        assert blocks == fresh
        assert provenance["origin"] == "generated"
        assert provenance["reuse_rejected"]["mismatched_fields"] == [mismatch]


@pytest.mark.parametrize("changed", ["macro_rays", "incidence_gradient"])
def test_source_incidence_partition_rechecks_inputs_on_the_same_grids(tmp_path, changed):
    import jax
    from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
    from TinyLensGpu.PhysicalModel import PhysicalModel, SIS
    from examples.potential_correction._scientific_partitions import (
        geometry_fingerprint,
        prepare_partition,
    )
    from examples.potential_correction.calibrate_single_step import source_incidence_blocks

    def renderer(theta_e):
        macro = PhysicalModel(
            lens_mass=[SIS(theta_E=theta_e, center_x=0.0, center_y=0.0)],
            source_light=[], lens_light=[]
        )
        return PotentialCorrectionRenderer(
            macro, sim, 4, (-0.4, 0.4, -0.4, 0.4),
            source_interpolation="cubic_hermite",
        )

    def prepare(model, gx, gy, config):
        return prepare_partition(
            model, config,
            lambda: source_incidence_blocks(
                model, model.source_mapping(), gx, gy, source_tile=2, block_size=8
            ),
            source="initial_nodal", method="source_incidence",
            incidence_gradients=(gx, gy),
        )

    with jax.experimental.enable_x64():
        sim = SimulatorConfig(0.1, 6, np.ones((1, 1)))
        base = renderer(0.10)
        ones, zeros = np.ones(36), np.zeros(36)
        config = {"source_tile": 2, "block_size": 8}
        original, saved = prepare(base, ones, zeros, config)
        path = tmp_path / "partition.json"
        path.write_text(json.dumps(saved))

        current = renderer(0.35) if changed == "macro_rays" else base
        gx, gy = (ones, zeros) if changed == "macro_rays" else (zeros, ones)
        assert geometry_fingerprint(base) == geometry_fingerprint(current)
        rebuilt, _ = prepare(current, gx, gy, config)
        assert rebuilt != original
        chosen, provenance = prepare(
            current, gx, gy, {**config, "partition_file": path}
        )
    assert chosen == rebuilt
    assert provenance["origin"] == "generated"
    assert provenance["reuse_rejected"]["mismatched_fields"] == [f"{changed}_sha256"]


@pytest.mark.integration
def test_bundled_legacy_partition_reuses_only_matching_bilinear_geometry():
    import jax
    import jax.numpy as jnp

    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from examples.potential_correction import iterative_scientific
    from examples.potential_correction._scientific_partitions import (
        geometry_fingerprint,
        prepare_partition,
    )
    from examples.potential_correction.calibrate_single_step import source_incidence_blocks
    from examples.potential_correction.mock import MockConfig, generate_mock_pair

    directory = Path(iterative_scientific.__file__).parent
    config_path = directory / "iterative_scientific_configuration.json"
    config = json.loads(config_path.read_text())
    archive = directory / "scientific_partition.json"
    recorded = json.loads(archive.read_text())
    mock = MockConfig.from_dict(config["mock"])
    observation = generate_mock_pair(mock).control
    with jax.experimental.enable_x64():
        sim = SimulatorConfig(
            mock.dpix, mock.npix, observation.psf_kernel,
            config["inversion_nsub"], observation.mask,
        )
        args = (
            observation.macro, sim, config["source_n"], tuple(config["source_bbox"])
        )
        renderer = PotentialCorrectionRenderer(
            *args, potential_spacing=config["potential_spacing"]
        )
        hermite = PotentialCorrectionRenderer(
            *args,
            potential_grid=renderer.potential_grid,
            source_interpolation="cubic_hermite",
        )
    regenerated = [
        [index] for index in range(renderer.n_source + renderer.potential_grid.n_unknown)
    ]
    # Reproduce the paired iterative initialization from its bundled config.
    with jax.experimental.enable_x64():
        mapping = renderer.source_mapping()
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        initial = observation.source_truth.brightness(sx, sy).ravel()
        gradients = renderer.source_spatial_gradient(initial, mapping)
    selected, provenance = prepare_partition(
        renderer,
        config,
        lambda: regenerated,
        source="initial_nodal",
        method="source_incidence",
        default_file=archive,
        incidence_gradients=gradients,
    )
    # CUDA matches the archived hash. CPU can differ in floating-point
    # coordinates even though the physical configuration is unchanged.
    if geometry_fingerprint(renderer) == recorded["geometry_sha256"]:
        assert selected == recorded["indices"]
        assert provenance["legacy_bilinear"] is True
        assert provenance["origin"] == "automatic_reuse"
        with jax.experimental.enable_x64():
            mapping = renderer.source_mapping()
            frozen = observation.source_truth.gradient(mapping.beta_x, mapping.beta_y)
            rebuilt, reference = prepare_partition(
                renderer, config,
                lambda: source_incidence_blocks(
                    renderer, mapping, *frozen, source_tile=config["source_tile"],
                    block_size=config["block_size"],
                ),
                source="frozen_macro", method="source_incidence",
                incidence_gradients=frozen,
            )
        assert rebuilt == selected
        for key in ("macro_rays_sha256", "incidence_gradient_sha256"):
            assert provenance[key] == reference[key]
        chosen, rejected = prepare_partition(
            renderer, config, lambda: regenerated,
            source="initial_nodal", method="source_incidence", default_file=archive,
            incidence_gradients=(np.ones(renderer.query_x.shape), np.zeros(renderer.query_y.shape)),
        )
        assert chosen == regenerated
        assert rejected["reuse_rejected"]["mismatched_fields"] == [
            "incidence_gradient_sha256"
        ]
    else:
        assert selected == regenerated
        assert "geometry_sha256" in provenance["reuse_rejected"]["mismatched_fields"]
    selected, provenance = prepare_partition(
        hermite,
        config,
        lambda: regenerated,
        source="initial_nodal",
        method="source_incidence",
        default_file=archive,
        incidence_gradients=gradients,
    )
    assert selected == regenerated
    assert "source_interpolation" in provenance["reuse_rejected"]["mismatched_fields"]


def _assert_metrics(actual, expected):
    """Compare public diagnostic outputs with pre-refactor saved values."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            _assert_metrics(actual[name], value)
    elif isinstance(expected, (bool, str)) or expected is None:
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("layout", "mode"),
    [("morton", None), ("source_incidence", None), ("source_incidence", "cubic_hermite")],
)
def test_small_calibration_exports_raw_fields_domains_and_candidate_status(
    tmp_path, layout, mode
):
    from examples.potential_correction.calibrate_single_step import run_calibration
    from examples.potential_correction.mock import (
        MockConfig,
        SourceTruth,
        generate_mock_pair,
    )
    from examples.potential_correction.result_products import summarize_result

    configuration = {
        "mock": asdict(
            MockConfig(
                npix=12,
                dpix=0.2,
                nsub=1,
                theta_E=0.7,
                subhalo_x=0.65,
                subhalo_y=0.1,
                b_sub=0.4,
                mask_inner=0.4,
                mask_outer=1.0,
                psf_size=3,
                noise_sigma=0.03,
                source=SourceTruth(clump_scale=0.8),
            )
        ),
        "source_n": 9,
        "source_bbox": [-0.4, 0.4, -0.4, 0.4],
        "potential_spacing": 0.4,
        "lambda_amplitudes": [100.0, 1000.0],
        "source_strengths": [0.001, 0.01],
        "potential_strengths": [10.0, 1000.0],
        "log_bounds": np.log([[0.001, 0.01], [10.0, 1000.0]]).tolist(),
        "block_size": 16,
        "block_layout": layout,
        "source_tile": 4,
        "pcg_max_iter": 2000,
        "pcg_rtol": 1e-11,
        "solve_rtol": 1e-8,
        "direct_maxfun": 8,
        "direct_len_tol": 1e-2,
        "boundary_rtol": 0.1,
    }
    if mode is not None:
        configuration.update(source_interpolation=mode, lambda_amplitudes=[1000.0])
    expected_mode = mode or "bilinear"
    baseline = json.loads((CONSUMER_FIXTURE / "baseline.json").read_text())
    observations = generate_mock_pair(MockConfig.from_dict(configuration["mock"]))
    run = run_calibration(configuration, output=tmp_path)
    record = json.loads((tmp_path / "calibration.json").read_text())
    assert record["configuration"] == configuration
    assert record["block_partition"]["method"] == layout
    assert record["block_partition"]["source_interpolation"] == expected_mode
    assert len(record["runs"]) == 2 * len(configuration["lambda_amplitudes"])
    with np.load(tmp_path / "inputs.npz", allow_pickle=False) as inputs:
        assert np.count_nonzero(inputs["control_potential_truth"]) == 0
        np.testing.assert_array_equal(inputs["subhalo_noise"], inputs["control_noise"])
    with np.load(tmp_path / "reconstructions.npz", allow_pickle=False) as saved:
        for key, fits in run.reconstructions.items():
            row = next(row for row in record["runs"] if row["key"] == key)
            if mode is None:
                expected = baseline["calibration"][key]
                assert sorted(row["selected"]["dense"]) == expected["schema"]
                _assert_metrics(
                    {
                        name: row["selected"]["dense"][name]
                        for name in expected["metrics"]
                    },
                    expected["metrics"],
                )
            assert row["scan"]
            assert row["order_check"]["passed"]
            assert row["failure_check"]["objective"] is None
            assert row["failure_check"]["success"] is False
            assert row["failure_check"]["failure_reason"] == "pcg_iteration_limit"
            for backend in ("dense", "operator"):
                fit = fits[backend]
                assert fit.success
                assert fit.configuration["source_interpolation"] == expected_mode
                assert row["selected"][backend]["optimizer_success"] is False
                assert row["selected"][backend]["boundary_sides"]
                assert row["selected"][backend]["determinant_method"] == (
                    "full_cholesky"
                    if backend == "dense"
                    else "principal_block_cholesky"
                )
                assert fit.strengths.lambda_amplitude == row["lambda_amplitude"]
                assert (
                    row["selected"][backend]["initialization_provenance"]
                    == fit.initialization_provenance
                )
                for field in (
                    "source_values",
                    "potential_values",
                    "convergence",
                    "linear_model_image",
                    "nonlinear_model_image",
                    "derivative_valid",
                ):
                    np.testing.assert_array_equal(
                        saved[f"{key}_{backend}_{field}"], getattr(fit, field)
                    )
                assert np.isnan(
                    saved[f"{key}_{backend}_convergence"][~fit.derivative_valid]
                ).all()
                case = key.split("_", 1)[0]
                summary, _ = summarize_result(
                    fit, getattr(observations, case), affine_diagnostics=True
                )
                calibration = row["selected"][backend]
                for name in (
                    "source_relative_l2",
                    "source_rmse",
                    "potential_rmse_raw_center_zero_truth",
                    "potential_rmse_mean_centered_truth",
                    "potential_rms",
                    "potential_mean",
                    "deflection_vector_rmse_discrete_truth",
                ):
                    assert calibration[name] == pytest.approx(summary[name], abs=1e-14)
                for domain, metrics in calibration["convergence"].items():
                    _assert_metrics(
                        {
                            name: summary["convergence"][domain][name]
                            for name in metrics
                        },
                        metrics,
                    )


@pytest.mark.parametrize("case", ["subhalo", "control"])
def test_saved_public_results_keep_pre_review_truth_domains_and_recovery(case):
    from examples.potential_correction.mock import MockConfig, generate_mock_pair
    from examples.potential_correction.result_products import (
        load_result,
        summarize_result,
        truth_products,
    )

    baseline = json.loads((CONSUMER_FIXTURE / "baseline.json").read_text())
    observation = getattr(
        generate_mock_pair(MockConfig.from_dict(baseline["configuration"]["mock"])),
        case,
    )
    result = load_result(CONSUMER_FIXTURE / case)
    assert result.configuration["source_interpolation"] == "bilinear"
    assert result.configuration["source_boundary"] == "hard_zero_outside_bbox"
    summary, truth = summarize_result(result, observation, affine_diagnostics=True)
    _assert_metrics(summary, baseline["scientific"][case])
    products, local_radius = truth_products(result, observation)
    assert local_radius == pytest.approx(1.058300524425836)
    assert local_radius > 2 * result.potential_grid.spacing
    assert {name: row["count"] for name, row in summary["convergence"].items()} == {
        "all_core": 37,
        "illuminated_core": 15,
        "local_core": 18,
    }
    if case == "control":
        assert observation.subhalo_truth.b_sub == 0.0
        assert observation.config.b_sub == 0.4
    with np.load(
        CONSUMER_FIXTURE / case / "reconstruction.npz", allow_pickle=False
    ) as saved:
        for name, values in truth.items():
            np.testing.assert_allclose(values, saved[name], rtol=2e-12, atol=2e-14)
        for name, values in products.items():
            np.testing.assert_array_equal(values, truth[name])
        valid = np.asarray(result.derivative_valid)
        for name in (
            "truth_discrete_deflection_x",
            "truth_discrete_deflection_y",
            "truth_discrete_convergence",
            "truth_analytic_deflection_x",
            "truth_analytic_deflection_y",
            "truth_analytic_convergence",
        ):
            assert np.isnan(products[name][~valid]).all()
        # Consumer diagnostics preserve the signed fit and its image/evidence.
        assert np.any(result.source_values < 0)
        assert np.any(result.potential_values < 0)
        for name in (
            "source_values",
            "potential_values",
            "linear_model_image",
            "nonlinear_model_image",
        ):
            np.testing.assert_array_equal(getattr(result, name), saved[name])
    metadata = json.loads((CONSUMER_FIXTURE / case / "result.json").read_text())
    assert float(result.log_evidence) == metadata["log_evidence"]
