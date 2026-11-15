"""Relocated potential-correction layout: CLIs, defaults, and saved datasets."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "examples" / "potential_correction"

CLI_SCRIPTS = (
    "mock/generate_mock.py",
    "mock/single_step/run.py",
    "mock/single_step/calibrate.py",
    "mock/single_step/tune.py",
    "mock/single_step/demos/operator.py",
    "mock/iterative/run.py",
    "mock/iterative/tune.py",
    "mock/iterative/demos/fixed_strength.py",
    "mock/lbfgs/run.py",
    "j0946/run_pipeline.py",
)


@pytest.mark.parametrize("script", CLI_SCRIPTS)
def test_relocated_cli_help_runs_directly_from_any_cwd(tmp_path, script):
    result = subprocess.run(
        [sys.executable, str(PACKAGE / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_module_invocation_resolves_package(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "examples.potential_correction.mock.generate_mock",
         "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr


def test_default_config_files_are_inversion_only():
    for name in (
        "mock/single_step/configuration.json",
        "mock/iterative/configuration.json",
        "mock/lbfgs/configuration.json",
    ):
        assert "mock" not in json.loads((PACKAGE / name).read_text())
    assert json.loads((PACKAGE / "mock/configuration.json").read_text())["npix"] == 200


class _Sentinel(Exception):
    pass


@pytest.mark.parametrize(
    ("module_name", "function"),
    (
        (
            "examples.potential_correction.mock.single_step.run",
            "run_scientific",
        ),
        (
            "examples.potential_correction.mock.iterative.run",
            "run_iterative_scientific",
        ),
        (
            "examples.potential_correction.mock.lbfgs.run",
            "run_lbfgs_mock",
        ),
    ),
)
def test_runner_cli_defaults_use_sibling_config_dataset_and_output(
    tmp_path, monkeypatch, module_name, function
):
    import importlib

    module = importlib.import_module(module_name)
    captured = {}

    def spy(configuration, **kwargs):
        captured.update(configuration=configuration, **kwargs)
        raise _Sentinel

    monkeypatch.setattr(module, function, spy)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(_Sentinel):
        module.main([])
    assert "mock" not in captured["configuration"]
    algorithm = Path(module.__file__).resolve().parent
    assert captured["output"] == algorithm / "output" / "default"
    assert captured["data"] is None


def test_generator_cli_defaults_use_sibling_config_and_default_data(
    tmp_path, monkeypatch
):
    from examples.potential_correction.mock import generate_mock as generator

    captured = {}

    def fake_generate(config):
        captured["config"] = config
        return "pair"

    def fake_save(pair, output):
        captured["pair"] = pair
        captured["output"] = Path(output)
        return captured["output"]

    monkeypatch.setattr(generator, "generate_mock_pair", fake_generate)
    monkeypatch.setattr(generator, "save_mock_pair", fake_save)
    monkeypatch.chdir(tmp_path)
    generator.main([])
    from examples.potential_correction.mock import MockConfig

    module = Path(generator.__file__).resolve().parent
    assert captured["output"] == module / "data" / "default"
    expected = json.loads((module / "configuration.json").read_text())
    assert captured["config"] == MockConfig.from_dict(expected)


@pytest.fixture(scope="module")
def saved_dataset(tmp_path_factory):
    from examples.potential_correction.mock import (
        MockConfig,
        SourceTruth,
        generate_mock_pair,
        save_mock_pair,
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
    data = save_mock_pair(
        generate_mock_pair(config), tmp_path_factory.mktemp("data") / "dataset"
    )
    return config, data


def _dataset_inversion_configuration():
    from test_potential_correction_scientific import small_configuration

    config = small_configuration()
    config["mock"]["nsub"] = 1
    mock = config.pop("mock")
    return config, mock


DATASET_INPUTS = ("image_data", "noise_map", "noise_realization", "psf_kernel", "mask")


def _assert_inputs_match_dataset(output, key, data):
    case = key.split("/")[0]
    with np.load(
        output / key / "reconstruction.npz", allow_pickle=False
    ) as saved, np.load(
        data / f"{case}.npz", allow_pickle=False
    ) as stored:
        for name in DATASET_INPUTS:
            assert name in saved.files
            np.testing.assert_array_equal(saved[name], stored[name])
        if "model_image" in saved.files:
            np.testing.assert_array_equal(
                saved["model_image"], stored["model_image"]
            )
        elif "mock_model_image" in saved.files:
            np.testing.assert_array_equal(
                saved["mock_model_image"], stored["model_image"]
            )


def _assert_dataset_provenance(configuration, data):
    dataset = configuration["dataset"]
    assert Path(dataset["path"]) == data.resolve()
    for name in ("configuration.json", "subhalo.npz", "control.npz"):
        digest = hashlib.sha256((data / name).read_bytes()).hexdigest()
        assert dataset["sha256"][name] == digest


def _forbid_generation(monkeypatch):
    from examples.potential_correction.mock import dataset

    def forbidden(*args, **kwargs):
        raise AssertionError("dataset mode must not regenerate inputs")

    monkeypatch.setattr(dataset, "generate_mock_pair", forbidden)


@pytest.mark.integration
def test_single_step_consumes_saved_dataset_with_provenance(
    tmp_path, saved_dataset, monkeypatch
):
    from examples.potential_correction.mock.single_step.run import run_scientific
    from examples.potential_correction.shared.result_products import load_result

    config, mock = _dataset_inversion_configuration()
    _, data = saved_dataset
    _forbid_generation(monkeypatch)
    config.update(backends=["dense"], make_plots=False)
    run = run_scientific(config, output=tmp_path / "run", data=data)
    assert run.record["success"]
    _assert_dataset_provenance(run.record["configuration"], data)
    assert run.record["configuration"]["mock"] == mock
    for key, result in run.results.items():
        restored = load_result(tmp_path / "run" / key)
        assert restored.configuration == result.configuration
        _assert_inputs_match_dataset(tmp_path / "run", key, data)


@pytest.mark.integration
def test_iterative_consumes_saved_dataset_with_provenance(
    tmp_path, saved_dataset, monkeypatch
):
    from examples.potential_correction.mock.iterative.run import (
        run_iterative_scientific,
    )

    config, mock = _dataset_inversion_configuration()
    _, data = saved_dataset
    config.update(
        backends=["dense"],
        lambda_amplitude=1e6,
        source_regularization="zero-order",
        partition_source="initial_nodal",
        solve_atol=1e-8,
        iteration_options={"gradient_atol": 1e-4, "max_iter": 80},
        make_plots=False,
    )
    _forbid_generation(monkeypatch)
    run = run_iterative_scientific(config, output=tmp_path / "run", data=data)
    assert run.record["success"]
    _assert_dataset_provenance(run.record["configuration"], data)
    assert run.record["configuration"]["mock"] == mock
    for key in run.results:
        _assert_inputs_match_dataset(tmp_path / "run", key, data)


@pytest.mark.integration
def test_lbfgs_consumes_saved_dataset_with_provenance(
    tmp_path, saved_dataset, monkeypatch
):
    from examples.potential_correction.mock.lbfgs.run import run_lbfgs_mock

    _, data = saved_dataset
    _, mock = _dataset_inversion_configuration()
    configuration = {
        "source_n": 9,
        "source_bbox": [-0.4, 0.4, -0.4, 0.4],
        "potential_spacing": 0.4,
        "source_regularization": "first-order",
        "strengths": {
            "lambda_source": 0.001,
            "lambda_potential": 10000.0,
            "lambda_amplitude": 10000.0,
        },
        "make_plots": False,
    }
    _forbid_generation(monkeypatch)
    run = run_lbfgs_mock(configuration, output=tmp_path / "run", data=data)
    assert run.record["success"]
    _assert_dataset_provenance(run.record, data)
    written = json.loads((tmp_path / "run" / "configuration.json").read_text())
    _assert_dataset_provenance(written, data)
    assert written["mock"] == mock
    for key in run.results:
        _assert_inputs_match_dataset(tmp_path / "run", key, data)


@pytest.mark.integration
def test_dataset_failures_precede_any_output_write(tmp_path, saved_dataset):
    from examples.potential_correction.mock.dataset import prepare_mock_input
    from examples.potential_correction.mock.single_step.run import run_scientific

    _, data = saved_dataset
    config, _ = _dataset_inversion_configuration()
    with pytest.raises(OSError):
        prepare_mock_input(config, data=tmp_path / "missing", output=tmp_path / "out")
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="separate"):
        prepare_mock_input(config, data=data, output=data / "nested")
    assert not (data / "nested").exists()
    with pytest.raises(ValueError, match="separate"):
        run_scientific(config, output=data, data=data)
    sibling = tmp_path / "nonempty"
    sibling.mkdir()
    (sibling / "stale.txt").write_text("x")
    with pytest.raises(FileExistsError):
        run_scientific(config, output=sibling, data=data)


@pytest.mark.integration
def test_saved_run_configuration_reuses_recorded_dataset(
    tmp_path, saved_dataset, monkeypatch
):
    from examples.potential_correction.mock import dataset
    from examples.potential_correction.mock.dataset import prepare_mock_input

    config, _ = _dataset_inversion_configuration()
    _, data = saved_dataset
    captured, pair = prepare_mock_input(
        config, data=data, output=tmp_path / "first"
    )
    assert "mock" in captured and "dataset" in captured

    def forbidden(*args, **kwargs):
        raise AssertionError("saved run configuration must not regenerate")

    monkeypatch.setattr(dataset, "generate_mock_pair", forbidden)
    reused, pair_again = prepare_mock_input(captured, output=tmp_path / "second")
    assert reused["dataset"] == captured["dataset"]
    for case in ("subhalo", "control"):
        for name in DATASET_INPUTS + ("model_image",):
            np.testing.assert_array_equal(
                np.asarray(getattr(getattr(pair_again, case), name)),
                np.asarray(getattr(getattr(pair, case), name)),
            )

    shifted = tmp_path / "shifted"
    shifted.mkdir()
    for name in ("configuration.json", "subhalo.npz", "control.npz"):
        (shifted / name).write_bytes((data / name).read_bytes())
    shifted_config, _ = prepare_mock_input(
        config, data=shifted, output=tmp_path / "shifted-run"
    )
    with (shifted / "configuration.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="checksum"):
        prepare_mock_input(shifted_config, output=tmp_path / "third")
    assert not (tmp_path / "third").exists()

    other = tmp_path / "other"
    other.mkdir()
    for name in ("configuration.json", "subhalo.npz", "control.npz"):
        (other / name).write_bytes((data / name).read_bytes())
    updated, _ = prepare_mock_input(captured, data=other, output=tmp_path / "fourth")
    assert updated["dataset"]["path"] == str(other.resolve())


@pytest.mark.parametrize("recorded", (False, True))
def test_single_step_cli_rejects_simulation_overrides_in_dataset_mode(
    tmp_path, capsys, recorded
):
    from examples.potential_correction.mock.single_step.run import main

    config, _ = _dataset_inversion_configuration()
    if recorded:
        config["mock"] = {}
        config["dataset"] = {"path": str(tmp_path / "missing"), "sha256": {}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--config",
                str(path),
                "--npix",
                "12",
                "--output",
                str(tmp_path / "out"),
            ]
        )
    assert excinfo.value.code == 2
    assert "Simulation overrides" in capsys.readouterr().err
