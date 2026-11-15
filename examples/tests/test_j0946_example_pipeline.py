"""Layout and mechanics tests for the self-contained J0946 example workflow."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from test_j0946_single_step_workflow import _handoff


ROOT = Path(__file__).resolve().parents[2]
J0946 = ROOT / "examples" / "potential_correction" / "j0946"


class _Sentinel(Exception):
    pass


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_j0946_cli_help_runs_directly_from_any_cwd(tmp_path):
    for script in (
        "macro_model.py",
        "pt.py",
        "pt_iter_lbfgs.py",
        "run_pipeline.py",
    ):
        result = subprocess.run(
            [sys.executable, str(J0946 / script), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (script, result.stderr)


@pytest.mark.parametrize(
    "module_name",
    (
        "examples.potential_correction.j0946.macro_model",
        "examples.potential_correction.j0946.pt",
        "examples.potential_correction.j0946.pt_iter_lbfgs",
        "examples.potential_correction.j0946.run_pipeline",
    ),
)
def test_j0946_module_help_smoke(module_name):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (module_name, result.stderr)


def _capture_defaults(module, entry):
    original = argparse.ArgumentParser.parse_args
    captured = {}

    def spy(self, args=None, namespace=None):
        parsed = original(self, args, namespace)
        captured.update(vars(parsed))
        raise _Sentinel

    argparse.ArgumentParser.parse_args = spy
    try:
        with pytest.raises(_Sentinel):
            entry([])
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured


def test_pt_and_lbfgs_defaults_use_output_subdirectories():
    pt = _load("j0946_pt_defaults", J0946 / "pt.py")
    captured = _capture_defaults(pt, pt.main)
    assert captured["macro_dir"] == J0946 / "output" / "macro"
    assert captured["output"] == J0946 / "output" / "single_step"

    lbfgs = _load("j0946_lbfgs_defaults", J0946 / "pt_iter_lbfgs.py")
    captured = _capture_defaults(lbfgs, lbfgs.main)
    assert captured["macro_dir"] == J0946 / "output" / "macro"
    assert captured["output"] == J0946 / "output" / "lbfgs"


def test_macro_cli_defaults_use_sibling_data_and_output_dirs():
    macro = _load("j0946_macro_defaults", J0946 / "macro_model.py")
    captured = _capture_defaults(macro, macro.cli)
    assert captured["data_dir"] == J0946 / "data"
    assert captured["out_dir"] is None
    assert macro.OUT_DIR == J0946 / "output" / "macro"
    assert macro.DATA_DIR == J0946 / "data"


def _raw_data(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("image.fits", "noise.fits", "psf.fits"):
        (directory / name).write_bytes(f"{name}-bytes".encode())
    return directory


def _pipeline(tmp_path, monkeypatch=None):
    pipeline = _load("j0946_pipeline_test", J0946 / "run_pipeline.py")
    calls = []
    if monkeypatch is not None:
        def record(command, **kwargs):
            calls.append(command)

        def fail_load(macro_dir):
            raise AssertionError("load_handoff must be stubbed")

        monkeypatch.setattr(pipeline.subprocess, "run", record)
        monkeypatch.setattr(pipeline, "load_handoff", fail_load)
    return pipeline, calls


def test_pipeline_dispatches_macro_then_corrections(tmp_path, monkeypatch):
    data = _raw_data(tmp_path / "data")
    macro = tmp_path / "out" / "macro"
    pipeline, calls = _pipeline(tmp_path, monkeypatch)
    seen = []

    def record(command, **kwargs):
        calls.append(command)
        if "macro_model.py" in command[1]:
            macro.parent.mkdir(parents=True, exist_ok=True)
            _handoff(macro)

    monkeypatch.setattr(pipeline.subprocess, "run", record)
    monkeypatch.setattr(
        pipeline, "load_handoff", lambda path: seen.append(Path(path))
    )
    pipeline.main(
        ["--data-dir", str(data), "--output-root", str(tmp_path / "out")]
    )
    assert calls == [
        [
            sys.executable,
            str(J0946 / "macro_model.py"),
            "--data-dir",
            str(data.resolve()),
            "--output",
            str(macro),
        ],
        [
            sys.executable,
            str(J0946 / "pt.py"),
            "--macro-dir",
            str(macro),
            "--output",
            str(tmp_path / "out" / "single_step"),
        ],
        [
            sys.executable,
            str(J0946 / "pt_iter_lbfgs.py"),
            "--macro-dir",
            str(macro),
            "--output",
            str(tmp_path / "out" / "lbfgs"),
        ],
    ]
    assert seen == [macro]


@pytest.mark.parametrize(
    ("correction", "script"),
    (("single-step", "pt.py"), ("lbfgs", "pt_iter_lbfgs.py")),
)
def test_pipeline_branch_selection(tmp_path, monkeypatch, correction, script):
    data = _raw_data(tmp_path / "data")
    handoff = tmp_path / "handoff"
    _handoff(handoff)
    pipeline, calls = _pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "load_handoff", lambda path: None)
    pipeline.main(
        [
            "--data-dir",
            str(data),
            "--output-root",
            str(tmp_path / "out"),
            "--macro-dir",
            str(handoff),
            "--correction",
            correction,
        ]
    )
    assert [Path(call[1]).name for call in calls] == [script]


def test_pipeline_prepare_only_runs_macro_only(tmp_path, monkeypatch):
    data = _raw_data(tmp_path / "data")
    pipeline, calls = _pipeline(tmp_path, monkeypatch)
    pipeline.main(
        [
            "--data-dir",
            str(data),
            "--output-root",
            str(tmp_path / "out"),
            "--prepare-only",
        ]
    )
    assert len(calls) == 1
    command = calls[0]
    assert command[1] == str(J0946 / "macro_model.py")
    assert "--prepare-only" in command


def test_pipeline_resume_macro_flag_propagates(tmp_path, monkeypatch):
    data = _raw_data(tmp_path / "data")
    macro = tmp_path / "out" / "macro"
    macro.mkdir(parents=True)
    pipeline, calls = _pipeline(tmp_path, monkeypatch)
    pipeline.main(
        [
            "--data-dir",
            str(data),
            "--output-root",
            str(tmp_path / "out"),
            "--resume-macro",
            "--prepare-only",
        ]
    )
    assert "--resume" in calls[0] and "--prepare-only" in calls[0]


@pytest.mark.parametrize(
    "argv",
    (
        ["--macro-dir", "x", "--resume-macro"],
        ["--macro-dir", "x", "--prepare-only"],
        ["--macro-dir", "x", "--source-interpolation", "bilinear"],
    ),
)
def test_pipeline_rejects_macro_dir_flag_conflicts(tmp_path, argv):
    pipeline, _ = _pipeline(tmp_path, None)
    with pytest.raises(SystemExit) as excinfo:
        pipeline.main(argv)
    assert excinfo.value.code == 2


def test_pipeline_rejects_missing_raw_before_subprocess(tmp_path, capsys):
    pipeline, calls = _pipeline(tmp_path, None)
    with pytest.raises(SystemExit) as excinfo:
        pipeline.main(
            [
                "--data-dir",
                str(tmp_path / "data"),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
    assert excinfo.value.code == 2
    assert "Missing raw FITS" in capsys.readouterr().err
    assert calls == []


def test_pipeline_rejects_existing_correction_output(tmp_path, capsys):
    data = _raw_data(tmp_path / "data")
    (tmp_path / "out" / "single_step").mkdir(parents=True)
    pipeline, calls = _pipeline(tmp_path, None)
    with pytest.raises(SystemExit) as excinfo:
        pipeline.main(
            [
                "--data-dir",
                str(data),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
    assert excinfo.value.code == 2
    assert "new correction output" in capsys.readouterr().err
    assert calls == []


def test_pipeline_rejects_output_overlapping_data(tmp_path, capsys):
    data = _raw_data(tmp_path / "data")
    pipeline, calls = _pipeline(tmp_path, None)
    with pytest.raises(SystemExit) as excinfo:
        pipeline.main(
            [
                "--data-dir",
                str(data),
                "--output-root",
                str(data / "nested"),
            ]
        )
    assert excinfo.value.code == 2
    assert "separate from the raw data" in capsys.readouterr().err
    assert calls == []


def test_pipeline_stops_corrections_when_macro_fails(tmp_path, monkeypatch):
    data = _raw_data(tmp_path / "data")
    pipeline, _ = _pipeline(tmp_path, monkeypatch)
    calls = []

    def record(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(pipeline.subprocess, "run", record)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.main(
            ["--data-dir", str(data), "--output-root", str(tmp_path / "out")]
        )
    assert len(calls) == 1


def test_pipeline_stops_lbfgs_when_single_step_fails(tmp_path, monkeypatch):
    data = _raw_data(tmp_path / "data")
    macro = tmp_path / "out" / "macro"
    pipeline, _ = _pipeline(tmp_path, monkeypatch)
    calls = []

    def record(command, **kwargs):
        calls.append(command)
        if "macro_model.py" in command[1]:
            macro.parent.mkdir(parents=True, exist_ok=True)
            _handoff(macro)
        elif "pt.py" in command[1]:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(pipeline.subprocess, "run", record)
    monkeypatch.setattr(pipeline, "load_handoff", lambda path: None)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.main(
            ["--data-dir", str(data), "--output-root", str(tmp_path / "out")]
        )
    assert [Path(call[1]).name for call in calls] == [
        "macro_model.py",
        "pt.py",
    ]


def test_pipeline_rejects_corrupt_handoff_after_macro(tmp_path, monkeypatch):
    data = _raw_data(tmp_path / "data")
    macro = tmp_path / "out" / "macro"
    pipeline, calls = _pipeline(tmp_path)

    def record(command, **kwargs):
        calls.append(command)
        if "macro_model.py" in command[1]:
            macro.parent.mkdir(parents=True, exist_ok=True)
            _handoff(macro)
            with (macro / "m3_handoff.npz").open("ab") as handle:
                handle.write(b" ")

    monkeypatch.setattr(pipeline.subprocess, "run", record)
    with pytest.raises(RuntimeError, match="Incomplete or changed"):
        pipeline.main(
            ["--data-dir", str(data), "--output-root", str(tmp_path / "out")]
        )
    assert len(calls) == 1


def test_macro_prepare_only_records_true_input_hashes(tmp_path):
    macro = _load("j0946_macro_prepare", J0946 / "macro_model.py")
    data = tmp_path / "data"
    data.mkdir()
    expected = {}
    for name in ("image.fits", "noise.fits", "psf.fits"):
        payload = f"{name}-payload".encode()
        (data / name).write_bytes(payload)
        expected[name] = hashlib.sha256(payload).hexdigest()
    output = tmp_path / "macro"
    macro.cli(
        [
            "--data-dir",
            str(data),
            "--output",
            str(output),
            "--prepare-only",
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["settings"]["inputs_sha256"] == expected
    assert not (output / "m3_handoff.npz").exists()


def test_macro_input_dir_copies_into_data_dir(tmp_path):
    macro = _load("j0946_macro_copy", J0946 / "macro_model.py")
    source = _raw_data(tmp_path / "external")
    data = tmp_path / "local-data"
    macro.cli(
        [
            "--data-dir",
            str(data),
            "--input-dir",
            str(source),
            "--output",
            str(tmp_path / "macro"),
            "--prepare-only",
        ]
    )
    for name in ("image.fits", "noise.fits", "psf.fits"):
        assert (data / name).read_bytes() == (source / name).read_bytes()


def test_macro_missing_and_changed_inputs_reject_without_writes(tmp_path):
    macro = _load("j0946_macro_reject", J0946 / "macro_model.py")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        macro.cli(
            [
                "--data-dir",
                str(empty),
                "--output",
                str(tmp_path / "macro"),
                "--prepare-only",
            ]
        )
    assert excinfo.value.code == 2

    data = _raw_data(tmp_path / "data")
    source = _raw_data(tmp_path / "external")
    (source / "image.fits").write_bytes(b"different")
    before = (data / "image.fits").read_bytes()
    with pytest.raises(RuntimeError, match="Input changed"):
        macro.cli(
            [
                "--data-dir",
                str(data),
                "--input-dir",
                str(source),
                "--output",
                str(tmp_path / "macro"),
                "--prepare-only",
            ]
        )
    assert (data / "image.fits").read_bytes() == before


def test_handoff_accepts_str_and_path_and_rejects_bad_archives(tmp_path):
    from examples.potential_correction.j0946._handoff import load_handoff

    macro = tmp_path / "macro"
    _handoff(macro)
    metadata, arrays = load_handoff(macro)
    assert load_handoff(str(macro))[0] == metadata
    assert "image_data" in arrays

    broken = tmp_path / "broken"
    _handoff(broken)
    record = json.loads((broken / "m3_handoff.json").read_text())
    record["complete"] = False
    (broken / "m3_handoff.json").write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="Incomplete or changed"):
        load_handoff(broken)

    changed = tmp_path / "changed"
    _handoff(changed)
    with (changed / "m3_handoff.npz").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(RuntimeError, match="Incomplete or changed"):
        load_handoff(changed)


def test_importing_lbfgs_does_not_import_pt():
    code = (
        "import os\n"
        "import sys\n"
        "import examples.potential_correction.j0946.pt_iter_lbfgs\n"
        "assert 'examples.potential_correction.j0946.pt' not in sys.modules\n"
        "assert os.environ['OMP_NUM_THREADS'] == '1'\n"
        "assert os.environ['MKL_NUM_THREADS'] == '1'\n"
        "assert os.environ['NVIDIA_TF32_OVERRIDE'] == '0'\n"
        "print('isolated')\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NVIDIA_TF32_OVERRIDE")
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "isolated" in result.stdout


@pytest.mark.integration
def test_tiny_lbfgs_search_qualifies_stationary_candidate(tmp_path):
    import jax

    lbfgs = _load("j0946_lbfgs_tiny", J0946 / "pt_iter_lbfgs.py")

    macro = tmp_path / "macro"
    _handoff(macro)
    archive = np.load(macro / "m3_handoff.npz")
    arrays = {name: archive[name] for name in archive.files}
    archive.close()
    arrays.update(
        source_values=np.zeros(4, dtype=np.float32),
        source_scale=np.ones(4, dtype=np.float32),
        image_data=np.zeros((4, 4), dtype=np.float32),
        lens_subtracted_image=np.zeros((4, 4), dtype=np.float32),
        macro_model_image=np.zeros((4, 4), dtype=np.float32),
        lens_light_model=np.zeros((4, 4), dtype=np.float32),
        noise_map=np.ones((4, 4), dtype=np.float32),
    )
    np.savez_compressed(macro / "m3_handoff.npz", **arrays)
    metadata = json.loads((macro / "m3_handoff.json").read_text())
    metadata["selected_sample_index"] = 0
    metadata["handoff_sha256"] = hashlib.sha256(
        (macro / "m3_handoff.npz").read_bytes()
    ).hexdigest()
    (macro / "m3_handoff.json").write_text(json.dumps(metadata))

    output = tmp_path / "lbfgs"
    output.mkdir()
    with jax.experimental.enable_x64(), jax.default_matmul_precision("highest"):
        problem, saved_metadata, observation = lbfgs.prepare_problem(macro, 0.2)
        assert saved_metadata["selected_sample_index"] == 0
        bounds = np.asarray([[-1.1, -0.9], [-0.8, -0.6], [-0.1, 0.1]])
        best, result, summary = lbfgs.search(
            problem,
            bounds,
            output,
            de_options=dict(
                maxiter=0, popsize=2, seed=947, polish=False, workers=1
            ),
            fit_options=dict(max_iter=20, gn_stationarity_tol=0.001),
        )
    assert result is not None and best is not None
    assert summary["qualified"] >= 1
    assert np.isfinite(best["log_evidence"])
    ledger = [
        json.loads(line)
        for line in (output / "evaluations.jsonl").read_text().splitlines()
    ]
    qualified = [row for row in ledger if row["qualified"]]
    assert qualified and best["log_evidence"] == max(
        row["log_evidence"] for row in qualified
    )

    destination = output / "maximum_evidence"
    lbfgs.save_reconstruction(destination, result, best, observation)
    record = json.loads((destination / "result.json").read_text())
    with np.load(destination / "reconstruction.npz") as exported:
        for name in (
            "image_data",
            "lens_subtracted_image",
            "macro_model_image",
            "lens_light_model",
            "noise_map",
            "mask",
            "psf_kernel",
            "source_scale",
        ):
            np.testing.assert_array_equal(exported[name], arrays[name])
        assert exported["potential_values"].dtype == np.float64
    assert record["reconstruction_sha256"] == hashlib.sha256(
        (destination / "reconstruction.npz").read_bytes()
    ).hexdigest()


def test_j0946_sources_have_no_workspace_or_machine_paths():
    for script in (
        "macro_model.py",
        "pt.py",
        "pt_iter_lbfgs.py",
        "run_pipeline.py",
        "_handoff.py",
        "_macro_provenance.py",
    ):
        path = J0946 / script
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("workspace"), (
                        script,
                        alias.name,
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("workspace"), (
                    script,
                    node.module,
                )
        assert "/home/" not in path.read_text(), script
