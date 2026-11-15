"""State/provenance and coordinate regressions for experiment drivers.

These tests exercise CLI behavior, persisted run/summary artifacts and
plotted coordinate extents with tiny generated fixtures under tmp_path.
They do not rerun any numerical campaign; the fake external auditor used
here is a test fixture, not a scientific alignment PASS.
"""

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / ".scratch/potential-correction/experiments"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(REPO))

import compare_single_step_precision as compare_driver  # noqa: E402
import plot_dense_three_strength_median as plot_driver  # noqa: E402
import sample_dense_three_strengths_precision as sample_driver  # noqa: E402


def tree_bytes(root):
    """Snapshot every file's bytes below root for mutation checks."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def call_main(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["driver"] + [str(a) for a in argv])
    module.main()


# --------------------------------------------------------------------------
# Sampler driver state guards
# --------------------------------------------------------------------------

NAMES = ["lambda_source", "lambda_potential", "lambda_amplitude"]

F64_POLICY = {
    "primary_dtype": "float64",
    "primary_solve_rtol": 1e-8,
    "reference_dtype": "float64",
    "reference_solve_rtol": 1e-8,
    "matmul_precision": "highest",
    "fallback_dtype": None,
}


def small_cfg():
    """Small scientific configuration recipe for the sampler success test."""
    return dict(
        mock=dict(
            npix=12,
            dpix=0.2,
            nsub=2,
            theta_E=0.7,
            macro_center_x=0.0,
            macro_center_y=0.0,
            gamma1=0.05,
            gamma2=0.02,
            b_sub=0.03,
            subhalo_x=0.65,
            subhalo_y=0.1,
            noise_sigma=0.03,
            seed=17,
            psf_sigma=0.04,
            psf_size=3,
            mask_inner=0.4,
            mask_outer=1.0,
            dtype="float64",
            source=dict(
                center_x=0.05,
                center_y=0.03,
                sigma_x=0.16,
                sigma_y=0.12,
                amplitude=1.0,
                clump_fraction=0.35,
                clump_dx=-0.13,
                clump_dy=0.1,
                clump_scale=0.7,
            ),
        ),
        inversion_nsub=1,
        source_n=9,
        source_bbox=[-0.4, 0.4, -0.4, 0.4],
        potential_spacing=0.4,
        lambda_amplitude=1000.0,
        solve_rtol=1e-8,
    )


def default_settings(dtype="float64"):
    return dict(
        n_live=300,
        n_eff=5000,
        batch_size=8,
        seed=271828,
        case="subhalo",
        dtype=dtype,
        log10_prior_bounds=[[-8.0, -1.0], [1.0, 11.0], [0.0, 8.0]],
        f_live=0.01,
        discard_exploration=True,
    )


def make_root(tmp_path, cfg):
    """Temporary ROOT holding the config and earlier-posterior fixtures."""
    root = tmp_path / "root"
    config = root / "examples/potential_correction/mock/single_step/configurations/scientific.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(cfg))
    research = root / ".scratch/potential-correction/research/three-strength-nautilus"
    research.mkdir(parents=True)
    np.savez_compressed(
        research / "posterior.npz",
        log10_strengths=np.array([[-4.0, 5.0, 4.0]]),
        weights=np.array([1.0]),
    )
    return root


def make_run(tmp_path, cfg, *, policy="__legacy__", mutate=None):
    """A persisted run directory with manifest and sentinel checkpoint."""
    out = tmp_path / "run"
    out.mkdir()
    saved_cfg = copy.deepcopy(cfg)
    settings = default_settings()
    order = list(NAMES)
    if mutate == "seed":
        saved_cfg["mock"]["seed"] += 1
    elif mutate == "source_n":
        saved_cfg["source_n"] += 1
    elif mutate == "sampler":
        settings["n_live"] += 1
    elif mutate == "parameter_order":
        order = list(reversed(NAMES))
    manifest = dict(
        date="2026-11-01T00:00:00+00:00",
        scientific_configuration=saved_cfg,
        sampler_settings=settings,
        parameter_order=order,
        target="test",
        fallback="test",
        driver_sha256="historical-driver-hash",
    )
    if mutate == "policy":
        manifest["precision_policy"] = dict(F64_POLICY, primary_solve_rtol=1e-4)
    elif policy == "explicit":
        manifest["precision_policy"] = dict(F64_POLICY)
    (out / "manifest.json").write_text(json.dumps(manifest))
    (out / "checkpoint.h5").write_bytes(b"SENTINEL")
    return out


RESUME_MISMATCH = {
    "seed": "scientific_configuration",
    "source_n": "scientific_configuration",
    "sampler": "sampler_settings",
    "parameter_order": "parameter_order",
    "policy": "precision policy",
}


@pytest.mark.parametrize("mutate", sorted(RESUME_MISMATCH))
def test_resume_rejects_mismatched_state(tmp_path, monkeypatch, mutate):
    cfg = small_cfg()
    root = make_root(tmp_path, cfg)
    out = make_run(tmp_path, cfg, mutate=mutate)
    before = tree_bytes(out)
    monkeypatch.setattr(sample_driver, "ROOT", root)
    with pytest.raises(ValueError, match=RESUME_MISMATCH[mutate]):
        call_main(
            sample_driver,
            [
                "--output",
                out,
                "--resume",
                "--validate-only",
                "--dtype",
                "float64",
            ],
            monkeypatch,
        )
    assert tree_bytes(out) == before


def test_resume_missing_directory_creates_nothing(tmp_path, monkeypatch):
    root = make_root(tmp_path, small_cfg())
    out = tmp_path / "absent"
    monkeypatch.setattr(sample_driver, "ROOT", root)
    with pytest.raises(SystemExit):
        call_main(sample_driver, ["--output", out, "--resume"], monkeypatch)
    assert not out.exists()


def test_resume_missing_manifest_writes_nothing(tmp_path, monkeypatch):
    root = make_root(tmp_path, small_cfg())
    out = tmp_path / "run"
    out.mkdir()
    monkeypatch.setattr(sample_driver, "ROOT", root)
    with pytest.raises(SystemExit):
        call_main(sample_driver, ["--output", out, "--resume"], monkeypatch)
    assert list(out.iterdir()) == []


def test_median_reference_only_missing_directory(tmp_path, monkeypatch):
    root = make_root(tmp_path, small_cfg())
    out = tmp_path / "absent"
    monkeypatch.setattr(sample_driver, "ROOT", root)
    with pytest.raises(SystemExit):
        call_main(
            sample_driver,
            ["--output", out, "--median-reference-only"],
            monkeypatch,
        )
    assert not out.exists()


def gpu_available():
    import jax

    try:
        return jax.devices()[0].platform == "gpu"
    except Exception:
        return False


@pytest.mark.skipif(not gpu_available(), reason="GPU validation requires CUDA")
@pytest.mark.parametrize("policy", ["__legacy__", "explicit"])
def test_resume_validate_only_success(tmp_path, monkeypatch, policy):
    cfg = small_cfg()
    root = make_root(tmp_path, cfg)
    out = make_run(tmp_path, cfg, policy=policy)
    monkeypatch.setattr(sample_driver, "ROOT", root)
    call_main(
        sample_driver,
        ["--output", out, "--resume", "--validate-only", "--dtype", "float64"],
        monkeypatch,
    )
    assert (out / "resume-validation.json").is_file()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["scientific_configuration"] == cfg
    assert manifest["driver_sha256"] == "historical-driver-hash"
    assert manifest["precision_policy"] == F64_POLICY
    assert len(manifest["resumptions"]) == 1


# --------------------------------------------------------------------------
# Comparison driver cache/summary consistency
# --------------------------------------------------------------------------

CASE_LAMBDAS = {"subhalo": 1.6e7, "control": 1e11}
NODES = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])


def fixture_metadata(case, solve_rtol):
    return dict(
        name="scientific200",
        observation=case,
        potential_spacing=0.075,
        lambda_source=0.0006,
        lambda_potential=CASE_LAMBDAS[case],
        lambda_amplitude=1e4,
        solve_rtol=solve_rtol,
    )


def write_input(path, case, solve_rtol, masked_nan=False):
    image = np.array([1.0, 2.0, 3.0, 4.0])
    if masked_nan:
        image[3] = np.nan
    np.savez_compressed(
        path,
        node_indices=NODES,
        image_data=image,
        metadata=json.dumps(fixture_metadata(case, solve_rtol)),
    )


def write_worker(output, key, backend, case, dtype, fixture):
    metadata = json.loads(str(dict(np.load(fixture))["metadata"]))
    np.savez_compressed(
        output / f"{key}.npz",
        potential_node_indices=NODES,
        derivative_valid=np.array([True] * 4),
        source_values=np.array([1.0, 2.0]),
        potential_values=np.array([0.1, 0.2, 0.3, 0.4]),
        convergence=np.array([0.1, 0.2, 0.3, 0.4]),
        noise_active=np.array([1.0, 1.0]),
        nonlinear_image_active=np.array([1.0, 2.0]),
    )
    row = dict(
        worker_returncode=0,
        algorithm="single_step",
        backend=backend,
        case="scientific200",
        observation=case,
        dtype=dtype,
        profile="scientific",
        configuration=metadata,
        input_path=str(fixture),
        input_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        result=dict(
            success=True,
            log_evidence=1.0,
            configuration=dict(solve_rtol=metadata["solve_rtol"]),
        ),
        timing=dict(warm_fit_seconds=[1.0]),
        counts={},
        dtype_audit={"psf_fft": "complex64" if dtype == "float32" else "complex128"},
        geometry={},
        software={},
    )
    (output / f"{key}.json").write_text(json.dumps(row))


def populate(output, fixtures, *, probe, masked_nan=False):
    """Write strict (and optionally cached probe) inputs and worker records."""
    for case in ("subhalo", "control"):
        strict = fixtures / f"scientific200-{case}.npz"
        write_input(strict, case, 1e-8, masked_nan=masked_nan)
        profiles = [("strict", strict)]
        if probe:
            derived = output / f"{case}-residual-probe-inputs.npz"
            write_input(derived, case, 1e-5, masked_nan=masked_nan)
            profiles.append(("residual_probe", derived))
        for profile, fixture in profiles:
            for backend in ("dense", "operator"):
                for dtype in ("float64", "float32"):
                    write_worker(
                        output,
                        f"{backend}-{profile}-{case}-{dtype}",
                        backend,
                        case,
                        dtype,
                        fixture,
                    )


def compare_argv(fixtures, output, summary, rtol="1e-5"):
    return [
        "--fixtures",
        fixtures,
        "--output",
        output,
        "--probe-solve-rtol",
        rtol,
        "--summarize-only",
        "--summary",
        summary,
    ]


def test_summarize_only_consistent_probe(tmp_path, monkeypatch):
    fixtures, output = tmp_path / "fixtures", tmp_path / "out"
    fixtures.mkdir()
    output.mkdir()
    populate(output, fixtures, probe=True)
    before = tree_bytes(fixtures) | tree_bytes(output)
    summary = tmp_path / "summary.json"
    call_main(compare_driver, compare_argv(fixtures, output, summary), monkeypatch)
    record = json.loads(summary.read_text())
    assert record["residual_probe_rtol"] == 1e-5
    assert len(record["runs"]) == 16
    assert tree_bytes(fixtures) | tree_bytes(output) == before


@pytest.mark.parametrize("probe", [False, True])
def test_summarize_only_relocated_campaign_uses_current_fixtures(
    tmp_path, monkeypatch, probe
):
    original = tmp_path / "original"
    fixtures, output = original / "fixtures", original / "out"
    fixtures.mkdir(parents=True)
    output.mkdir()
    populate(output, fixtures, probe=probe)
    moved = original.rename(tmp_path / "relocated")
    fixtures, output = moved / "fixtures", moved / "out"
    for path in output.glob("*.json"):
        assert not Path(json.loads(path.read_text())["input_path"]).exists()
    before = tree_bytes(moved)
    summary = tmp_path / "summary.json"
    argv = [
        "--fixtures",
        fixtures,
        "--output",
        output,
        "--summarize-only",
        "--summary",
        summary,
    ]
    if probe:
        argv.extend(["--probe-solve-rtol", "1e-5"])
    call_main(compare_driver, argv, monkeypatch)
    record = json.loads(summary.read_text())
    assert len(record["runs"]) == (16 if probe else 8)
    assert len(record["comparisons"]) == (8 if probe else 4)
    for comparison in record["comparisons"].values():
        assert comparison["fields"]["source_values"]["max_absolute"] == 0
    assert tree_bytes(moved) == before


def test_summarize_only_rtol_mismatch_rejected(tmp_path, monkeypatch):
    fixtures, output = tmp_path / "fixtures", tmp_path / "out"
    fixtures.mkdir()
    output.mkdir()
    populate(output, fixtures, probe=True)
    before = tree_bytes(fixtures) | tree_bytes(output)
    summary = tmp_path / "summary.json"
    with pytest.raises(ValueError):
        call_main(
            compare_driver,
            compare_argv(fixtures, output, summary, rtol="1e-3"),
            monkeypatch,
        )
    assert not summary.exists()
    assert tree_bytes(fixtures) | tree_bytes(output) == before


def test_summarize_only_missing_probe_not_created(tmp_path, monkeypatch):
    fixtures, output = tmp_path / "fixtures", tmp_path / "out"
    fixtures.mkdir()
    output.mkdir()
    populate(output, fixtures, probe=False)
    summary = tmp_path / "summary.json"
    with pytest.raises(FileNotFoundError):
        call_main(compare_driver, compare_argv(fixtures, output, summary), monkeypatch)
    assert not (output / "subhalo-residual-probe-inputs.npz").exists()
    assert not summary.exists()


def test_summarize_only_missing_output_dir(tmp_path, monkeypatch):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    output = tmp_path / "absent"
    with pytest.raises(SystemExit):
        call_main(
            compare_driver,
            compare_argv(fixtures, output, tmp_path / "s.json"),
            monkeypatch,
        )
    assert not output.exists()


@pytest.mark.parametrize("tamper", ["solve_rtol", "input_sha256", "configuration"])
def test_tampered_worker_rejected(tmp_path, monkeypatch, tamper):
    fixtures, output = tmp_path / "fixtures", tmp_path / "out"
    fixtures.mkdir()
    output.mkdir()
    populate(output, fixtures, probe=True)
    target = output / "dense-strict-subhalo-float64.json"
    row = json.loads(target.read_text())
    if tamper == "solve_rtol":
        row["result"]["configuration"]["solve_rtol"] = 1e-4
    elif tamper == "input_sha256":
        row["input_sha256"] = "0" * 64
    else:
        row["configuration"]["lambda_potential"] = 1.0
    target.write_text(json.dumps(row))
    before = tree_bytes(fixtures) | tree_bytes(output)
    with pytest.raises(ValueError):
        call_main(
            compare_driver,
            compare_argv(fixtures, output, tmp_path / "s.json"),
            monkeypatch,
        )
    assert not (tmp_path / "s.json").exists()
    assert tree_bytes(fixtures) | tree_bytes(output) == before


def test_summarize_only_paired_nan_accepted(tmp_path, monkeypatch):
    """Corresponding NaNs in masked pixels are equal inputs, not tampering."""
    fixtures, output = tmp_path / "fixtures", tmp_path / "out"
    fixtures.mkdir()
    output.mkdir()
    populate(output, fixtures, probe=True, masked_nan=True)
    before = tree_bytes(fixtures) | tree_bytes(output)
    summary = tmp_path / "summary.json"
    call_main(compare_driver, compare_argv(fixtures, output, summary), monkeypatch)
    record = json.loads(summary.read_text())
    assert record["residual_probe_rtol"] == 1e-5
    assert tree_bytes(fixtures) | tree_bytes(output) == before


def test_tampered_probe_array_rejected(tmp_path, monkeypatch):
    fixtures, output = tmp_path / "fixtures", tmp_path / "out"
    fixtures.mkdir()
    output.mkdir()
    populate(output, fixtures, probe=True)
    derived = output / "subhalo-residual-probe-inputs.npz"
    with np.load(derived) as a:
        values = dict(a)
    values["image_data"] = values["image_data"] + 1
    np.savez_compressed(derived, **values)
    with pytest.raises(ValueError):
        call_main(
            compare_driver,
            compare_argv(fixtures, output, tmp_path / "s.json"),
            monkeypatch,
        )
    assert not (tmp_path / "s.json").exists()


# --------------------------------------------------------------------------
# Plot driver coordinates, output directory and external auditor
# --------------------------------------------------------------------------

FIXTURE_AUDITOR = '''"""TEST FIXTURE auditor: records extents; not a scientific PASS."""
import json
from pathlib import Path


def require_matplotlib_panel_alignment(fig, *, json_out=None, strict=False,
                                       axes=None, **kwargs):
    assert strict is True, "fixture auditor must be called with strict=True"
    extents = [
        [float(v) for v in im.get_extent()]
        for ax in fig.axes
        for im in ax.get_images()
    ]
    if json_out is not None:
        Path(json_out).write_text(
            json.dumps({"fixture_auditor": True, "extents": extents})
        )
    return {"fixture_auditor": True}
'''


def make_auditor_dir(path):
    path.mkdir(parents=True)
    (path / "audit_panel_alignment.py").write_text(FIXTURE_AUDITOR)
    return path


def make_run_dir(tmp_path):
    run = tmp_path / "run"
    median = run / "posterior-median"
    median.mkdir(parents=True)
    (run / "completion.json").write_text(json.dumps(dict(complete=True, n_live=4)))
    (run / "manifest.json").write_text(
        json.dumps(
            dict(sampler_settings=dict(log10_prior_bounds=[[-8, -1], [1, 11], [0, 8]]))
        )
    )
    rng = np.random.default_rng(17)
    points = rng.normal(loc=[-4, 5, 4], scale=[0.1, 0.2, 0.3], size=(256, 3))
    np.savez_compressed(
        run / "posterior.npz",
        log10_strengths=points,
        weights=np.ones(256),
        parameter_order=np.array(NAMES),
    )
    (median / "result.json").write_text(
        json.dumps(
            dict(
                success=True,
                backend="dense",
                mock_configuration=dict(npix=4, dpix=0.5, subhalo_x=0.3, subhalo_y=0.2),
                potential_geometry=dict(spacing=0.25),
            )
        )
    )
    np.savez_compressed(
        median / "reconstruction.npz",
        mask=np.zeros((4, 4), dtype=bool),
        image_data=np.arange(16.0).reshape(4, 4) + 1,
        linear_model_image=np.arange(16.0).reshape(4, 4) + 1.1,
        linear_normalized_residual=np.ones((4, 4)),
        source_bbox=np.array([-2.0, 2.0, 1.0, 5.0]),
        source_x_axis=np.array([-2.0, 0.0, 2.0]),
        source_y_axis=np.array([1.0, 3.0, 5.0]),
        source_values=np.arange(9.0) + 1,
        truth_source_values=np.arange(9.0) + 1.5,
        derivative_valid=np.array([True] * 4),
        convergence=np.array([0.1, 0.2, 0.3, 0.4]),
        truth_discrete_convergence=np.array([0.4, 0.3, 0.2, 0.1]),
        potential_node_indices=NODES,
    )
    return run


def test_render_extents_explicit_auditor(tmp_path, monkeypatch):
    run = make_run_dir(tmp_path)
    audit_dir = make_auditor_dir(tmp_path / "audit-scripts")
    out = tmp_path / "products"
    inputs = {
        p: p.read_bytes()
        for p in (run / "posterior.npz", run / "posterior-median/reconstruction.npz")
    }
    call_main(
        plot_driver,
        [run, "--output", out, "--audit-scripts-dir", audit_dir],
        monkeypatch,
    )
    alignment = json.loads(
        (out / "posterior_median_recovery.alignment.json").read_text()
    )
    assert alignment["fixture_auditor"] is True
    source_extents = alignment["extents"][3:6]
    assert source_extents == [[-3.0, 3.0, 0.0, 6.0]] * 3
    # Pixel centers of a 3x3 image inside [-3,3]x[0,6] are the saved nodes.
    left, right, bottom, top = source_extents[0]
    centers_x = [left + (i + 0.5) * (right - left) / 3 for i in range(3)]
    centers_y = [bottom + (i + 0.5) * (top - bottom) / 3 for i in range(3)]
    assert centers_x == [-2.0, 0.0, 2.0]
    assert centers_y == [1.0, 3.0, 5.0]
    corner = json.loads((out / "posterior_corner.alignment.json").read_text())
    assert corner["fixture_auditor"] is True
    for suffix in ("png", "pdf", "svg"):
        assert (out / f"posterior_median_recovery.{suffix}").is_file()
    assert (out / "posterior-summary.json").is_file()
    for path, before in inputs.items():
        assert path.read_bytes() == before


FAILING_AUDITOR = '''"""TEST FIXTURE auditor that always fails; not a scientific PASS."""


class FixtureAuditError(RuntimeError):
    pass


def require_matplotlib_panel_alignment(fig, *, json_out=None, strict=False,
                                       axes=None, **kwargs):
    raise FixtureAuditError("fixture auditor rejection")
'''


def test_failing_auditor_blocks_delivery(tmp_path, monkeypatch):
    run = make_run_dir(tmp_path)
    audit_dir = tmp_path / "audit-scripts"
    audit_dir.mkdir()
    (audit_dir / "audit_panel_alignment.py").write_text(FAILING_AUDITOR)
    out = tmp_path / "products"
    with pytest.raises(RuntimeError, match="fixture auditor rejection"):
        call_main(
            plot_driver,
            [run, "--output", out, "--audit-scripts-dir", audit_dir],
            monkeypatch,
        )
    for suffix in ("png", "pdf", "svg"):
        assert not list(out.glob(f"*.{suffix}"))


def test_output_directory_must_be_new(tmp_path, monkeypatch):
    run = make_run_dir(tmp_path)
    audit_dir = make_auditor_dir(tmp_path / "audit-scripts")
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        call_main(
            plot_driver,
            [run, "--output", out, "--audit-scripts-dir", audit_dir],
            monkeypatch,
        )


def test_explicit_invalid_audit_dir_fails(tmp_path, monkeypatch):
    home = tmp_path / "home"
    make_auditor_dir(home / ".config/devin/skills/nature-figure/scripts")
    monkeypatch.setenv("HOME", str(home))
    run = make_run_dir(tmp_path)
    with pytest.raises(ValueError):
        call_main(
            plot_driver,
            [
                run,
                "--output",
                tmp_path / "o",
                "--audit-scripts-dir",
                tmp_path / "not-auditor",
            ],
            monkeypatch,
        )
    assert not (tmp_path / "o").exists()


def test_default_auditor_discovery(tmp_path, monkeypatch):
    home = tmp_path / "home"
    make_auditor_dir(home / ".config/devin/skills/nature-figure/scripts")
    monkeypatch.setenv("HOME", str(home))
    run = make_run_dir(tmp_path)
    out = tmp_path / "products"
    call_main(plot_driver, [run, "--output", out], monkeypatch)
    assert (out / "posterior_corner.alignment.json").is_file()


def test_missing_default_auditor_fails_before_outputs(tmp_path, monkeypatch):
    home = tmp_path / "empty-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    run = make_run_dir(tmp_path)
    out = tmp_path / "products"
    with pytest.raises(ValueError):
        call_main(plot_driver, [run, "--output", out], monkeypatch)
    assert not out.exists()


def test_help_succeeds_without_auditor(tmp_path, monkeypatch):
    home = tmp_path / "empty-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["driver", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        plot_driver.main()
    assert exit_info.value.code == 0
