"""Joint/subhalo support keeps sampling controls and stage artifact contracts."""
import importlib.util
import json
from pathlib import Path
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.Inference import StagePosterior
from TinyLensGpu.example_support import stage_from_payload
from test_pixelized_support import _small_likelihood


def load_case(
    examples_dir, relative, monkeypatch, tmp_path, *, family="pix_src_demo_operator",
):
    path = examples_dir / family / relative
    spec = importlib.util.spec_from_file_location('joint_support_case', path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as import_context:
        import_context.chdir(path.parent)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'OUT_DIR', tmp_path)
    return module


@pytest.mark.parametrize("family", ["pix_src_demo", "pix_src_demo_operator"])
@pytest.mark.parametrize("requested", [None, "cubic_hermite"], ids=["legacy", "hermite"])
def test_joint_stages_keep_interpolation_and_unit_lens_light(
    examples_dir, monkeypatch, tmp_path, family, requested,
):
    import caskade as ck
    from test_survey_shared_support import ControlledSampler

    case = load_case(
        examples_dir, "pipe/simult_fit_lens_source/mge_lens_light/model.py",
        monkeypatch, tmp_path, family=family,
    )
    for name, value in dict(
        Sampler=ControlledSampler, DPIX=0.08, NSUB=1, NSUB_PIX=1,
        N_GAUSSIANS_SRC=2, N_GAUSSIANS_LENS=2,
    ).items():
        monkeypatch.setattr(case, name, value)
    # Keep a small inversion while exercising the real joint stage builder.
    monkeypatch.setattr(case, "NSRC", 4, raising=False)
    y, x = np.indices((8, 8))
    image = 0.06 + 0.12 * np.exp(-((x - 3.1) ** 2 + (y - 3.8) ** 2) / 5)
    noise, psf = np.full_like(image, 0.1), np.ones((1, 1))
    mask = np.zeros_like(image, dtype=bool)
    mask[0, 0] = True
    monkeypatch.setattr(case, "load_lens_data", lambda **kw: (image, noise, psf, None))
    monkeypatch.setattr(case, "_position_likelihood_from_stage_a", lambda *a: {"positions": []})
    monkeypatch.setattr(case, "_build_arc_mask", lambda *a: mask)
    monkeypatch.setattr(case, "plot_model_results", lambda *a, **kw: None)
    reconstructions = []

    def observe_joint_fit(tag, likelihood, medians, names, path, **kwargs):
        theta = jnp.asarray([medians[name] for name in names])
        result, diagnostics = likelihood.evaluate_with_diagnostics(
            theta, include_components=True, include_effective_dof=True,
        )
        assert diagnostics.valid
        with ck.ActiveContext(likelihood):
            likelihood.fill_params(theta)
            assert all(
                float(basis.flux.value) == 1 for basis in likelihood.phys_model.lens_light
            )
            matrix = likelihood.sim_obj.build_lens_light_matrix()
        np.testing.assert_allclose(
            diagnostics.lens_light_image[~mask], matrix @ result.lens_amplitudes,
            atol=1e-6, rtol=2e-5,
        )
        np.testing.assert_allclose(
            result.model_image[~mask],
            (diagnostics.source_image + diagnostics.lens_light_image)[~mask],
            atol=1e-6, rtol=2e-5,
        )
        reconstructions.append((likelihood.source_model.interpolation, result))

    monkeypatch.setattr(case, "_plot_stage_b", observe_joint_fit)
    case.main(source_interpolation=requested)
    mode = requested or "bilinear"
    assert len(reconstructions) == 1
    assert reconstructions[0][0] == mode
    for tag in ("a", "b"):
        path = tmp_path / f"stage_{tag}.pkl"
        payload = pickle.loads(path.read_bytes())
        extra = payload["extra"]
        assert extra["source_interpolation"] == mode
        assert extra["source_boundary"] == (
            "C1_cardinal_bicubic_fixed_zero_exterior_nodes" if requested
            else "hard_zero_outside_bbox"
        )
        if requested is None:
            extra.pop("source_interpolation")
            extra.pop("source_boundary")
            path.write_bytes(pickle.dumps(payload))

    if requested is not None:
        # A records the choice before any joint pixelized fit is complete.
        (tmp_path / "stage_b.pkl").unlink()
        case.main(skip_done=True)
        restored_mode, restored = reconstructions[-1]
        assert restored_mode == mode
        np.testing.assert_allclose(
            restored.model_image, reconstructions[0][1].model_image,
            rtol=2e-5, atol=1e-7,
        )
        np.testing.assert_allclose(
            restored.lens_amplitudes, reconstructions[0][1].lens_amplitudes,
            rtol=2e-5, atol=1e-7,
        )
    before = {path.name: path.read_bytes() for path in tmp_path.glob("stage_*.pkl")}
    count = len(reconstructions)
    case.main(skip_done=True)
    assert len(reconstructions) == count
    assert {path.name: path.read_bytes() for path in tmp_path.glob("stage_*.pkl")} == before
    monkeypatch.setattr(
        case, "load_lens_data", lambda **kw: pytest.fail("conflict must precede stage reuse"),
    )
    conflict = "bilinear" if requested else "cubic_hermite"
    with pytest.raises(ValueError, match="stage_a.pkl.*Start a fresh run"):
        case.main(skip_done=True, source_interpolation=conflict)


@pytest.mark.parametrize("requested", [None, "cubic_hermite"], ids=["legacy", "hermite"])
def test_j0946_resumes_before_and_after_source_package(
    examples_dir, monkeypatch, tmp_path, requested,
):
    from test_survey_shared_support import ControlledSampler

    case = load_case(
        examples_dir, "pipe/subhalo_0946/model.py", monkeypatch, tmp_path,
    )
    for name, value in dict(
        Sampler=ControlledSampler, DPIX=0.35, NSUB=1, NSUB_PIX=1, NSRC=4,
        N_GAUSSIANS_SRC=2, N_GAUSSIANS_LENS=2,
        PIXEL_REGULARIZATION_TYPE="zero-order", SUBHALO_POSITION_MODE="gaussian_local",
    ).items():
        monkeypatch.setattr(case, name, value)
    y, x = np.indices((10, 10))
    image = 0.04 + 0.09 * np.exp(-((x - 8.6) ** 2 + (y - 5.2) ** 2) / 4)
    noise, psf = np.full_like(image, 0.1), np.ones((1, 1))
    mask = np.zeros_like(image, dtype=bool)
    mask[4:6, 4:6] = True
    monkeypatch.setattr(case, "load_lens_data", lambda **kw: (image, noise, psf, None))
    monkeypatch.setattr(case, "_position_likelihood_from_stage_a", lambda *a: None)
    monkeypatch.setattr(case, "run_stage_b", lambda *a, **kw: mask)
    monkeypatch.setattr(case, "arc_mask_from", lambda *a, **kw: mask)
    monkeypatch.setattr(case, "plot_model_results", lambda *a, **kw: None)
    reports = {}

    def observe_fit(tag, likelihood, medians, names, path):
        result, diagnostics = likelihood.evaluate_with_diagnostics(
            jnp.asarray([medians[name] for name in names])
        )
        assert diagnostics.valid and diagnostics.evidence_valid
        reports[tag] = (
            likelihood.source_model.interpolation, result, diagnostics.regularization_scale,
            diagnostics.evidence_logdet,
        )
        Path(path).touch()

    monkeypatch.setattr(case, "_plot_pix_stage", observe_fit)

    class InterruptedAtM0(Exception):
        pass

    def stop_after_m0(*args, **kwargs):
        raise InterruptedAtM0

    with monkeypatch.context() as interrupted:
        interrupted.setattr(case, "run_stage_m1", stop_after_m0)
        with pytest.raises(InterruptedAtM0):
            case.main(source_interpolation=requested)
    mode = requested or "bilinear"
    early = case._load_stage("m0")["extra"]
    assert early["source_interpolation"] == mode
    assert "s1" not in early
    assert not (tmp_path / "stage_m1.pkl").exists()
    assert reports["stage-M0"][0] == mode

    stages = ("a", "m0", "m1", "m2", "m3", "m4", "m5")
    case.main(skip_done=True)
    assert {report[0] for report in reports.values()} == {mode}
    assert set(reports) == {f"stage-M{n}" for n in range(6)}
    source = case._load_stage("m1")["extra"]["s1"]
    assert source["source_interpolation"] == mode
    assert np.any(source["source_pixels"] != 0)
    assert source["adapt_image"].shape == image.shape
    np.testing.assert_allclose(
        source["adapt_image"], reports["stage-M1"][1].model_image, rtol=2e-5, atol=1e-7,
    )
    for tag, (_, result, scale, score_method) in reports.items():
        np.testing.assert_allclose(result.source_bbox, source["source_bbox"])
        assert score_method == "overlap"
        if tag in ("stage-M2", "stage-M3", "stage-M4", "stage-M5"):
            assert np.any(scale > 1.0)
    first_reports = reports.copy()
    for tag in stages:
        path = tmp_path / f"stage_{tag}.pkl"
        payload = pickle.loads(path.read_bytes())
        assert payload["extra"]["source_interpolation"] == mode
        if tag != "a":
            assert payload["extra"]["objective"]["evidence_logdet"] == "overlap"
    m4 = case._load_stage("m4")["extra"]
    m5 = case._load_stage("m5")["extra"]
    assert m4["pipeline_schema"] == m5["pipeline_schema"]
    for key in ("reference_id", "adaptation", "grid", "observation", "fixed_strengths"):
        assert m4["objective"][key] == m5["objective"][key]
    assert m4["objective"]["adaptation"]["recipe"] == "native_source_arc_v1"
    null_priors = {prior["name"]: prior for prior in m4["objective"]["priors"]}
    candidate_priors = {prior["name"]: prior for prior in m5["objective"]["priors"]}
    assert null_priors == {name: candidate_priors[name] for name in null_priors}

    # Rebuild diagnostics from cached fits and the original S1 intensities.
    reports.clear()
    for tag in stages[1:]:
        (tmp_path / f"stage_{tag}_model.png").unlink()
    with monkeypatch.context() as restored:
        restored.setattr(
            case, "_run_sampler", lambda *a, **kw: pytest.fail("reused fit was sampled"),
        )
        case.main(skip_done=True)
    assert reports.keys() == first_reports.keys()
    for tag, (resumed_mode, result, scale, _) in reports.items():
        assert resumed_mode == mode
        np.testing.assert_allclose(
            result.model_image, first_reports[tag][1].model_image, rtol=2e-5, atol=1e-7
        )
        np.testing.assert_allclose(scale, first_reports[tag][2], rtol=2e-5)

    m5_output = tmp_path / "m5-only"
    # The null evidence also applies to a deliberately different candidate-position prior.
    monkeypatch.setattr(case, "SUBHALO_POSITION_MODE", "lensing_square")
    reference_candidate = tmp_path / "stage_m5.pkl"
    if requested is None:
        saved_candidate = reference_candidate.read_bytes()
        reference_candidate.unlink()
    summary = case.run_m5_from_cache(tmp_path, m5_output)
    assert ("reference_log_z_m5" in summary) is (requested is not None)
    if requested is None:
        reference_candidate.write_bytes(saved_candidate)
    assert summary["source_interpolation"] == mode
    assert summary["reference_m4_schema_matches"] is True
    assert summary["evidence_logdet"] == {"m4": "overlap", "m5": "overlap"}
    assert np.isfinite(summary["delta_log_z_m5_m4"])
    assert case._load_stage("m5")["extra"]["source_interpolation"] == mode
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        case.run_m5_from_cache(tmp_path, m5_output)

    if requested is None:
        m4_path = tmp_path / "stage_m4.pkl"
        stale = pickle.loads(m4_path.read_bytes())
        # A fixed-template checkpoint had no native-arc objective record.
        stale["extra"].pop("objective")
        m4_path.write_bytes(pickle.dumps(stale))
    else:
        # A valid Block null uses the same input arc but a different determinant.
        with monkeypatch.context() as block_null:
            block_null.setattr(case, "OUT_DIR", tmp_path)
            m2, m3 = (stage_from_payload(case._load_stage(tag)) for tag in ("m2", "m3"))
            case.run_stage_m4(
                image, noise, psf, mask, case._load_stage("a")["extra"]["lens_light_model"],
                m2, m3, None, source, source_interpolation=mode, evidence_logdet="block",
                upstream=m4["objective"]["upstream"],
            )
            assert case._load_stage("m4")["extra"]["objective"]["evidence_logdet"] == "block"
    rejected = tmp_path / "m5-rejected"
    with pytest.raises(ValueError, match="M4.*refit"):
        case.run_m5_from_cache(tmp_path, rejected)
    assert not rejected.exists()

    # A matched refit in the full pipeline must replace the stale null.
    monkeypatch.setattr(case, "SUBHALO_POSITION_MODE", "gaussian_local")
    case.main(skip_done=True, out_dir=tmp_path)
    repaired = case._load_stage("m4")["extra"]
    assert repaired["objective"]["evidence_logdet"] == "overlap"
    assert repaired["pipeline_schema"] == case._load_stage("m5")["extra"]["pipeline_schema"]


@pytest.mark.parametrize("stage", ["m0", "m1", "m4", "m5"])
def test_j0946_rejects_incompatible_comparison_caches_before_reuse(
    examples_dir, monkeypatch, tmp_path, stage,
):
    case = load_case(
        examples_dir, "pipe/subhalo_0946/model.py", monkeypatch, tmp_path,
    )
    extra = {"source_interpolation": "cubic_hermite"}
    if stage == "m1":
        # A legacy package cannot accompany a Hermite stage checkpoint.
        extra["s1"] = {}
    case._dump_stage(stage, None, None, [], -2.0, extra=extra)
    monkeypatch.setattr(
        case, "load_lens_data", lambda **kw: pytest.fail("conflict must precede data and evidence reuse"),
    )
    requested = None if stage == "m1" else "bilinear"
    match = rf"stage_{stage}\.pkl.*Start a fresh run"
    with pytest.raises(ValueError, match=match):
        case.main(skip_done=True, source_interpolation=requested)
    output = tmp_path / "m5-only"
    with pytest.raises(ValueError, match=match):
        case.run_m5_from_cache(tmp_path, output, source_interpolation=requested)
    assert not output.exists()


@pytest.mark.parametrize("family", ["pix_src_demo", "pix_src_demo_operator"])
@pytest.mark.parametrize("case", ["pix_src_lens_light", "pix_src_lens_light_mge"])
@pytest.mark.parametrize("mode", [None, "cubic_hermite"], ids=["default", "hermite"])
def test_simple_joint_exports_source_model_choice(
    examples_dir, monkeypatch, tmp_path, family, case, mode,
):
    import ast
    import gzip
    import os

    from TinyLensGpu.Inference import ParamU
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel
    from TinyLensGpu.example_support import source_interpolation_metadata

    # Exercise the constructor and persisted posterior of the top-level
    # script without running its full simulated lens analysis.
    path = examples_dir / family / "simple" / case / "fit_joint.py"
    nodes = ast.parse(path.read_text()).body
    model = next(node for node in nodes if (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pix_src" for target in node.targets)
    ))
    export = next(node for node in nodes if (
        isinstance(node, ast.With) and "fit_results.pkl.gz" in ast.unparse(node.items[0].context_expr)
    ))
    if mode is None:
        monkeypatch.delenv("TINYLENS_SOURCE_INTERPOLATION", raising=False)
    else:
        monkeypatch.setenv("TINYLENS_SOURCE_INTERPOLATION", mode)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    namespace = dict(
        os=os, jnp=jnp, gzip=gzip, pickle=pickle,
        PixelizedSourceModel=PixelizedSourceModel, ParamU=ParamU,
        source_interpolation_metadata=source_interpolation_metadata,
        samples=np.zeros((1, 1)), weights=np.ones(1),
        log_z=-1.0, param_names=["theta_E"],
    )
    code = ast.Module(body=[model, export], type_ignores=[])
    exec(compile(code, str(path), "exec"), namespace)
    assert namespace["pix_src"].interpolation == (mode or "bilinear")
    with gzip.open(tmp_path / "output" / "fit_results.pkl.gz", "rb") as stream:
        saved = pickle.load(stream)
    assert saved["source_interpolation"] == (mode or "bilinear")
    assert saved["source_boundary"] == (
        "C1_cardinal_bicubic_fixed_zero_exterior_nodes" if mode else "hard_zero_outside_bbox"
    )
    assert saved["param_names"] == ["theta_E"]


@pytest.mark.parametrize('relative,batch,schema', [
    ('pipe/simult_fit_lens_source/mge_lens_light/model.py', 7, False),
    ('pipe/subhalo_0946/model.py', 100, True),
    ('pipe/subhalo_B1938/model.py', 100, True),
])
def test_joint_sampling_and_cache_contract(examples_dir, monkeypatch, tmp_path,
                                          relative, batch, schema):
    case = load_case(examples_dir, relative, monkeypatch, tmp_path)
    instances = []

    class Sampler:
        log_z = -8.25
        n_like = 13
        n_eff = 3.0

        def __init__(self, prior, loglike, n_dim, **kwargs):
            self.theta = np.asarray(prior(np.full(n_dim, 0.5)))
            self.kwargs = kwargs
            self.loglike = loglike
            instances.append(self)

        def run(self, **kwargs):
            self.run_kwargs = kwargs

        def posterior(self):
            return np.tile(self.theta, (3, 1)), np.log([0.2, 0.3, 0.5]), np.zeros(3)

    monkeypatch.setattr(case, 'Sampler', Sampler)
    likelihood = _small_likelihood()
    stage = case._run_sampler(likelihood, 7, 11, 'stage-M2')
    assert isinstance(stage, StagePosterior)
    assert stage.param_names == ['theta_E']
    np.testing.assert_allclose(stage.samples, [[0.125]] * 3, atol=1e-7)
    np.testing.assert_allclose(stage.weights, [0.2, 0.3, 0.5], atol=1e-7)
    sampler = instances[-1]
    assert sampler.kwargs['n_batch'] == batch
    assert sampler.kwargs['n_live'] == 7
    assert sampler.run_kwargs == {'verbose': True, 'n_eff': 11}
    values = np.asarray(sampler.loglike(np.tile([0.125], (11, 1))))
    assert values.shape == (11,)
    assert np.isfinite(values).all()
    if 'B1938' in relative:
        assert sampler.kwargs['seed'] == 1938 + sum(map(ord, 'm2')) - ord('a')
        assert sampler.kwargs['filepath'] == str(tmp_path / 'stage_m2_sampler.h5')
        assert sampler.kwargs['resume'] is case.RESUME_SAMPLERS
        metrics = json.loads((tmp_path / 'stage_m2_sampling.json').read_text())
        assert metrics == dict(n_live=7, requested_n_eff=11, actual_n_eff=3.0,
                               n_like=13, log_z=-8.25)
    kwargs = {'stage': stage} if schema else {}
    case._dump_stage('m2', stage.samples, stage.weights, stage.param_names, stage.log_z,
                     extra={'medians': stage.medians()}, **kwargs)
    payload = case._load_stage('m2')
    assert set(payload) == {'samples', 'weights', 'param_names', 'log_z', 'extra'} | (
        {'stage_schema'} if schema else set())
    restored = stage_from_payload(payload)
    np.testing.assert_array_equal(restored.samples, stage.samples)
    assert restored.log_z == stage.log_z
    assert restored.param_names == stage.param_names
    payload.pop('stage_schema', None)
    with (tmp_path / 'stage_legacy.pkl').open('wb') as handle:
        pickle.dump(payload, handle)
    legacy = stage_from_payload(case._load_stage('legacy'))
    np.testing.assert_array_equal(legacy.samples, stage.samples)


def test_b1938_protected_m5_and_schema_rejection(examples_dir, monkeypatch, tmp_path):
    case = load_case(examples_dir, 'pipe/subhalo_B1938/model.py', monkeypatch, tmp_path)
    cache = tmp_path / 'cache'
    output = tmp_path / 'rerun'
    cache.mkdir()
    output.mkdir()
    with pytest.raises(ValueError, match='must differ'):
        case.run_m5_from_cache(str(cache), str(cache))
    protected = output / 'stage_m5.pkl'
    protected.write_bytes(b'reference science result')
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        case.run_m5_from_cache(str(cache), str(output))
    assert protected.read_bytes() == b'reference science result'
    protected.unlink()
    for tag in ('a', 'm1', 'm2', 'm3', 'm4', 'm5'):
        with (cache / f'stage_{tag}.pkl').open('wb') as handle:
            pickle.dump({'extra': {'pipeline_schema': 'old' if tag == 'm4' else 'new'}}, handle)
    with pytest.raises(ValueError, match='different comparison schemas'):
        case.run_m5_from_cache(str(cache), str(output))
    assert not protected.exists()
    assert not list(output.iterdir())


@pytest.mark.parametrize('relative', ['pipe/subhalo_0946/model.py',
                                     'pipe/subhalo_B1938/model.py'])
def test_subhalo_source_package_normalization_and_identity(
    examples_dir, monkeypatch, tmp_path, relative,
):
    case = load_case(examples_dir, relative, monkeypatch, tmp_path)
    monkeypatch.setattr(case, 'NSRC', 3)
    package = dict(source_pixels=np.arange(1, 10, dtype=float),
                   source_bbox=(-0.2, 0.2, -0.2, 0.2), n=3,
                   source_x_axis=np.linspace(-0.2, 0.2, 3),
                   source_y_axis=np.linspace(-0.2, 0.2, 3),
                   lambda_best=1.0, log_lambda_best=0.0)
    if relative == "pipe/subhalo_0946/model.py":
        with pytest.raises(ValueError, match="missing verifiable reference provenance"):
            case._fixed_source_kwargs(package)
        return
    validated = case._validate_source_package(package)
    assert validated is package
    assert package['scale_map'].dtype == np.float32
    assert package['scale_map'].shape == (9,)
    assert np.isfinite(package['scale_map']).all()
    supplied = np.full(9, 2.0, dtype=np.float64)
    package['scale_map'] = supplied
    assert case._validate_source_package(package) is package
    np.testing.assert_array_equal(package['scale_map'], np.full(9, 2.0, dtype=np.float32))
    fixed = case._fixed_source_kwargs(package)
    assert fixed['fixed_source_bbox'] == package['source_bbox']
    np.testing.assert_array_equal(fixed['fixed_reg_template'], package['source_pixels'])
    package['nx'] = 3
    with pytest.raises(KeyError, match='legacy source-grid keys'):
        case._validate_source_package(package)


def test_joint_pipeline_renders_actual_components_with_unavailable_dof(
    examples_dir, monkeypatch, tmp_path,
):
    import matplotlib.pyplot as plt
    from test_operator_diagnostics_plotting import (
        _small_joint_operator, _capture_saved_figures,
    )

    case = load_case(examples_dir,
                     'pipe/simult_fit_lens_source/mge_lens_light/model.py',
                     monkeypatch, tmp_path)
    likelihood, medians, theta = _small_joint_operator(positions=True)
    result, diagnostics = likelihood.evaluate_with_diagnostics(
        theta, include_components=True, include_effective_dof=True,
    )
    assert diagnostics.valid
    assert diagnostics.reduced_chi2 is None
    figures = _capture_saved_figures(monkeypatch)
    path = tmp_path / 'joint.png'
    try:
        case._plot_stage_b('stage-B', likelihood, medians, list(medians), str(path))
        assert path.stat().st_size > 0
        # Repeated float32 PCG solves can differ by about 1e-7 in source intensities.
        images = [ax.images[0].get_array() for ax in figures[0].axes if ax.images]
        np.testing.assert_allclose(images[1], result.model_image, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(images[3], np.asarray(result.source_pixels).reshape(5, 5),
                                   rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(images[4], diagnostics.source_image, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(images[5], diagnostics.lens_light_image, rtol=1e-5, atol=1e-7)
        assert np.any(np.asarray(images[5]) != 0)
        assert any('unavailable' in ax.get_title() for ax in figures[0].axes)
    finally:
        plt.close('all')
