"""Survey entry points retain stage artifacts while using common support."""
import importlib.util
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.Inference import StagePosterior

PIPE = Path(__file__).resolve().parents[1] / 'pix_src_demo_operator' / 'pipe'
CASES = (
    'slacs_examples/pipeline.py', 'euclid_examples/pipeline.py',
    'pre_subtract_lens/pix_src_pipe/model.py',
    'pre_subtract_lens/pix_src_pipe_sersic/model.py',
    'pre_subtract_lens/pix_src_pipe_mge/model.py',
    'pre_subtract_lens/pix_src_pipe_mge/model_split.py',
)


def load_case(relative_path, monkeypatch):
    monkeypatch.chdir(PIPE)
    spec = importlib.util.spec_from_file_location('survey_' + relative_path.replace('/', '_'), PIPE / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ControlledSampler:
    log_z = -12.0
    instances = []

    def __init__(self, prior, loglike, *, n_dim, **kwargs):
        self.prior, self.n_dim, self.kwargs = prior, n_dim, kwargs
        self.instances.append(self)

    def run(self, **kwargs):
        self.run_kwargs = kwargs

    def posterior(self):
        samples = np.tile(np.asarray(self.prior(jnp.full(self.n_dim, 0.5))), (3, 1))
        return samples, np.log([0.2, 0.3, 0.5]), np.zeros(3)


@pytest.mark.parametrize("recorded", [None, "cubic_hermite"], ids=["legacy", "hermite"])
def test_slacs_rejects_interpolation_conflict_before_stage_reuse(recorded, tmp_path, monkeypatch):
    from TinyLensGpu.example_support import StageStorageConfig, save_stage

    module = load_case(CASES[0], monkeypatch)
    config = StageStorageConfig(tmp_path, include_schema=True)
    extra = {"log_lambda_best": 0.0}
    if recorded is not None:
        extra["source_interpolation"] = recorded
    save_stage(config, "m0", None, None, ["log_lambda_reg"], -12.0, extra=extra)

    def unexpected_load(**kwargs):
        pytest.fail("the interpolation conflict must precede data and stage reuse")

    monkeypatch.setattr(module, "load_lens_data", unexpected_load)
    requested = "cubic_hermite" if recorded is None else "bilinear"
    with pytest.raises(ValueError, match=r"stage_m0\.pkl.*Start a fresh run"):
        module.main(skip_done=True, out_dir=tmp_path, source_interpolation=requested)


def test_slacs_checks_source_package_provenance_before_restore(tmp_path, monkeypatch):
    from TinyLensGpu.example_support import StageStorageConfig, save_stage

    module = load_case(CASES[0], monkeypatch)
    storage = StageStorageConfig(tmp_path, include_schema=True)
    # An explicitly Hermite checkpoint cannot point to a legacy bilinear S1.
    save_stage(
        storage, "m1", None, None, ["log_lambda_reg"], -12.0,
        extra={"source_interpolation": "cubic_hermite", "s1": {}},
    )
    with pytest.raises(ValueError, match="S1 source package.*Start a fresh run"):
        module.main(skip_done=True, out_dir=tmp_path)


@pytest.mark.parametrize("case,late_tag", [(CASES[1], "m3"), (CASES[4], "m3"), (CASES[5], "m2")])
def test_survey_rejects_late_stage_conflicts_before_any_restore(
    case, late_tag, tmp_path, monkeypatch,
):
    module = load_case(case, monkeypatch)
    monkeypatch.setattr(module, "OUT_DIR", tmp_path)
    module._dump_stage("a", None, None, ["theta_E"], 0.0)
    module._dump_stage(
        late_tag, None, None, ["log_lambda_reg"], -12.0,
        source_interpolation="cubic_hermite",
    )

    def unexpected_load(**kwargs):
        pytest.fail("all checkpoint modes must be checked before loading data")

    monkeypatch.setattr(module, "load_lens_data", unexpected_load)
    with pytest.raises(ValueError, match=rf"stage_{late_tag}\.pkl.*Start a fresh run"):
        module.main(skip_done=True)


@pytest.mark.parametrize("requested", [None, "cubic_hermite"], ids=["legacy", "hermite"])
def test_slacs_saves_and_resumes_one_source_model(requested, tmp_path, monkeypatch):
    module = load_case(CASES[0], monkeypatch)
    for key, value in dict(
        Sampler=ControlledSampler, NSRC=3, NSUB=1, NSUB_PIX=1,
        N_GAUSSIANS_SRC=2, N_GAUSSIANS_LENS=2, DPIX=0.08,
        PIXEL_REGULARIZATION_TYPE="zero-order",
    ).items():
        monkeypatch.setattr(module, key, value)
    y, x = np.indices((8, 8))
    image = 0.06 + 0.08 * np.exp(-((x - 6) ** 2 + (y - 4) ** 2) / 3)
    noise, psf = np.full_like(image, 0.1), np.array([[1.0]])
    feature_mask = np.zeros_like(image, dtype=bool)
    feature_mask[3:5, 3:5] = True
    monkeypatch.setattr(module, "load_lens_data", lambda **kw: (image, noise, psf, None))
    monkeypatch.setattr(module, "run_stage_b", lambda *a, **kw: feature_mask)
    monkeypatch.setattr(module, "_position_likelihood_from_stage_a", lambda *a: None)
    monkeypatch.setattr(module, "plot_model_results", lambda *a, **kw: None)
    reports = {}

    def reconstruct(tag, likelihood, medians, names, path):
        result, diagnostics = likelihood.evaluate_with_diagnostics(
            jnp.asarray([medians[name] for name in names])
        )
        assert diagnostics.valid
        reports[tag] = (likelihood.source_model.interpolation, result)
        Path(path).touch()

    monkeypatch.setattr(module, "_plot_pix_stage", reconstruct)
    seed = None
    if requested is not None:
        # A saved pre-feature source supplies nodal intensities for a fresh
        # Hermite M0 solve while its interpolation provenance stays bilinear.
        seed = dict(
            n=3, source_pixels=np.linspace(0.1, 0.9, 9),
            source_bbox=(-0.55, 0.55, -0.55, 0.55),
            source_x_axis=np.linspace(-0.55, 0.55, 3),
            source_y_axis=np.linspace(-0.55, 0.55, 3),
        )
    module.main(out_dir=tmp_path, source_interpolation=requested, initial_source=seed)
    fresh_reports = reports.copy()
    assert set(fresh_reports) == {"stage-M0", "stage-M1", "stage-M2", "stage-M3"}
    mode = requested or "bilinear"
    boundary = (
        "C1_cardinal_bicubic_fixed_zero_exterior_nodes"
        if requested else "hard_zero_outside_bbox"
    )
    stages = ("a", "l", "m0", "m1", "m2", "m3")
    for tag in stages:
        path = tmp_path / f"stage_{tag}.pkl"
        payload = pickle.loads(path.read_bytes())
        extra = payload["extra"]
        assert extra["source_interpolation"] == mode
        assert extra["source_boundary"] == boundary
        if tag.startswith("m"):
            assert extra["objective"]["evidence_logdet"] == "overlap"
        if tag == "m0":
            assert "s1" not in extra
            if seed is not None:
                np.testing.assert_array_equal(extra["initial_source"]["source_pixels"], seed["source_pixels"])
                np.testing.assert_allclose(fresh_reports["stage-M0"][1].source_bbox, seed["source_bbox"])
        if tag == "m1":
            s1 = extra["s1"]
            assert s1["adapt_image"].shape == image.shape
            assert s1["source_image"].shape == (3, 3)
            assert s1["source_interpolation"] == mode
            assert s1["source_boundary"] == boundary
            np.testing.assert_allclose(s1["source_bbox"], fresh_reports["stage-M2"][1].source_bbox)

    if requested:
        # Reopen the run at a checkpoint which has M0 evidence and no S1.
        for tag in ("m1", "m2", "m3"):
            (tmp_path / f"stage_{tag}.pkl").unlink()
        reports.clear()
        module.main(skip_done=True, out_dir=tmp_path)
        assert {value[0] for value in reports.values()} == {mode}
        assert module._load_stage("m1")["extra"]["s1"]["source_interpolation"] == mode

    original = {tag: (tmp_path / f"stage_{tag}.pkl").read_bytes() for tag in stages}
    for tag in ("m0", "m1", "m2", "m3"):
        (tmp_path / f"stage_{tag}_model.png").unlink()
    reports.clear()
    module.main(skip_done=True, out_dir=tmp_path)
    for tag in stages:
        assert (tmp_path / f"stage_{tag}.pkl").read_bytes() == original[tag]
    assert reports.keys() == fresh_reports.keys()
    for tag, (saved_mode, resumed) in reports.items():
        assert saved_mode == fresh_reports[tag][0] == mode
        fresh = fresh_reports[tag][1]
        np.testing.assert_allclose(resumed.source_bbox, fresh.source_bbox)
        np.testing.assert_allclose(resumed.model_image, fresh.model_image, rtol=2e-5, atol=1e-7)

    # A stage fitted under the previous objective must be sampled again.
    m2_path = tmp_path / "stage_m2.pkl"
    old_m2 = pickle.loads(m2_path.read_bytes())
    old_m2["extra"].pop("objective")
    m2_path.write_bytes(pickle.dumps(old_m2))
    runs_before = len(ControlledSampler.instances)
    module.main(skip_done=True, out_dir=tmp_path)
    assert len(ControlledSampler.instances) > runs_before
    assert "objective" in module._load_stage("m2")["extra"]


@pytest.mark.parametrize('case', CASES)
def test_survey_stage_a_artifact_and_recovery(case, tmp_path, monkeypatch):
    module = load_case(case, monkeypatch)
    ControlledSampler.instances.clear()
    monkeypatch.setattr(module, 'Sampler', ControlledSampler)
    monkeypatch.setattr(module, 'OUT_DIR', tmp_path)
    monkeypatch.setattr(module, 'NSUB', 1)
    for name in ('N_GAUSSIANS_SRC', 'N_GAUSSIANS_LENS'):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, 2)
    monkeypatch.setattr(module, 'plot_model_results', lambda *a, **kw: None)
    image, noise = jnp.full((8, 8), 0.1), jnp.full((8, 8), 0.2)
    psf = jnp.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    returned = module.run_stage_a(image, noise, psf)
    with (tmp_path / 'stage_a.pkl').open('rb') as stream:
        payload = pickle.load(stream)
    schema_case = not case.endswith('model_split.py')
    assert set(payload) == {'samples', 'weights', 'param_names', 'log_z', 'extra'} | ({'stage_schema'} if schema_case else set())
    assert payload['samples'].dtype == np.float64
    assert payload['weights'].dtype == np.float64
    np.testing.assert_allclose(payload['weights'], [0.2, 0.3, 0.5])
    assert payload['log_z'] == -12.0
    assert payload['extra']['lens_light_model'].shape == (8, 8)
    sampler = ControlledSampler.instances[-1]
    assert sampler.kwargs['n_batch'] == 100
    assert sampler.kwargs['vectorized'] is True
    assert sampler.run_kwargs['n_eff'] > 0
    if schema_case:
        assert isinstance(returned[0], StagePosterior)
        restored = module._stage_from_payload(module._load_stage('a'))
        assert list(restored.param_names) == payload['param_names']
        assert restored.medians() == payload['extra']['medians']
    else:
        assert returned[2] == payload['param_names']
        np.testing.assert_array_equal(returned[0], payload['samples'])


@pytest.mark.parametrize('case', CASES[:-1])
def test_survey_template_validation_keeps_identity(case, monkeypatch):
    module = load_case(case, monkeypatch)
    monkeypatch.setattr(module, 'NSRC', 3)
    package = dict(source_pixels=np.arange(9, dtype=np.float64),
                   source_bbox=(-1., 1., -1., 1.), n=3,
                   source_x_axis=np.linspace(-1, 1, 3), source_y_axis=np.linspace(-1, 1, 3),
                   lambda_best=2., log_lambda_best=np.log(2.))
    assert module._validate_source_package(package) is package
    assert 'scale_map' not in package
    package['nx'] = 3
    with pytest.raises(KeyError, match='legacy'):
        module._validate_source_package(package)


@pytest.mark.parametrize('case', CASES[:-1])
@pytest.mark.parametrize("interpolation", [None, "cubic_hermite"], ids=["legacy", "hermite"])
def test_survey_source_transition_and_render_data(case, tmp_path, monkeypatch, interpolation):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from TinyLensGpu.example_support import export_source_package

    module = load_case(case, monkeypatch)
    for key, value in dict(NSRC=3, DPIX=0.08, NSUB_PIX=1, SOLVER_TYPE='pcg').items():
        monkeypatch.setattr(module, key, value)
    medians = dict(theta_E=0.17, gamma=2.1, e1_mass=0.1, e2_mass=0.03,
                   center_x_mass=0., center_y_mass=0., gamma1=0.02, gamma2=-0.01,
                   log_lambda_reg=0.)
    image, noise = np.full((10, 10), 0.05), np.full((10, 10), 0.1)
    mask = np.zeros((10, 10), dtype=bool)
    mask[:2] = True
    psf = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    a_names = ["theta_E", "e1_mass", "e2_mass", "center_x_mass", "center_y_mass", "gamma1", "gamma2"]
    a_samples = np.tile([medians[name] for name in a_names], (3, 1))
    stage_a = StagePosterior(a_samples, np.ones(3), a_names)
    producer = module.build_stage_m1_likelihood(
        image, noise, psf, mask, stage_a, None, 0.0,
        source_interpolation=interpolation,
    )
    producer_names = [param.name for param in producer.get_dynamic_params()]
    exported = export_source_package(
        producer, medians, producer_names, n=3, record_adapt_image=True,
    )
    values = dict(log_lambda_reg=0.0, adaptive_reg_rho=module.ADAPTIVE_REG_RHO)
    names = list(values)
    prepared = exported
    assert exported["source_interpolation"] == (interpolation or "bilinear")
    prepared.update(lambda_best=1.0, log_lambda_best=0.0)
    next_stage = module.build_stage_m2_likelihood(image, noise, psf, mask, medians, None, prepared)
    assert next_stage.source_model.interpolation == exported["source_interpolation"]
    result, diagnostics = next_stage.evaluate_with_diagnostics(include_effective_dof=True)
    assert diagnostics.valid
    np.testing.assert_allclose(result.source_bbox, prepared['source_bbox'])
    rendered = []
    original = Axes.imshow

    def capture(ax, array, *args, **kwargs):
        rendered.append(np.ma.asarray(array).copy())
        return original(ax, array, *args, **kwargs)

    monkeypatch.setattr(Axes, 'imshow', capture)
    # Critical-curve drawing has its own tests; capture the five data products.
    monkeypatch.setattr(module, 'overlay_critical_and_caustics', lambda **kw: None)
    out = tmp_path / 'stage_m2.png'
    module._plot_pix_stage('M2', next_stage, values, names, out)
    assert out.stat().st_size > 0
    assert len(rendered) == 5
    np.testing.assert_allclose(rendered[1][~mask], np.asarray(result.model_image)[~mask], rtol=2e-5)
    np.testing.assert_allclose(rendered[3], np.asarray(result.source_pixels).reshape(3, 3), rtol=2e-5)
    np.testing.assert_allclose(rendered[4], diagnostics.regularization_scale, rtol=2e-5)
    plt.close('all')
