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
    assert package['scale_map'].dtype == np.float32
    assert package['scale_map'].shape == (9,)
    assert np.all(package['scale_map'] > 0)
    package['nx'] = 3
    with pytest.raises(KeyError, match='legacy'):
        module._validate_source_package(package)


@pytest.mark.parametrize('case', [CASES[0], CASES[1], CASES[4]])
def test_survey_source_transition_and_render_data(case, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from TinyLensGpu.example_support import export_source_package, validate_source_package

    module = load_case(case, monkeypatch)
    for key, value in dict(NSRC=3, DPIX=0.08, NSUB_PIX=1, SOLVER_TYPE='pcg').items():
        monkeypatch.setattr(module, key, value)
    medians = dict(theta_E=0.17, gamma=2.1, e1_mass=0.1, e2_mass=0.03,
                   center_x_mass=0., center_y_mass=0., gamma1=0.02, gamma2=-0.01,
                   log_lambda_reg=0.)
    package = dict(source_pixels=np.arange(9, dtype=float), n=3,
                   source_bbox=(-0.3, 0.3, -0.3, 0.3),
                   source_x_axis=np.linspace(-0.3, 0.3, 3),
                   source_y_axis=np.linspace(-0.3, 0.3, 3),
                   lambda_best=1., log_lambda_best=0.)
    image, noise = np.full((10, 10), 0.05), np.full((10, 10), 0.1)
    mask = np.zeros((10, 10), dtype=bool)
    mask[:2] = True
    psf = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    initial = module.build_stage_m2_likelihood(image, noise, psf, mask, medians, None, package)
    names = [p.name for p in initial.get_dynamic_params()]
    values = {p.name: float(np.asarray(p.value)) for p in initial.get_dynamic_params()}
    exported = export_source_package(initial, values, names, n=3)
    exported.update(lambda_best=1., log_lambda_best=0.)
    prepared = validate_source_package(exported, expected_n=3, rho=module.ADAPTIVE_REG_RHO)
    next_stage = module.build_stage_m2_likelihood(image, noise, psf, mask, medians, None, prepared)
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
