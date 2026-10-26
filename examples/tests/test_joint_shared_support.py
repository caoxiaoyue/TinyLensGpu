"""Joint/subhalo support keeps sampling controls and stage artifact contracts."""
import importlib.util
import json
import pickle

import numpy as np
import pytest

from TinyLensGpu.Inference import StagePosterior
from TinyLensGpu.example_support import stage_from_payload
from test_pixelized_support import _small_likelihood


def load_case(examples_dir, relative, monkeypatch, tmp_path):
    path = examples_dir / 'pix_src_demo_operator' / relative
    spec = importlib.util.spec_from_file_location('joint_support_case', path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as import_context:
        import_context.chdir(path.parent)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'OUT_DIR', tmp_path)
    return module


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
